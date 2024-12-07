import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class LSTMModel(BaseModel):
    def __init__(self,
                 sequence_length: int = 30,
                 n_features: Optional[int] = None,
                 lstm_units: List[int] = [32, 16],
                 dropout_rate: float = 0.3,
                 recurrent_dropout: float = 0.3):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.scaler = StandardScaler()
        with tf.device('/CPU:0'):
            self.model = None

    def _build_model(self) -> None:
        """Build the LSTM model architecture"""
        try:
            with tf.device('/CPU:0'):
                self.model = Sequential()

                # First LSTM layer
                self.model.add(Input(shape=(self.sequence_length, self.n_features)))
                self.model.add(LSTM(
                    units=self.lstm_units[0],
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout
                ))

                # Additional LSTM layers
                for units in self.lstm_units[1:]:
                    self.model.add(LSTM(
                        units=units,
                        return_sequences=False,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout
                    ))
                    self.model.add(Dropout(self.dropout_rate))

                # Dense layers
                self.model.add(Dense(units=max(16, self.lstm_units[-1]//2), activation='relu'))
                self.model.add(Dropout(self.dropout_rate))
                self.model.add(Dense(units=1, activation='sigmoid'))

                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series, epochs: int = 50, batch_size: int = 32) -> None:
        """Train LSTM model"""
        try:
            if self.n_features is None:
                self.n_features = X.shape[1]

            # Scale features
            scaled_data = self.scaler.fit_transform(X)

            # Prepare sequences
            X_seq = np.array([scaled_data[i:i + self.sequence_length]
                              for i in range(len(scaled_data) - self.sequence_length)])
            y_seq = y[self.sequence_length:].values

            # Build and train model
            self._build_model()
            if self.model is not None:
                # Add early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )

                # Train model
                self.model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )

        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            self.model = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                return np.array([])

            # Scale features
            scaled_data = self.scaler.transform(X)

            # Prepare sequences
            X_seq = np.array([scaled_data[i:i + self.sequence_length]
                              for i in range(len(scaled_data) - self.sequence_length)])

            if len(X_seq) == 0:
                return np.array([])

            # Make predictions
            predictions = self.model.predict(X_seq, verbose=0)

            # Add NaN for the initial sequence_length timestamps
            full_predictions = np.full(len(X), np.nan)
            full_predictions[self.sequence_length:] = predictions.flatten()

            return full_predictions

        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {e}")
            return np.array([])

    def save(self, path: str) -> bool:
        """Save LSTM model and scaler"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            # Save keras model with correct extension
            keras_path = f"{path}_keras.keras"  # Add .keras extension
            self.model.save(keras_path)

            # Save scaler
            import joblib
            joblib.dump(self.scaler, f"{path}_scaler.pkl")

            return True
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load LSTM model and scaler"""
        try:
            # Load keras model with correct extension
            keras_path = f"{path}_keras.keras"  # Add .keras extension
            self.model = tf.keras.models.load_model(keras_path)

            # Load scaler
            import joblib
            self.scaler = joblib.load(f"{path}_scaler.pkl")

            return True
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            return False