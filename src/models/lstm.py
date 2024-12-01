import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""

    def __init__(self,
                 sequence_length: int = 60,
                 n_features: Optional[int] = None):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = StandardScaler()

    def _build_model(self) -> None:
        """Build LSTM architecture"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True,
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def _prepare_sequences(self,
                           data: pd.DataFrame) -> np.ndarray:
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data.iloc[i:i + self.sequence_length].values)
        return np.array(sequences)

    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              epochs: int = 50,
              batch_size: int = 32) -> None:
        """Train LSTM model"""
        try:
            if self.n_features is None:
                self.n_features = X.shape[1]

            # Scale features
            scaled_data = self.scaler.fit_transform(X)

            # Prepare sequences
            X_seq = self._prepare_sequences(pd.DataFrame(scaled_data))
            y_seq = y[self.sequence_length:]

            # Build and train model
            self._build_model()
            self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )

        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            # Scale features
            scaled_data = self.scaler.transform(X)

            # Prepare sequences
            X_seq = self._prepare_sequences(pd.DataFrame(scaled_data))

            # Make predictions
            predictions = self.model.predict(X_seq)

            # Add NaN for the initial sequence_length timestamps
            full_predictions = np.full(len(X), np.nan)
            full_predictions[self.sequence_length:] = predictions.flatten()

            return full_predictions
        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {e}")
            return np.array([])