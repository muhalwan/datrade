import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, sequence_length: int = 60, n_features: Optional[int] = None):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = StandardScaler()
        with tf.device('/CPU:0'):
            self.model = None

    def _build_model(self) -> None:
        try:
            with tf.device('/CPU:0'):
                self.model = Sequential([
                    Input(shape=(self.sequence_length, self.n_features)),
                    LSTM(units=50, return_sequences=True),
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
                self.model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
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