import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .base import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for sequence prediction"""

    def __init__(self, config: Dict = None):
        super().__init__("lstm", config or {})
        self.sequence_length = self.config.get('sequence_length', 60)
        self.scaler = MinMaxScaler()

        # Default configurations
        self.layers = self.config.get('layers', [128, 64, 32])
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)

    def _build_model(self, input_shape: Tuple):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM"""
        try:
            # Scale features
            scaled_data = self.scaler.fit_transform(data)

            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])  # First column is target (price)

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train LSTM model"""
        try:
            if self.model is None:
                self.model = self._build_model((X.shape[1], X.shape[2]))

            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                validation_split=0.2,
                verbose=1
            )

            self.is_trained = True

            return {
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'mae': history.history['mae'][-1]
            }

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            predictions = self.model.predict(X)

            # Inverse transform predictions
            if hasattr(self.scaler, 'scale_'):
                predictions = predictions * self.scaler.scale_[0]

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise