import logging
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib

from .base import BaseModel, ModelConfig

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Adjust sequence length based on data size
        self.sequence_length = config.params.get('sequence_length', 3)  # Reduced from 60
        self.batch_size = config.params.get('batch_size', 2)  # Reduced from 32
        self.epochs = config.params.get('epochs', 20)  # Reduced from 50
        self.validation_split = config.params.get('validation_split', 0.1)  # Reduced from 0.2
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        try:
            # Ensure we have enough data points
            if len(data) <= self.sequence_length:
                self.logger.warning(f"Data length ({len(data)}) is less than sequence length ({self.sequence_length})")
                # Adjust sequence length to be 1/3 of data length
                self.sequence_length = max(2, len(data) // 3)
                self.logger.info(f"Adjusted sequence length to {self.sequence_length}")

            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length, 0])  # Predict close price

            X = np.array(X)
            y = np.array(y)

            self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model"""
        try:
            self.logger.info("Starting LSTM model training...")
            self.logger.info(f"Input data shape: {df.shape}")

            # Handle small datasets
            if len(df) < 10:  # Minimum data requirement
                raise ValueError(f"Insufficient data points ({len(df)}) for training. Need at least 10.")

            X, y = self.preprocess(df)

            # Adjust batch size if necessary
            if len(X) < self.batch_size:
                self.batch_size = max(1, len(X) // 2)
                self.logger.warning(f"Adjusted batch size to {self.batch_size}")

            # Create a simpler model for small datasets
            model = Sequential([
                LSTM(32, return_sequences=True,
                     input_shape=(self.sequence_length, len(self.config.features))),
                Dropout(0.1),
                LSTM(16, return_sequences=False),
                Dense(8),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            self.logger.info("\nModel Architecture:")
            model.summary(print_fn=self.logger.info)

            history = model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=1
            )

            self.model = model

            self.logger.info("\nTraining completed:")
            self.logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
            if 'val_loss' in history.history:
                self.logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise