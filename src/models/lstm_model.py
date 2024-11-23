import logging
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

from .base import BaseModel, ModelConfig

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sequence_length = config.params.get('sequence_length', 60)
        self.batch_size = config.params.get('batch_size', 32)
        self.epochs = config.params.get('epochs', 50)
        self.validation_split = config.params.get('validation_split', 0.2)
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM"""
        # Select features and scale
        features = df[self.config.features].values

        # Initialize and fit scaler on first call
        if self.scaler is None:
            self.scaler = tf.keras.preprocessing.MinMaxScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # Create sequences
        X, y = self._create_sequences(features_scaled)
        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model"""
        try:
            X, y = self.preprocess(df)

            model = Sequential([
                LSTM(128, return_sequences=True,
                     input_shape=(self.sequence_length, len(self.config.features))),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=1
            )

            self.model = model

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            X, _ = self.preprocess(df)
            predictions_scaled = self.model.predict(X)

            # Inverse transform predictions
            predictions = np.zeros((len(predictions_scaled), len(self.config.features)))
            predictions[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(predictions)[:, 0]

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save LSTM model"""
        try:
            model_path = f"{path}/lstm_model"
            scaler_path = f"{path}/lstm_scaler.pkl"

            save_model(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load LSTM model"""
        try:
            model_path = f"{path}/lstm_model"
            scaler_path = f"{path}/lstm_scaler.pkl"

            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            raise