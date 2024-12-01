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
import os

from .base import BaseModel, ModelConfig

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sequence_length = config.params.get('sequence_length', 3)
        self.batch_size = config.params.get('batch_size', 2)
        self.epochs = config.params.get('epochs', 20)
        self.validation_split = config.params.get('validation_split', 0.1)
        self.scaler = MinMaxScaler()
        self.model = None
        self.logger = logging.getLogger(__name__)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        try:
            if len(data) <= self.sequence_length:
                self.logger.warning(f"Data length ({len(data)}) is less than sequence length ({self.sequence_length})")
                self.sequence_length = max(2, len(data) // 3)
                self.logger.info(f"Adjusted sequence length to {self.sequence_length}")

            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length, 0])

            X = np.array(X)
            y = np.array(y)

            self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM"""
        try:
            self.logger.info(f"Preprocessing data of shape: {df.shape}")

            # Select features
            feature_data = df[self.config.features].fillna(method='ffill').fillna(method='bfill')
            features = feature_data.values

            # Scale features
            if not hasattr(self.scaler, 'n_features_in_'):  # If scaler not fitted
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)

            # Create sequences
            X, y = self._create_sequences(features_scaled)

            return X, y

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model"""
        try:
            self.logger.info("Starting LSTM model training...")
            self.logger.info(f"Input data shape: {df.shape}")

            # Handle small datasets
            if len(df) < 10:
                raise ValueError(f"Insufficient data points ({len(df)}) for training. Need at least 10.")

            X, y = self.preprocess(df)

            # Adjust batch size if necessary
            if len(X) < self.batch_size:
                self.batch_size = max(1, len(X) // 2)
                self.logger.warning(f"Adjusted batch size to {self.batch_size}")

            # Create model
            model = Sequential([
                LSTM(32, return_sequences=True,
                     input_shape=(self.sequence_length, len(self.config.features))),
                Dropout(0.1),
                LSTM(16, return_sequences=False),
                Dense(8),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # Log model summary
            self.logger.info("\nModel Architecture:")
            model.summary(print_fn=self.logger.info)

            # Train model
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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            # Preprocess test data
            feature_data = df[self.config.features].fillna(method='ffill').fillna(method='bfill')
            features = feature_data.values
            features_scaled = self.scaler.transform(features)

            # Create sequences
            X = []
            for i in range(len(features_scaled) - self.sequence_length):
                X.append(features_scaled[i:(i + self.sequence_length)])
            X = np.array(X)

            # Make predictions
            predictions_scaled = self.model.predict(X)

            # Prepare for inverse transform
            pred_full = np.zeros((len(predictions_scaled), features.shape[1]))
            pred_full[:, 0] = predictions_scaled.flatten()

            # Inverse transform
            predictions = self.scaler.inverse_transform(pred_full)[:, 0]

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save LSTM model"""
        try:
            os.makedirs(path, exist_ok=True)
            model_path = f"{path}/model"
            scaler_path = f"{path}/scaler.pkl"

            # Save model and scaler
            save_model(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load LSTM model"""
        try:
            model_path = f"{path}/model"
            scaler_path = f"{path}/scaler.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Model files not found in {path}")

            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            raise