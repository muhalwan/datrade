import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    Bidirectional, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, TimeDistributed
)

from .base import BaseModel, ModelConfig

class LSTMModel(BaseModel):
    """Optimized LSTM model for time series prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Model parameters
        self.sequence_length = config.params.get('sequence_length', 10)
        self.batch_size = config.params.get('batch_size', 32)
        self.epochs = config.params.get('epochs', 100)
        self.validation_split = config.params.get('validation_split', 0.2)
        self.learning_rate = config.params.get('learning_rate', 0.001)
        self.weight_decay = config.params.get('weight_decay', 0.004)

        # Add tensorboard directory
        self.tensorboard_dir = Path('logs/tensorboard')
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.scaler = MinMaxScaler()
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.history = None
        self.training_time = None

    def build_model(self, input_shape: tuple) -> Sequential:
        """Build advanced LSTM architecture with attention"""
        try:
            # Input shape is (sequence_length, n_features)
            inputs = Input(shape=input_shape)

            # First Bidirectional LSTM layer
            x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Self-attention mechanism
            attention = MultiHeadAttention(
                num_heads=4,
                key_dim=32,
            )(x, x)
            x = LayerNormalization()(attention + x)  # Skip connection

            # Second Bidirectional LSTM layer
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            # Global pooling to reduce sequence dimension
            x = GlobalAveragePooling1D()(x)

            # Dense layers for final processing
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)

            # Output layer
            outputs = Dense(1)(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compile with AdamW optimizer and Huber loss
            optimizer = AdamW(
                learning_rate=self.config.params.get('learning_rate', 0.001),
                weight_decay=self.config.params.get('weight_decay', 0.004)
            )

            model.compile(
                optimizer=optimizer,
                loss='huber'
            )

            return model

        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences efficiently"""
        total_samples = len(data) - self.sequence_length

        X = np.zeros((total_samples, self.sequence_length, data.shape[1]))
        y = np.zeros(total_samples)

        for i in range(total_samples):
            X[i] = data[i:(i + self.sequence_length)]
            y[i] = data[i + self.sequence_length, 0]  # Predict next close price

        self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data efficiently"""
        try:
            self.logger.info(f"Preprocessing data of shape: {df.shape}")

            # Select and clean features
            feature_data = df[self.config.features].copy()
            feature_data = feature_data.ffill().bfill()

            # Scale features
            if not hasattr(self.scaler, 'n_features_in_'):
                features_scaled = self.scaler.fit_transform(feature_data)
            else:
                features_scaled = self.scaler.transform(feature_data)

            # Create sequences
            X, y = self._create_sequences(features_scaled)

            # Verify data
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Insufficient data for sequence creation")

            return X, y

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model with optimizations"""
        try:
            self.logger.info("Starting LSTM model training...")
            start_time = pd.Timestamp.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Adjust batch size if needed
            if len(X) < self.batch_size:
                self.batch_size = max(1, len(X) // 4)
                self.logger.warning(f"Adjusted batch size to {self.batch_size}")

            # Build model
            self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

            # Setup callbacks
            callbacks = [
                # Early stopping with restore
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    mode='min'
                ),
                # Learning rate reduction
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    mode='min',
                    verbose=1
                ),
                # Model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='models/temp/lstm_checkpoint',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=0
                ),
                # Tensorboard logging
                tf.keras.callbacks.TensorBoard(
                    log_dir=f'logs/tensorboard/lstm_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                    histogram_freq=1
                )
            ]

            # Train model with gradient clipping
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False,  # Important for time series
                clip_norm=1.0  # Gradient clipping
            )

            self.training_time = (pd.Timestamp.now() - start_time).total_seconds()

            # Log results
            self._log_training_results()

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def _log_training_results(self):
        """Log training results and metrics"""
        try:
            self.logger.info(f"Training completed: {len(self.history.history['loss'])} epochs")
            self.logger.info(f"Final loss: {self.history.history['loss'][-1]:.4f}")
            self.logger.info(f"Final val_loss: {self.history.history['val_loss'][-1]:.4f}")
            self.logger.info(f"Training time: {self.training_time:.2f} seconds")

            # Calculate additional metrics
            best_epoch = np.argmin(self.history.history['val_loss'])
            self.logger.info(f"Best epoch: {best_epoch + 1}")
            self.logger.info(f"Best val_loss: {self.history.history['val_loss'][best_epoch]:.4f}")

            # Learning rate progression
            lr_history = self.history.history.get('lr', [])
            if lr_history:
                self.logger.info(f"Initial LR: {lr_history[0]:.6f}")
                self.logger.info(f"Final LR: {lr_history[-1]:.6f}")

        except Exception as e:
            self.logger.error(f"Error logging training results: {str(e)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with proper error handling"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            # Preprocess test data
            feature_data = df[self.config.features].copy()
            feature_data = feature_data.ffill().bfill()
            features_scaled = self.scaler.transform(feature_data.values)

            # Create sequences
            X = np.array([
                features_scaled[i:(i + self.sequence_length)]
                for i in range(len(features_scaled) - self.sequence_length)
            ])

            # Make predictions
            predictions_scaled = self.model.predict(X, batch_size=self.batch_size)

            # Inverse transform predictions
            predictions_full = np.zeros((len(predictions_scaled), features_scaled.shape[1]))
            predictions_full[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(predictions_full)[:, 0]

            # Pad beginning
            full_predictions = np.full(len(df), np.nan)
            full_predictions[self.sequence_length:] = predictions

            return full_predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save model and artifacts"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save model
            model_path = f"{path}/model.keras"
            self.model.save(model_path)

            # Save scaler
            scaler_path = f"{path}/scaler.pkl"
            joblib.dump(self.scaler, scaler_path)

            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'features': self.config.features,
                'training_time': self.training_time,
                'history': self.history.history if self.history else None
            }
            joblib.dump(metadata, f"{path}/metadata.pkl")

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load model and artifacts"""
        try:
            model_path = f"{path}/model.keras"
            scaler_path = f"{path}/scaler.pkl"
            metadata_path = f"{path}/metadata.pkl"

            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                raise FileNotFoundError(f"Model files not found in {path}")

            # Load model and artifacts
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)

            # Load metadata
            metadata = joblib.load(metadata_path)
            self.sequence_length = metadata['sequence_length']
            self.batch_size = metadata['batch_size']
            self.config.features = metadata['features']
            self.training_time = metadata['training_time']

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            raise