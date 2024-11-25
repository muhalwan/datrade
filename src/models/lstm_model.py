from pathlib import Path
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
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import AdamW
from .base import BaseModel, ModelConfig
from src.utils.gpu_utils import setup_gpu, monitor_gpu_usage, get_gpu_memory_info

def setup_gpu():
    """Configure GPU settings"""
    gpu_logger = logging.getLogger(__name__)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_logger.info(f"Found {len(gpus)} GPU(s)")
            return True
        else:
            gpu_logger.warning("No GPU found, using CPU")
            return False
    except Exception as e:
        gpu_logger.error(f"Error configuring GPU: {e}")
        return False

class LSTMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()

        # Model params
        self.sequence_length = config.params.get('sequence_length', 10)
        self.batch_size = 32
        self.epochs = config.params.get('epochs', 100)
        self.validation_split = config.params.get('validation_split', 0.2)
        self.learning_rate = config.params.get('learning_rate', 0.001)
        self.weight_decay = config.params.get('weight_decay', 0.004)

        # Dirs
        self.tensorboard_dir = Path('logs/tensorboard')
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # GPU setup
        self.using_gpu = setup_gpu()
        if self.using_gpu:
            tf.keras.backend.clear_session()
            tf.keras.backend.set_floatx('float16')
            self.batch_size = 16

    def build_model(self, input_shape: tuple) -> Sequential:
        """Build advanced LSTM architecture with attention"""
        try:
            inputs = Input(shape=input_shape)
            # Add memory optimization for smaller GPUs
            tf.keras.backend.clear_session()
            tf.keras.backend.set_floatx('float16')

            # Reduce batch size if memory is constrained
            if self.using_gpu:
                gpu_mem = get_gpu_memory_info()
                if gpu_mem and gpu_mem['total_memory'] < 6000:  # Less than 6GB
                    self.batch_size = min(self.batch_size, 32)
                    self.logger.info(f"Adjusted batch size to {self.batch_size} for GPU memory")

            # First LSTM layer with bidirectional wrapper
            x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # Self-attention mechanism
            attention = MultiHeadAttention(
                num_heads=4,
                key_dim=32
            )(x, x)
            x = LayerNormalization()(attention + x)

            # Second LSTM layer
            x = Bidirectional(LSTM(32, return_sequences=True))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            # Global pooling and dense layers
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)

            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)

            outputs = Dense(1)(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
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
        """Create sequences for LSTM"""
        total_samples = len(data) - self.sequence_length

        X = np.zeros((total_samples, self.sequence_length, data.shape[1]))
        y = np.zeros(total_samples)

        for i in range(total_samples):
            X[i] = data[i:(i + self.sequence_length)]
            y[i] = data[i + self.sequence_length, 0]

        self.logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data efficiently"""
        try:
            feature_data = df[self.config.features].astype('float32')  # Use float32
            # Scale to [-1,1] range for better stability
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            features_scaled = self.scaler.fit_transform(feature_data)
            self.logger.info(f"Preprocessing data of shape: {df.shape}")

            # Verify all required features are present
            missing_features = [f for f in self.config.features if f not in df.columns]
            if missing_features:
                self.logger.error(f"Missing features: {missing_features}")
                self.logger.info("Available features:")
                self.logger.info(", ".join(sorted(df.columns)))
                raise ValueError(f"Missing required features: {missing_features}")

            # Select and clean features
            feature_data = df[self.config.features].copy()
            feature_data = feature_data.ffill().bfill()

            # Scale features
            if not hasattr(self.scaler, 'n_featuresf_in_'):
                self.scaler = self.scaler.fit(feature_data)
            features_scaled = self.scaler.transform(feature_data)

            # Create sequences
            total_samples = len(features_scaled) - self.sequence_length
            X = np.zeros((total_samples, self.sequence_length, len(self.config.features)))
            y = np.zeros(total_samples)

            for i in range(total_samples):
                X[i] = features_scaled[i:(i + self.sequence_length)]
                y[i] = feature_data.iloc[i + self.sequence_length][self.config.target]

            return X, y

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        try:
            self.logger.info("Starting LSTM model training...")
            if self.using_gpu:
                initial_usage = monitor_gpu_usage()
                self.logger.info(f"Initial GPU memory usage: {initial_usage}")
            start_time = pd.Timestamp.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Adjust batch size if needed
            if len(X) < self.batch_size:
               self.batch_size = max(1, len(X) // 4)
               self.logger.warning(f"Adjusted batch size to {self.batch_size}")

            # Build model
            self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

            class GPUMonitorCallback(tf.keras.callbacks.Callback):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger

                def on_epoch_end(self, epoch, logs=None):
                    usage = monitor_gpu_usage()
                    if usage:
                        self.logger.info(
                            f"Epoch {epoch} - GPU Memory: {usage['memory_used_mb']}MB, "
                            f"Temp: {usage['temperature_c']}°C"
                        )

            callbacks = [
                GPUMonitorCallback(self.logger),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(Path('models/temp/lstm_checkpoint.keras')),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=0
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.tensorboard_dir / f'lstm_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'),
                    histogram_freq=1
                )
            ]

            # Train model
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False
            )

            self.training_time = (pd.Timestamp.now() - start_time).total_seconds()
            self._log_training_results()

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def _log_training_results(self):
        """Log training results"""
        try:
            self.logger.info(f"Training completed: {len(self.history.history['loss'])} epochs")
            self.logger.info(f"Final loss: {self.history.history['loss'][-1]:.4f}")
            self.logger.info(f"Final val_loss: {self.history.history['val_loss'][-1]:.4f}")
            self.logger.info(f"Training time: {self.training_time:.2f} seconds")

            best_epoch = np.argmin(self.history.history['val_loss'])
            self.logger.info(f"Best epoch: {best_epoch + 1}")
            self.logger.info(f"Best val_loss: {self.history.history['val_loss'][best_epoch]:.4f}")

            lr_history = self.history.history.get('lr', [])
            if lr_history:
                self.logger.info(f"Initial LR: {lr_history[0]:.6f}")
                self.logger.info(f"Final LR: {lr_history[-1]:.6f}")

        except Exception as e:
            self.logger.error(f"Error logging results: {str(e)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
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
            self.logger.error(f"Error saving model: {str(e)}")
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
            self.logger.error(f"Error loading model: {str(e)}")
            raise