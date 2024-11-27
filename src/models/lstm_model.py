import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from tensorflow.keras.optimizers import AdamW
from .base import BaseModel, ModelConfig, ModelMetrics
from src.utils.gpu_utils import setup_gpu, get_gpu_info
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class LSTMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Initialize with CPU as fallback
        self.device = '/CPU:0'
        self.using_gpu = False

        # Try GPU setup
        if setup_gpu():
            self.device = '/GPU:0'
            self.using_gpu = True
            self.logger.info("Using GPU for LSTM training")
        else:
            self.logger.warning("GPU setup failed, using CPU instead")

        # Model params
        self.sequence_length = config.params.get('sequence_length', 5)
        self.batch_size = config.params.get('batch_size', 16)
        self.epochs = config.params.get('epochs', 50)
        self.validation_split = config.params.get('validation_split', 0.2)
        self.learning_rate = config.params.get('learning_rate', 0.0005)
        self.weight_decay = config.params.get('weight_decay', 0.0001)

        # Initialize components
        self.scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        self.model = None
        self.history = None

        # Clear any existing sessions
        tf.keras.backend.clear_session()

    def build_model(self, input_shape: tuple) -> Sequential:
        with tf.device('/CPU:0'):  # Force CPU to avoid CUDA issues
            model = Sequential()
            # Use Input layer first to avoid warning
            model.add(tf.keras.layers.Input(shape=input_shape))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        return model

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            self.logger.info(f"Preprocessing data of shape: {df.shape}")

            # Select features and clean
            feature_data = df[self.config.features].copy().astype('float32')
            feature_data = feature_data.ffill().bfill()

            # Remove inf values
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.dropna()

            # Scale data
            self.scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            features_scaled = self.scaler.fit_transform(feature_data)

            # Create sequences
            total_samples = len(features_scaled) - self.sequence_length
            X = np.zeros((total_samples, self.sequence_length, len(self.config.features)))
            y = np.zeros(total_samples)

            for i in range(total_samples):
                X[i] = features_scaled[i:(i + self.sequence_length)]
                y[i] = feature_data.iloc[i + self.sequence_length][self.config.target]

            # Validate
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("NaN values in preprocessed data")

            self.logger.info(f"Preprocessed shapes - X: {X.shape}, y: {y.shape}")
            return X, y

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        try:
            self.logger.info("Starting LSTM model training...")
            start_time = pd.Timestamp.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Build model
            self.model = self.build_model((self.sequence_length, X.shape[2]))
            self.logger.info(self.model.summary())

            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    mode='min',
                    min_lr=1e-6
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f'logs/tensorboard/lstm_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}',
                    histogram_freq=1
                )
            ]

            # Train model with proper error handling
            try:
                self.history = self.model.fit(
                    X, y,
                    validation_split=self.validation_split,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                if self.using_gpu:
                    self.logger.warning("GPU error encountered, falling back to CPU...")
                    self.device = '/CPU:0'
                    self.using_gpu = False
                    # Retry training on CPU
                    return self.train(df)
                else:
                    raise

            self.training_time = (pd.Timestamp.now() - start_time).total_seconds()
            self._log_training_results()

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
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
            with tf.device(self.device):
                predictions_scaled = self.model.predict(
                    X,
                    batch_size=self.batch_size,
                    verbose=0
                )

            # Inverse transform predictions
            predictions_full = np.zeros((len(predictions_scaled), features_scaled.shape[1]))
            predictions_full[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(predictions_full)[:, 0]

            # Pad beginning with NaN values
            full_predictions = np.full(len(df), np.nan)
            full_predictions[self.sequence_length:] = predictions

            return full_predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def _log_training_results(self):
        """Log comprehensive training results"""
        if not self.history:
            return

        self.logger.info("\nTraining Results:")
        self.logger.info(f"Training Time: {self.training_time:.2f} seconds")
        self.logger.info(f"Final Loss: {self.history.history['loss'][-1]:.4f}")
        self.logger.info(f"Final Val Loss: {self.history.history['val_loss'][-1]:.4f}")

        # Log best epoch
        best_epoch = np.argmin(self.history.history['val_loss'])
        self.logger.info(f"Best Epoch: {best_epoch + 1}")
        self.logger.info(f"Best Val Loss: {self.history.history['val_loss'][best_epoch]:.4f}")

        # Log metrics
        for metric in ['mse', 'mae']:
            if metric in self.history.history:
                self.logger.info(f"Final {metric.upper()}: {self.history.history[metric][-1]:.4f}")
                self.logger.info(f"Final Val {metric.upper()}: {self.history.history[f'val_{metric}'][-1]:.4f}")

    def save(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)

            # Save model
            model_path = f"{path}/model.keras"
            self.model.save(model_path)

            # Save scaler
            scaler_path = f"{path}/scaler.pkl"
            joblib.dump(self.scaler, scaler_path)

            # Save training history and metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'batch_size': self.batch_size,
                'features': self.config.features,
                'training_time': self.training_time,
                'device': self.device,
                'using_gpu': self.using_gpu,
                'history': self.history.history if self.history else None,
                'model_config': {
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    'validation_split': self.validation_split
                }
            }
            joblib.dump(metadata, f"{path}/metadata.pkl")

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        try:
            model_path = f"{path}/model.keras"
            scaler_path = f"{path}/scaler.pkl"
            metadata_path = f"{path}/metadata.pkl"

            if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
                raise FileNotFoundError(f"Model files not found in {path}")

            # Load model with proper compile
            self.model = load_model(model_path, compile=False)
            optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
            )
            self.model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mse', 'mae']
            )

            # Load scaler and metadata
            self.scaler = joblib.load(scaler_path)
            metadata = joblib.load(metadata_path)

            # Restore configuration
            self.sequence_length = metadata['sequence_length']
            self.batch_size = metadata['batch_size']
            self.config.features = metadata['features']
            self.training_time = metadata['training_time']
            self.device = metadata['device']
            self.using_gpu = metadata['using_gpu']

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def validate(self, df: pd.DataFrame) -> ModelMetrics:
        """Validate model performance"""
        try:
            predictions = self.predict(df)
            actuals = df[self.config.target].values

            # Calculate directional accuracy
            pred_direction = np.diff(predictions[~np.isnan(predictions)]) > 0
            true_direction = np.diff(actuals[~np.isnan(predictions)]) > 0
            directional_accuracy = np.mean(pred_direction == true_direction)

            # Remove NaN values for other metrics
            mask = ~np.isnan(predictions) & ~np.isnan(actuals)
            predictions = predictions[mask]
            actuals = actuals[mask]

            metrics = ModelMetrics(
                mse=float(np.mean((predictions - actuals) ** 2)),
                rmse=float(np.sqrt(np.mean((predictions - actuals) ** 2))),
                mae=float(np.mean(np.abs(predictions - actuals))),
                mape=float(np.mean(np.abs((predictions - actuals) / actuals)) * 100),
                directional_accuracy=float(directional_accuracy),
                training_time=self.training_time or 0.0,
                timestamp=pd.Timestamp.now()
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise