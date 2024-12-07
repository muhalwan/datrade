import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class LSTMModel(BaseModel):
    def __init__(self,
                 sequence_length: int = 30,
                 n_features: Optional[int] = None,
                 lstm_units: List[int] = [32, 16],
                 dropout_rate: float = 0.3,
                 recurrent_dropout: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 32):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scaler = StandardScaler()

        self.model_config = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'recurrent_dropout': recurrent_dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }

        with tf.device('/CPU:0'):
            self.model = None

    def _build_model(self) -> None:
        """Build LSTM model architecture with advanced features"""
        try:
            with tf.device('/CPU:0'):
                self.model = Sequential([
                    # Input layer
                    Input(shape=(self.sequence_length, self.n_features)),

                    # First LSTM layer
                    LSTM(
                        units=self.lstm_units[0],
                        return_sequences=True,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    ),
                    BatchNormalization(),

                    # Additional LSTM layers
                    *[
                        tf.keras.Sequential([
                            LSTM(
                                units=units,
                                return_sequences=i < len(self.lstm_units) - 2,
                                dropout=self.dropout_rate,
                                recurrent_dropout=self.recurrent_dropout,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)
                            ),
                            BatchNormalization(),
                            Dropout(self.dropout_rate)
                        ])
                        for i, units in enumerate(self.lstm_units[1:])
                    ],

                    # Dense layers
                    Dense(
                        units=max(16, self.lstm_units[-1]//2),
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    ),
                    BatchNormalization(),
                    Dropout(self.dropout_rate),

                    # Output layer
                    Dense(units=1, activation='sigmoid')
                ])

                # Custom learning rate schedule
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.learning_rate,
                    decay_steps=1000,
                    decay_rate=0.9
                )

                self.model.compile(
                    optimizer=Adam(learning_rate=lr_schedule),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            self.model = None

    def _prepare_sequences(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare sequences for LSTM input"""
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)

            # Create sequences
            sequences = []
            for i in range(len(X_scaled) - self.sequence_length):
                sequences.append(X_scaled[i:i + self.sequence_length])

            return np.array(sequences)
        except Exception as e:
            self.logger.error(f"Error preparing sequences: {e}")
            return np.array([])

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, epochs: int = 100) -> None:
        """Train LSTM model with advanced features"""
        try:
            if self.n_features is None:
                self.n_features = X.shape[1]

            # Prepare sequences
            X_seq = self._prepare_sequences(X)
            y_seq = y[self.sequence_length:].values

            # Create validation data
            val_split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:val_split_idx], X_seq[val_split_idx:]
            y_train, y_val = y_seq[:val_split_idx], y_seq[val_split_idx:]

            # Build model
            self._build_model()

            if self.model is not None:
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    ),
                    ModelCheckpoint(
                        'best_model.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True
                    )
                ]

                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=0
                )

                # Store training history
                self.training_history = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy']
                }

        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            self.model = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                return np.zeros(len(X))

            # Prepare sequences
            X_seq = self._prepare_sequences(X)

            if len(X_seq) == 0:
                return np.zeros(len(X))

            # Make predictions
            predictions = self.model.predict(X_seq, batch_size=self.batch_size, verbose=0)

            # Handle NaN values
            predictions = np.nan_to_num(predictions, nan=0.5)

            # Create full predictions array
            full_predictions = np.zeros(len(X))
            full_predictions[self.sequence_length:] = predictions.flatten()

            return full_predictions

        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on gradient analysis"""
        try:
            if self.model is None:
                return {}

            # Get layer weights
            layer_weights = [layer.get_weights()[0] for layer in self.model.layers if len(layer.get_weights()) > 0]

            # Calculate importance based on weight magnitudes
            if layer_weights:
                importance = np.abs(layer_weights[0]).mean(axis=(0, 1))
                importance = importance / importance.sum()

                return dict(enumerate(importance))

            return {}

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}