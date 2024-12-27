import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
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
                self.model = Sequential()

                # Input layer
                self.model.add(LSTM(
                    units=self.lstm_units[0],
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=l2(0.01),
                    input_shape=(self.sequence_length, self.n_features)
                ))
                self.model.add(BatchNormalization())

                # Additional LSTM layers
                for units in self.lstm_units[1:]:
                    self.model.add(LSTM(
                        units=units,
                        return_sequences=False,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        kernel_regularizer=l2(0.01)
                    ))
                    self.model.add(BatchNormalization())
                    self.model.add(Dropout(self.dropout_rate))

                # Dense layers
                self.model.add(Dense(
                    units=max(16, self.lstm_units[-1]//2),
                    activation='relu',
                    kernel_regularizer=l2(0.01)
                ))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(self.dropout_rate))

                # Output layer
                self.model.add(Dense(units=1, activation='sigmoid'))

                # Compile model with static learning rate
                optimizer = Adam(learning_rate=self.learning_rate)

                self.model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            self.model = None

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2, epochs: int = 100) -> None:
        """Train LSTM model with advanced features"""
        try:
            if self.n_features is None:
                self.n_features = X.shape[2]

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
                        'best_model.weights.h5',
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True
                    )
                ]

                # Train model
                history = self.model.fit(
                    X, y,
                    validation_split=validation_split,
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                return np.zeros(len(X))

            # Make predictions
            predictions = self.model.predict(X, batch_size=self.batch_size, verbose=0)

            # Handle NaN values
            predictions = np.nan_to_num(predictions, nan=0.5)

            # Flatten predictions
            return predictions.flatten()

        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on gradient analysis"""
        # LSTM models do not have straightforward feature importances.
        # Advanced techniques like SHAP or Integrated Gradients are recommended.
        # Here, we return an empty dictionary.
        return {}
