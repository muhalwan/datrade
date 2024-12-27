import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging

from .base import BaseModel

class LSTMModel(BaseModel):
    """
    LSTM-based Neural Network Model for Binary Classification in Trading.
    """

    def __init__(
            self,
            sequence_length: int = 30,
            n_features: Optional[int] = None,
            lstm_units: List[int] = [32, 16],
            dropout_rate: float = 0.3,
            # Set recurrent_dropout to 0.0 to enable cuDNN kernels
            recurrent_dropout: float = 0.0,
            learning_rate: float = 0.001,
            batch_size: int = 32
    ):
        """
        Initializes the LSTMModel with specified hyperparameters.

        Args:
            sequence_length (int): Number of time steps in each input sequence.
            n_features (Optional[int]): Number of features per time step.
            lstm_units (List[int]): List containing the number of units for each LSTM layer.
            dropout_rate (float): Dropout rate for input dropout.
            recurrent_dropout (float): Recurrent dropout rate for LSTM layers. Set to 0.0 to enable cuDNN.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
        """
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout  # Set to 0.0 to enable cuDNN
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.training_history = {}
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """
        Builds the LSTM model architecture.
        """
        self.model = Sequential()

        # Add the first LSTM layer with input shape
        self.model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,  # Return sequences for stacking LSTM layers
            dropout=self.dropout_rate,
            # Remove or set recurrent_dropout to 0.0 to enable cuDNN
            recurrent_dropout=self.recurrent_dropout,
            kernel_regularizer=l2(0.01),
            input_shape=(self.sequence_length, self.n_features)
        ))

        # Add additional LSTM layers if specified
        for units in self.lstm_units[1:]:
            # Set return_sequences=True for all but the last LSTM layer
            return_seq = units != self.lstm_units[-1]
            self.model.add(LSTM(
                units=units,
                return_sequences=return_seq,
                dropout=self.dropout_rate,
                # Remove or set recurrent_dropout to 0.0 to enable cuDNN
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l2(0.01)
            ))

        # Add the output layer
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model with Adam optimizer and binary crossentropy loss
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.logger.info("LSTM model built successfully with cuDNN-optimized kernels.")

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            validation_split: float = 0.2,
            epochs: int = 100
    ) -> None:
        """
        Trains the LSTM model on the provided data.

        Args:
            X (np.ndarray): Input feature sequences of shape (samples, sequence_length, n_features).
            y (np.ndarray): Binary target labels of shape (samples,).
            validation_split (float): Fraction of data to use for validation.
            epochs (int): Number of training epochs.
        """
        try:
            if self.n_features is None:
                # Determine n_features from the input data shape
                self.n_features = X.shape[2] if len(X.shape) == 3 else 1
                self.logger.debug(f"Number of features set to {self.n_features} based on input data.")

            self._build_model()

            if self.model is not None:
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

                self.logger.info("Starting model training...")
                history = self.model.fit(
                    X, y,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=1  # Set to 1 for progress output, change to 0 to suppress
                )

                self.training_history = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy']
                }

                self.logger.info("LSTM model training completed successfully.")

        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            self.model = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions using the trained LSTM model.

        Args:
            X (np.ndarray): Input feature sequences of shape (samples, sequence_length, n_features).

        Returns:
            np.ndarray: Flattened prediction probabilities.
        """
        try:
            if self.model is None:
                self.logger.warning("LSTM model is not trained. Returning zeros.")
                return np.zeros(len(X))

            preds = self.model.predict(X, batch_size=self.batch_size, verbose=0)
            preds = np.nan_to_num(preds, nan=0.5)
            return preds.flatten()

        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance scores. (Not implemented for LSTM)

        Returns:
            Dict[str, float]: Empty dictionary as LSTM models do not provide feature importances.
        """
        return {}
