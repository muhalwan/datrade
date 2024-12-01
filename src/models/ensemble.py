# src/models/base.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"model.{name}")
        self.model = None
        self.is_trained = False

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model"""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def save(self, path: str):
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.model, save_path)
            self.logger.info(f"Model saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str):
        """Load model from disk"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            self.model = joblib.load(load_path)
            self.is_trained = True
            self.logger.info(f"Model loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

# src/models/lstm.py

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
        super().__init__("lstm", config)
        self.sequence_length = config.get('sequence_length', 60)
        self.scaler = MinMaxScaler()

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

# src/models/xgboost_model.py

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model for feature-based prediction"""

    def __init__(self, config: Dict = None):
        super().__init__("xgboost", config)
        self.scaler = StandardScaler()

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for XGBoost"""
        try:
            # Extract features and target
            y = data['close'].pct_change().shift(-1).dropna()
            X = data.iloc[:-1]  # Remove last row as it has no target

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, y.values

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost model"""
        try:
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 1000),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 6),
                min_child_weight=self.config.get('min_child_weight', 1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=42
            )

            # Train model
            self.model.fit(
                X, y,
                eval_set=[(X, y)],
                early_stopping_rounds=50,
                verbose=False
            )

            self.is_trained = True

            return {
                'best_score': self.model.best_score,
                'best_iteration': self.model.best_iteration,
                'feature_importance': dict(zip(
                    range(X.shape[1]),
                    self.model.feature_importances_
                ))
            }

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

# src/models/ensemble.py

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel

class EnsembleModel(BaseModel):
    """Ensemble model combining LSTM and XGBoost"""

    def __init__(self, config: Dict = None):
        super().__init__("ensemble", config or {})

        # Initialize sub-models with their specific configs
        self.lstm = LSTMModel(
            config.get('lstm', {}) if config else {}
        )
        self.xgboost = XGBoostModel(
            config.get('xgboost', {}) if config else {}
        )

        # Initialize weights
        weights = config.get('weights', {}) if config else {}
        self.weights = {
            'lstm': weights.get('lstm', 0.5),
            'xgboost': weights.get('xgboost', 0.5)
        }

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for all models"""
        try:
            # Preprocess for each model
            lstm_X, lstm_y = self.lstm.preprocess(data)
            xgb_X, xgb_y = self.xgboost.preprocess(data)

            return {
                'lstm': (lstm_X, lstm_y),
                'xgboost': (xgb_X, xgb_y)
            }

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, X: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict:
        """Train all models in ensemble"""
        try:
            # Train LSTM
            lstm_metrics = self.lstm.train(X['lstm'], y['lstm'])

            # Train XGBoost
            xgb_metrics = self.xgboost.train(X['xgboost'], y['xgboost'])

            self.is_trained = True

            return {
                'lstm_metrics': lstm_metrics,
                'xgboost_metrics': xgb_metrics
            }

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            # Get predictions from each model
            lstm_pred = self.lstm.predict(X['lstm'])
            xgb_pred = self.xgboost.predict(X['xgboost'])

            # Weighted average
            ensemble_pred = (
                    self.weights['lstm'] * lstm_pred +
                    self.weights['xgboost'] * xgb_pred
            )

            return ensemble_pred

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def optimize_weights(self, X_val: Dict[str, np.ndarray],
                         y_val: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights using validation data"""
        try:
            # Get predictions from each model
            lstm_pred = self.lstm.predict(X_val['lstm'])
            xgb_pred = self.xgboost.predict(X_val['xgboost'])

            # Grid search for optimal weights
            best_score = float('inf')
            best_weights = self.weights.copy()

            for lstm_weight in np.arange(0.1, 1.0, 0.1):
                xgb_weight = 1 - lstm_weight

                ensemble_pred = lstm_weight * lstm_pred + xgb_weight * xgb_pred
                score = np.mean((ensemble_pred - y_val) ** 2)

                if score < best_score:
                    best_score = score
                    best_weights = {
                        'lstm': lstm_weight,
                        'xgboost': xgb_weight
                    }

            self.weights = best_weights
            return best_weights

        except Exception as e:
            self.logger.error(f"Weight optimization error: {str(e)}")
            raise