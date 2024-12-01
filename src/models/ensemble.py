import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from .prophet_model import ProphetModel

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("ensemble")
        self.models = {
            'lstm': LSTMModel(),
            'xgboost': XGBoostModel(),
            'prophet': ProphetModel()
        }
        self.weights = weights or {
            'lstm': 0.4,
            'xgboost': 0.4,
            'prophet': 0.2
        }

    def train(self,
              X: pd.DataFrame,
              y: pd.Series) -> None:
        """Train all models in ensemble"""
        try:
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model...")
                model.train(X, y)
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions"""
        try:
            predictions = {}

            # Get predictions from each model
            for name, model in self.models.items():
                predictions[name] = model.predict(X)

            # Calculate weighted average
            weighted_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weighted_pred += pred * self.weights[name]

            # Convert to binary predictions
            return (weighted_pred > 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])

    def save(self, path: str) -> bool:
        """Save all models in ensemble"""
        try:
            for name, model in self.models.items():
                model_path = f"{path}_{name}"
                if not model.save(model_path):
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load all models in ensemble"""
        try:
            for name, model in self.models.items():
                model_path = f"{path}_{name}"
                if not model.load(model_path):
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            return False