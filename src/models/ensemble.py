from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging
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
        try:
            predictions = {}
            working_models = 0

            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    if len(pred) == len(X):
                        predictions[name] = pred
                        working_models += 1
                except Exception as e:
                    self.logger.warning(f"Model {name} failed to predict: {e}")

            if working_models == 0:
                return np.zeros(len(X))

            # Recalculate weights for working models
            working_weights = {k: self.weights[k] for k in predictions.keys()}
            weight_sum = sum(working_weights.values())
            if weight_sum > 0:
                working_weights = {k: v/weight_sum for k, v in working_weights.items()}

            # Calculate weighted average
            weighted_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weighted_pred += pred * working_weights[name]

            return (weighted_pred > 0.5).astype(int)

        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {e}")
            return np.zeros(len(X))

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