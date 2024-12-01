import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict

from .base import BaseModel, ModelConfig

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models = {}
        self.weights = config.params.get('weights', {})
        self.logger = logging.getLogger(__name__)

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight

    def preprocess(self, df: pd.DataFrame):
        """Not used in ensemble model"""
        return df

    def train(self, df: pd.DataFrame) -> None:
        """Train all models in the ensemble"""
        try:
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model...")
                model.train(df)

        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with ensemble"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(df)

            # Normalize weights
            total_weight = sum(self.weights.values())
            normalized_weights = {
                name: weight/total_weight
                for name, weight in self.weights.items()
            }

            # Calculate weighted average
            weighted_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
            for name, pred in predictions.items():
                weighted_pred += pred * normalized_weights[name]

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save ensemble model"""
        try:
            # Save individual models
            for name, model in self.models.items():
                model_path = f"{path}/{name}"
                model.save(model_path)

            # Save weights
            weights_path = f"{path}/ensemble_weights.pkl"
            joblib.dump(self.weights, weights_path)

        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load ensemble model"""
        try:
            # Load individual models
            for name, model in self.models.items():
                model_path = f"{path}/{name}"
                model.load(model_path)

            # Load weights
            weights_path = f"{path}/ensemble_weights.pkl"
            self.weights = joblib.load(weights_path)

        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            raise