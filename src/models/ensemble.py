import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
import os

from .base import BaseModel, ModelConfig
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel

class EnsembleModel(BaseModel):
    """Optimized ensemble model combining multiple base models"""

    def __init__(self, config: ModelConfig):
        """Initialize ensemble with configuration"""
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.weights = config.params.get('weights', {})
        self.logger = logging.getLogger(__name__)
        self.model_metrics = {}
        self.prediction_history = []

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Common preprocessing for ensemble predictions"""
        try:
            features = df[self.config.features].values
            target = df[self.config.target].values if self.config.target in df else None

            # Handle missing values
            if np.isnan(features).any():
                features = pd.DataFrame(features, columns=self.config.features).ffill().bfill().values

            return features, target

        except Exception as e:
            self.logger.error(f"Error in ensemble preprocessing: {str(e)}")
            raise

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble with validation"""
        try:
            if not isinstance(model, BaseModel):
                raise ValueError(f"Model {name} must be a BaseModel instance")

            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for model {name} must be a non-negative number")

            self.models[name] = model
            self.weights[name] = weight
            self.normalize_weights()

            self.logger.info(f"Added model {name} with weight {weight:.3f}")
            self.logger.info(f"Current weights: {self.weights}")

        except Exception as e:
            self.logger.error(f"Error adding model {name}: {str(e)}")
            raise

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble"""
        try:
            if name not in self.models:
                raise KeyError(f"Model {name} not found in ensemble")

            del self.models[name]
            del self.weights[name]
            self.normalize_weights()

            self.logger.info(f"Removed model {name}")
            self.logger.info(f"Current weights: {self.weights}")

        except Exception as e:
            self.logger.error(f"Error removing model {name}: {str(e)}")
            raise

    def normalize_weights(self) -> None:
        """Normalize model weights to sum to 1"""
        try:
            total = sum(self.weights.values())
            if total <= 0:
                raise ValueError("Sum of weights must be positive")

            self.weights = {
                name: weight/total
                for name, weight in self.weights.items()
            }

        except Exception as e:
            self.logger.error(f"Error normalizing weights: {str(e)}")
            raise

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model weights with validation"""
        try:
            # Validate models exist
            missing_models = set(new_weights.keys()) - set(self.models.keys())
            if missing_models:
                raise ValueError(f"Models not found: {missing_models}")

            # Validate weights are non-negative
            if any(w < 0 for w in new_weights.values()):
                raise ValueError("Weights must be non-negative")

            self.weights = new_weights
            self.normalize_weights()

            self.logger.info("Updated ensemble weights")
            for name, weight in self.weights.items():
                self.logger.info(f"{name}: {weight:.3f}")

        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train ensemble (trains individual models if needed)"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            for name, model in self.models.items():
                if not hasattr(model, 'model') or model.model is None:
                    self.logger.info(f"Training {name} model...")
                    model.train(df)
                    self.model_metrics[name] = {'train_time': datetime.now()}

        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions with error handling"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                try:
                    start_time = datetime.now()
                    pred = model.predict(df)
                    pred_time = (datetime.now() - start_time).total_seconds()

                    predictions[name] = pred
                    self.model_metrics[name] = {
                        'prediction_time': pred_time,
                        'last_prediction': datetime.now()
                    }
                except Exception as e:
                    self.logger.error(f"Error getting predictions from {name}: {str(e)}")
                    continue

            if not predictions:
                raise ValueError("No valid predictions from any model")

            # Calculate weighted average
            weighted_pred = np.zeros_like(next(iter(predictions.values())))
            weights_used = {}

            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                weighted_pred += pred * weight
                weights_used[name] = weight

            # Store prediction info
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'predictions': predictions,
                'weights_used': weights_used,
                'final_prediction': weighted_pred
            })

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {str(e)}")
            raise

    def get_model_contributions(self) -> Dict[str, float]:
        """Calculate each model's contribution to recent predictions"""
        try:
            if not self.prediction_history:
                return {}

            recent_pred = self.prediction_history[-1]
            contributions = {}

            for name, pred in recent_pred['predictions'].items():
                weight = recent_pred['weights_used'].get(name, 0)
                contribution = np.mean(np.abs(pred * weight)) / np.mean(np.abs(recent_pred['final_prediction']))
                contributions[name] = contribution

            return contributions

        except Exception as e:
            self.logger.error(f"Error calculating model contributions: {str(e)}")
            return {}

    def save(self, path: str) -> None:
        """Save ensemble model and configuration"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save individual models
            for name, model in self.models.items():
                model_path = os.path.join(path, name)
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)

            # Save ensemble configuration
            config = {
                'weights': self.weights,
                'metrics': self.model_metrics,
                'model_configs': {
                    name: model.config.__dict__
                    for name, model in self.models.items()
                }
            }

            config_path = os.path.join(path, 'ensemble_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, default=str)

            # Save prediction history (last 100 predictions)
            history_path = os.path.join(path, 'prediction_history.pkl')
            recent_history = self.prediction_history[-100:] if self.prediction_history else []
            joblib.dump(recent_history, history_path)

            self.logger.info(f"Ensemble saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving ensemble: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load ensemble model and configuration"""
        try:
            # Load configuration
            config_path = os.path.join(path, 'ensemble_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)

            self.weights = config['weights']
            self.model_metrics = config['metrics']

            # Load individual models
            for name, model_config in config['model_configs'].items():
                model_path = os.path.join(path, name)

                # Create appropriate model instance
                if model_config['type'] == 'LSTM':
                    model = LSTMModel(ModelConfig(**model_config))
                elif model_config['type'] == 'LIGHTGBM':
                    model = LightGBMModel(ModelConfig(**model_config))
                else:
                    raise ValueError(f"Unknown model type: {model_config['type']}")

                # Load model
                model.load(model_path)
                self.models[name] = model

            # Load prediction history
            history_path = os.path.join(path, 'prediction_history.pkl')
            if os.path.exists(history_path):
                self.prediction_history = joblib.load(history_path)

            self.logger.info(f"Ensemble loaded from {path}")
            self.logger.info(f"Loaded models: {list(self.models.keys())}")

        except Exception as e:
            self.logger.error(f"Error loading ensemble: {str(e)}")
            raise

    def get_metrics(self) -> Dict:
        """Get current model metrics"""
        return {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'metrics': self.model_metrics,
            'contributions': self.get_model_contributions()
        }