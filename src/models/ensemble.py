import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
import os
from collections import deque
import torch
import torch.nn as nn
from statistics import mean, stdev
from sklearn.preprocessing import StandardScaler
from .base import BaseModel, ModelConfig, ModelMetrics, ModelType


class AdaptiveWeights(nn.Module):
    """Neural network for adaptive weight computation"""
    def __init__(self, n_models: int, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_models),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class EnsembleModel(BaseModel):
    """Enhanced ensemble model with dynamic weighting"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.weights = config.params.get('weights', {})
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.model_metrics: Dict[str, List[ModelMetrics]] = {}
        self.prediction_history: List[Dict] = []
        self.error_history: Dict[str, deque] = {
            'mse': deque(maxlen=100),
            'mae': deque(maxlen=100),
            'directional': deque(maxlen=100)
        }

        # Dynamic weighting parameters
        self.performance_window = config.params.get('performance_window', 20)
        self.weight_update_frequency = config.params.get('weight_update_frequency', 10)
        self.min_weight = config.params.get('min_weight', 0.1)
        self.adaptive_weighting = config.params.get('adaptive_weighting', True)

        # Initialize components
        self.adaptive_network = None
        self.scaler = StandardScaler()
        self.last_weight_update = datetime.now()
        self.prediction_cache = {}

    def _initialize_adaptive_network(self, n_features: int):
        """Initialize the adaptive weighting network"""
        if self.adaptive_weighting and len(self.models) > 1:
            self.adaptive_network = AdaptiveWeights(
                n_models=len(self.models),
                n_features=n_features
            )
            self.optimizer = torch.optim.Adam(
                self.adaptive_network.parameters(),
                lr=0.001,
                weight_decay=0.0001
            )

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble"""
        try:
            if not isinstance(model, BaseModel):
                raise ValueError(f"Model {name} must be a BaseModel instance")

            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Weight for model {name} must be non-negative")

            self.models[name] = model
            self.weights[name] = weight
            self.model_metrics[name] = []
            self._normalize_weights()

            # Initialize adaptive network if needed
            if len(model.config.features) > 0:
                self._initialize_adaptive_network(len(model.config.features))

            self.logger.info(f"Added model {name} with weight {weight:.3f}")
            self.logger.info(f"Current weights: {self.weights}")

        except Exception as e:
            self.logger.error(f"Error adding model {name}: {str(e)}")
            raise

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1"""
        try:
            total = sum(self.weights.values())
            if total <= 0:
                raise ValueError("Sum of weights must be positive")

            self.weights = {
                name: max(self.min_weight, weight/total)
                for name, weight in self.weights.items()
            }

            # Re-normalize after applying min weight
            total = sum(self.weights.values())
            self.weights = {
                name: weight/total
                for name, weight in self.weights.items()
            }

        except Exception as e:
            self.logger.error(f"Error normalizing weights: {str(e)}")
            raise

    def _calculate_model_performance(self, model_name: str,
                                     predictions: np.ndarray,
                                     actuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        try:
            if len(predictions) < 2 or len(actuals) < 2:
                return {}

            # Remove NaN values
            mask = ~np.isnan(predictions) & ~np.isnan(actuals)
            predictions = predictions[mask]
            actuals = actuals[mask]

            if len(predictions) < 2:
                return {}

            # Calculate metrics
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))

            # Directional accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)

            # Calculate weighted error score
            error_score = (
                    0.4 * mse / np.std(actuals) +
                    0.3 * mae / np.mean(np.abs(actuals)) +
                    0.3 * (1 - directional_accuracy)
            )

            return {
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_accuracy,
                'error_score': error_score
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance for {model_name}: {str(e)}")
            return {}

    def _update_weights_dynamically(self, df: pd.DataFrame,
                                    recent_performances: Dict[str, Dict[str, float]]) -> None:
        """Update weights based on recent performance"""
        try:
            if not recent_performances or len(self.models) < 2:
                return

            if self.adaptive_weighting and self.adaptive_network is not None:
                # Use adaptive network for weight computation
                features = torch.FloatTensor(
                    self.scaler.transform(df[self.config.features].values)
                )
                with torch.no_grad():
                    weights = self.adaptive_network(features).mean(dim=0).numpy()
                self.weights = {
                    name: max(self.min_weight, weight)
                    for name, weight in zip(self.models.keys(), weights)
                }
            else:
                # Use performance-based weighting
                error_scores = {
                    name: perf['error_score']
                    for name, perf in recent_performances.items()
                    if 'error_score' in perf
                }

                if not error_scores:
                    return

                # Convert errors to weights
                max_error = max(error_scores.values())
                inv_errors = {
                    name: (max_error - error + 1e-6)
                    for name, error in error_scores.items()
                }

                # Calculate new weights
                total = sum(inv_errors.values())
                self.weights = {
                    name: max(self.min_weight, inv_error / total)
                    for name, inv_error in inv_errors.items()
                }

            self._normalize_weights()
            self.last_weight_update = datetime.now()

            # Log weight updates
            self.logger.info("Updated ensemble weights:")
            for name, weight in self.weights.items():
                self.logger.info(f"{name}: {weight:.3f}")

        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for ensemble"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            # Use first model's preprocessing as reference
            first_model = next(iter(self.models.values()))
            return first_model.preprocess(df)

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train the adaptive network if enabled"""
        if self.adaptive_weighting and self.adaptive_network is not None:
            try:
                # Prepare data
                features = torch.FloatTensor(
                    self.scaler.fit_transform(df[self.config.features].values)
                )
                actuals = torch.FloatTensor(df[self.config.target].values)

                # Train adaptive network
                self.adaptive_network.train()
                for epoch in range(100):
                    self.optimizer.zero_grad()
                    weights = self.adaptive_network(features)

                    # Get predictions from all models
                    model_predictions = []
                    for model in self.models.values():
                        preds = model.predict(df)
                        model_predictions.append(torch.FloatTensor(preds))

                    # Compute weighted predictions
                    ensemble_preds = sum(w * p for w, p in zip(weights.T, model_predictions))

                    # Compute loss
                    loss = nn.MSELoss()(ensemble_preds, actuals)
                    loss.backward()
                    self.optimizer.step()

                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

            except Exception as e:
                self.logger.error(f"Error training adaptive network: {str(e)}")
                self.adaptive_weighting = False

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions with dynamic weighting"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            current_time = datetime.now()
            predictions = {}
            performances = {}

            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    start_time = datetime.now()
                    pred = model.predict(df)
                    pred_time = (datetime.now() - start_time).total_seconds()

                    predictions[name] = pred
                    self.prediction_cache[name] = {
                        'predictions': pred,
                        'timestamp': current_time,
                        'prediction_time': pred_time
                    }

                    if self.config.target in df.columns:
                        perf = self._calculate_model_performance(
                            name,
                            pred[-self.performance_window:],
                            df[self.config.target].values[-self.performance_window:]
                        )
                        performances[name] = perf

                except Exception as e:
                    self.logger.error(f"Error getting predictions from {name}: {str(e)}")
                    continue

            if not predictions:
                raise ValueError("No valid predictions from any model")

            # Update weights if needed
            if (current_time - self.last_weight_update).seconds > self.weight_update_frequency:
                self._update_weights_dynamically(df, performances)

            # Calculate weighted predictions
            weighted_pred = np.zeros_like(next(iter(predictions.values())))
            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                weighted_pred += pred * weight

            # Store prediction info
            self.prediction_history.append({
                'timestamp': current_time,
                'predictions': predictions,
                'weights': self.weights.copy(),
                'final_prediction': weighted_pred,
                'performances': performances
            })

            # Maintain history length
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {str(e)}")
            raise

    def get_model_diagnostics(self) -> Dict:
        """Get comprehensive model diagnostics"""
        try:
            if not self.prediction_history:
                return {}

            recent_history = self.prediction_history[-self.performance_window:]

            diagnostics = {
                'model_weights': self.weights.copy(),
                'last_weight_update': self.last_weight_update.isoformat(),
                'model_performances': {},
                'prediction_times': {},
                'error_trends': {},
                'adaptive_network_status': {
                    'enabled': self.adaptive_weighting,
                    'initialized': self.adaptive_network is not None
                }
            }

            for name in self.models:
                model_perfs = [
                    h['performances'].get(name, {})
                    for h in recent_history
                    if 'performances' in h
                ]

                if model_perfs:
                    diagnostics['model_performances'][name] = {
                        metric: mean([p.get(metric, np.nan) for p in model_perfs])
                        for metric in ['mse', 'mae', 'directional_accuracy']
                    }

                if name in self.prediction_cache:
                    diagnostics['prediction_times'][name] = \
                        self.prediction_cache[name]['prediction_time']

            return diagnostics

        except Exception as e:
            self.logger.error(f"Error getting model diagnostics: {str(e)}")
            return {}

    def save(self, path: str) -> None:
        """Save ensemble model and its components"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save base model info
            super().save(path)

            # Save ensemble configuration
            config = {
                'weights': self.weights,
                'performance_window': self.performance_window,
                'weight_update_frequency': self.weight_update_frequency,
                'min_weight': self.min_weight,
                'adaptive_weighting': self.adaptive_weighting,
                'model_configs': {
                    name: model.config.to_dict()
                    for name, model in self.models.items()
                }
            }

            with open(os.path.join(path, 'ensemble_config.json'), 'w') as f:
                json.dump(config, f, indent=4)

            # Save prediction history
            joblib.dump(
                self.prediction_history[-1000:],
                os.path.join(path, 'prediction_history.pkl')
            )

            # Save adaptive network if exists
            if self.adaptive_network is not None:
                torch.save(
                    {
                        'state_dict': self.adaptive_network.state_dict(),
                        'scaler': self.scaler
                    },
                    os.path.join(path, 'adaptive_network.pt')
                )

            # Save component models
            for name, model in self.models.items():
                model_path = os.path.join(path, name)
                model.save(model_path)

            self.logger.info(f"Ensemble model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving ensemble: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Implement abstract load method"""
        try:
            # Load base model info
            super().load(path)

            # Load ensemble specific data
            ensemble_path = os.path.join(path, 'ensemble_config.json')
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'r') as f:
                    config = json.load(f)
                    self.weights = config.get('weights', {})
                    self.performance_window = config.get('performance_window', 20)
                    self.weight_update_frequency = config.get('weight_update_frequency', 10)

            # Load component models
            models_path = os.path.join(path, 'models')
            if os.path.exists(models_path):
                for model_dir in os.listdir(models_path):
                    model_path = os.path.join(models_path, model_dir)
                    if os.path.isdir(model_path):
                        config_path = os.path.join(model_path, 'config.json')
                        if os.path.exists(config_path):
                            with open(config_path) as f:
                                model_config = json.load(f)
                                model = self._create_model(model_config)
                                model.load(model_path)
                                self.models[model_dir] = model

        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            raise