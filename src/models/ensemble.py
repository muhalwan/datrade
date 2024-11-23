import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import datetime, timedelta
from .base import BaseModel, ModelConfig
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel

class EnsembleModel(BaseModel):
    """Enhanced ensemble model with dynamic weighting"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.weights = config.params.get('weights', {})
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.model_metrics = {}
        self.prediction_history = []
        self.performance_window = config.params.get('performance_window', 20)

        # Dynamic weighting parameters
        self.weight_update_frequency = config.params.get('weight_update_frequency', 10)
        self.min_weight = config.params.get('min_weight', 0.1)
        self.prediction_cache = {}
        self.last_weight_update = datetime.now()

        # Error tracking
        self.error_history = {
            'mse': deque(maxlen=100),
            'mae': deque(maxlen=100),
            'directional': deque(maxlen=100)
        }

    def calculate_model_performance(self, model_name: str,
                                    predictions: np.ndarray,
                                    actuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        try:
            if len(predictions) < 2 or len(actuals) < 2:
                return {}

            # Calculate basic error metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)

            # Calculate directional accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)

            # Calculate weighted error score
            # Lower is better
            error_score = (0.4 * mse / np.std(actuals) +
                           0.3 * mae / np.mean(np.abs(actuals)) +
                           0.3 * (1 - directional_accuracy))

            return {
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_accuracy,
                'error_score': error_score
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance for {model_name}: {str(e)}")
            return {}

    def update_weights_dynamically(self, recent_performances: Dict[str, Dict[str, float]]):
        """Update model weights based on recent performance"""
        try:
            if not recent_performances:
                return

            # Calculate weight adjustments based on error scores
            error_scores = {
                name: perf['error_score']
                for name, perf in recent_performances.items()
                if 'error_score' in perf
            }

            if not error_scores:
                return

            # Convert errors to weights (lower error = higher weight)
            max_error = max(error_scores.values())
            inv_errors = {
                name: (max_error - error + 1e-6)  # Add small constant to avoid division by zero
                for name, error in error_scores.items()
            }

            # Calculate new weights
            total_inv_error = sum(inv_errors.values())
            new_weights = {
                name: max(self.min_weight, inv_error / total_inv_error)
                for name, inv_error in inv_errors.items()
            }

            # Normalize weights
            total_weight = sum(new_weights.values())
            self.weights = {
                name: weight / total_weight
                for name, weight in new_weights.items()
            }

            self.last_weight_update = datetime.now()

            # Log weight updates
            self.logger.info("Updated ensemble weights:")
            for name, weight in self.weights.items():
                self.logger.info(f"{name}: {weight:.3f}")

        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced prediction with dynamic weighting"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            current_time = datetime.now()
            predictions = {}
            performances = {}

            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    start_time = time.time()
                    pred = model.predict(df)
                    pred_time = time.time() - start_time

                    predictions[name] = pred
                    self.prediction_cache[name] = {
                        'predictions': pred,
                        'timestamp': current_time,
                        'prediction_time': pred_time
                    }

                    # Calculate performance if we have actual values
                    if 'close' in df.columns:
                        perf = self.calculate_model_performance(
                            name, pred[-self.performance_window:],
                            df['close'].values[-self.performance_window:]
                        )
                        performances[name] = perf

                except Exception as e:
                    self.logger.error(f"Error getting predictions from {name}: {str(e)}")
                    continue

            if not predictions:
                raise ValueError("No valid predictions from any model")

            # Update weights if enough time has passed
            if (current_time - self.last_weight_update).seconds > self.weight_update_frequency:
                self.update_weights_dynamically(performances)

            # Calculate weighted predictions
            weighted_pred = np.zeros_like(next(iter(predictions.values())))
            weights_used = {}

            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                weighted_pred += pred * weight
                weights_used[name] = weight

            # Store prediction info
            self.prediction_history.append({
                'timestamp': current_time,
                'predictions': predictions,
                'weights_used': weights_used,
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
                'model_weights': self.weights,
                'last_weight_update': self.last_weight_update,
                'model_performances': {},
                'prediction_times': {},
                'error_trends': {}
            }

            for name in self.models:
                model_perfs = [h['performances'].get(name, {})
                               for h in recent_history
                               if 'performances' in h]

                if model_perfs:
                    diagnostics['model_performances'][name] = {
                        metric: np.mean([p.get(metric, np.nan) for p in model_perfs])
                        for metric in ['mse', 'mae', 'directional_accuracy']
                    }

                # Get prediction times
                if name in self.prediction_cache:
                    diagnostics['prediction_times'][name] = \
                        self.prediction_cache[name]['prediction_time']

            return diagnostics

        except Exception as e:
            self.logger.error(f"Error getting model diagnostics: {str(e)}")
            return {}

    def save(self, path: str) -> None:
        """Enhanced save with diagnostics"""
        try:
            super().save(path)

            # Save additional diagnostic information
            diagnostics_path = os.path.join(path, 'diagnostics.json')
            diagnostics = self.get_model_diagnostics()

            with open(diagnostics_path, 'w') as f:
                json.dump(diagnostics, f, indent=4, default=str)

            self.logger.info(f"Saved ensemble diagnostics to {diagnostics_path}")

        except Exception as e:
            self.logger.error(f"Error saving ensemble diagnostics: {str(e)}")
            raise