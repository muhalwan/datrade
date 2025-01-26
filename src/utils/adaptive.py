import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ModelPerformance:
    accuracy: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    drawdown: float = 0.0
    confidence: float = 0.0

class AdaptiveLearningSystem:
    """Manages adaptive learning rates and model adjustments"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learning_rates = {
            'lstm': 0.001,
            'xgboost': 0.1
        }
        self.window_size = 100
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.confidence_thresholds: Dict[str, float] = {}
        self.model_weights: Dict[str, float] = {}
        self.position_sizes: Dict[str, float] = {}

    def adjust_learning_rate(
            self,
            model_name: str,
            current_performance: ModelPerformance,
            min_lr: float = 0.0001,
            max_lr: float = 0.1
    ) -> float:
        """Adjust learning rate based on performance"""
        try:
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []

            self.performance_history[model_name].append(current_performance)

            # Keep only recent history
            if len(self.performance_history[model_name]) > self.window_size:
                self.performance_history[model_name].pop(0)

            # Calculate performance trend
            if len(self.performance_history[model_name]) > 1:
                prev_perf = self.performance_history[model_name][-2]

                # Calculate composite score
                current_score = self._calculate_composite_score(current_performance)
                prev_score = self._calculate_composite_score(prev_perf)

                # Adjust learning rate
                if current_score > prev_score:
                    new_lr = self.learning_rates[model_name] * 1.1
                else:
                    new_lr = self.learning_rates[model_name] * 0.8

                # Clip learning rate
                new_lr = np.clip(new_lr, min_lr, max_lr)
                self.learning_rates[model_name] = new_lr

                self.logger.debug(f"Adjusted learning rate for {model_name}: {new_lr}")

            return self.learning_rates.get(model_name, 0.001)

        except Exception as e:
            self.logger.error(f"Error adjusting learning rate: {e}")
            return self.learning_rates.get(model_name, 0.001)

    def _calculate_composite_score(self, performance: ModelPerformance) -> float:
        """Calculate composite performance score"""
        try:
            weights = {
                'accuracy': 0.3,
                'sharpe': 0.3,
                'profit_factor': 0.2,
                'win_rate': 0.1,
                'drawdown': -0.1  # Negative weight for drawdown
            }

            score = (
                    performance.accuracy * weights['accuracy'] +
                    performance.sharpe * weights['sharpe'] +
                    performance.profit_factor * weights['profit_factor'] +
                    performance.win_rate * weights['win_rate'] +
                    (1 - performance.drawdown) * abs(weights['drawdown'])
            )

            return max(0, score)  # Ensure non-negative score

        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return 0.0

    def calculate_confidence(
            self,
            model_name: str,
            predictions: np.ndarray,
            features: pd.DataFrame
    ) -> np.ndarray:
        """Calculate prediction confidence scores"""
        try:
            confidence_scores = np.ones_like(predictions)

            # Get recent performance
            if model_name in self.performance_history and self.performance_history[model_name]:
                recent_perf = self.performance_history[model_name][-1]
                baseline_confidence = recent_perf.accuracy
            else:
                baseline_confidence = 0.5

            # Adjust confidence based on prediction strength
            prediction_strength = np.abs(predictions - 0.5) * 2
            confidence_scores *= prediction_strength

            # Adjust based on volatility if available
            if 'volatility_20' in features.columns:
                volatility = features['volatility_20'].values
                confidence_scores *= np.exp(-volatility)

            # Adjust based on trend strength if available
            if 'trend_strength_20' in features.columns:
                trend_strength = np.abs(features['trend_strength_20'].values)
                confidence_scores *= (1 + trend_strength)

            # Apply baseline confidence
            confidence_scores *= baseline_confidence

            # Normalize to [0, 1]
            confidence_scores = (confidence_scores - confidence_scores.min()) / (
                    confidence_scores.max() - confidence_scores.min() + 1e-10
            )

            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return np.ones_like(predictions) * 0.5

    def get_position_sizes(
            self,
            confidence_scores: np.ndarray,
            current_price: float,
            available_capital: float,
            max_position: float = 1.0,
            min_position: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Calculate position sizes based on confidence"""
        try:
            # Base position sizes on confidence
            position_sizes = confidence_scores * (max_position - min_position) + min_position

            # Apply risk-based scaling
            avg_confidence = np.mean(confidence_scores)
            metrics = {
                'average_confidence': avg_confidence,
                'max_position_size': max_position,
                'allocated_capital': 0.0
            }

            # Adjust based on confidence regime
            if avg_confidence < 0.3:  # Low confidence
                position_sizes *= 0.5
                metrics['regime'] = 'low_confidence'
            elif avg_confidence > 0.7:  # High confidence
                position_sizes *= 1.2
                metrics['regime'] = 'high_confidence'
            else:
                metrics['regime'] = 'normal'

            # Calculate capital allocation
            position_capital = position_sizes * available_capital
            metrics['allocated_capital'] = position_capital.sum()

            # Ensure maximum position size
            position_sizes = np.minimum(position_sizes, max_position)

            return position_sizes, metrics

        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            return np.ones_like(confidence_scores) * min_position, {}

    def update_model_weights(
            self,
            performance: Dict[str, ModelPerformance],
            lookback: int = 20
    ) -> None:
        """Update model weights based on recent performance"""
        try:
            scores = {}
            for model_name, perf in performance.items():
                # Get recent performance history
                history = self.performance_history.get(model_name, [])[-lookback:]
                if not history:
                    continue

                # Calculate average performance
                avg_performance = ModelPerformance(
                    accuracy=np.mean([p.accuracy for p in history]),
                    sharpe=np.mean([p.sharpe for p in history]),
                    profit_factor=np.mean([p.profit_factor for p in history]),
                    win_rate=np.mean([p.win_rate for p in history]),
                    drawdown=np.mean([p.drawdown for p in history]),
                    confidence=np.mean([p.confidence for p in history])
                )

                scores[model_name] = self._calculate_composite_score(avg_performance)

            # Normalize weights
            total_score = sum(scores.values())
            if total_score > 0:
                self.model_weights = {
                    name: score/total_score
                    for name, score in scores.items()
                }
                self.logger.debug(f"Updated model weights: {self.model_weights}")

        except Exception as e:
            self.logger.error(f"Error updating model weights: {e}")

    def reset_daily_metrics(self) -> None:
        """Reset daily tracking metrics"""
        try:
            for model_name in self.performance_history:
                # Keep only recent history
                self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]

            # Reset position sizes
            self.position_sizes = {}

        except Exception as e:
            self.logger.error(f"Error resetting metrics: {e}")
