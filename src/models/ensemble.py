from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from .prophet_model import ProphetModel
from src.utils.metrics import calculate_trading_metrics

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models with improved robustness"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("ensemble")
        # Initialize models with optimized parameters
        self.models = {
            'lstm': LSTMModel(
                sequence_length=30,
                lstm_units=[32, 16],
                dropout_rate=0.3,
                recurrent_dropout=0.3
            ),
            'xgboost': XGBoostModel(params={
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1.0
            }),
            'prophet': ProphetModel(params={
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 5,
                'seasonality_mode': 'multiplicative',
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': False
            })
        }

        # Initial model weights
        self.weights = weights or {
            'lstm': 0.3,
            'xgboost': 0.5,
            'prophet': 0.2
        }

        # Additional attributes for feature selection and model performance
        self.feature_importance: Dict[str, float] = {}
        self.prediction_threshold = 0.55
        self.cv_scores: Dict[str, List[float]] = {}
        self.selected_features: List[str] = []

        # Configure logger
        self.logger = logging.getLogger(__name__)

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select important features using mutual information"""
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
            selector.fit(X, y)

            # Calculate feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            self.feature_importance = feature_scores

            # Select features above mean importance
            mean_score = np.mean(list(feature_scores.values()))
            selected_features = [col for col, score in feature_scores.items()
                                 if score > mean_score]

            self.logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return list(X.columns)

    def _perform_cross_validation(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform time-series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = {model_name: [] for model_name in self.models.keys()}

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                self.logger.info(f"Training fold {fold + 1}/{n_splits}")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Get validation set close prices
                val_close = X_val['close'] if 'close' in X_val else None

                for model_name, model in self.models.items():
                    try:
                        # Train model on fold
                        model.train(X_train, y_train)
                        val_pred = model.predict(X_val)

                        # Calculate validation metrics
                        if val_close is not None:
                            val_metrics = calculate_trading_metrics(
                                y_val.values,
                                val_pred,
                                val_close.values
                            )
                            score = val_metrics.get('sharpe_ratio', 0)
                        else:
                            # Fallback to accuracy if no price data
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_val, val_pred)

                        cv_scores[model_name].append(score)
                    except Exception as e:
                        self.logger.error(f"Error in fold {fold} for model {model_name}: {e}")
                        cv_scores[model_name].append(0)

            return cv_scores

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {model_name: [0.0] for model_name in self.models.keys()}

    def _adjust_weights(self, cv_scores: Dict[str, List[float]]) -> None:
        """Adjust model weights based on cross-validation performance"""
        try:
            # Calculate mean scores for each model
            mean_scores = {
                name: np.mean(scores) if scores else 0
                for name, scores in cv_scores.items()
            }

            # Convert negative scores to zero
            mean_scores = {name: max(0, score) for name, score in mean_scores.items()}

            # Calculate total score
            total_score = sum(mean_scores.values())

            if total_score > 0:
                # Update weights based on relative performance
                self.weights = {
                    name: score / total_score
                    for name, score in mean_scores.items()
                }
            else:
                # Fallback to default weights if all scores are negative
                self.logger.warning("All models performed poorly, using default weights")

            self.logger.info(f"Adjusted model weights: {self.weights}")

        except Exception as e:
            self.logger.error(f"Error adjusting weights: {e}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train ensemble with feature selection and cross-validation"""
        try:
            self.logger.info("Starting ensemble training...")

            # Select features
            self.selected_features = self._select_features(X, y)
            X_selected = X[self.selected_features]

            # Perform cross-validation
            self.cv_scores = self._perform_cross_validation(X_selected, y)

            # Adjust model weights
            self._adjust_weights(self.cv_scores)

            # Final training on full dataset
            for name, model in self.models.items():
                self.logger.info(f"Final training of {name} model...")
                model.train(X_selected, y)

            self.logger.info("Ensemble training completed")

        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with confidence threshold"""
        try:
            # Use selected features
            if self.selected_features:
                X = X[self.selected_features]

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
                self.logger.error("All models failed to predict")
                return np.zeros(len(X))

            # Calculate weighted predictions
            weighted_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weighted_pred += pred * self.weights[name]

            # Apply confidence threshold
            return (weighted_pred > self.prediction_threshold).astype(int)

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return np.zeros(len(X))

    def save(self, path: str) -> bool:
        """Save all models and ensemble parameters"""
        try:
            # Save each model
            for name, model in self.models.items():
                model_path = f"{path}_{name}"
                if not model.save(model_path):
                    self.logger.error(f"Failed to save {name} model")
                    return False

            # Save ensemble parameters
            params = {
                'weights': self.weights,
                'feature_importance': self.feature_importance,
                'prediction_threshold': self.prediction_threshold,
                'selected_features': self.selected_features
            }

            np.save(f"{path}_params.npy", params)
            return True

        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load all models and ensemble parameters"""
        try:
            # Load each model
            for name, model in self.models.items():
                model_path = f"{path}_{name}"
                if not model.load(model_path):
                    self.logger.error(f"Failed to load {name} model")
                    return False

            # Load ensemble parameters
            params = np.load(f"{path}_params.npy", allow_pickle=True).item()
            self.weights = params['weights']
            self.feature_importance = params['feature_importance']
            self.prediction_threshold = params['prediction_threshold']
            self.selected_features = params['selected_features']

            return True

        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            return False