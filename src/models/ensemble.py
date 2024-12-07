# src/models/xgboost_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model with advanced features"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__("xgboost")

        self.params = params or {
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
        }

        self.num_boost_rounds = 1000
        self.early_stopping_rounds = 50
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> None:
        """Train XGBoost model with validation"""
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train = X_scaled[:split_idx]
            y_train = y[:split_idx]
            X_val = X_scaled[split_idx:]
            y_val = y[split_idx:]

            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

            # Setup evaluation list
            evallist = [(dtrain, 'train'), (dval, 'eval')]

            # Train model
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_rounds,
                evals=evallist,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )

            # Store training history
            self.training_history = {
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score
            }

        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
            self.model = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        try:
            if self.model is None:
                return np.zeros(len(X))

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Create DMatrix
            dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)

            # Make predictions
            return self.model.predict(dtest)

        except Exception as e:
            self.logger.error(f"Error making XGBoost predictions: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            if self.model is None:
                return {}

            # Get importance scores
            scores = self.model.get_score(importance_type='gain')

            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}

            return scores

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}

# src/models/ensemble.py

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import logging
from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from ..features.selector import FeatureSelector

class EnhancedEnsemble(BaseModel):
    """Enhanced ensemble model with adaptive features"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("enhanced_ensemble")

        self.config = config or {
            'lstm': {
                'sequence_length': 30,
                'lstm_units': [32, 16],
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.3,
                'learning_rate': 0.001
            },
            'xgboost': {
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1
            }
        }

        # Initialize models
        self.models = {
            'lstm': LSTMModel(**self.config['lstm']),
            'xgboost': XGBoostModel(self.config['xgboost'])
        }

        # Initialize components
        self.feature_selector = FeatureSelector()
        self.scaler = StandardScaler()

        # Model weights and performance tracking
        self.weights = {'lstm': 0.4, 'xgboost': 0.6}
        self.model_performance = {}
        self.validation_metrics = {}

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Prepare and select features"""
        try:
            # Select features
            X_selected, feature_importance = self.feature_selector.select_features(X, y)

            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)

            return X_scaled, feature_importance

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return X, {}

    def _validate_model(
            self,
            model_name: str,
            model: BaseModel,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 5
    ) -> Dict[str, float]:
        """Perform time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'sharpe': [],
                'sortino': [],
                'max_drawdown': []
            }

            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # Train model
                model.train(X_train, y_train)

                # Get predictions
                val_pred = model.predict(X_val)

                # Calculate metrics
                metrics['accuracy'].append(self._calculate_accuracy(y_val, val_pred))
                metrics['precision'].append(self._calculate_precision(y_val, val_pred))
                metrics['recall'].append(self._calculate_recall(y_val, val_pred))
                metrics['f1'].append(self._calculate_f1(y_val, val_pred))
                metrics['sharpe'].append(self._calculate_sharpe(val_pred, X_val['returns']))
                metrics['sortino'].append(self._calculate_sortino(val_pred, X_val['returns']))
                metrics['max_drawdown'].append(self._calculate_max_drawdown(val_pred, X_val['returns']))

            # Average metrics
            return {k: np.mean(v) for k, v in metrics.items()}

        except Exception as e:
            self.logger.error(f"Error validating {model_name}: {e}")
            return {}

    def _adjust_weights(self, performance: Dict[str, Dict[str, float]]) -> None:
        """Adjust model weights based on performance"""
        try:
            scores = {}
            for model_name, metrics in performance.items():
                # Calculate composite score
                score = (
                        metrics.get('accuracy', 0) * 0.3 +
                        metrics.get('sharpe', 0) * 0.3 +
                        metrics.get('f1', 0) * 0.2 +
                        (1 - metrics.get('max_drawdown', 0)) * 0.2
                )
                scores[model_name] = max(0, score)

            # Normalize weights
            total_score = sum(scores.values())
            if total_score > 0:
                self.weights = {
                    name: score/total_score
                    for name, score in scores.items()
                }

        except Exception as e:
            self.logger.error(f"Error adjusting weights: {e}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train ensemble with validation"""
        try:
            self.logger.info("Starting ensemble training...")

            # Prepare data
            X_prepared, feature_importance = self._prepare_data(X, y)

            # Validate and train models
            performance = {}
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")

                # Validate model
                metrics = self._validate_model(name, model, X_prepared, y)
                performance[name] = metrics

                # Train final model
                model.train(X_prepared, y)

            # Adjust weights
            self._adjust_weights(performance)

            # Store metadata
            self.model_performance = performance
            self.feature_importance = feature_importance

            self.logger.info("Ensemble training completed")

        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with confidence scores"""
        try:
            # Prepare features
            X_prepared = self.feature_selector.transform(X)
            X_scaled = self.scaler.transform(X_prepared)
            X_scaled = pd.DataFrame(X_scaled, columns=X_prepared.columns, index=X_prepared.index)

            predictions = {}
            working_models = 0

            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)
                    if len(pred) == len(X):
                        predictions[name] = pred
                        working_models += 1
                except Exception as e:
                    self.logger.warning(f"Model {name} failed to predict: {e}")

            if working_models == 0:
                return np.zeros(len(X))

            # Calculate weighted predictions
            weighted_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weighted_pred += pred * self.weights[name]

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance"""
        try:
            importance = {}
            for name, model in self.models.items():
                model_importance = model.get_feature_importance()
                for feature, score in model_importance.items():
                    importance[feature] = importance.get(feature, 0) + score * self.weights[name]

            return importance

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}