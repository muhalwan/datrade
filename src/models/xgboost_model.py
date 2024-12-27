import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import xgboost as xgb
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
        self.model = None
        self.feature_names: List[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> None:
        """Train XGBoost model with validation"""
        try:
            # Ensure no duplicate feature names
            if len(X.columns) != len(set(X.columns)):
                raise ValueError("Duplicate feature names detected. Ensure all feature names are unique.")

            # Store feature names
            self.feature_names = X.columns.tolist()

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]

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

            # Ensure no duplicate feature names
            if len(X.columns) != len(set(X.columns)):
                raise ValueError("Duplicate feature names detected. Ensure all feature names are unique.")

            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features for prediction: {missing_features}")

            # Create DMatrix
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)

            # Make predictions
            preds = self.model.predict(dtest)

            # Convert probabilities to binary predictions
            return (preds > 0.5).astype(int)

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
                scores = {k: v / total for k, v in scores.items()}

            return scores

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
