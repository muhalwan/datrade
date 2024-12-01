import numpy as np
import pandas as pd
from typing import Optional, Dict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model for classification"""
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("xgboost")
        # Remove n_estimators from params and use num_boost_round in training
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        self.num_boost_rounds = 100
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost model"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            dtrain = xgb.DMatrix(X_scaled, label=y)
            # Use num_boost_round instead of n_estimators
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_rounds
            )
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)

            # Create DMatrix
            dtest = xgb.DMatrix(X_scaled)

            # Make predictions
            return self.model.predict(dtest)
        except Exception as e:
            self.logger.error(f"Error making XGBoost predictions: {e}")
            return np.array([])