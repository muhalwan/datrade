import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model for feature-based prediction"""

    def __init__(self, config: Dict = None):
        super().__init__("xgboost", config)
        self.scaler = StandardScaler()

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for XGBoost"""
        try:
            # Extract features and target
            y = data['close'].pct_change().shift(-1).dropna()
            X = data.iloc[:-1]  # Remove last row as it has no target

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, y.values

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost model"""
        try:
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 1000),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 6),
                min_child_weight=self.config.get('min_child_weight', 1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=42
            )

            # Train model
            self.model.fit(
                X, y,
                eval_set=[(X, y)],
                early_stopping_rounds=50,
                verbose=False
            )

            self.is_trained = True

            return {
                'best_score': self.model.best_score,
                'best_iteration': self.model.best_iteration,
                'feature_importance': dict(zip(
                    range(X.shape[1]),
                    self.model.feature_importances_
                ))
            }

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise