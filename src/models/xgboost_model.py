import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

from .base import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost Classifier Model.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initializes the XGBoostModel with specified hyperparameters.

        Args:
            params (Optional[Dict]): Hyperparameters for XGBoost.
        """
        super().__init__("xgboost")
        self.params = params if params else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series, class_weights: dict = None):
        try:
            if X.empty or len(X) == 0:
                self.logger.error("Empty training data. Aborting XGBoost training.")
                self.model = None
                return

            self.logger.info("Training XGBoost model...")
            if class_weights:
                # Ensure all classes in y are present
                present_classes = y.unique()
                for cls in present_classes:
                    if cls not in class_weights:
                        class_weights[cls] = 1.0  # Default weight
                sample_weights = y.map(class_weights)
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                self.model.fit(X, y)
            self.logger.info("XGBoost model training completed.")
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
            self.model = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained XGBoost model.

        Args:
            X (pd.DataFrame): Feature DataFrame.

        Returns:
            np.ndarray: Prediction probabilities.
        """
        try:
            if self.model is None:
                self.logger.warning("XGBoost model is not trained. Returning zeros.")
                return np.zeros(len(X))
            preds = self.model.predict_proba(X)[:, 1]
            return preds
        except Exception as e:
            self.logger.error(f"Error making XGBoost predictions: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance scores from XGBoost model.

        Returns:
            Dict[str, float]: Feature importances.
        """
        try:
            importance = self.model.get_booster().get_score(importance_type='weight')
            importance = {k.replace('f', ''): v for k, v in importance.items()}
            return importance
        except Exception as e:
            self.logger.error(f"Error retrieving feature importance from XGBoost: {e}")
            return {}