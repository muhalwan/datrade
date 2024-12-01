import logging
from typing import Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseModel, ModelConfig

class LightGBMModel(BaseModel):
    """LightGBM model for prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.params = config.params.get('lgb_params', {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'num_threads': 4
        })
        self.logger = logging.getLogger(__name__)

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LightGBM"""
        X = df[self.config.features].values
        y = df[self.config.target].values
        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM model"""
        try:
            X, y = self.preprocess(df)

            train_data = lgb.Dataset(X, label=y)

            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                verbose_eval=10
            )

        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            X = df[self.config.features].values
            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save LightGBM model"""
        try:
            model_path = f"{path}/lgb_model.txt"
            self.model.save_model(model_path)

        except Exception as e:
            self.logger.error(f"Error saving LightGBM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load LightGBM model"""
        try:
            model_path = f"{path}/lgb_model.txt"
            self.model = lgb.Booster(model_file=model_path)

        except Exception as e:
            self.logger.error(f"Error loading LightGBM model: {str(e)}")
            raise