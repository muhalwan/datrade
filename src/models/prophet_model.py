from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseModel

class ProphetModel(BaseModel):
    """Prophet model for time series prediction"""

    def __init__(self, params: Optional[dict] = None):
        super().__init__("prophet")
        self.params = params or {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'seasonality_mode': 'multiplicative',
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train prophet model"""
        try:
            # Format dates properly for Prophet
            df = pd.DataFrame({
                'ds': pd.to_datetime(X.index).strftime('%Y-%m-%d %H:%M:%S'),
                'y': y
            })

            self.model = Prophet(**self.params)
            self.model.fit(df)
        except Exception as e:
            self.logger.error(f"Error training Prophet: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        try:
            future = pd.DataFrame({'ds': pd.to_datetime(X.index).strftime('%Y-%m-%d %H:%M:%S')})
            forecast = self.model.predict(future)
            return (forecast['yhat'] > 0).astype(int).values
        except Exception as e:
            self.logger.error(f"Error making Prophet predictions: {e}")
            return np.array([])
