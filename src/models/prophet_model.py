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

    def train(self,
              X: pd.DataFrame,
              y: pd.Series) -> None:
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': X.index,
                'y': y
            })

            # Initialize and train model
            self.model = Prophet(**self.params)
            self.model.fit(df)

        except Exception as e:
            self.logger.error(f"Error training Prophet: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        try:
            # Prepare future dataframe
            future = pd.DataFrame({'ds': X.index})

            # Make predictions
            forecast = self.model.predict(future)

            # Convert predictions to binary signals
            predictions = (forecast['yhat'] > 0).astype(int).values

            return predictions
        except Exception as e:
            self.logger.error(f"Error making Prophet predictions: {e}")
            return np.array([])