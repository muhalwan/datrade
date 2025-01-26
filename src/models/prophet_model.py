from typing import Optional, Dict
import pandas as pd
import logging
from fbprophet import Prophet

from .base import BaseModel

class ProphetModel(BaseModel):
    """
    Prophet Time Series Forecasting Model.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initializes the ProphetModel with specified hyperparameters.

        Args:
            params (Optional[Dict]): Hyperparameters for Prophet.
        """
        super().__init__("prophet")
        self.params = params if params else {}
        self.model = Prophet(**self.params)
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Prophet model.

        Args:
            X (pd.DataFrame): Feature DataFrame with a datetime index.
            y (pd.Series): Target variable.
        """
        try:
            self.logger.info("Training Prophet model...")
            df = X.copy()
            df['y'] = y
            df.reset_index(inplace=True)
            df.rename(columns={'trade_time': 'ds'}, inplace=True)
            self.model.fit(df[['ds', 'y']])
            self.logger.info("Prophet model training completed.")
        except Exception as e:
            self.logger.error(f"Error training Prophet: {e}")
            self.model = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates forecasts using the trained Prophet model.

        Args:
            X (pd.DataFrame): Future dates for forecasting.

        Returns:
            pd.Series: Forecasted values.
        """
        try:
            if self.model is None:
                self.logger.warning("Prophet model is not trained. Returning zeros.")
                return pd.Series([0.0] * len(X))

            future = X.copy().reset_index()
            future.rename(columns={'trade_time': 'ds'}, inplace=True)
            forecast = self.model.predict(future[['ds']])
            return forecast['yhat']
        except Exception as e:
            self.logger.error(f"Error making Prophet predictions: {e}")
            return pd.Series([0.0] * len(X))