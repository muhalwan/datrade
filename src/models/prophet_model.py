import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Tuple
import logging
from datetime import datetime, timedelta

from .base import BaseModel

class ProphetModel(BaseModel):
    """Prophet model for time series prediction"""

    def __init__(self, config: Dict = None):
        super().__init__("prophet", config)
        self.forecast_periods = config.get('forecast_periods', 24)
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_mode = config.get('seasonality_mode', 'multiplicative')

    def _create_prophet_model(self) -> Prophet:
        """Create and configure Prophet model"""
        return Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )

    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess data for Prophet"""
        try:
            # Prophet requires 'ds' (date) and 'y' (target) columns
            prophet_df = pd.DataFrame({
                'ds': data.index,
                'y': data['close']
            })

            # Add additional regressors if configured
            for regressor in self.config.get('additional_regressors', []):
                if regressor in data.columns:
                    prophet_df[regressor] = data[regressor]

            # Split into training and validation
            train_size = int(len(prophet_df) * 0.8)
            train_df = prophet_df[:train_size]
            val_df = prophet_df[train_size:]

            return train_df, val_df

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict:
        """Train Prophet model"""
        try:
            self.model = self._create_prophet_model()

            # Add additional regressors
            for regressor in self.config.get('additional_regressors', []):
                if regressor in train_df.columns:
                    self.model.add_regressor(regressor)

            # Fit model
            self.model.fit(train_df)
            self.is_trained = True

            # Calculate metrics if validation data is provided
            metrics = {}
            if val_df is not None:
                forecast = self.model.predict(val_df)
                metrics = self._calculate_metrics(val_df, forecast)

            return metrics

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def _calculate_metrics(self, actual: pd.DataFrame,
                           forecast: pd.DataFrame) -> Dict:
        """Calculate forecast performance metrics"""
        try:
            y_true = actual['y'].values
            y_pred = forecast['yhat'].values

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            return {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }

        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return {}

    def predict(self, periods: int = None) -> pd.DataFrame:
        """Make future predictions"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            # Create future dataframe
            periods = periods or self.forecast_periods
            future = self.model.make_future_dataframe(
                periods=periods,
                freq='H'
            )

            # Make prediction
            forecast = self.model.predict(future)

            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def get_components(self) -> Dict:
        """Get trend and seasonality components"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")

            return self.model.component_modes

        except Exception as e:
            self.logger.error(f"Component extraction error: {str(e)}")
            raise