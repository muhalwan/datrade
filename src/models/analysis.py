import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error



class ModelAnalyzer:
    """Analyze model performance and feature importance"""

    def __init__(self, model_trainer):
        self.logger = logging.getLogger(__name__)
        self.model_trainer = model_trainer

    def analyze_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Analyze feature importance for model"""
        try:
            model = self.model_trainer.models[model_name]

            if model_name == 'xgboost':
                importance = pd.DataFrame({
                    'feature': range(len(model.model.feature_importances_)),
                    'importance': model.model.feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False)

            elif model_name == 'lstm':
                # Use integrated gradients or SHAP for LSTM
                importance = pd.DataFrame()  # Placeholder

            else:
                importance = pd.DataFrame()

            return importance

        except Exception as e:
            self.logger.error(f"Feature importance analysis error: {str(e)}")
            return pd.DataFrame()

    def analyze_prediction_errors(self,
                                  model_name: str,
                                  test_data: pd.DataFrame) -> Dict:
        """Analyze prediction errors"""
        try:
            model = self.model_trainer.models[model_name]

            if not model.is_trained:
                raise ValueError("Model not trained")

            # Get predictions
            if model_name == 'prophet':
                predictions = model.predict()['yhat']
                actuals = test_data['close']
            else:
                X, y = model.preprocess(test_data)
                predictions = model.predict(X)
                actuals = y

            # Calculate error metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'mape': mape
            }

        except Exception as e:
            self.logger.error(f"Error analysis error: {str(e)}")
            return {}

    def analyze_model_stability(self, model_name: str,
                                time_periods: List[Tuple[datetime, datetime]]) -> Dict:
        """Analyze model stability across different time periods"""
        try:
            stability_metrics = []

            for start_time, end_time in time_periods:
                data = self.model_trainer.feature_pipeline.get_historical_features(
                    start_time, end_time)

                metrics = self.analyze_prediction_errors(model_name, data)
                metrics['period_start'] = start_time
                metrics['period_end'] = end_time
                stability_metrics.append(metrics)

            return {
                'metrics': stability_metrics,
                'consistency': self._calculate_consistency(stability_metrics)
            }

        except Exception as e:
            self.logger.error(f"Stability analysis error: {str(e)}")
            return {}

    def _calculate_consistency(self, metrics: List[Dict]) -> Dict:
        """Calculate consistency metrics"""
        try:
            df = pd.DataFrame(metrics)
            return {
                'rmse_std': df['rmse'].std(),
                'rmse_cv': df['rmse'].std() / df['rmse'].mean(),
                'mape_std': df['mape'].std(),
                'mape_cv': df['mape'].std() / df['mape'].mean()
            }
        except Exception as e:
            self.logger.error(f"Consistency calculation error: {str(e)}")
            return {}