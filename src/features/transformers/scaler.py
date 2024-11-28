import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path

class FeatureScaler:
    """Handles feature scaling for different data types"""

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.scalers = {}
        self.fitted = False

    def fit(self, data: pd.DataFrame, columns: Optional[Dict[str, str]] = None):
        """Fit scalers to data"""
        try:
            # Default column types if not specified
            if columns is None:
                columns = self._infer_column_types(data)

            for col, scale_type in columns.items():
                if col not in data.columns:
                    continue

                scaler = self._get_scaler(scale_type)
                scaler.fit(data[col].values.reshape(-1, 1))
                self.scalers[col] = {
                    'scaler': scaler,
                    'type': scale_type
                }

            self.fitted = True

        except Exception as e:
            self.logger.error(f"Scaler fitting error: {str(e)}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        if not self.fitted:
            raise ValueError("Scalers not fitted")

        try:
            result = data.copy()

            for col, scaler_info in self.scalers.items():
                if col not in result.columns:
                    continue

                scaler = scaler_info['scaler']
                result[col] = scaler.transform(
                    result[col].values.reshape(-1, 1)).ravel()

            return result

        except Exception as e:
            self.logger.error(f"Transform error: {str(e)}")
            return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scalers not fitted")

        try:
            result = data.copy()

            for col, scaler_info in self.scalers.items():
                if col not in result.columns:
                    continue

                scaler = scaler_info['scaler']
                result[col] = scaler.inverse_transform(
                    result[col].values.reshape(-1, 1)).ravel()

            return result

        except Exception as e:
            self.logger.error(f"Inverse transform error: {str(e)}")
            return data

    def _get_scaler(self, scale_type: str):
        """Get scaler instance based on type"""
        if scale_type == 'standard':
            return StandardScaler()
        elif scale_type == 'minmax':
            return MinMaxScaler()
        elif scale_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()

    def _infer_column_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """Infer appropriate scaling for columns"""
        column_types = {}

        for col in data.columns:
            # Price and volume data usually works better with robust scaling
            if any(key in col.lower() for key in ['price', 'volume', 'size']):
                column_types[col] = 'robust'
            # Indicators often work better with standard scaling
            elif any(key in col.lower() for key in ['rsi', 'macd', 'momentum']):
                column_types[col] = 'standard'
            # Percentage values work better with minmax scaling
            elif any(key in col.lower() for key in ['pct', 'ratio', 'percent']):
                column_types[col] = 'minmax'
            else:
                column_types[col] = 'standard'

        return column_types

    def save(self, path: str):
        """Save fitted scalers"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.scalers, save_path)
            self.logger.info(f"Scalers saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving scalers: {str(e)}")

    def load(self, path: str):
        """Load fitted scalers"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {load_path}")

            self.scalers = joblib.load(load_path)
            self.fitted = True
            self.logger.info(f"Scalers loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading scalers: {str(e)}")