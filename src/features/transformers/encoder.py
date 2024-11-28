import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
from pathlib import Path

class FeatureEncoder:
    """Handles categorical feature encoding"""

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.encoders = {}
        self.fitted = False

    def fit(self, data: pd.DataFrame, categorical_columns: List[str]):
        """Fit encoders to categorical data"""
        try:
            for col in categorical_columns:
                if col not in data.columns:
                    continue

                # Choose encoder type based on cardinality
                unique_values = data[col].nunique()

                if unique_values <= self.config.get('onehot_max_categories', 10):
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoder.fit(data[col].values.reshape(-1, 1))
                    encoder_type = 'onehot'
                else:
                    encoder = LabelEncoder()
                    encoder.fit(data[col].values)
                    encoder_type = 'label'

                self.encoders[col] = {
                    'encoder': encoder,
                    'type': encoder_type
                }

            self.fitted = True

        except Exception as e:
            self.logger.error(f"Encoder fitting error: {str(e)}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features"""
        if not self.fitted:
            raise ValueError("Encoders not fitted")

        try:
            result = data.copy()

            for col, encoder_info in self.encoders.items():
                if col not in result.columns:
                    continue

                encoder = encoder_info['encoder']
                encoder_type = encoder_info['type']

                if encoder_type == 'onehot':
                    encoded = encoder.transform(result[col].values.reshape(-1, 1))
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                        index=result.index
                    )
                    result = pd.concat([result.drop(col, axis=1), encoded_df], axis=1)
                else:
                    result[col] = encoder.transform(result[col].values)

            return result

        except Exception as e:
            self.logger.error(f"Transform error: {str(e)}")
            return data

    def save(self, path: str):
        """Save fitted encoders"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.encoders, save_path)
            self.logger.info(f"Encoders saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving encoders: {str(e)}")

    def load(self, path: str):
        """Load fitted encoders"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Encoder file not found: {load_path}")

            self.encoders = joblib.load(load_path)
            self.fitted = True
            self.logger.info(f"Encoders loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading encoders: {str(e)}")