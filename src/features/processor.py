from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .technical import TechnicalFeatureCalculator
from .sentiment import SentimentAnalyzer

class FeatureProcessor:
    """Processes and combines all features for model input"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_calculator = TechnicalFeatureCalculator()
        self.sentiment_analyzer = SentimentAnalyzer()

    def prepare_features(self,
                         price_data: pd.DataFrame,
                         orderbook_data: pd.DataFrame,
                         news_data: Optional[List[Dict]] = None,
                         target_minutes: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training"""
        try:
            # Calculate technical features
            technical_features = self.technical_calculator.calculate_features(price_data)

            # Calculate sentiment features
            sentiment_features = self.sentiment_analyzer.calculate_market_sentiment(
                price_data,
                orderbook_data,
                news_data
            )

            # Combine features
            features = pd.concat([
                technical_features,
                sentiment_features
            ], axis=1)

            # Prepare target (future price movement)
            target = self._prepare_target(price_data, target_minutes)

            # Clean and align data
            features, target = self._align_and_clean_data(features, target)

            return features, target

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()

    def _prepare_target(self,
                        price_data: pd.DataFrame,
                        target_minutes: int) -> pd.Series:
        """Prepare target variable (future returns)"""
        try:
            # Calculate future returns
            future_price = price_data['close'].shift(-target_minutes)
            target = (future_price - price_data['close']) / price_data['close']

            # Convert to binary classification
            target = (target > 0).astype(int)

            return target
        except Exception as e:
            self.logger.error(f"Error preparing target: {e}")
            return pd.Series()

    def _align_and_clean_data(self,
                              features: pd.DataFrame,
                              target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, handle missing values"""
        try:
            # Align indices
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]

            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill')

            # Remove remaining NA rows
            valid_rows = features.notna().all(axis=1) & target.notna()
            features = features[valid_rows]
            target = target[valid_rows]

            return features, target
        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            return pd.DataFrame(), pd.Series()