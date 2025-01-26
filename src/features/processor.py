import numpy as np
import pandas as pd
from typing import Tuple
import logging
from .technical import TechnicalIndicators
from .sentiment import SentimentAnalyzer

class FeatureProcessor:
    """
    Processes raw price and orderbook data into features for model training.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()

    def process(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            self.logger.info("Processing features from price and orderbook data.")
            price_data, orderbook_data = self._align_data(price_data, orderbook_data)

            # Validate input data
            if price_data.empty or orderbook_data.empty:
                self.logger.error("Aligned data is empty. Check input data sources.")
                return pd.DataFrame(), pd.Series()

            # Technical features with enhanced NaN handling
            tech_features = self.technical_indicators.calculate(price_data)
            tech_features = tech_features.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaNs with column means
            tech_features = tech_features.fillna(tech_features.mean())

            # Sentiment features with default values
            sentiment_features = self.sentiment_analyzer.analyze(orderbook_data)
            sentiment_features = sentiment_features.fillna(0)

            # Merge features with validation
            features = tech_features.join(sentiment_features, how='left')
            features = features.fillna(0)

            # Final NaN check and cleanup
            if features.isnull().any().any():
                self.logger.warning(f"Final NaN removal: {features.isnull().sum().sum()} values")
                features = features.dropna()

            # Target creation with strict index alignment
            aligned_price = price_data.reindex(features.index, method='ffill')
            features = features[~aligned_price['close'].isnull()]
            aligned_price = aligned_price.dropna()

            if len(features) == 0 or len(aligned_price) == 0:
                self.logger.error("Final dataset validation failed.")
                return pd.DataFrame(), pd.Series()

            features['future_close'] = aligned_price['close'].shift(-1)
            features = features.dropna(subset=['future_close'])
            features['target'] = (features['future_close'] > features['close']).astype(int)

            # Final validation
            target = features.pop('target')
            if len(features) != len(target):
                self.logger.error("Final feature/target length mismatch")
                return pd.DataFrame(), pd.Series()

            return features, target

        except Exception as e:
            self.logger.error(f"Feature processing failed: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def _align_data(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Resample using 'min' instead of deprecated 'T'
        price_data_resampled = price_data.resample('1min').last()
        orderbook_data_resampled = orderbook_data.resample('1min').last()

        # Align by index (timestamps)
        aligned_price_data = price_data_resampled.reindex(orderbook_data_resampled.index, method='ffill')
        aligned_orderbook_data = orderbook_data_resampled.reindex(price_data_resampled.index, method='ffill')
        return aligned_price_data, aligned_orderbook_data