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
        """Process raw data into features and targets with strict alignment"""
        try:
            self.logger.info("Processing features from price and orderbook data")

            # 1. Align datasets by timestamp
            price_data, orderbook_data = self._align_data(price_data, orderbook_data)

            # Validate input data
            if price_data.empty or orderbook_data.empty:
                self.logger.error("Aligned data is empty. Check input sources")
                return pd.DataFrame(), pd.Series()

            # 2. Calculate technical indicators with NaN handling
            tech_features = self.technical_indicators.calculate(price_data)
            tech_features = tech_features.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaNs with column means
            tech_features = tech_features.fillna(tech_features.mean())

            if tech_features.empty:
                self.logger.error("Technical features calculation failed")
                return pd.DataFrame(), pd.Series()

            # 3. Calculate sentiment features with fallback values
            sentiment_features = self.sentiment_analyzer.analyze(orderbook_data)
            required_cols = ['bid_ask_spread', 'volume_imbalance',
                             'bid_ask_spread_norm', 'volume_imbalance_norm']
            for col in required_cols:
                if col not in sentiment_features.columns:
                    self.logger.warning(f"Missing {col} - filling with zeros")
                    sentiment_features[col] = 0.0
            sentiment_features = sentiment_features.fillna(0)

            # 4. Merge features with strict index alignment
            features = tech_features.join(sentiment_features, how='inner')
            if features.empty:
                self.logger.error("Feature merge failed - empty DataFrame")
                return pd.DataFrame(), pd.Series()

            # 5. Final NaN cleanup
            features = features.ffill().bfill()
            if features.isnull().any().any():
                self.logger.warning(f"Dropping {features.isnull().any(axis=1).sum()} rows with NaNs")
                features = features.dropna()

            # 6. Create target variable with strict alignment
            aligned_price = price_data.reindex(features.index, method='ffill')
            features['future_close'] = aligned_price['close'].shift(-1)

            # Remove final row with NaN target
            features = features.dropna(subset=['future_close'])
            features['target'] = (features['future_close'] > features['close']).astype(int)

            # 7. Final validation
            target = features.pop('target')
            features = features.drop(columns=['future_close'])

            # Ensure perfect index alignment
            features = features.loc[target.index]
            aligned_price = aligned_price.loc[target.index]

            if len(features) != len(target):
                self.logger.error("Final feature/target mismatch - aborting")
                return pd.DataFrame(), pd.Series()

            self.logger.info(f"Successfully processed {len(features)} samples")
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