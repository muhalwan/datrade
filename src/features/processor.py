from typing import Tuple
import pandas as pd
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

            # Ensure both dataframes have the same index (timestamps)
            if not price_data.index.equals(orderbook_data.index):
                self.logger.warning("Aligning price and orderbook data by index...")
                price_data, orderbook_data = self._align_data(price_data, orderbook_data)

            # Technical features
            tech_features = self.technical_indicators.calculate(price_data)

            # Sentiment features
            sentiment_features = self.sentiment_analyzer.analyze(orderbook_data)

            # Merge features
            features = tech_features.join(sentiment_features, how='inner')

            # Handle missing values
            features.ffill(inplace=True)
            features.bfill(inplace=True)

            # Create target using FUTURE close price
            features['future_close'] = features['close'].shift(-1)
            features['target'] = (features['future_close'] > features['close']).astype(int)

            # Remove last row with NaN target and align price_data
            features = features.iloc[:-1]
            price_data = price_data.iloc[:-1]  # Critical alignment fix
            target = features.pop('target')

            self.logger.info("Feature processing completed successfully.")
            return features, target

        except Exception as e:
            self.logger.error(f"Error during feature processing: {e}")
            return pd.DataFrame(), pd.Series()

    def _align_data(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aligns the price and orderbook data by index (timestamps).

        Args:
            price_data (pd.DataFrame): Price data to align.
            orderbook_data (pd.DataFrame): Orderbook data to align.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned price and orderbook data.
        """
        # Resample both dataframes to a common frequency (e.g., 1 minute) if needed
        price_data_resampled = price_data.resample('1T').last()  # Resample to 1-minute intervals
        orderbook_data_resampled = orderbook_data.resample('1T').last()

        # Align by index (timestamps)
        aligned_price_data = price_data_resampled.reindex(orderbook_data_resampled.index)
        aligned_orderbook_data = orderbook_data_resampled.reindex(price_data_resampled.index)

        return aligned_price_data, aligned_orderbook_data
