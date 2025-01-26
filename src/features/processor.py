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

            # Align data first
            price_data, orderbook_data = self._align_data(price_data, orderbook_data)

            # Check if data is empty after alignment
            if price_data.empty or orderbook_data.empty:
                self.logger.error("Aligned data is empty. Check input data sources.")
                return pd.DataFrame(), pd.Series()

            # Technical features
            tech_features = self.technical_indicators.calculate(price_data)
            if tech_features.empty:
                self.logger.error("Technical features calculation failed.")
                return pd.DataFrame(), pd.Series()

            # Sentiment features
            sentiment_features = self.sentiment_analyzer.analyze(orderbook_data)
            if sentiment_features.empty:
                self.logger.error("Sentiment features calculation failed.")
                return pd.DataFrame(), pd.Series()

            # Merge features
            features = tech_features.join(sentiment_features, how='inner')
            if features.empty:
                self.logger.error("No features after merging technical and sentiment data.")
                return pd.DataFrame(), pd.Series()

            # Handle missing values
            features.ffill(inplace=True)
            features.bfill(inplace=True)
            features.dropna(inplace=True)
            if features.empty:
                self.logger.error("Features DataFrame is empty after handling missing values.")
                return pd.DataFrame(), pd.Series()

            # Ensure price_data is aligned with features after processing
            aligned_price_data = price_data.reindex(features.index, method='ffill')
            if aligned_price_data.empty:
                self.logger.error("Aligned price data is empty.")
                return pd.DataFrame(), pd.Series()

            # Create target using FUTURE close price
            features['future_close'] = aligned_price_data['close'].shift(-1)
            features['target'] = (features['future_close'] > features['close']).astype(int)

            # Remove rows with NaN target (last row)
            features = features.dropna(subset=['target'])
            if features.empty:
                self.logger.error("No valid targets after processing.")
                return pd.DataFrame(), pd.Series()

            # Align price_data with features index
            aligned_price_data = aligned_price_data.loc[features.index]

            # Extract target
            target = features.pop('target')

            return features, target

        except Exception as e:
            self.logger.error(f"Error during feature processing: {str(e)}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def _align_data(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Resample using 'min' instead of deprecated 'T'
        price_data_resampled = price_data.resample('1min').last()
        orderbook_data_resampled = orderbook_data.resample('1min').last()

        # Align by index (timestamps)
        aligned_price_data = price_data_resampled.reindex(orderbook_data_resampled.index, method='ffill')
        aligned_orderbook_data = orderbook_data_resampled.reindex(price_data_resampled.index, method='ffill')
        return aligned_price_data, aligned_orderbook_data
