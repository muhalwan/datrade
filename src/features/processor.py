from typing import Tuple
import pandas as pd
import numpy as np
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

            # Calculate technical indicators
            tech_features = self.technical_indicators.calculate(price_data)

            # Calculate sentiment features
            sentiment_features = self.sentiment_analyzer.analyze(orderbook_data)

            # Merge features
            features = tech_features.join(sentiment_features, how='inner')

            # Handle missing values
            features.fillna(method='ffill', inplace=True)
            features.fillna(method='bfill', inplace=True)

            # Generate target variable (e.g., price will go up)
            features['future_close'] = price_data['close'].shift(-1)
            features['target'] = (features['future_close'] > features['close']).astype(int)
            target = features['target'].iloc[:-1]
            features = features.iloc[:-1]

            self.logger.info("Feature processing completed successfully.")
            return features, target

        except Exception as e:
            self.logger.error(f"Error during feature processing: {e}")
            return pd.DataFrame(), pd.Series()
