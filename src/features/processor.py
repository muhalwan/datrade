import pandas as pd
import numpy as np
from typing import Tuple
import logging

from .technical import TechnicalIndicators
from .sentiment import SentimentAnalyzer
from .selector import FeatureSelector
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    """
    Processes raw price and orderbook data into features for model training.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_selector = FeatureSelector()
        self.scaler = StandardScaler()

    def process(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Processes and merges price and orderbook data to generate features and target.

        Args:
            price_data (pd.DataFrame): OHLCV price data.
            orderbook_data (pd.DataFrame): Orderbook snapshots.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target labels.
        """
        try:
            self.logger.info("Calculating technical features...")
            tech_features = self.tech_indicators.calculate(price_data)

            self.logger.info("Calculating sentiment features...")
            sentiment_features = self.sentiment_analyzer.calculate(orderbook_data)

            # Merge features
            self.logger.info("Merging technical and sentiment features...")
            features = pd.concat([tech_features, sentiment_features], axis=1)

            # Handle missing values and type inference
            self.logger.info("Handling missing values and inferring object types...")
            features = features.ffill().bfill().infer_objects()

            # Align with price data
            self.logger.info("Aligning features with price data for target generation...")
            features = features.loc[price_data.index]

            # Define target: Binary classification based on future returns
            self.logger.info("Generating target labels...")
            target = (price_data['close'].shift(-1) > price_data['close']).astype(int)
            target = target.loc[features.index]

            self.logger.info(f"Features shape: {features.shape}")
            self.logger.info(f"Target shape: {target.shape}")

            # Feature selection and scaling will be handled in the model training pipeline
            return features, target
        except Exception as e:
            self.logger.error(f"Error in feature processing: {e}")
            return pd.DataFrame(), pd.Series()
