import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List, Dict
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
                         target_minutes: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training"""
        try:
            # Debug input data
            self.logger.info(f"Price data shape: {price_data.shape}")
            self.logger.info(f"Price data columns: {price_data.columns.tolist()}")
            self.logger.info(f"Price data index: {price_data.index[:5]}")  # Log sample of index

            self.logger.info(f"Orderbook data shape: {orderbook_data.shape}")
            self.logger.info(f"Orderbook data columns: {orderbook_data.columns.tolist()}")

            # Ensure price_data has all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in price_data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in price_data.columns]
                raise ValueError(f"Price data missing required columns: {missing}")

            # Copy data to avoid modifications
            price_df = price_data.copy()
            orderbook_df = orderbook_data.copy()

            # Calculate technical features
            self.logger.info("Calculating technical features...")
            technical_features = self.technical_calculator.calculate_features(price_df)
            self.logger.info(f"Technical features shape: {technical_features.shape}")

            # Calculate sentiment features
            self.logger.info("Calculating sentiment features...")
            sentiment_features = self.sentiment_analyzer.calculate_market_sentiment(
                price_data=price_df,
                orderbook_data=orderbook_df
            )
            self.logger.info(f"Sentiment features shape: {sentiment_features.shape if not sentiment_features.empty else '(empty)'}")

            # Combine features
            features = pd.DataFrame(index=price_df.index)

            if not technical_features.empty:
                features = pd.concat([features, technical_features], axis=1)

            if not sentiment_features.empty:
                features = pd.concat([features, sentiment_features], axis=1)

            self.logger.info(f"Combined features shape: {features.shape}")

            # Prepare target
            target = self._prepare_target(price_df, target_minutes)
            self.logger.info(f"Target shape: {target.shape}")

            # Clean and align data
            features, target = self._align_and_clean_data(features, target)
            self.logger.info(f"Final features shape: {features.shape}")
            self.logger.info(f"Final target shape: {target.shape}")

            if features.empty:
                self.logger.error("No features were generated")
                return pd.DataFrame(), pd.Series()

            return features, target

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()

    def _prepare_target(self, price_data: pd.DataFrame, target_minutes: int) -> pd.Series:
        """Prepare target variable (future returns)"""
        try:
            # Calculate future price changes
            future_price = price_data['close'].shift(-target_minutes)
            target = (future_price - price_data['close']) / price_data['close']

            # Convert to binary classification (1 for positive returns, 0 for negative)
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
            if features.empty or target.empty:
                return pd.DataFrame(), pd.Series()

            # Forward fill missing values first
            features = features.ffill()
            # Then backward fill any remaining NaNs
            features = features.bfill()

            # Drop any columns that are all NaN
            features = features.dropna(axis=1, how='all')

            # Find valid rows (non-NaN)
            valid_rows = features.notna().all(axis=1) & target.notna()
            valid_count = valid_rows.sum()

            self.logger.info(f"Valid rows before cleaning: {len(features)}")
            self.logger.info(f"Valid rows after cleaning: {valid_count}")

            if valid_count < 100:  # Minimum required data points
                self.logger.error(f"Insufficient valid rows after cleaning: {valid_count}")
                return pd.DataFrame(), pd.Series()

            # Filter data
            features = features[valid_rows]
            target = target[valid_rows]

            # Remove features with low variance
            feature_std = features.std()
            valid_features = feature_std[feature_std > 0].index
            features = features[valid_features]

            self.logger.info(f"Features remaining after variance check: {len(features.columns)}")

            return features, target

        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            return pd.DataFrame(), pd.Series()