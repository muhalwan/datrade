import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from .technical import TechnicalFeatureCalculator
from .sentiment import SentimentAnalyzer

class FeatureProcessor:
    """Advanced feature processing and engineering system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_calculator = TechnicalFeatureCalculator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scaler = StandardScaler()
        self.logger.setLevel(logging.INFO)

    def prepare_features(
            self,
            price_data: pd.DataFrame,
            orderbook_data: pd.DataFrame,
            target_minutes: int = 5,
            include_sentiment: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training."""
        try:
            # Log input data
            self._log_input_data(price_data, orderbook_data)

            # Validate input data
            if not self._validate_input_data(price_data):
                raise ValueError("Invalid input data")

            # Copy data to avoid modifications
            price_df = price_data.copy()
            orderbook_df = orderbook_data.copy()

            # Calculate technical features
            self.logger.info("Calculating technical features...")
            tech_features = self.technical_calculator.calculate_features(price_df)
            self.logger.info(f"Technical features shape: {tech_features.shape}")

            # Calculate sentiment features
            if include_sentiment:
                self.logger.info("Calculating sentiment features...")
                sentiment_features = self.sentiment_analyzer.calculate_market_sentiment(
                    price_data=price_df,
                    orderbook_data=orderbook_df
                )
                if not sentiment_features.empty:
                    tech_features = pd.concat([tech_features, sentiment_features], axis=1)
                self.logger.info(f"Sentiment features shape: {sentiment_features.shape}")

            # Merge price data with technical and sentiment features
            combined_features = pd.concat([price_df, tech_features], axis=1)

            # Prepare target
            target = self._prepare_target(price_df, target_minutes)

            # Clean and align data
            features, target = self._align_and_clean_data(combined_features, target)

            # Log processing results
            self._log_processing_results(features, target)

            return features, target

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            self.logger.exception("Detailed error:")
            return pd.DataFrame(), pd.Series()

    def _log_input_data(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> None:
        """Log input data information."""
        self.logger.info(f"Price data shape: {price_data.shape}")
        self.logger.info(f"Price data columns: {price_data.columns.tolist()}")
        self.logger.info(f"Price data index: {price_data.index[:5]}")

        self.logger.info(f"Orderbook data shape: {orderbook_data.shape}")
        self.logger.info(f"Orderbook data columns: {orderbook_data.columns.tolist()}")

    def _validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """Validate input data requirements."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in price_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in price_data.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return False

        if price_data.empty:
            self.logger.error("Empty price data")
            return False

        return True

    def _prepare_target(self, price_data: pd.DataFrame, target_minutes: int) -> pd.Series:
        """Prepare target variable."""
        try:
            # Calculate future price changes
            future_price = price_data['close'].shift(-target_minutes)
            target = (future_price - price_data['close']) / price_data['close']

            # Convert to binary classification
            target = (target > 0).astype(int)

            return target

        except Exception as e:
            self.logger.error(f"Error preparing target: {e}")
            return pd.Series()

    def _align_and_clean_data(
            self,
            features: pd.DataFrame,
            target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align and clean features and target data."""
        try:
            if features.empty or target.empty:
                return pd.DataFrame(), pd.Series()

            # Handle missing values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill().infer_objects()
            target = target.ffill().bfill()

            # Remove low variance features
            feature_std = features.std()
            valid_features = feature_std[feature_std > 0].index
            features = features[valid_features]

            # Find valid rows
            valid_rows = features.notna().all(axis=1) & target.notna()
            valid_count = valid_rows.sum()

            self.logger.info(f"Valid rows before cleaning: {len(features)}")
            self.logger.info(f"Valid rows after cleaning: {valid_count}")

            if valid_count < 100:
                self.logger.error(f"Insufficient valid rows after cleaning: {valid_count}")
                return pd.DataFrame(), pd.Series()

            # Filter data
            features = features[valid_rows]
            target = target[valid_rows]

            return features, target

        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            return pd.DataFrame(), pd.Series()

    def _log_processing_results(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Log processing results."""
        self.logger.info(f"Features remaining after variance check: {len(features.columns)}")
        self.logger.info(f"Final features shape: {features.shape}")
        self.logger.info(f"Final target shape: {target.shape}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features for model prediction."""
        try:
            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill().infer_objects()

            # Scale features
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return X
