import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from .technical import TechnicalFeatureCalculator
from .sentiment import SentimentAnalyzer

class FeatureProcessor:
    """Advanced feature processing and engineering system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_calculator = TechnicalFeatureCalculator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scaler = StandardScaler()
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}

    def prepare_features(
            self,
            price_data: pd.DataFrame,
            orderbook_data: pd.DataFrame,
            target_minutes: int = 5,
            include_sentiment: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training"""
        try:
            # Debug input data
            self._log_input_data(price_data, orderbook_data)

            # Validate input data
            if not self._validate_input_data(price_data):
                raise ValueError("Invalid input data")

            # Copy data to avoid modifications
            price_df = price_data.copy()
            orderbook_df = orderbook_data.copy()

            # Calculate features
            features = self._generate_all_features(price_df, orderbook_df, include_sentiment)

            # Prepare target
            target = self._prepare_target(price_df, target_minutes)

            # Clean and align data
            features, target = self._align_and_clean_data(features, target)

            # Log processing results
            self._log_processing_results(features, target)

            return features, target

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()

    def _log_input_data(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> None:
        """Log input data information"""
        self.logger.info(f"Price data shape: {price_data.shape}")
        self.logger.info(f"Price data columns: {price_data.columns.tolist()}")
        self.logger.info(f"Price data index: {price_data.index[:5]}")

        self.logger.info(f"Orderbook data shape: {orderbook_data.shape}")
        self.logger.info(f"Orderbook data columns: {orderbook_data.columns.tolist()}")

    def _validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """Validate input data requirements"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in price_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in price_data.columns]
            self.logger.error(f"Missing required columns: {missing}")
            return False

        if price_data.empty:
            self.logger.error("Empty price data")
            return False

        return True

    def _generate_all_features(
            self,
            price_data: pd.DataFrame,
            orderbook_data: pd.DataFrame,
            include_sentiment: bool = True
    ) -> pd.DataFrame:
        """Generate all feature sets"""
        try:
            self.logger.info("Calculating technical features...")
            features = self.technical_calculator.calculate_features(price_data)
            self.logger.info(f"Technical features shape: {features.shape}")

            # Add price action features
            features = self._add_price_action_features(features, price_data)

            # Add orderbook features
            features = self._add_orderbook_features(features, orderbook_data)

            # Add time features
            features = self._add_time_features(features)

            if include_sentiment:
                self.logger.info("Calculating sentiment features...")
                sentiment_features = self.sentiment_analyzer.calculate_market_sentiment(
                    price_data=price_data,
                    orderbook_data=orderbook_data
                )
                if not sentiment_features.empty:
                    features = pd.concat([features, sentiment_features], axis=1)
                self.logger.info(f"Sentiment features shape: {sentiment_features.shape}")

            self.logger.info(f"Combined features shape: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return pd.DataFrame()

    def _add_price_action_features(
            self,
            features: pd.DataFrame,
            price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add price action based features"""
        try:
            df = features.copy()

            # Candle patterns
            df['body_size'] = (price_data['close'] - price_data['open']).abs()
            df['upper_shadow'] = price_data['high'] - price_data[['open', 'close']].max(axis=1)
            df['lower_shadow'] = price_data[['open', 'close']].min(axis=1) - price_data['low']
            df['body_to_shadow_ratio'] = df['body_size'] / (df['upper_shadow'] + df['lower_shadow']).replace(0, np.nan)

            # Gap analysis
            df['gap_up'] = (price_data['low'] > price_data['high'].shift(1)).astype(int)
            df['gap_down'] = (price_data['high'] < price_data['low'].shift(1)).astype(int)

            # Price levels
            for window in [20, 50]:
                df[f'distance_to_high_{window}'] = (
                                                           price_data['high'].rolling(window).max() - price_data['close']
                                                   ) / price_data['close']
                df[f'distance_to_low_{window}'] = (
                                                          price_data['close'] - price_data['low'].rolling(window).min()
                                                  ) / price_data['close']

            return df

        except Exception as e:
            self.logger.error(f"Error adding price action features: {e}")
            return features

    def _add_orderbook_features(
            self,
            features: pd.DataFrame,
            orderbook_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add orderbook-based features"""
        try:
            df = features.copy()

            if orderbook_data.empty:
                return df

            # Group by timestamp and side
            grouped = orderbook_data.groupby(['timestamp', 'side'])['quantity'].sum().unstack()

            if 'bid' in grouped and 'ask' in grouped:
                # Calculate order book imbalance
                df['orderbook_imbalance'] = (
                        (grouped['bid'] - grouped['ask']) /
                        (grouped['bid'] + grouped['ask'])
                )

                # Calculate spread and depth
                bid_prices = orderbook_data[orderbook_data['side'] == 'bid'].groupby('timestamp')['price'].max()
                ask_prices = orderbook_data[orderbook_data['side'] == 'ask'].groupby('timestamp')['price'].min()
                df['spread'] = (ask_prices - bid_prices) / bid_prices
                df['depth_ratio'] = grouped['bid'].rolling(20).mean() / grouped['ask'].rolling(20).mean()

            return df

        except Exception as e:
            self.logger.error(f"Error adding orderbook features: {e}")
            return features

    def _add_time_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df = features.copy()

            # Extract time components
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day

            # Trading session indicators (assuming UTC)
            df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

            # Session overlap periods
            df['asia_europe_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
            df['europe_us_overlap'] = ((df['hour'] >= 15) & (df['hour'] < 17)).astype(int)

            return df

        except Exception as e:
            self.logger.error(f"Error adding time features: {e}")
            return features

    def _prepare_target(self, price_data: pd.DataFrame, target_minutes: int) -> pd.Series:
        """Prepare target variable"""
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
        """Align features and target, handle missing values"""
        try:
            if features.empty or target.empty:
                return pd.DataFrame(), pd.Series()

            # Handle missing values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill')

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
        """Log processing results"""
        self.logger.info(f"Features remaining after variance check: {len(features.columns)}")
        self.logger.info(f"Final features shape: {features.shape}")
        self.logger.info(f"Final target shape: {target.shape}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            # Select features if available
            if self.selected_features:
                X = X[self.selected_features]

            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill()  # Replace fillna with ffill/bfill

            # Scale features
            if not hasattr(self.scaler, 'mean_'):  # Check if scaler is fitted
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return X