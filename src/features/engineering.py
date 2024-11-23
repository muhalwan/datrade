import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta

from src.features.config import FeatureConfig, FeatureType, DEFAULT_FEATURES

class FeatureEngineering:
    """Feature engineering pipeline for cryptocurrency trading data"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection
        self.default_features = DEFAULT_FEATURES

    def get_price_data(self, symbol: str, start_time: datetime,
                       end_time: datetime) -> pd.DataFrame:
        """Fetch price data from MongoDB"""
        try:
            collection = self.db.get_collection('price_data')
            query = {
                "symbol": symbol,
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }

            # Fetch data
            cursor = collection.find(query).sort("timestamp", 1)
            df = pd.DataFrame(list(cursor))

            if df.empty:
                self.logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Create OHLCV data
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df['price'].resample('1Min').first()
            ohlcv['high'] = df['price'].resample('1Min').max()
            ohlcv['low'] = df['price'].resample('1Min').min()
            ohlcv['close'] = df['price'].resample('1Min').last()
            ohlcv['volume'] = df['quantity'].resample('1Min').sum()

            # Forward fill missing values
            ohlcv = ohlcv.ffill()  # Using ffill() instead of fillna(method='ffill')

            # Add basic returns
            ohlcv['returns'] = ohlcv['close'].pct_change()

            # Log data shape and sample
            self.logger.info(f"OHLCV data shape: {ohlcv.shape}")
            self.logger.info(f"Sample OHLCV data:\n{ohlcv.head()}")

            return ohlcv

        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            features = df.copy()

            # Log initial feature calculation
            self.logger.info(f"Calculating technical features for shape: {features.shape}")

            for feature in self.default_features:
                if not feature.enabled:
                    continue

                try:
                    if feature.name == "SMA":
                        for window in feature.params["windows"]:
                            features[f'sma_{window}'] = ta.trend.sma_indicator(
                                features['close'],
                                window=window,
                                fillna=True
                            )

                    elif feature.name == "EMA":
                        for window in feature.params["windows"]:
                            features[f'ema_{window}'] = ta.trend.ema_indicator(
                                features['close'],
                                window=window,
                                fillna=True
                            )

                    elif feature.name == "RSI":
                        features['rsi'] = ta.momentum.rsi(
                            features['close'],
                            window=feature.params["window"],
                            fillna=True
                        )

                    elif feature.name == "MACD":
                        macd = ta.trend.MACD(
                            close=features['close'],
                            window_slow=feature.params["window_slow"],
                            window_fast=feature.params["window_fast"],
                            window_sign=feature.params["window_sign"],
                            fillna=True
                        )
                        features['macd'] = macd.macd()
                        features['macd_signal'] = macd.macd_signal()
                        features['macd_diff'] = macd.macd_diff()

                    elif feature.name == "BB":
                        bb = ta.volatility.BollingerBands(
                            close=features['close'],
                            window=feature.params["window"],
                            window_dev=feature.params["window_dev"],
                            fillna=True
                        )
                        features['bb_high'] = bb.bollinger_hband()
                        features['bb_mid'] = bb.bollinger_mavg()
                        features['bb_low'] = bb.bollinger_lband()

                    elif feature.name == "OBV":
                        features['obv'] = ta.volume.on_balance_volume(
                            close=features['close'],
                            volume=features['volume'],
                            fillna=True
                        )

                    self.logger.debug(f"Calculated {feature.name} features")

                except Exception as e:
                    self.logger.error(f"Error calculating {feature.name}: {str(e)}")
                    continue

            # Check for any remaining NaN values
            nan_cols = features.columns[features.isna().any()].tolist()
            if nan_cols:
                self.logger.warning(f"NaN values found in columns: {nan_cols}")
                features = features.ffill().bfill()  # Forward fill then backward fill

            self.logger.info(f"Final features shape: {features.shape}")
            self.logger.info(f"Features generated: {features.columns.tolist()}")

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {str(e)}")
            return df

    def generate_features(self, symbol: str, start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Generate all features for the given symbol and time range"""
        try:
            # Get price data
            price_data = self.get_price_data(symbol, start_time, end_time)
            if price_data.empty:
                self.logger.error("No price data available for feature generation")
                return pd.DataFrame()

            # Calculate technical features
            features = self.calculate_technical_features(price_data)

            # Verify we have enough valid data
            if len(features) < 2:
                self.logger.error("Insufficient data points for feature generation")
                return pd.DataFrame()

            # Final validation
            if features.empty:
                self.logger.error("Feature generation produced empty DataFrame")
                return pd.DataFrame()

            if features.isna().any().any():
                self.logger.warning("NaN values present in features - filling with forward fill")
                features = features.ffill().bfill()

            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return pd.DataFrame()