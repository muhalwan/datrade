import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.features.config import FeatureConfig, FeatureType, DEFAULT_FEATURES
from src.utils.logging import LoggerConfig

class FeatureEngineering:
    """Optimized feature engineering pipeline"""

    def __init__(self, db_connection, max_workers: int = 4):
        self.logger = LoggerConfig.get_logger(__name__)
        self.db = db_connection
        self.max_workers = max_workers
        self.default_features = DEFAULT_FEATURES
        self._setup_feature_calculators()

    def _setup_feature_calculators(self):
        """Setup feature calculation functions"""
        self.feature_funcs = {
            'price': self._calculate_price_features,
            'volume': self._calculate_volume_features,
            'technical': self._calculate_technical_features,
            'orderbook': self._calculate_orderbook_features
        }

    def _get_price_data(self, symbol: str, start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        """Fetch and prepare price data efficiently"""
        try:
            collection = self.db.get_collection('price_data')
            pipeline = [
                {
                    '$match': {
                        'symbol': symbol,
                        'timestamp': {
                            '$gte': start_time,
                            '$lte': end_time
                        }
                    }
                },
                {
                    '$sort': {'timestamp': 1}
                },
                {
                    '$project': {
                        '_id': 0,
                        'timestamp': 1,
                        'price': 1,
                        'quantity': 1
                    }
                }
            ]

            cursor = collection.aggregate(pipeline)
            df = pd.DataFrame(list(cursor))

            if df.empty:
                self.logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()

            # Create OHLCV data efficiently
            df.set_index('timestamp', inplace=True)
            ohlcv = pd.DataFrame()

            # Use efficient resampling
            grouped = df.resample('1Min')
            ohlcv['open'] = grouped['price'].first()
            ohlcv['high'] = grouped['price'].max()
            ohlcv['low'] = grouped['price'].min()
            ohlcv['close'] = grouped['price'].last()
            ohlcv['volume'] = grouped['quantity'].sum()

            # Forward fill and drop NaN rows
            ohlcv = ohlcv.ffill().dropna()

            self.logger.info(f"OHLCV data shape: {ohlcv.shape}")
            return ohlcv

        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()

    def _get_orderbook_data(self, symbol: str, timestamp: datetime,
                            levels: int = 10) -> pd.DataFrame:
        """Fetch orderbook data efficiently"""
        try:
            collection = self.db.get_collection('order_book')
            pipeline = [
                {
                    '$match': {
                        'symbol': symbol,
                        'timestamp': {
                            '$lte': timestamp
                        }
                    }
                },
                {
                    '$sort': {'timestamp': -1}
                },
                {
                    '$limit': levels * 2  # Both sides of the book
                },
                {
                    '$project': {
                        '_id': 0,
                        'timestamp': 1,
                        'side': 1,
                        'price': 1,
                        'quantity': 1
                    }
                }
            ]

            cursor = collection.aggregate(pipeline)
            return pd.DataFrame(list(cursor))

        except Exception as e:
            self.logger.error(f"Error fetching orderbook data: {str(e)}")
            return pd.DataFrame()

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            features = df.copy()

            # Basic returns
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])

            # Rolling volatility
            features['volatility_5'] = features['returns'].rolling(5).std()
            features['volatility_15'] = features['returns'].rolling(15).std()

            # Price momentum
            features['momentum_5'] = features['close'].pct_change(5)
            features['momentum_15'] = features['close'].pct_change(15)

            return features
        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            features = df.copy()

            # Volume momentum
            features['volume_momentum_5'] = features['volume'].pct_change(5)
            features['volume_momentum_15'] = features['volume'].pct_change(15)

            # Volume moving averages
            features['volume_sma_5'] = ta.volume.volume_sma_indicator(
                close=features['close'],
                volume=features['volume'],
                window=5
            )

            # Money flow
            features['mfi'] = ta.volume.money_flow_index(
                high=features['high'],
                low=features['low'],
                close=features['close'],
                volume=features['volume'],
                window=14
            )

            return features
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return df

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators efficiently"""
        try:
            features = df.copy()

            # Calculate features in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                feature_results = list(executor.map(
                    partial(self._calculate_single_indicator, df=features),
                    self.default_features
                ))

            # Combine results
            for result in feature_results:
                if result is not None:
                    features.update(result)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {str(e)}")
            return df

    def _calculate_single_indicator(self, feature: FeatureConfig,
                                    df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate a single technical indicator"""
        try:
            if not feature.enabled:
                return None

            result = pd.DataFrame(index=df.index)

            if feature.name == "SMA":
                for window in feature.params["windows"]:
                    result[f'sma_{window}'] = ta.trend.sma_indicator(
                        df['close'], window=window, fillna=True
                    )

            elif feature.name == "EMA":
                for window in feature.params["windows"]:
                    result[f'ema_{window}'] = ta.trend.ema_indicator(
                        df['close'], window=window, fillna=True
                    )

            elif feature.name == "RSI":
                result['rsi'] = ta.momentum.rsi(
                    df['close'],
                    window=feature.params["window"],
                    fillna=True
                )

            elif feature.name == "MACD":
                macd = ta.trend.MACD(
                    close=df['close'],
                    window_slow=feature.params["window_slow"],
                    window_fast=feature.params["window_fast"],
                    window_sign=feature.params["window_sign"]
                )
                result['macd'] = macd.macd()
                result['macd_signal'] = macd.macd_signal()
                result['macd_diff'] = macd.macd_diff()

            elif feature.name == "BB":
                bb = ta.volatility.BollingerBands(
                    close=df['close'],
                    window=feature.params["window"],
                    window_dev=feature.params["window_dev"]
                )
                result['bb_high'] = bb.bollinger_hband()
                result['bb_mid'] = bb.bollinger_mavg()
                result['bb_low'] = bb.bollinger_lband()

            return result

        except Exception as e:
            self.logger.error(f"Error calculating {feature.name}: {str(e)}")
            return None

    def _calculate_orderbook_features(self, df: pd.DataFrame,
                                      orderbook_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate orderbook-based features"""
        try:
            features = df.copy()

            if orderbook_data.empty:
                return features

            # Split bids and asks
            bids = orderbook_data[orderbook_data['side'] == 'bid']
            asks = orderbook_data[orderbook_data['side'] == 'ask']

            # Calculate spread
            features['spread'] = asks['price'].min() - bids['price'].max()
            features['spread_pct'] = features['spread'] / features['close']

            # Calculate imbalance
            bid_volume = bids['quantity'].sum()
            ask_volume = asks['quantity'].sum()
            features['orderbook_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating orderbook features: {str(e)}")
            return df

    def generate_features(self, symbol: str, start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Generate all features with parallel processing"""
        try:
            # Get price data
            price_data = self._get_price_data(symbol, start_time, end_time)
            if price_data.empty:
                return pd.DataFrame()

            # Calculate features in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_feature = {
                    executor.submit(func, price_data): name
                    for name, func in self.feature_funcs.items()
                }

                features = price_data.copy()
                for future in future_to_feature:
                    name = future_to_feature[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            features = pd.concat([features, result], axis=1)
                    except Exception as e:
                        self.logger.error(f"Error processing {name} features: {str(e)}")

            # Remove duplicate columns and handle missing values
            features = features.loc[:, ~features.columns.duplicated()]
            features = features.ffill().bfill()

            self.logger.info(f"Generated features shape: {features.shape}")
            self.logger.info(f"Features: {features.columns.tolist()}")

            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return pd.DataFrame()