import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class FeatureEngineering:
    """Feature engineering pipeline"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection

        # Technical features configuration
        self.tech_features = {
            'sma': [7, 14, 21, 50, 200],
            'ema': [7, 14, 21, 50, 200],
            'bb_window': 20,
            'bb_std': 2,
            'rsi_window': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_window': 20
        }

    def get_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data"""
        try:
            # Basic data cleaning
            df = df.copy()
            df = df.dropna()
            df = df.sort_index()

            return df
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return pd.DataFrame()


    def _get_price_data(self, symbol: str, start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        """Get price data with efficient MongoDB aggregation"""
        try:
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
                    '$group': {
                        '_id': {
                            '$dateTrunc': {
                                'date': '$timestamp',
                                'unit': 'minute'
                            }
                        },
                        'open': {'$first': '$price'},
                        'high': {'$max': '$price'},
                        'low': {'$min': '$price'},
                        'close': {'$last': '$price'},
                        'volume': {'$sum': '$quantity'}
                    }
                },
                {
                    '$sort': {'_id': 1}
                }
            ]

            collection = self.db.get_collection('price_data')
            data = list(collection.aggregate(pipeline))

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df.rename(columns={'_id': 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error getting price data: {str(e)}")
            return pd.DataFrame()

    def _get_orderbook_data(self, symbol: str, start_time: datetime,
                            end_time: datetime) -> pd.DataFrame:
        """Get orderbook data with efficient MongoDB aggregation"""
        try:
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
                    '$group': {
                        '_id': {
                            'time': {
                                '$dateTrunc': {
                                    'date': '$timestamp',
                                    'unit': 'minute'
                                }
                            },
                            'side': '$side'
                        },
                        'depth': {'$sum': '$quantity'},
                        'avg_price': {'$avg': '$price'},
                        'orders': {'$sum': 1}
                    }
                }
            ]

            collection = self.db.get_collection('order_book')
            data = list(collection.aggregate(pipeline))

            if not data:
                return pd.DataFrame()

            # Transform data
            records = []
            for record in data:
                time = record['_id']['time']
                side = record['_id']['side']
                prefix = 'bid_' if side == 'bid' else 'ask_'
                records.append({
                    'timestamp': time,
                    f'{prefix}depth': record['depth'],
                    f'{prefix}price': record['avg_price'],
                    f'{prefix}orders': record['orders']
                })

            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error getting orderbook data: {str(e)}")
            return pd.DataFrame()

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features"""
        try:
            self.logger.info(f"Generating features from data shape: {df.shape}")
            features = self.get_data(df)

            if features.empty:
                self.logger.warning("No data available for feature generation")
                return pd.DataFrame()

            # Generate features in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_feature = {
                    executor.submit(self._calculate_price_features, features): 'price',
                    executor.submit(self._calculate_volume_features, features): 'volume',
                    executor.submit(self._calculate_technical_features, features): 'technical',
                    executor.submit(self._calculate_advanced_features, features): 'advanced',
                    executor.submit(self._calculate_order_flow_features, features): 'order_flow'
                }

                # Collect results
                for future in future_to_feature:
                    name = future_to_feature[future]
                    try:
                        result = future.result()
                        if result is not None:
                            features = pd.concat([features, result], axis=1)
                            self.logger.info(f"Generated {name} features")
                    except Exception as e:
                        self.logger.error(f"Error generating {name} features: {str(e)}")

            # Remove duplicate columns and handle missing values
            features = features.loc[:, ~features.columns.duplicated()]
            features = features.ffill().bfill()

            self.logger.info(f"Final features shape: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return pd.DataFrame()

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Returns
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])

            # Price momentum
            for window in [5, 15, 30]:
                features[f'momentum_{window}'] = df['close'].pct_change(window)

            # Price volatility
            for window in [5, 15, 30]:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return None

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Volume momentum
            features['volume_momentum'] = df['volume'].pct_change()

            # Volume moving averages
            for window in [5, 15, 30]:
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()

            # Money flow index
            features['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=self.tech_features['rsi_window'],
                fillna=True
            )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return None

    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            features = pd.DataFrame(index=df.index)

            # Moving averages
            for window in self.tech_features['sma']:
                features[f'sma_{window}'] = ta.trend.sma_indicator(
                    df['close'], window=window, fillna=True
                )
                features[f'ema_{window}'] = ta.trend.ema_indicator(
                    df['close'], window=window, fillna=True
                )

            # RSI
            features['rsi'] = ta.momentum.rsi(
                df['close'],
                window=self.tech_features['rsi_window'],
                fillna=True
            )

            # MACD
            macd = ta.trend.MACD(
                close=df['close'],
                window_slow=self.tech_features['macd_slow'],
                window_fast=self.tech_features['macd_fast'],
                window_sign=self.tech_features['macd_signal'],
                fillna=True
            )
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=self.tech_features['bb_window'],
                window_dev=self.tech_features['bb_std'],
                fillna=True
            )
            features['bb_high'] = bb.bollinger_hband()
            features['bb_middle'] = bb.bollinger_mavg()
            features['bb_low'] = bb.bollinger_lband()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {str(e)}")
            return None

    def _calculate_advanced_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate advanced technical features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Pivot Points
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate Pivot Point (PP)
            pp = (high + low + close) / 3
            features['pivot_point'] = pp

            # Calculate Support and Resistance levels
            features['r1'] = 2 * pp - low  # Resistance level 1
            features['s1'] = 2 * pp - high  # Support level 1

            # Price position relative to PP
            features['price_to_pivot'] = close / pp - 1

            # Volatility features
            features['true_range'] = self._calculate_true_range(df)
            features['volatility_index'] = features['true_range'].rolling(14).mean()

            # Volume profile
            volume = df['volume']
            price_bins = pd.qcut(close, q=10, labels=False, duplicates='drop')
            vol_profile = volume.groupby(price_bins).sum()
            features['volume_concentration'] = volume / vol_profile[price_bins].values

            # Market efficiency
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            features['market_efficiency'] = abs(returns.rolling(20).sum()) / (returns.abs().rolling(20).sum())
            features['volatility_ratio'] = volatility / volatility.rolling(100).mean()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating advanced features: {str(e)}")
            return None

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        ranges = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close),
            'lc': abs(low - close)
        })

        return ranges.max(axis=1)

    def _calculate_order_flow_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate order flow based features"""
        try:
            features = pd.DataFrame(index=df.index)

            volume = df['volume']
            close = df['close']

            # Volume momentum
            features['volume_momentum'] = volume.pct_change()

            # Price-volume correlation
            features['price_volume_corr'] = (
                close.rolling(20)
                .corr(volume)
                .fillna(0)
            )

            # Volume weighted average price (VWAP)
            features['vwap'] = (volume * close).cumsum() / volume.cumsum()
            features['price_to_vwap'] = close / features['vwap'] - 1

            return features

        except Exception as e:
            self.logger.error(f"Error calculating order flow features: {str(e)}")
            return None

    def _calculate_orderbook_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate orderbook features"""
        try:
            features = pd.DataFrame(index=df.index)

            if 'bid_depth' not in df.columns or 'ask_depth' not in df.columns:
                return features

            # Basic features
            features['spread'] = df['ask_price'] - df['bid_price']
            features['spread_pct'] = features['spread'] / df['close']
            features['mid_price'] = (df['ask_price'] + df['bid_price']) / 2

            # Imbalance features
            total_depth = df['bid_depth'] + df['ask_depth']
            features['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / total_depth

            # Order imbalance
            total_orders = df['bid_orders'] + df['ask_orders']
            features['order_imbalance'] = (df['bid_orders'] - df['ask_orders']) / total_orders

            return features

        except Exception as e:
            self.logger.error(f"Error calculating orderbook features: {str(e)}")
            return None