from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from dataclasses import dataclass
from enum import Enum

class FeatureType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    SENTIMENT = "sentiment"

@dataclass
class FeatureConfig:
    name: str
    type: FeatureType
    params: Dict
    enabled: bool = True

class FeatureEngineering:
    """Feature engineering pipeline for cryptocurrency trading data"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection

        # Default feature configurations
        self.default_features = [
            FeatureConfig(
                name="SMA",
                type=FeatureType.TREND,
                params={"windows": [7, 14, 21, 50, 200]}
            ),
            FeatureConfig(
                name="EMA",
                type=FeatureType.TREND,
                params={"windows": [7, 14, 21, 50, 200]}
            ),
            FeatureConfig(
                name="RSI",
                type=FeatureType.MOMENTUM,
                params={"window": 14}
            ),
            FeatureConfig(
                name="MACD",
                type=FeatureType.MOMENTUM,
                params={"window_slow": 26, "window_fast": 12, "window_sign": 9}
            ),
            FeatureConfig(
                name="BB",
                type=FeatureType.VOLATILITY,
                params={"window": 20, "window_dev": 2}
            ),
            FeatureConfig(
                name="OBV",
                type=FeatureType.VOLUME,
                params={}
            )
        ]

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

            cursor = collection.find(query).sort("timestamp", 1)
            df = pd.DataFrame(list(cursor))

            if df.empty:
                self.logger.warning(f"No price data found for {symbol}")
                return pd.DataFrame()

            # Convert to OHLCV format
            ohlcv = df.groupby(pd.Grouper(key='timestamp', freq='1Min')).agg({
                'price': ['first', 'max', 'min', 'last'],
                'quantity': 'sum'
            }).droplevel(0, axis=1)

            ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
            return ohlcv

        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using ta library"""
        try:
            features = df.copy()

            for feature in self.default_features:
                if not feature.enabled:
                    continue

                if feature.name == "SMA":
                    for window in feature.params["windows"]:
                        features[f'sma_{window}'] = ta.trend.sma_indicator(
                            features['close'], window=window
                        )

                elif feature.name == "EMA":
                    for window in feature.params["windows"]:
                        features[f'ema_{window}'] = ta.trend.ema_indicator(
                            features['close'], window=window
                        )

                elif feature.name == "RSI":
                    features['rsi'] = ta.momentum.rsi(
                        features['close'],
                        window=feature.params["window"]
                    )

                elif feature.name == "MACD":
                    macd = ta.trend.MACD(
                        close=features['close'],
                        window_slow=feature.params["window_slow"],
                        window_fast=feature.params["window_fast"],
                        window_sign=feature.params["window_sign"]
                    )
                    features['macd'] = macd.macd()
                    features['macd_signal'] = macd.macd_signal()
                    features['macd_diff'] = macd.macd_diff()

                elif feature.name == "BB":
                    bb = ta.volatility.BollingerBands(
                        close=features['close'],
                        window=feature.params["window"],
                        window_dev=feature.params["window_dev"]
                    )
                    features['bb_high'] = bb.bollinger_hband()
                    features['bb_mid'] = bb.bollinger_mavg()
                    features['bb_low'] = bb.bollinger_lband()

                elif feature.name == "OBV":
                    features['obv'] = ta.volume.on_balance_volume(
                        close=features['close'],
                        volume=features['volume']
                    )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {str(e)}")
            return df

    def calculate_orderbook_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Calculate order book features"""
        try:
            collection = self.db.get_collection('order_book')

            # Get recent orderbook data
            window = timestamp - timedelta(minutes=1)
            query = {
                "symbol": symbol,
                "timestamp": {"$gte": window, "$lte": timestamp}
            }

            # Separate bids and asks
            bids = list(collection.find({**query, "side": "bid"}))
            asks = list(collection.find({**query, "side": "ask"}))

            if not bids or not asks:
                return {}

            # Calculate features
            bid_prices = [x['price'] for x in bids]
            ask_prices = [x['price'] for x in asks]
            bid_quantities = [x['quantity'] for x in bids]
            ask_quantities = [x['quantity'] for x in asks]

            features = {
                'bid_ask_spread': min(ask_prices) - max(bid_prices),
                'bid_depth': sum(bid_quantities),
                'ask_depth': sum(ask_quantities),
                'bid_ask_ratio': sum(bid_quantities) / sum(ask_quantities),
                'weighted_mid_price': (
                                              sum(p * q for p, q in zip(bid_prices, bid_quantities)) +
                                              sum(p * q for p, q in zip(ask_prices, ask_quantities))
                                      ) / (sum(bid_quantities) + sum(ask_quantities))
            }

            return features

        except Exception as e:
            self.logger.error(f"Error calculating orderbook features: {str(e)}")
            return {}

    def generate_features(self, symbol: str, start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Generate all features for the given symbol and time range"""
        try:
            # Get price data and calculate technical features
            price_data = self.get_price_data(symbol, start_time, end_time)
            if price_data.empty:
                return pd.DataFrame()

            features = self.calculate_technical_features(price_data)

            # Calculate orderbook features for each timestamp
            orderbook_features = []
            for timestamp in features.index:
                ob_feats = self.calculate_orderbook_features(symbol, timestamp)
                orderbook_features.append(ob_feats)

            # Combine all features
            if orderbook_features:
                ob_df = pd.DataFrame(orderbook_features, index=features.index)
                features = pd.concat([features, ob_df], axis=1)

            # Drop rows with NaN values from initialization of indicators
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return pd.DataFrame()