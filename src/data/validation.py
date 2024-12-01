import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class DataValidator:
    """Validates incoming market data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_trade(self, trade: Dict) -> bool:
        """Validate trade data with enhanced checks"""
        try:
            # Required fields check
            required = {'timestamp', 'symbol', 'price', 'quantity'}
            if not required.issubset(trade.keys()):
                return False

            # Numeric validation
            if not self._validate_numeric_fields(trade, ['price', 'quantity']):
                return False

            # Timestamp validation
            if not self._validate_timestamp(trade['timestamp']):
                return False

            # Additional trade validations
            price = float(trade['price'])
            quantity = float(trade['quantity'])

            # Price and quantity must be positive
            if price <= 0 or quantity <= 0:
                return False

            # Check for unreasonable values
            if price > 1e10 or quantity > 1e10:  # Arbitrary large value check
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Trade validation failed: {str(e)}")
            return False

    def validate_orderbook(self, order: Dict) -> bool:
        """Validate orderbook data"""
        try:
            # Required fields check
            required = {'timestamp', 'symbol', 'side', 'price', 'quantity'}
            if not required.issubset(order.keys()):
                return False

            # Numeric validation
            if not self._validate_numeric_fields(order, ['price', 'quantity']):
                return False

            # Timestamp validation
            if not self._validate_timestamp(order['timestamp']):
                return False

            # Side validation
            if str(order['side']).lower() not in {'bid', 'ask'}:
                return False

            # Price and quantity validation
            price = float(order['price'])
            quantity = float(order['quantity'])

            if price <= 0 or quantity <= 0:
                return False

            # Check for unreasonable values
            if price > 1e10 or quantity > 1e10:
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Orderbook validation failed: {str(e)}")
            return False

    def _validate_numeric_fields(self, data: Dict, fields: List[str]) -> bool:
        """Validate numeric fields"""
        try:
            for field in fields:
                value = float(data[field])
                if not np.isfinite(value):  # Checks for NaN and Inf
                    return False
            return True
        except (ValueError, TypeError):
            return False

    def _validate_timestamp(self, timestamp) -> bool:
        """Validate timestamp is within acceptable range"""
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = pd.to_datetime(timestamp, unit='ms')
            elif not isinstance(timestamp, datetime):
                return False

            now = datetime.now()
            min_time = now - timedelta(days=1)
            max_time = now + timedelta(minutes=1)

            return min_time <= timestamp <= max_time

        except Exception:
            return False

class DataCleaner:
    """Cleans and normalizes market data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_trade(self, trade: Dict) -> Optional[Dict]:
        """Clean trade data"""
        try:
            timestamp = self._standardize_timestamp(trade['timestamp'])
            if timestamp is None:
                return None

            return {
                'timestamp': timestamp,
                'symbol': str(trade['symbol']).upper(),
                'price': float(trade['price']),
                'quantity': float(trade['quantity']),
                'is_buyer_maker': bool(trade.get('is_buyer_maker', False)),
                'trade_id': trade.get('trade_id'),
                'trade_time': timestamp
            }

        except Exception as e:
            self.logger.error(f"Trade cleaning error: {str(e)}")
            return None

    def clean_orderbook(self, order: Dict) -> Optional[Dict]:
        """Clean orderbook data"""
        try:
            timestamp = self._standardize_timestamp(order['timestamp'])
            if timestamp is None:
                return None

            return {
                'timestamp': timestamp,
                'symbol': str(order['symbol']).upper(),
                'side': str(order['side']).lower(),
                'price': float(order['price']),
                'quantity': float(order['quantity']),
                'update_id': int(order.get('update_id', 0))
            }

        except Exception as e:
            self.logger.error(f"Orderbook cleaning error: {str(e)}")
            return None

    def _standardize_timestamp(self, timestamp) -> Optional[datetime]:
        """Convert timestamp to standard datetime format"""
        try:
            if isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, str):
                return pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                return pd.to_datetime(timestamp, unit='ms')
            return None
        except Exception:
            return None

    def clean_ohlcv(self, candle: Dict) -> Optional[Dict]:
        """Clean OHLCV candle data"""
        try:
            timestamp = self._standardize_timestamp(candle['timestamp'])
            if timestamp is None:
                return None

            return {
                'timestamp': timestamp,
                'symbol': str(candle['symbol']).upper(),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume']),
                'trade_count': int(candle.get('trade_count', 0)),
                'vwap': self._calculate_vwap(candle),
                'typical_price': (float(candle['high']) + float(candle['low']) +
                                  float(candle['close'])) / 3,
                'buy_volume': float(candle.get('buy_volume', 0)),
                'sell_volume': float(candle.get('sell_volume', 0))
            }

        except Exception as e:
            self.logger.error(f"OHLCV cleaning error: {str(e)}")
            return None

    def _calculate_vwap(self, candle: Dict) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (float(candle['high']) + float(candle['low']) +
                             float(candle['close'])) / 3
            return typical_price * float(candle['volume'])
        except Exception:
            return 0.0