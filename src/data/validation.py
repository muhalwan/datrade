import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_trade(self, trade: Dict) -> bool:
        try:
            required = {'price', 'quantity', 'timestamp', 'symbol'}
            if not required.issubset(trade.keys()):
                return False

            price = float(trade['price'])
            quantity = float(trade['quantity'])
            if price <= 0 or quantity <= 0:
                return False

            now = datetime.now()
            ts = pd.to_datetime(trade['timestamp']) if isinstance(trade['timestamp'], datetime) \
                else pd.to_datetime(trade['timestamp'], unit='ms')

            if ts > now + timedelta(minutes=1) or ts < now - timedelta(days=1):
                return False

            return True
        except Exception:
            return False

    def validate_orderbook(self, order: Dict) -> bool:
        try:
            required = {'timestamp', 'symbol', 'side', 'price', 'quantity'}
            if not required.issubset(order.keys()):
                return False

            price = float(order['price'])
            quantity = float(order['quantity'])
            if price <= 0 or quantity <= 0:
                return False

            if str(order['side']).lower() not in {'bid', 'ask'}:
                return False

            now = datetime.now()
            ts = pd.to_datetime(order['timestamp']) if isinstance(order['timestamp'], datetime) \
                else pd.to_datetime(order['timestamp'], unit='ms')

            if ts > now + timedelta(minutes=1) or ts < now - timedelta(minutes=1):
                return False

            return True
        except Exception:
            return False

class DataCleaner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_trade(self, trade: Dict) -> Optional[Dict]:
        try:
            timestamp = pd.to_datetime(trade['timestamp']) if isinstance(trade['timestamp'], datetime) \
                else pd.to_datetime(trade['timestamp'], unit='ms')

            return {
                'timestamp': timestamp,
                'symbol': str(trade['symbol']).upper(),
                'price': float(trade['price']),
                'quantity': float(trade['quantity']),
                'is_buyer_maker': bool(trade.get('m', False)),
                'trade_time': timestamp
            }
        except Exception as e:
            self.logger.error(f"Error cleaning trade: {e}")
            return None

    def clean_orderbook(self, order: Dict) -> Optional[Dict]:
        try:
            timestamp = pd.to_datetime(order['timestamp']) if isinstance(order['timestamp'], datetime) \
                else pd.to_datetime(order['timestamp'], unit='ms')

            return {
                'timestamp': timestamp,
                'symbol': str(order['symbol']).upper(),
                'side': str(order['side']).lower(),
                'price': float(order['price']),
                'quantity': float(order['quantity']),
                'update_id': int(order.get('update_id', 0))
            }
        except Exception as e:
            self.logger.error(f"Error cleaning orderbook: {e}")
            return None