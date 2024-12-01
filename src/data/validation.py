import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

class DataValidator:
    """Validates trading data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_trade(self, trade: Dict) -> bool:
        """Validate trade data"""
        try:
            # Check required fields
            required = ['price', 'quantity', 'timestamp', 'symbol']
            if not all(k in trade for k in required):
                self.logger.debug(f"Missing required fields in trade data: {trade}")
                return False

            # Validate numeric values
            price = float(trade['price'])
            quantity = float(trade['quantity'])
            if price <= 0 or quantity <= 0:
                self.logger.debug(f"Invalid price or quantity in trade: price={price}, quantity={quantity}")
                return False

            # Validate timestamp
            if isinstance(trade['timestamp'], (int, float)):
                ts = pd.to_datetime(trade['timestamp'], unit='ms')
            else:
                ts = pd.to_datetime(trade['timestamp'])

            # Check if timestamp is reasonable
            now = datetime.now()
            if ts > now + timedelta(minutes=1) or ts < now - timedelta(days=1):
                self.logger.debug(f"Invalid timestamp in trade: {ts}")
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Trade validation failed: {e}")
            return False

    def validate_orderbook(self, order: Dict) -> bool:
        """Validate orderbook data"""
        try:
            # Check required fields
            required = ['timestamp', 'symbol', 'side', 'price', 'quantity']
            if not all(k in order for k in required):
                self.logger.debug(f"Missing required fields in order: {order}")
                return False

            # Validate numeric values
            try:
                price = float(order['price'])
                quantity = float(order['quantity'])
            except (ValueError, TypeError):
                self.logger.debug(f"Invalid price or quantity format in order: price={order['price']}, quantity={order['quantity']}")
                return False

            if price <= 0 or quantity <= 0:
                self.logger.debug(f"Non-positive price or quantity in order: price={price}, quantity={quantity}")
                return False

            # Validate side
            if str(order['side']).lower() not in ['bid', 'ask']:
                self.logger.debug(f"Invalid side in order: {order['side']}")
                return False

            # Validate timestamp
            if isinstance(order['timestamp'], (int, float)):
                ts = pd.to_datetime(order['timestamp'], unit='ms')
            else:
                ts = pd.to_datetime(order['timestamp'])

            now = datetime.now()
            if ts > now + timedelta(minutes=1) or ts < now - timedelta(minutes=1):
                self.logger.debug(f"Invalid timestamp in order: {ts}")
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Orderbook validation failed: {e}")
            return False

class DataCleaner:
    """Cleans and normalizes trading data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_trade(self, trade: Dict) -> Optional[Dict]:
        """Clean trade data"""
        try:
            # Handle timestamp conversion
            if isinstance(trade['timestamp'], (int, float)):
                timestamp = pd.to_datetime(trade['timestamp'], unit='ms')
            else:
                timestamp = pd.to_datetime(trade['timestamp'])

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
        """Clean orderbook data"""
        try:
            # Handle timestamp conversion
            if isinstance(order['timestamp'], (int, float)):
                timestamp = pd.to_datetime(order['timestamp'], unit='ms')
            else:
                timestamp = pd.to_datetime(order['timestamp'])

            cleaned_order = {
                'timestamp': timestamp,
                'symbol': str(order['symbol']).upper(),
                'side': str(order['side']).lower(),
                'price': float(order['price']),
                'quantity': float(order['quantity'])
            }

            # Add update_id if available
            if 'update_id' in order:
                cleaned_order['update_id'] = int(order['update_id'])

            return cleaned_order
        except Exception as e:
            self.logger.error(f"Error cleaning orderbook: {e}")
            return None