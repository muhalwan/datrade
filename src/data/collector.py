import logging
from typing import List, Dict, Any
import requests
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from datetime import datetime, timedelta
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import numpy as np
import threading
import queue
import time

from .validation import DataValidator, DataCleaner
from typing import Optional, Dict, List
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from datetime import datetime, timedelta

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class BinanceDataCollector:
    """Binance data collector with improved error handling and recovery"""

    def __init__(self, config: Dict):
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        self.connection_string = config.get('connection_string')
        self.db_name = config.get('name')

        if not self.connection_string or not self.db_name:
            raise ValueError("Missing required configuration")

        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.logger = logging.getLogger(__name__)
        self._collections = {}

        # Initialize components
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

        # Queues for batch processing
        self.trade_queue = queue.Queue()
        self.orderbook_queue = queue.Queue()
        self.batch_lock = threading.Lock()

        # Market state tracking
        self.market_state = {
            'last_price': {},
            'bid_depth': {},
            'ask_depth': {},
            'volume_24h': {},
            'price_change_24h': {}
        }

        # Performance tracking
        self.stats = {
            'trades_processed': 0,
            'orderbook_updates': 0,
            'errors': 0,
            'start_time': None,
            'last_update': None
        }

        # Historical data
        self.trade_history = []
        self.depth_history = {
            'bid': [],
            'ask': []
        }
        self.max_history = 60

        # Setup connections
        self._setup_connections()
        self._start_batch_processor()

    def _setup_connections(self):
        """Setup API connections with retry logic"""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Setup Binance client
                self.client = Client(
                    api_key=self.auth.api_key,
                    api_secret=self.auth.secret_key,
                    testnet=self.use_testnet
                )

                # Setup websocket manager
                self.websocket = ThreadedWebsocketManager(
                    api_key=self.auth.api_key,
                    api_secret=self.auth.secret_key,
                    testnet=self.use_testnet
                )

                # Initialize MongoDB collections
                self.price_collection = self.db.get_collection('price_data')
                self.orderbook_collection = self.db.get_collection('order_book')

                if not self._validate_connections():
                    raise ConnectionError("Failed to validate connections")

                self.logger.info("Successfully connected to all services")
                return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)

    def _validate_connections(self) -> bool:
        """Validate all connections are working"""
        try:
            if not self.price_collection or not self.orderbook_collection:
                return False

            # Test MongoDB collections
            self.price_collection.find_one({})
            self.orderbook_collection.find_one({})

            # Test Binance connection
            self.client.ping()

            return True

        except Exception as e:
            self.logger.error(f"Connection validation error: {str(e)}")
            return False

    def _start_batch_processor(self):
        """Start batch processing thread"""
        def process_batches():
            last_log = time.time()

            while True:
                try:
                    current_time = time.time()

                    # Log queue sizes periodically
                    if current_time - last_log >= 10:
                        self.logger.info(f"Queue sizes - Trades: {self.trade_queue.qsize()}, Orders: {self.orderbook_queue.qsize()}")
                        last_log = current_time

                    # Process trades
                    trades = []
                    while not self.trade_queue.empty() and len(trades) < self.batch_size:
                        trade = self.trade_queue.get_nowait()
                        if trade:
                            trades.append(trade)

                    if trades:
                        try:
                            result = self.price_collection.insert_many(trades, ordered=False)
                            self.logger.info(f"Inserted {len(result.inserted_ids)} trades")
                        except Exception as e:
                            self.logger.error(f"Trade insertion error: {str(e)}")

                    # Process orderbook
                    orders = []
                    while not self.orderbook_queue.empty() and len(orders) < self.batch_size:
                        order = self.orderbook_queue.get_nowait()
                        if order:
                            orders.append(order)

                    if orders:
                        try:
                            result = self.orderbook_collection.insert_many(orders, ordered=False)
                            self.logger.info(f"Inserted {len(result.inserted_ids)} orders")
                        except Exception as e:
                            self.logger.error(f"Order insertion error: {str(e)}")

                    time.sleep(0.1)  # Prevent CPU overuse

                except Exception as e:
                    self.logger.error(f"Batch processing error: {str(e)}")
                    time.sleep(1)

        # Start processor thread
        self.batch_processor = threading.Thread(target=process_batches, daemon=True)
        self.batch_processor.start()
        self.logger.info("Batch processor started")

    def _handle_trade(self, msg: Dict):
        """Process trade message"""
        try:
            if 'data' in msg:
                msg = msg['data']

            trade_data = {
                'timestamp': datetime.fromtimestamp(int(msg['T']) / 1000),
                'symbol': msg['s'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'is_buyer_maker': bool(msg['m']),
                'trade_id': msg['t']
            }

            if self.validator.validate_trade(trade_data):
                clean_trade = self.cleaner.clean_trade(trade_data)
                if clean_trade:
                    self.trade_queue.put(clean_trade)

                    # Update trade history
                    volume = float(msg['q']) * float(msg['p'])
                    self.trade_history.append(volume)
                    if len(self.trade_history) > self.max_history:
                        self.trade_history = self.trade_history[-self.max_history:]

                    self.stats['trades_processed'] += 1
                    self.stats['last_update'] = datetime.now()

        except Exception as e:
            self.logger.error(f"Trade processing error: {str(e)}")
            self.stats['errors'] += 1

    def _handle_depth(self, msg: Dict):
        """Process orderbook depth message"""
        try:
            if 'data' in msg:
                msg = msg['data']

            timestamp = datetime.fromtimestamp(int(msg['T']) / 1000)
            symbol = msg['s']

            # Process bids
            for bid in msg.get('b', []):
                price, quantity = float(bid[0]), float(bid[1])
                order_data = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'bid',
                    'price': price,
                    'quantity': quantity,
                    'update_id': msg.get('u', 0)
                }

                if self.validator.validate_orderbook(order_data):
                    clean_order = self.cleaner.clean_orderbook(order_data)
                    if clean_order:
                        self.orderbook_queue.put(clean_order)
                        self.stats['orderbook_updates'] += 1
                        self.stats['last_update'] = datetime.now()

            # Process asks
            for ask in msg.get('a', []):
                price, quantity = float(ask[0]), float(ask[1])
                order_data = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'ask',
                    'price': price,
                    'quantity': quantity,
                    'update_id': msg.get('u', 0)
                }

                if self.validator.validate_orderbook(order_data):
                    clean_order = self.cleaner.clean_orderbook(order_data)
                    if clean_order:
                        self.orderbook_queue.put(clean_order)
                        self.stats['orderbook_updates'] += 1
                        self.stats['last_update'] = datetime.now()

        except Exception as e:
            self.logger.error(f"Depth processing error: {str(e)}")
            self.stats['errors'] += 1

    def _message_handler(self, msg):
        """Handle websocket messages"""
        try:
            self.logger.debug(f"Received message: {msg}")

            if 'e' in msg:
                event_type = msg['e']
                if event_type == 'trade':
                    self._handle_trade(msg)
                elif event_type == 'depthUpdate':
                    self._handle_depth(msg)
            elif 'data' in msg and 'e' in msg['data']:
                event_type = msg['data']['e']
                if event_type == 'trade':
                    self._handle_trade(msg)
                elif event_type == 'depthUpdate':
                    self._handle_depth(msg)

        except Exception as e:
            self.logger.error(f"Message handling error: {str(e)}")
            self.stats['errors'] += 1

    def start_collection(self):
        """Start data collection"""
        try:
            self.stats['start_time'] = datetime.now()
            self.logger.info("Starting WebSocket manager...")
            self.websocket.start()

            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                streams.extend([
                    f"{symbol_lower}@trade",
                    f"{symbol_lower}@depth@100ms"
                ])

            self.logger.info(f"Starting multiplex socket with streams: {streams}")
            self.websocket.start_multiplex_socket(
                streams=streams,
                callback=self._message_handler
            )

            # Verify connection
            time.sleep(5)
            if not self.websocket.is_alive():
                raise ConnectionError("WebSocket failed to start")

            self.logger.info(f"Started data collection for symbols: {self.symbols}")
            self.logger.info(f"Active streams: {streams}")

        except Exception as e:
            self.logger.error(f"Error starting collection: {str(e)}")
            raise

    def stop(self):
        """Stop data collection"""
        try:
            if self.websocket:
                self.websocket.stop()
            self.logger.info("Data collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping collection: {str(e)}")
            raise

    # Utility methods for monitoring
    def get_connection_status(self) -> str:
        """Get WebSocket connection status"""
        try:
            return "active" if self.websocket.is_alive() else "inactive"
        except:
            return "error"

    def get_db_status(self) -> str:
        """Get database connection status"""
        try:
            return "active" if self.db.validate_collections() else "inactive"
        except:
            return "error"

    def get_warnings(self) -> List[str]:
        """Get current warnings"""
        warnings = []
        if self.stats.get('errors', 0) > 0:
            warnings.append(f"Errors detected: {self.stats['errors']}")
        return warnings

    def get_volume_history(self) -> List[float]:
        """Get recent trade volume history"""
        return self.trade_history[-60:] if self.trade_history else [0] * 60

    def get_depth_history(self, side: str) -> List[float]:
        """Get recent orderbook depth history"""
        return self.depth_history[side][-60:] if side in self.depth_history else [0] * 60

    def get_collection_rate(self) -> float:
        """Calculate current collection rate"""
        if not self.stats.get('start_time'):
            return 0.0

        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        if elapsed <= 0:
            return 0.0

        total_updates = (self.stats.get('trades_processed', 0) +
                         self.stats.get('orderbook_updates', 0))
        return total_updates / elapsed