import logging
from typing import List, Dict, Any
import requests
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from datetime import datetime
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import threading
import time
import queue

from .auth import BinanceAuth, AuthType
from .database.connection import MongoDBConnection
from .validation import DataValidator, DataCleaner

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class BinanceDataCollector:
    def __init__(self, auth: BinanceAuth, symbols: List[str], db: MongoDBConnection,
                 use_testnet: bool = True, batch_size: int = 100):
        self.auth = auth
        self.symbols = symbols
        self.db = db
        self.use_testnet = use_testnet
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # Batch processing
        self.trade_queue = queue.Queue()
        self.orderbook_queue = queue.Queue()
        self.batch_lock = threading.Lock()

        # Initialize components
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self._setup_connections()

        # Start batch processor
        self._start_batch_processor()

        # Add statistics tracking
        self.stats = {
            'trades_processed': 0,
            'orderbook_updates': 0,
            'errors': 0,
            'start_time': None
        }

        # Add history tracking
        self.trade_history = []
        self.depth_history = {
            'bid': [],
            'ask': []
        }
        self.max_history = 60

    def _setup_connections(self):
        """Setup API connections with retries"""
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

                # Setup websocket
                self.websocket = ThreadedWebsocketManager(
                    api_key=self.auth.api_key,
                    api_secret=self.auth.secret_key,
                    testnet=self.use_testnet
                )

                # Get MongoDB collections
                self.price_collection = self.db.get_collection('price_data')
                self.orderbook_collection = self.db.get_collection('order_book')

                # Validate collections properly
                if self.price_collection is None or self.orderbook_collection is None:
                    raise ConnectionError("Failed to get MongoDB collections")

                # Test collections with a simple query
                self.price_collection.find_one({})  # Test price collection
                self.orderbook_collection.find_one({})  # Test orderbook collection

                # Test Binance connection
                self.client.ping()

                self.logger.info("Successfully connected to all services")
                return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)

    def get_connection_status(self) -> str:
        """Get websocket connection status"""
        try:
            return "active" if self.websocket.is_alive() else "error"
        except:
            return "error"

    def get_db_status(self) -> str:
        """Get database connection status"""
        try:
            return "active" if self.db.validate_collections() else "error"
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

    def _validate_collections(self) -> bool:
        """Validate MongoDB collections are accessible"""
        try:
            if self.price_collection is None or self.orderbook_collection is None:
                return False

            # Test both collections with simple queries
            self.price_collection.find_one({})
            self.orderbook_collection.find_one({})
            return True

        except Exception as e:
            self.logger.error(f"Collection validation error: {str(e)}")
            return False

    def _start_batch_processor(self):
        def process_batches():
            while True:
                try:
                    # Process trades
                    trades = []
                    while not self.trade_queue.empty() and len(trades) < self.batch_size:
                        trades.append(self.trade_queue.get_nowait())

                    if trades:
                        self.price_collection.insert_many(trades, ordered=False)

                    # Process orderbook
                    orders = []
                    while not self.orderbook_queue.empty() and len(orders) < self.batch_size:
                        orders.append(self.orderbook_queue.get_nowait())

                    if orders:
                        self.orderbook_collection.insert_many(orders, ordered=False)

                    time.sleep(0.1)  # Prevent CPU overuse

                except Exception as e:
                    self.logger.error(f"Batch processing error: {str(e)}")
                    time.sleep(1)

        threading.Thread(target=process_batches, daemon=True).start()

    def _handle_trade(self, msg: Dict):
        """Enhanced trade handler with statistics"""
        try:
            if msg.get('e') != 'trade':
                return

            trade_data = {
                'timestamp': datetime.fromtimestamp(int(msg['E']) / 1000),
                'symbol': msg['s'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'is_buyer_maker': bool(msg.get('m', False))
            }

            if not self.validator.validate_trade(trade_data):
                return

            clean_trade = self.cleaner.clean_trade(trade_data)
            if clean_trade:
                # Update statistics
                self.stats['trades_processed'] += 1

                # Update history
                volume = float(msg['q']) * float(msg['p'])
                self.trade_history.append(volume)
                if len(self.trade_history) > self.max_history:
                    self.trade_history = self.trade_history[-self.max_history:]

                # Add to database queue
                self.trade_queue.put(clean_trade)

        except Exception as e:
            self.logger.error(f"Trade processing error: {str(e)}")
            self.stats['errors'] += 1

    def _handle_depth(self, msg: Dict):
        """Enhanced depth handler with statistics"""
        try:
            timestamp = datetime.fromtimestamp(int(msg.get('E', time.time() * 1000)) / 1000)
            symbol = msg.get('s')

            if not symbol:
                return

            # Track depth
            bid_depth = 0
            ask_depth = 0

            for bid in msg.get('b', []):
                if len(bid) >= 2:
                    price, quantity = float(bid[0]), float(bid[1])
                    bid_depth += quantity

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

            for ask in msg.get('a', []):
                if len(ask) >= 2:
                    price, quantity = float(ask[0]), float(ask[1])
                    ask_depth += quantity

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

            # Update depth history
            self.depth_history['bid'].append(bid_depth)
            self.depth_history['ask'].append(ask_depth)

            # Maintain history size
            if len(self.depth_history['bid']) > self.max_history:
                self.depth_history['bid'] = self.depth_history['bid'][-self.max_history:]
            if len(self.depth_history['ask']) > self.max_history:
                self.depth_history['ask'] = self.depth_history['ask'][-self.max_history:]

        except Exception as e:
            self.logger.error(f"Depth processing error: {str(e)}")
            self.stats['errors'] += 1

    def _message_handler(self, msg):
        try:
            if 'data' not in msg:
                return

            data = msg['data']
            stream = msg.get('stream', '')

            if 'trade' in stream:
                self._handle_trade(data)
            elif 'depth' in stream:
                self._handle_depth(data)

        except Exception as e:
            self.logger.error(f"Message handling error: {str(e)}")

    def start_collection(self):
        """Enhanced start_collection with timestamp"""
        try:
            self.stats['start_time'] = datetime.now()
            self.websocket.start()

            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                streams.extend([
                    f"{symbol_lower}@trade",
                    f"{symbol_lower}@depth@100ms"
                ])

            self.websocket.start_multiplex_socket(
                callback=self._message_handler,
                streams=streams
            )

            self.logger.info(f"Started data collection for symbols: {self.symbols}")

        except Exception as e:
            self.logger.error(f"Error starting collection: {str(e)}")
            raise

    def stop(self):
        try:
            self.websocket.stop()
            self.logger.info("Data collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping collection: {str(e)}")
            raise