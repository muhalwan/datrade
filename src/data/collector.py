import asyncio
import logging
from typing import List, Dict, Any
import requests
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from datetime import datetime, timedelta
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import json
import time
from .auth import BinanceAuth, AuthType
from .database.connection import MongoDBConnection
from .validation import DataValidator, DataCleaner

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class BinanceDataCollector:
    """Optimized data collector for Binance exchange"""

    def __init__(self, auth: BinanceAuth, symbols: List[str], db: MongoDBConnection, use_testnet: bool = True):
        """Initialize with additional monitoring fields"""
        self.auth = auth
        self.symbols = symbols
        self.db = db
        self.use_testnet = use_testnet
        self.logger = logging.getLogger(__name__)

        # Initialize validator and cleaner
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

        # Add timestamp tracking
        self.last_message_time = datetime.now()

        # Statistics
        self.stats = {
            'trades_processed': 0,
            'trades_invalid': 0,
            'orderbook_updates': 0,
            'orderbook_invalid': 0,
            'errors': 0,
            'start_time': None,
            'reconnects': 0
        }

        # Initialize connections
        self.client = self._setup_client()
        self.websocket = self._setup_websocket()
        self.price_collection = self.db.get_collection('price_data')
        self.orderbook_collection = self.db.get_collection('order_book')

        if self.price_collection is None or self.orderbook_collection is None:
            raise ConnectionError("Failed to connect to MongoDB collections")

    def _setup_client(self) -> Client:
        """Setup Binance client"""
        return Client(
            api_key=self.auth.api_key,
            api_secret=self.auth.secret_key,
            testnet=self.use_testnet,
            requests_params={
                'timeout': 30,
                'verify': False
            }
        )

    def _setup_websocket(self) -> ThreadedWebsocketManager:
        """Setup websocket manager with error handling"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            return ThreadedWebsocketManager(
                api_key=self.auth.api_key,
                api_secret=self.auth.secret_key,
                testnet=self.use_testnet
            )
        except Exception as e:
            self.logger.error(f"Error setting up websocket: {e}")
            raise

    def _handle_socket_message(self, msg):
        """Handle incoming websocket messages with validation"""
        try:
            if isinstance(msg, dict):
                data = msg.get('data', {})
                stream = msg.get('stream', '')

                # Update last message timestamp
                self.last_message_time = datetime.now()

                if not stream:
                    self.logger.debug(f"Empty stream type in message: {msg}")
                    return

                if 'trade' in stream:
                    self._handle_trade_message(data)
                elif 'depth' in stream:
                    self._handle_depth_message(data)
                else:
                    self.logger.debug(f"Unknown stream type: {stream}")

        except Exception as e:
            self.logger.error(f"Socket message error: {e}")
            self.stats['errors'] += 1

    def _handle_trade_message(self, msg: Dict[str, Any]):
        """Handle trade messages"""
        try:
            if msg.get('e') == 'trade':
                trade_data = {
                    'timestamp': datetime.fromtimestamp(int(msg['E']) / 1000),
                    'symbol': msg['s'],
                    'price': float(msg['p']),
                    'quantity': float(msg['q']),
                    'm': msg.get('m', False)
                }

                if not self.validator.validate_trade(trade_data):
                    self.stats['trades_invalid'] += 1
                    return

                cleaned_trade = self.cleaner.clean_trade(trade_data)
                if cleaned_trade:
                    self.price_collection.insert_one(cleaned_trade)
                    self.stats['trades_processed'] += 1
                    self.print_stats()

        except Exception as e:
            self.logger.error(f"Trade error: {e}")
            self.stats['errors'] += 1

    def _handle_depth_message(self, msg: Dict[str, Any]):
        """Handle order book messages"""
        try:
            if not isinstance(msg, dict):
                self.logger.debug(f"Invalid message format: {type(msg)}")
                return

            timestamp = datetime.fromtimestamp(int(msg.get('E', time.time() * 1000)) / 1000)
            symbol = msg.get('s', '')

            if not symbol:
                self.logger.debug("Missing symbol in depth message")
                return

            updates = 0
            invalid = 0

            # Process bids
            for bid in msg.get('b', []):
                if len(bid) >= 2:
                    try:
                        order_data = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'side': 'bid',
                            'price': float(bid[0]),
                            'quantity': float(bid[1]),
                            'update_id': msg.get('u', msg.get('U', 0))
                        }

                        # Handle deletion (quantity = 0)
                        if float(bid[1]) == 0:
                            self.orderbook_collection.delete_one({
                                'symbol': symbol,
                                'side': 'bid',
                                'price': float(bid[0])
                            })
                            updates += 1
                            continue

                        if self.validator.validate_orderbook(order_data):
                            cleaned_order = self.cleaner.clean_orderbook(order_data)
                            if cleaned_order:
                                self.orderbook_collection.replace_one(
                                    {
                                        'symbol': symbol,
                                        'side': 'bid',
                                        'price': float(bid[0])
                                    },
                                    cleaned_order,
                                    upsert=True
                                )
                                updates += 1
                        else:
                            invalid += 1

                    except Exception as e:
                        self.logger.debug(f"Error processing bid: {e}")
                        invalid += 1

            # Process asks
            for ask in msg.get('a', []):
                if len(ask) >= 2:
                    try:
                        order_data = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'side': 'ask',
                            'price': float(ask[0]),
                            'quantity': float(ask[1]),
                            'update_id': msg.get('u', msg.get('U', 0))
                        }

                        # Handle deletion (quantity = 0)
                        if float(ask[1]) == 0:
                            self.orderbook_collection.delete_one({
                                'symbol': symbol,
                                'side': 'ask',
                                'price': float(ask[0])
                            })
                            updates += 1
                            continue

                        if self.validator.validate_orderbook(order_data):
                            cleaned_order = self.cleaner.clean_orderbook(order_data)
                            if cleaned_order:
                                self.orderbook_collection.replace_one(
                                    {
                                        'symbol': symbol,
                                        'side': 'ask',
                                        'price': float(ask[0])
                                    },
                                    cleaned_order,
                                    upsert=True
                                )
                                updates += 1
                        else:
                            invalid += 1

                    except Exception as e:
                        self.logger.debug(f"Error processing ask: {e}")
                        invalid += 1

            self.stats['orderbook_updates'] += updates
            self.stats['orderbook_invalid'] += invalid

            if updates > 0 or invalid > 0:
                self.print_stats()

        except Exception as e:
            self.logger.error(f"Orderbook error: {e}")
            self.stats['errors'] += 1

    def print_stats(self):
        """Print collection statistics"""
        if not self.stats['start_time']:
            return

        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        if runtime > 0 and runtime % 10 < 1:  # Print every ~10 seconds
            self.logger.info(
                f"Stats after {int(runtime)}s:\n"
                f"Trades: {self.stats['trades_processed']} (invalid: {self.stats['trades_invalid']})\n"
                f"OrderBook: {self.stats['orderbook_updates']} (invalid: {self.stats['orderbook_invalid']})\n"
                f"Errors: {self.stats['errors']}"
            )

    def print_orderbook_summary(self):
        """Print orderbook summary"""
        try:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            for symbol in self.symbols:
                # Get best bids and asks
                best_bids = list(self.orderbook_collection.find({
                    'symbol': symbol,
                    'side': 'bid',
                    'timestamp': {'$gte': minute_ago}
                }).sort('price', -1).limit(5))

                best_asks = list(self.orderbook_collection.find({
                    'symbol': symbol,
                    'side': 'ask',
                    'timestamp': {'$gte': minute_ago}
                }).sort('price', 1).limit(5))

                if best_bids and best_asks:
                    bid_price = best_bids[0]['price']
                    ask_price = best_asks[0]['price']
                    spread = ask_price - bid_price

                    summary = f"\nOrderbook Summary for {symbol}:\n"
                    summary += f"Time: {now.strftime('%H:%M:%S.%f')[:-3]}\n"
                    summary += "\nTop 5 Bids:\n"
                    for bid in best_bids:
                        summary += f"${bid['price']:,.2f} ({bid['quantity']:.8f} BTC)\n"

                    summary += "\nTop 5 Asks:\n"
                    for ask in best_asks:
                        summary += f"${ask['price']:,.2f} ({ask['quantity']:.8f} BTC)\n"

                    summary += f"\nSpread: ${spread:,.2f} ({(spread/bid_price*100):.3f}%)"

                    self.logger.info(summary)
                else:
                    self.logger.info(f"\nNo recent orderbook data for {symbol}")

        except Exception as e:
            self.logger.error(f"Error printing orderbook summary: {str(e)}")

    def start_data_collection(self):
        """Start data collection"""
        try:
            if not self._test_connection():
                raise ConnectionError("Failed to connect to Binance API")

            self.websocket.start()
            self.stats['start_time'] = datetime.now()

            for symbol in self.symbols:
                symbol_lower = symbol.lower()

                # Individual streams
                streams = [
                    f"{symbol_lower}@trade",       # Trade stream
                    f"{symbol_lower}@depth@100ms"  # Depth stream with 100ms updates
                ]

                # Start combined stream
                self.websocket.start_multiplex_socket(
                    callback=self._handle_socket_message,
                    streams=streams
                )

            self.logger.info(f"Started data collection for symbols: {self.symbols}")
            self._start_monitoring()

        except Exception as e:
            self.logger.error(f"Error starting data collection: {str(e)}")
            raise

    def _start_monitoring(self):
        """Start monitoring in background thread with connection checking"""
        def monitor_loop():
            consecutive_fails = 0
            max_fails = 3

            while True:
                try:
                    time.sleep(10)

                    # Check if we're still receiving orderbook data
                    now = datetime.now()
                    minute_ago = now - timedelta(minutes=1)

                    recent_orders = self.orderbook_collection.count_documents({
                        'timestamp': {'$gte': minute_ago}
                    })

                    if recent_orders == 0:
                        consecutive_fails += 1
                        self.logger.warning(f"No recent orderbook data (fail {consecutive_fails}/{max_fails})")

                        if consecutive_fails >= max_fails:
                            self.logger.error("Connection appears to be lost, restarting streams...")
                            self._restart_streams()
                            consecutive_fails = 0
                    else:
                        consecutive_fails = 0
                        self.print_orderbook_summary()

                except Exception as e:
                    self.logger.error(f"Monitor error: {e}")
                    time.sleep(5)

        import threading
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _restart_streams(self):
        """Restart WebSocket streams with proper cleanup"""
        try:
            # Stop existing websocket
            try:
                self.websocket.stop()
                time.sleep(1)
            except Exception as e:
                self.logger.warning(f"Error stopping websocket: {e}")

            # Clean up old event loop if exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
                loop.close()
            except Exception as e:
                self.logger.warning(f"Error cleaning up event loop: {e}")

            # Set up new event loop
            asyncio.set_event_loop(asyncio.new_event_loop())

            # Create new websocket
            self.websocket = ThreadedWebsocketManager(
                api_key=self.auth.api_key,
                api_secret=self.auth.secret_key,
                testnet=self.use_testnet
            )

            # Start new websocket
            self.websocket.start()
            time.sleep(1)  # Wait for connection

            # Restart streams
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                streams = [
                    f"{symbol_lower}@trade",
                    f"{symbol_lower}@depth@100ms"
                ]

                # Add retry logic for stream connection
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        self.websocket.start_multiplex_socket(
                            callback=self._handle_socket_message,
                            streams=streams
                        )
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise
                        self.logger.warning(f"Retry {retry + 1} for stream connection: {e}")
                        time.sleep(1)

            self.logger.info("Successfully restarted WebSocket streams")
            self.stats['reconnects'] += 1

        except Exception as e:
            self.logger.error(f"Error restarting streams: {e}", exc_info=True)
            # Force complete restart if stream restart fails
            self.force_restart()

    def force_restart(self):
        """Force a complete restart of the collector"""
        try:
            self.logger.warning("Forcing complete collector restart...")

            # Stop everything
            self.stop()
            time.sleep(2)

            # Reset event loop
            asyncio.set_event_loop(asyncio.new_event_loop())

            # Reinitialize connections
            self.client = self._setup_client()
            self.websocket = self._setup_websocket()

            # Test connection
            if not self._test_connection():
                raise ConnectionError("Failed to reconnect to Binance API")

            # Restart data collection
            self.start_data_collection()
            self.logger.info("Successfully performed complete restart")

        except Exception as e:
            self.logger.error(f"Error during force restart: {e}")
            raise

    def _test_connection(self) -> bool:
        """Test API connection"""
        try:
            self.client.ping()
            server_time = self.client.get_server_time()
            self.logger.info(f"Connected to Binance {'testnet' if self.use_testnet else 'mainnet'}")
            self.logger.info(f"Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False

    def stop(self):
        """Stop data collection with improved cleanup"""
        try:
            # Stop websocket
            if hasattr(self, 'websocket'):
                try:
                    self.websocket.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping websocket: {e}")

            # Clean up event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
                loop.close()
            except Exception as e:
                self.logger.warning(f"Error cleaning up event loop: {e}")

            self.logger.info("Data collection stopped")

            if hasattr(self, 'stats') and self.stats.get('start_time'):
                runtime = (datetime.now() - self.stats['start_time']).total_seconds()
                self.logger.info(
                    f"\nFinal Statistics after {int(runtime)}s:\n"
                    f"Total Trades: {self.stats['trades_processed']}\n"
                    f"Invalid Trades: {self.stats['trades_invalid']}\n"
                    f"Total Orderbook Updates: {self.stats['orderbook_updates']}\n"
                    f"Invalid Orderbook Updates: {self.stats['orderbook_invalid']}\n"
                    f"Total Reconnects: {self.stats.get('reconnects', 0)}\n"
                    f"Total Errors: {self.stats['errors']}"
                )

        except Exception as e:
            self.logger.error(f"Error stopping collection: {str(e)}")
            raise
