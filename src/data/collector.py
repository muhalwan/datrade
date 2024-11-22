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
        """Initialize the Binance data collector"""
        self.auth = auth
        self.symbols = symbols
        self.db = db
        self.use_testnet = use_testnet
        self.logger = logging.getLogger(__name__)

        # Initialize validator and cleaner
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

        # Statistics
        self.stats = {
            'trades_processed': 0,
            'trades_invalid': 0,
            'orderbook_updates': 0,
            'orderbook_invalid': 0,
            'errors': 0,
            'start_time': None
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
        """Setup websocket manager"""
        return ThreadedWebsocketManager(
            api_key=self.auth.api_key,
            api_secret=self.auth.secret_key,
            testnet=self.use_testnet
        )

    def _handle_socket_message(self, msg):
        """Handle incoming websocket messages"""
        try:
            if isinstance(msg, dict):
                data = msg.get('data', {})
                stream = msg.get('stream', '')

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

            # Log the raw message for debugging
            self.logger.debug(f"Raw depth message: {msg}")

            # Extract timestamp and symbol
            timestamp = datetime.fromtimestamp(int(msg.get('E', time.time() * 1000)) / 1000)
            symbol = msg.get('s', '')

            if not symbol:
                self.logger.debug("Missing symbol in depth message")
                return

            updates = 0
            invalid = 0

            # Process bids
            for bid in msg.get('b', []):  # Bids
                if len(bid) >= 2:
                    try:
                        # Log raw bid data
                        self.logger.debug(f"Processing bid: price={bid[0]}, quantity={bid[1]}")

                        order_data = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'side': 'bid',
                            'price': float(bid[0]),
                            'quantity': float(bid[1]),
                            'update_id': msg.get('u', msg.get('U', 0))
                        }

                        # If quantity is 0, it's a deletion
                        if float(bid[1]) == 0:
                            self.logger.debug(f"Deleting bid at price {bid[0]}")
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
                                # Update or insert the order
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
                                self.logger.debug(f"Successfully processed bid: {cleaned_order}")
                        else:
                            invalid += 1
                            self.logger.debug(f"Invalid bid data: {order_data}")

                    except Exception as e:
                        self.logger.debug(f"Error processing bid: {e}")
                        invalid += 1

            # Process asks
            for ask in msg.get('a', []):  # Asks
                if len(ask) >= 2:
                    try:
                        # Log raw ask data
                        self.logger.debug(f"Processing ask: price={ask[0]}, quantity={ask[1]}")

                        order_data = {
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'side': 'ask',
                            'price': float(ask[0]),
                            'quantity': float(ask[1]),
                            'update_id': msg.get('u', msg.get('U', 0))
                        }

                        # If quantity is 0, it's a deletion
                        if float(ask[1]) == 0:
                            self.logger.debug(f"Deleting ask at price {ask[0]}")
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
                                # Update or insert the order
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
                                self.logger.debug(f"Successfully processed ask: {cleaned_order}")
                        else:
                            invalid += 1
                            self.logger.debug(f"Invalid ask data: {order_data}")

                    except Exception as e:
                        self.logger.debug(f"Error processing ask: {e}")
                        invalid += 1

            self.stats['orderbook_updates'] += updates
            self.stats['orderbook_invalid'] += invalid

            # Only print stats if we have updates
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
                # Get best bids
                best_bids = list(self.orderbook_collection.find({
                    'symbol': symbol,
                    'side': 'bid',
                    'timestamp': {'$gte': minute_ago}
                }).sort('price', -1).limit(5))

                # Get best asks
                best_asks = list(self.orderbook_collection.find({
                    'symbol': symbol,
                    'side': 'ask',
                    'timestamp': {'$gte': minute_ago}
                }).sort('price', 1).limit(5))

                if best_bids and best_asks:
                    bid_price = best_bids[0]['price']
                    ask_price = best_asks[0]['price']
                    spread = ask_price - bid_price

                    # Build detailed orderbook view
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

                    # Debug information
                    total_bids = self.orderbook_collection.count_documents({
                        'symbol': symbol,
                        'side': 'bid'
                    })
                    total_asks = self.orderbook_collection.count_documents({
                        'symbol': symbol,
                        'side': 'ask'
                    })
                    self.logger.debug(f"Total bids in DB: {total_bids}")
                    self.logger.debug(f"Total asks in DB: {total_asks}")

                    # Check most recent entry
                    latest = self.orderbook_collection.find_one(
                        {'symbol': symbol},
                        sort=[('timestamp', -1)]
                    )
                    if latest:
                        self.logger.debug(f"Most recent entry: {latest}")

        except Exception as e:
            self.logger.error(f"Error printing orderbook summary: {str(e)}")
            self.logger.debug(f"Error details:", exc_info=True)

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

            # Start monitoring thread
            self._start_monitoring()

        except Exception as e:
            self.logger.error(f"Error starting data collection: {str(e)}")
            raise

    def _start_monitoring(self):
        """Start monitoring in background thread"""
        def monitor_loop():
            while True:
                try:
                    # Wait a bit before first summary
                    time.sleep(10)

                    # Print summary every 10 seconds
                    self.print_orderbook_summary()
                    time.sleep(10)

                except Exception as e:
                    self.logger.error(f"Monitor error: {e}")
                    time.sleep(5)

        import threading
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

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
        """Stop data collection"""
        try:
            self.websocket.stop()
            self.logger.info("Data collection stopped")

            # Print final stats
            runtime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
            self.logger.info(
                f"\nFinal Statistics after {int(runtime)}s:\n"
                f"Total Trades: {self.stats['trades_processed']}\n"
                f"Invalid Trades: {self.stats['trades_invalid']}\n"
                f"Total Orderbook Updates: {self.stats['orderbook_updates']}\n"
                f"Invalid Orderbook Updates: {self.stats['orderbook_invalid']}\n"
                f"Total Errors: {self.stats['errors']}"
            )

        except Exception as e:
            self.logger.error(f"Error stopping collection: {str(e)}")
            raise