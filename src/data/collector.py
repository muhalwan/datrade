import logging
from typing import List, Dict, Any
import requests
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from datetime import datetime
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import json
from .auth import BinanceAuth, AuthType
from .database.connection import MongoDBConnection

# Suppress only the single InsecureRequestWarning
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
        self.stats = {
            'klines_processed': 0,
            'trades_processed': 0,
            'orderbook_updates': 0,
            'start_time': None
        }

        # Initialize client
        self.client = self._setup_client()

        # Initialize websocket manager
        self.websocket = self._setup_websocket()

        # Get collection references
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
            # Convert string message to dict if needed
            if isinstance(msg, str):
                msg = json.loads(msg)

            # Process the message
            if isinstance(msg, dict):
                event_type = msg.get('e')
                if event_type == 'kline':
                    self._handle_kline_message(msg)
                elif event_type == 'trade':
                    self._handle_trade_message(msg)
                elif event_type == 'depthUpdate':
                    self._handle_depth_message(msg)
                else:
                    self.logger.debug(f"Received unknown message type: {event_type}")
            else:
                self.logger.warning(f"Received non-dict message: {type(msg)}")

        except Exception as e:
            self.logger.error(f"Error processing socket message: {str(e)}, message: {msg}")

    def _handle_kline_message(self, msg: Dict[str, Any]):
        """Handle incoming kline/candlestick messages"""
        try:
            if msg.get('e') == 'kline':
                k = msg['k']

                price_data = {
                    'timestamp': datetime.fromtimestamp(msg['E'] / 1000),
                    'symbol': msg['s'],
                    'interval': k['i'],
                    'open_time': datetime.fromtimestamp(k['t'] / 1000),
                    'close_time': datetime.fromtimestamp(k['T'] / 1000),
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v']),
                    'quote_volume': float(k['q']),
                    'trades': int(k['n']),
                    'taker_buy_volume': float(k.get('V', 0)),
                    'taker_buy_quote_volume': float(k.get('Q', 0))
                }

                self.price_collection.insert_one(price_data)
                self.stats['klines_processed'] += 1
                self.print_stats()

        except Exception as e:
            self.logger.error(f"Error processing kline message: {str(e)}")

    def _handle_depth_message(self, msg: Dict[str, Any]):
        """Handle incoming order book messages"""
        try:
            timestamp = datetime.fromtimestamp(msg['E'] / 1000) if 'E' in msg else datetime.now()
            symbol = msg.get('s', '')

            # Process bids and asks
            updates = 0
            for side, data in [('bid', msg.get('b', [])), ('ask', msg.get('a', []))]:
                for price_qty in data:
                    if len(price_qty) >= 2:
                        try:
                            order_data = {
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'side': side,
                                'price': float(price_qty[0]),
                                'quantity': float(price_qty[1]),
                                'update_id': msg.get('u', 0)
                            }
                            self.orderbook_collection.insert_one(order_data)
                            updates += 1
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"Error processing {side} data: {str(e)}")

            self.stats['orderbook_updates'] += updates
            self.print_stats()

        except Exception as e:
            self.logger.error(f"Error processing depth message: {str(e)}")

    def _handle_trade_message(self, msg: Dict[str, Any]):
        """Handle incoming trade messages"""
        try:
            if msg.get('e') == 'trade':
                try:
                    trade_data = {
                        'timestamp': datetime.fromtimestamp(int(msg['E']) / 1000),
                        'symbol': msg['s'],
                        'trade_id': msg['t'],
                        'price': float(msg['p']),
                        'quantity': float(msg['q']),
                        'buyer_order_id': msg.get('b', ''),
                        'seller_order_id': msg.get('a', ''),
                        'trade_time': datetime.fromtimestamp(int(msg['T']) / 1000),
                        'is_buyer_maker': msg.get('m', False)
                    }

                    self.price_collection.insert_one(trade_data)
                    self.stats['trades_processed'] += 1
                    self.print_stats()

                except (ValueError, TypeError, KeyError) as e:
                    self.logger.error(f"Error parsing trade data: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error processing trade message: {str(e)}")

    def print_stats(self):
        """Print collection statistics"""
        if not self.stats['start_time']:
            return

        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        if runtime > 0 and runtime % 10 < 1:  # Print every ~10 seconds
            self.logger.info(
                f"Stats after {int(runtime)}s - "
                f"Klines: {self.stats['klines_processed']}, "
                f"Trades: {self.stats['trades_processed']}, "
                f"OrderBook: {self.stats['orderbook_updates']}"
            )

    def start_data_collection(self):
        """Start all data collection streams"""
        try:
            if not self._test_connection():
                raise ConnectionError("Failed to connect to Binance API")

            self.websocket.start()
            self.stats['start_time'] = datetime.now()

            for symbol in self.symbols:
                symbol_lower = symbol.lower()

                # Start individual streams
                self.websocket.start_kline_socket(
                    symbol=symbol_lower,
                    interval='1m',
                    callback=self._handle_socket_message
                )

                self.websocket.start_depth_socket(
                    symbol=symbol_lower,
                    callback=self._handle_socket_message
                )

                self.websocket.start_trade_socket(
                    symbol=symbol_lower,
                    callback=self._handle_socket_message
                )

            self.logger.info(f"Started data collection for symbols: {self.symbols}")
            self.logger.info("Press Ctrl+C to stop collection")

        except Exception as e:
            self.logger.error(f"Error starting data collection: {str(e)}")
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
        """Stop data collection"""
        try:
            self.websocket.stop()
            self.logger.info("Data collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping data collection: {str(e)}")
            raise