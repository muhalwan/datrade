import logging
from typing import Optional, Dict
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
import pandas as pd
from pymongo.collection import Collection

class MongoDBConnection:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.connection_string = config['connection_string']
        self.db_name = config['name']
        self.client = None
        self.db = None

    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=30000)
            self.db = self.client[self.db_name]
            # Trigger a server selection to catch connection errors early
            self.client.server_info()
            self.logger.info(f"MongoDB connection established successfully to database: {self.db_name}")
            self.setup_indexes()
            return True
        except PyMongoError as e:
            self.logger.error(f"Error connecting to MongoDB: {e}")
            return False

    def setup_indexes(self):
        try:
            collection = self.db.price_data
            collection.create_index([("symbol", ASCENDING), ("trade_time", ASCENDING)])
            self.logger.info("Index created for 'price_data' collection.")
            collection = self.db.order_book
            collection.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)])
            self.logger.info("Index created for 'order_book' collection.")
            self.logger.info("MongoDB indexes setup completed successfully.")
        except PyMongoError as e:
            self.logger.error(f"Error setting up indexes: {e}")

    def fetch_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetches and converts trade data to 1-minute OHLCV candles"""
        try:
            if self.db is None:
                self.logger.warning("Database not connected. Attempting to connect...")
                if not self.connect():
                    self.logger.error("Failed to connect to MongoDB.")
                    return None

            # Fetch raw trade data
            collection: Collection = self.db.price_data
            query = {
                "symbol": symbol,
                "trade_time": {"$gte": start_date, "$lte": end_date}
            }
            cursor = collection.find(query).sort("trade_time", ASCENDING)
            trades_df = pd.DataFrame(list(cursor))

            if trades_df.empty:
                self.logger.warning(f"No price data found for {symbol} in the given date range")
                return pd.DataFrame()

            # Convert to OHLCV
            trades_df['trade_time'] = pd.to_datetime(trades_df['trade_time'])
            trades_df.set_index('trade_time', inplace=True)

            # Resample with modern pandas syntax
            ohlcv_df = trades_df.resample('1min').agg(
                open=('price', 'first'),
                high=('price', 'max'),
                low=('price', 'min'),
                close=('price', 'last'),
                volume=('quantity', 'sum')
            )

            # Handle missing data
            ohlcv_df[['open', 'high', 'low', 'close']] = ohlcv_df[['open', 'high', 'low', 'close']].ffill()
            ohlcv_df['volume'] = ohlcv_df['volume'].fillna(0)

            self.logger.info(f"Generated {len(ohlcv_df)} OHLCV records from raw trades")
            return ohlcv_df

        except PyMongoError as e:
            self.logger.error(f"MongoDB error fetching price data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing price data: {e}")
            return None

    def fetch_orderbook_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetches and aggregates order book data to best bid/ask"""
        try:
            if self.db is None:
                self.logger.warning("Database not connected. Attempting to connect...")
                if not self.connect():
                    self.logger.error("Failed to connect to MongoDB.")
                    return None

            # Fetch raw order book updates
            collection: Collection = self.db.order_book
            query = {
                "symbol": symbol,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }
            cursor = collection.find(query).sort("timestamp", ASCENDING)
            ob_df = pd.DataFrame(list(cursor))

            if ob_df.empty:
                self.logger.warning(f"No order book data found for {symbol} in the given date range")
                return pd.DataFrame()

            # Process order book data
            ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp'])
            ob_df.set_index('timestamp', inplace=True)

            # Modern groupby and aggregation
            grouped = ob_df.groupby([pd.Grouper(freq='1s'), 'side']).agg(
                price=('price', 'last'),
                quantity=('quantity', 'sum')
            )

            # Unstack and flatten columns
            aggregated = grouped.unstack(level='side')
            aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]

            # Create final dataframe with proper column names
            final_df = pd.DataFrame(index=aggregated.index)
            final_df['best_bid'] = aggregated.get('price_bid', pd.Series(index=aggregated.index))
            final_df['best_ask'] = aggregated.get('price_ask', pd.Series(index=aggregated.index))
            final_df['bid_volume'] = aggregated.get('quantity_bid', pd.Series(index=aggregated.index)).fillna(0)
            final_df['ask_volume'] = aggregated.get('quantity_ask', pd.Series(index=aggregated.index)).fillna(0)

            # Forward fill prices and resample to 1 minute
            final_df = final_df.reindex(columns=[
                'best_bid', 'best_ask',
                'bid_volume', 'ask_volume'
            ], fill_value=0)

            self.logger.info(f"Aggregated {len(final_df)} order book snapshots")
            return final_df

        except PyMongoError as e:
            self.logger.error(f"MongoDB error fetching order book data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing order book data: {e}")
            return None

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        try:
            if self.db is None:
                self.logger.warning("Database not connected. Attempting to connect...")
                if not self.connect():
                    self.logger.error("Failed to connect to MongoDB.")
                    return None

            if collection_name in self.db.list_collection_names():
                self.logger.info(f"Retrieved collection '{collection_name}'.")
                return self.db[collection_name]
            else:
                self.logger.warning(f"Collection '{collection_name}' does not exist.")
                return None

        except PyMongoError as e:
            self.logger.error(f"Error retrieving collection '{collection_name}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving collection '{collection_name}': {e}")
            return None

    def close(self):
        try:
            if self.client:
                self.client.close()
                self.logger.info("MongoDB connection closed successfully.")
            else:
                self.logger.warning("MongoDB client was not connected.")
        except PyMongoError as e:
            self.logger.error(f"Error closing MongoDB connection: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error closing MongoDB connection: {e}")
