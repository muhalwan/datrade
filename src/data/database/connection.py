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
        try:
            if self.db is None:
                self.logger.warning("Database not connected. Attempting to connect...")
                if not self.connect():
                    self.logger.error("Failed to connect to MongoDB.")
                    return None

            collection: Collection = self.db.price_data
            query = {
                "symbol": symbol,
                "trade_time": {"$gte": start_date, "$lte": end_date}
            }
            cursor = collection.find(query).sort("trade_time", ASCENDING)
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df.set_index('trade_time', inplace=True)
                self.logger.info(f"Fetched {len(df)} records from 'price_data' collection.")
                return df
            else:
                self.logger.warning(f"No data found for symbol {symbol} in the given date range.")
                return pd.DataFrame()

        except PyMongoError as e:
            self.logger.error(f"Error fetching price data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching price data: {e}")
            return None

    def fetch_orderbook_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            if self.db is None:
                self.logger.warning("Database not connected. Attempting to connect...")
                if not self.connect():
                    self.logger.error("Failed to connect to MongoDB.")
                    return None

            collection: Collection = self.db.order_book
            query = {
                "symbol": symbol,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }
            cursor = collection.find(query).sort("timestamp", ASCENDING)
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                self.logger.info(f"Fetched {len(df)} records from 'order_book' collection.")
                return df
            else:
                self.logger.warning(f"No data found for symbol {symbol} in the given date range.")
                return pd.DataFrame()

        except PyMongoError as e:
            self.logger.error(f"Error fetching orderbook data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching orderbook data: {e}")
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
