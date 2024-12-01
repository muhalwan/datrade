from typing import Optional, Dict, List
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError
from datetime import datetime, timedelta

class MongoDBConnection:
    """MongoDB connection manager with proper initialization"""

    def __init__(self, config: Dict):
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        self.connection_string = config.get('connection_string')
        self.db_name = config.get('name')

        if not self.connection_string or not self.db_name:
            raise ValueError("Missing required configuration: connection_string and name")

        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.logger = logging.getLogger(__name__)
        self._collections = {}

    def connect(self) -> bool:
        """Establish MongoDB connection with retries"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000
            )

            # Test connection
            self.client.server_info()
            self.db = self.client[self.db_name]

            # Setup indexes
            self._setup_indexes()

            self.logger.info(f"MongoDB connection established to database: {self.db_name}")
            return True

        except Exception as e:
            self.logger.error(f"MongoDB connection error: {str(e)}")
            return False

    def _setup_indexes(self):
        """Setup database indexes"""
        try:
            # Price data indexes
            price_collection = self.db['price_data']
            price_collection.create_index([
                ("timestamp", DESCENDING),
                ("symbol", ASCENDING)
            ])
            price_collection.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ])

            # Order book indexes
            order_collection = self.db['order_book']
            order_collection.create_index([
                ("timestamp", DESCENDING),
                ("symbol", ASCENDING),
                ("side", ASCENDING)
            ])
            order_collection.create_index([
                ("symbol", ASCENDING),
                ("side", ASCENDING),
                ("timestamp", DESCENDING)
            ])

            self.logger.info("MongoDB indexes created successfully")
        except Exception as e:
            self.logger.error(f"Error creating indexes: {str(e)}")
            raise

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get MongoDB collection with validation"""
        try:
            if collection_name not in self._collections:
                if self.db is None:
                    if not self.connect():
                        return None

                collection = self.db[collection_name]

                # Test collection
                collection.find_one({})

                self._collections[collection_name] = collection

            return self._collections.get(collection_name)

        except Exception as e:
            self.logger.error(f"Error getting collection {collection_name}: {str(e)}")
            return None

    def get_latest_data(self, collection_name: str,
                        symbol: Optional[str] = None,
                        limit: int = 1000) -> List[Dict]:
        """Get latest data from collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return []

            # Build query
            query = {}
            if symbol:
                query['symbol'] = symbol

            # Get latest documents
            cursor = collection.find(
                query,
                sort=[('timestamp', DESCENDING)],
                limit=limit
            )

            return list(cursor)

        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            return []

    def get_data_range(self, collection_name: str,
                       start_time: datetime,
                       end_time: Optional[datetime] = None,
                       symbol: Optional[str] = None) -> List[Dict]:
        """Get data within time range"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return []

            # Build query
            query = {
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time or datetime.now()
                }
            }
            if symbol:
                query['symbol'] = symbol

            # Get documents
            cursor = collection.find(query).sort('timestamp', ASCENDING)
            return list(cursor)

        except Exception as e:
            self.logger.error(f"Error getting data range: {str(e)}")
            return []

    def validate_collections(self) -> bool:
        """Validate all required collections"""
        try:
            collections = ['price_data', 'order_book']

            for collection_name in collections:
                collection = self.get_collection(collection_name)
                if collection is None:
                    self.logger.error(f"Failed to validate collection: {collection_name}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Collection validation error: {str(e)}")
            return False

    def close(self):
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.db = None
                self._collections.clear()
                self.logger.info("MongoDB connection closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {str(e)}")