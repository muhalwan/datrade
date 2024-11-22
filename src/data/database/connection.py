from typing import Optional
import logging
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError

class MongoDBConnection:
    """Handles MongoDB Atlas connection and collection management"""

    def __init__(self, config: dict):
        """
        Initialize MongoDB connection

        Args:
            config: Database configuration dictionary containing connection_string and name
        """
        self.connection_string = config.get('connection_string')
        self.db_name = config.get('name')
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Establish MongoDB connection"""
        try:
            if not self.connection_string:
                raise ValueError("MongoDB connection string is required")

            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )

            # Test connection
            self.client.server_info()
            self.db = self.client[self.db_name]

            # Create indexes for better query performance
            self._setup_indexes()

            self.logger.info(f"MongoDB connection established successfully to database: {self.db_name}")
            return True

        except ServerSelectionTimeoutError as e:
            self.logger.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
            return False
        except PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            return False

    def _setup_indexes(self):
        """Setup indexes for collections"""
        try:
            # Price data indexes
            self.db.price_data.create_index([
                ("timestamp", -1),
                ("symbol", 1)
            ])

            # Order book indexes
            self.db.order_book.create_index([
                ("timestamp", -1),
                ("symbol", 1),
                ("side", 1)
            ])

            self.logger.info("MongoDB indexes created successfully")

        except PyMongoError as e:
            self.logger.error(f"Error creating indexes: {str(e)}")

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get MongoDB collection

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection object
        """
        if self.db is None:
            if not self.connect():
                return None
        return self.db[collection_name] if self.db is not None else None

    def close(self):
        """Close MongoDB connection"""
        if self.client is not None:
            self.client.close()
            self.logger.info("MongoDB connection closed")