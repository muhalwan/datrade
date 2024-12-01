import logging
from pymongo import MongoClient
import pandas as pd

def check_database_content():
    """Check database content for debugging"""
    logger = logging.getLogger(__name__)

    try:
        client = MongoClient('mongodb://localhost:27017')
        db = client['crypto_trading']

        # Check collections
        logger.info(f"Collections: {db.list_collection_names()}")

        # Check price data
        price_data = list(db.price_data.find().limit(1))
        if price_data:
            logger.info(f"Sample price data: {price_data[0]}")
            logger.info(f"Price data columns: {list(price_data[0].keys())}")
        else:
            logger.warning("No price data found")

        # Check data count
        count = db.price_data.count_documents({})
        logger.info(f"Total documents: {count}")

    except Exception as e:
        logger.error(f"Database check error: {str(e)}")