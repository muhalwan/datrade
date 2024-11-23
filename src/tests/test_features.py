import os
import sys
from datetime import datetime, timedelta
import logging
from src.data.database.connection import MongoDBConnection
from src.features.engineering import FeatureEngineering
from src.config import settings

def test_feature_engineering():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Initialize database connection using existing config
        db_config = {
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        }
        db = MongoDBConnection(db_config)

        if not db.connect():
            logger.error("Failed to connect to database")
            return

        # Initialize feature engineering
        feature_eng = FeatureEngineering(db)

        # Set time range for testing
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        # Generate features for each configured symbol
        for symbol in settings.trading_symbols:
            logger.info(f"\nGenerating features for {symbol}")
            features = feature_eng.generate_features(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

            if features.empty:
                logger.warning(f"No features generated for {symbol}")
                continue

            # Print feature information
            logger.info(f"Generated {len(features)} rows of features")
            logger.info(f"Features available: {features.columns.tolist()}")

            # Print sample of features
            logger.info("\nFeature sample:")
            logger.info(features.head())

            # Print basic statistics
            logger.info("\nFeature statistics:")
            logger.info(features.describe())

    except Exception as e:
        logger.error(f"Error testing feature engineering: {str(e)}")

if __name__ == "__main__":
    test_feature_engineering()