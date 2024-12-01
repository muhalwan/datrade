import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
from src.config import settings
from src.data.database.connection import MongoDBConnection
from src.models.training import ModelTrainer

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ml_training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    return logging.getLogger(__name__)

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info("\nShutdown signal received. Cleaning up...")
    sys.exit(0)

def get_data_timerange(db, logger, symbol: str) -> tuple:
    """Get the actual time range of available data"""
    try:
        collection = db.get_collection('price_data')

        first_record = collection.find_one(
            {"symbol": symbol},
            sort=[("timestamp", 1)]
        )
        last_record = collection.find_one(
            {"symbol": symbol},
            sort=[("timestamp", -1)]
        )

        if first_record and last_record:
            start_time = first_record['timestamp']
            end_time = last_record['timestamp']

            # Add small buffer
            start_time = start_time + timedelta(minutes=1)
            end_time = end_time - timedelta(minutes=1)

            data_count = collection.count_documents({
                "symbol": symbol,
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            })

            logger.info(f"Data range for {symbol}:")
            logger.info(f"Start time: {start_time}")
            logger.info(f"End time: {end_time}")
            logger.info(f"Total records: {data_count}")

            return start_time, end_time, data_count

        return None, None, 0

    except Exception as e:
        logger.error(f"Error getting data timerange: {str(e)}")
        return None, None, 0

def main():
    db = None
    logger = setup_logging()

    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Initialize database
        logger.info("Connecting to database...")
        db_config = {
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        }
        db = MongoDBConnection(db_config)

        if not db.connect():
            logger.error("Failed to connect to database. Exiting...")
            return

        # Process each symbol
        for symbol in settings.trading_symbols:
            logger.info(f"\nChecking data for {symbol}")

            # Get actual data time range
            start_time, end_time, data_count = get_data_timerange(db, logger, symbol)

            if not start_time or not end_time:
                logger.warning(f"No data available for {symbol}")
                continue

            # For training/testing split, use 80% of data for training
            time_span = end_time - start_time
            train_end = start_time + (time_span * 0.8)

            logger.info(f"Training period: {start_time} to {train_end}")
            logger.info(f"Testing period: {train_end} to {end_time}")

            try:
                # Initialize model trainer
                trainer = ModelTrainer(db)

                # Train models
                logger.info("Training models...")
                models = trainer.train(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=train_end
                )

                # Generate test features
                test_features = trainer.feature_eng.generate_features(
                    symbol=symbol,
                    start_time=train_end,
                    end_time=end_time
                )

                if test_features.empty:
                    logger.warning(f"No test data available for {symbol}")
                    continue

                # Evaluate models
                results = trainer.evaluate_models(models, test_features)

                # Create models directory
                os.makedirs("models", exist_ok=True)

                # Save models
                trainer.save_models(models, symbol)

                logger.info(f"Successfully completed processing for {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        if db:
            db.close()
            logger.info("ML Training complete")

if __name__ == "__main__":
    main()