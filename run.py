import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from src.config import settings
from src.data.collector import BinanceDataCollector
from src.data.auth import BinanceAuth, AuthType
from src.data.database.connection import MongoDBConnection

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"collector_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    # Set debug level for collector
    logging.getLogger('src.data.collector').setLevel(logging.DEBUG)

    return logging.getLogger(__name__)

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info("\nShutdown signal received. Cleaning up...")
    sys.exit(0)

def main():
    collector = None
    db = None

    try:
        # Setup logging
        logger = setup_logging()

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

        # Initialize auth
        auth = BinanceAuth(
            api_key=settings.binance_api_key,
            auth_type=AuthType.HMAC,
            secret_key=settings.binance_secret_key
        )

        # Initialize collector
        collector = BinanceDataCollector(
            auth=auth,
            symbols=settings.trading_symbols,
            db=db,
            use_testnet=settings.use_testnet
        )

        # Start collection
        logger.info("Starting data collection...")
        collector.start_data_collection()

        # Keep script running
        while True:
            signal.pause()

    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        if collector:
            try:
                collector.stop()
            except Exception as e:
                logger.error(f"Error stopping collector: {e}")

        if db:
            try:
                db.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")

        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()
