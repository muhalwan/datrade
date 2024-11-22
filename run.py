import yaml
import logging
import os
from pathlib import Path
from src.data.collector import BinanceDataCollector
from src.data.auth import BinanceAuth, AuthType
from src.data.database.connection import MongoDBConnection
import signal
import sys

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger = logging.getLogger(__name__)
    logger.info("\nStopping data collection...")
    sys.exit(0)

def main():
    collector = None
    db = None
    try:
        # Load config
        with open('config/config.yaml') as f:
            config = yaml.safe_load(f)

        # Setup logging
        logger = setup_logging(config)

        # Setup signal handler
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize auth
        auth = BinanceAuth(
            api_key=config['exchange']['api_key'],
            auth_type=AuthType.HMAC,
            secret_key=config['exchange']['secret_key']
        )

        # Initialize database
        db = MongoDBConnection(config['database'])
        if not db.connect():
            logger.error("Failed to connect to database. Exiting...")
            return

        # Initialize collector
        collector = BinanceDataCollector(
            auth=auth,
            symbols=config['exchange']['symbols'],
            db=db,
            use_testnet=config['exchange'].get('use_testnet', True)
        )

        # Start collection
        logger.info("Starting data collection...")
        collector.start_data_collection()

        # Keep the script running and show stats
        while True:
            pass

    except KeyboardInterrupt:
        logger.info("\nStopping data collection...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        if collector:
            collector.stop()
        if db:
            db.close()

if __name__ == "__main__":
    main()