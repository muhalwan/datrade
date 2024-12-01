import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
import os
import gc
import time
from src.config import settings
from src.data.collector import BinanceDataCollector
from src.data.auth import BinanceAuth, AuthType
from src.data.database.connection import MongoDBConnection

def setup_logging():
    """Setup logging configuration for Heroku"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info("\nShutdown signal received. Cleaning up...")
    sys.exit(0)

def run_with_error_handling():
    """Run collector with error handling and reconnection logic"""
    collector = None
    db = None
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            # Initialize database
            logger.info("Connecting to database...")
            db_config = {
                'connection_string': settings.mongodb_uri,
                'name': settings.db_name
            }
            db = MongoDBConnection(db_config)

            if not db.connect():
                raise ConnectionError("Failed to connect to database")

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

            # Reset retry count on successful connection
            retry_count = 0

            # Keep script running with periodic cleanup
            while True:
                time.sleep(60)  # Check every minute
                gc.collect()  # Run garbage collection
                
                # Log memory usage
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
                logger.info(f"Current memory usage: {memory_usage:.2f} MB")
                
                if memory_usage > 450:  # 450MB threshold
                    logger.warning("Memory usage high, performing cleanup...")
                    if collector:
                        collector.stop()
                    if db:
                        db.close()
                    gc.collect()
                    return  # Exit to let Heroku restart the dyno

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            retry_count += 1
            
            if retry_count >= max_retries:
                logger.error("Max retries reached, exiting...")
                break
                
            wait_time = min(300, 30 * retry_count)  # Exponential backoff
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            
            # Cleanup before retry
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
            gc.collect()

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

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    while True:
        try:
            run_with_error_handling()
            logger.info("Restarting collection process...")
            time.sleep(10)  # Wait before restart
        except Exception as e:
            logger.error(f"Critical error: {e}")
            time.sleep(30)  # Wait longer on critical errors

if __name__ == "__main__":
    logger = setup_logging()
    main()
