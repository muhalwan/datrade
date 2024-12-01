import logging
import signal
import sys
import threading
import asyncio
import gc
import time
from datetime import datetime, timedelta
from pathlib import Path
import os
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
    """Run collector with improved error handling"""
    collector = None
    db = None
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            if collector:
                collector.stop()
            if db:
                db.close()
            
            # Clear any existing event loops
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
                loop.close()
            except Exception:
                pass

            asyncio.set_event_loop(asyncio.new_event_loop())
            
            # Initialize database with fresh connection
            logger.info("Connecting to database...")
            db_config = {
                'connection_string': settings.mongodb_uri,
                'name': settings.db_name
            }
            db = MongoDBConnection(db_config)

            if not db.connect():
                raise ConnectionError("Failed to connect to database")

            # Initialize collector
            auth = BinanceAuth(
                api_key=settings.binance_api_key,
                auth_type=AuthType.HMAC,
                secret_key=settings.binance_secret_key
            )

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
            
            # Monitor memory usage and connections
            while True:
                time.sleep(30)
                
                # Check connections
                if not collector._check_db_connection() or not collector._check_websocket_connection():
                    logger.error("Connection check failed, restarting...")
                    break
                
                # Check memory usage
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
                    logger.info(f"Current memory usage: {memory_usage:.2f} MB")
                    
                    if memory_usage > 450:  # 450MB threshold
                        logger.warning("Memory usage high, performing cleanup...")
                        gc.collect()
                        if memory_usage > 490:  # Force restart if still too high
                            logger.warning("Memory still too high, forcing restart...")
                            break
                except Exception as e:
                    logger.error(f"Error checking memory: {e}")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            retry_count += 1
            
            if retry_count >= max_retries:
                logger.error("Max retries reached, waiting longer...")
                time.sleep(300)  # Wait 5 minutes before resetting retry count
                retry_count = 0
            else:
                wait_time = min(300, 30 * retry_count)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        finally:
            # Cleanup
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
            
            # Force garbage collection
            gc.collect()

def print_system_info():
    """Print system information for debugging"""
    try:
        import psutil
        logger.info("\nSystem Information:")
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Total Memory: {psutil.virtual_memory().total / (1024.0 ** 3):.1f} GB")
        logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024.0 ** 3):.1f} GB")
        logger.info(f"Process Count: {len(psutil.pids())}")
        
        # Python version and implementation
        import platform
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Python Implementation: {platform.python_implementation()}")
        
        # Environment variables (excluding sensitive info)
        env_vars = {k: v for k, v in os.environ.items() 
                   if not any(sensitive in k.lower() 
                            for sensitive in ['key', 'secret', 'password', 'token'])}
        logger.info("Environment Variables:")
        for k, v in env_vars.items():
            logger.info(f"  {k}: {v}")
            
    except Exception as e:
        logger.error(f"Error printing system info: {e}")

def main():
    """Main entry point with improved error handling and monitoring"""
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Print system information at startup
    print_system_info()
    
    # Initialize error tracking
    consecutive_errors = 0
    last_error_time = None
    
    while True:
        try:
            logger.info("\nStarting data collection process...")
            run_with_error_handling()
            
            # Reset error tracking on successful run
            consecutive_errors = 0
            last_error_time = None
            
            logger.info("Collection process completed, restarting...")
            time.sleep(10)  # Wait before restart
            
        except Exception as e:
            logger.error(f"Critical error in main: {e}")
            consecutive_errors += 1
            current_time = datetime.now()
            
            # If we're getting frequent errors, wait longer
            if (last_error_time and 
                (current_time - last_error_time).total_seconds() < 300 and 
                consecutive_errors > 3):
                wait_time = 600  # 10 minutes
                logger.error(f"Too many errors too quickly, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                consecutive_errors = 0
            else:
                time.sleep(30)  # Normal error wait
                
            last_error_time = current_time

if __name__ == "__main__":
    logger = setup_logging()
    main()
