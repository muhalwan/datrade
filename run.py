#!/usr/bin/env python3

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
import uvicorn
import os
from src.config import settings
from src.data.collector import BinanceDataCollector
from src.data.auth import BinanceAuth, AuthType
from src.data.database.connection import MongoDBConnection
from src.monitoring.api import app
from src.globals import set_collector, get_collector
from src.utils.logging import setup_logging


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TradingPlatform:
    """Main trading platform controller"""
    def __init__(self):
        self.logger = setup_logging()
        self.collector = None
        self.db = None
        self.monitoring_thread = None
        self.is_running = True

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("\nShutdown signal received. Cleaning up...")
        self.is_running = False
        self.cleanup()
        sys.exit(0)

    def setup_database(self) -> bool:
        """Setup database connection"""
        try:
            self.logger.info("Connecting to database...")
            self.db = MongoDBConnection({
                'connection_string': settings.mongodb_uri,
                'name': settings.db_name
            })

            if not self.db.connect():
                self.logger.error("Failed to connect to database")
                return False

            if not self.db.validate_collections():
                self.logger.error("Failed to validate database collections")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Database setup error: {str(e)}")
            return False

    def setup_collector(self) -> bool:
        """Setup data collector"""
        try:
            auth = BinanceAuth(
                api_key=settings.binance_api_key,
                auth_type=AuthType.HMAC,
                secret_key=settings.binance_secret_key
            )

            self.collector = BinanceDataCollector(
                auth=auth,
                symbols=settings.trading_symbols,
                db=self.db,
                use_testnet=settings.use_testnet
            )

            # Set global collector instance for monitoring
            set_collector(self.collector)

            # Verify collector is set
            if get_collector() is None:
                raise ValueError("Failed to set global collector instance")

            return True

        except Exception as e:
            self.logger.error(f"Collector setup error: {str(e)}")
            return False

    def start_monitoring(self):
        """Start monitoring dashboard"""
        def run_monitoring():
            try:
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=8000,
                    log_level="error",
                    access_log=False
                )
            except Exception as e:
                self.logger.error(f"Monitoring server error: {str(e)}")

        self.monitoring_thread = threading.Thread(
            target=run_monitoring,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Monitoring dashboard started at http://localhost:8000")

    def start_collection(self):
        """Start data collection"""
        try:
            self.logger.info("Starting data collection...")
            self.collector.start_collection()
            self.logger.info("Data collection started successfully")
        except Exception as e:
            self.logger.error(f"Collection start error: {str(e)}")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        """Cleanup resources"""
        if self.collector:
            try:
                self.collector.stop()
            except Exception as e:
                self.logger.error(f"Error stopping collector: {e}")

        if self.db:
            try:
                self.db.close()
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")

        self.logger.info("Cleanup complete")

    def run(self):
        """Main run loop"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()

            # Setup database
            if not self.setup_database():
                self.logger.error("Database setup failed. Exiting...")
                return

            # Setup collector
            if not self.setup_collector():
                self.logger.error("Collector setup failed. Exiting...")
                return

            # Start monitoring
            self.start_monitoring()

            # Start collection
            self.start_collection()

            # Main loop
            while self.is_running:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    self.logger.info("\nShutdown requested...")
                    break

        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    platform = TradingPlatform()
    platform.run()

if __name__ == "__main__":
    main()