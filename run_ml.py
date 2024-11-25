#!/usr/bin/env python3

import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
import yaml
from typing import Dict, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import json
from src.config import settings
from src.data.database.connection import MongoDBConnection
from src.features.engineering import FeatureEngineering
from src.models.training import ModelTrainer
from src.monitoring.metrics import PerformanceMonitor
from src.utils.gpu_utils import get_gpu_info

class MLPipeline:
    def __init__(self):
        self.logger = self._setup_logging()

        # Check GPU status
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            self.logger.info(f"GPU available: {gpu_info}")
        else:
            self.logger.warning("No GPU available, using CPU only")
        self.db = None
        self.monitor = None
        self.trainer = None
        self.feature_eng = None
        self.running = True
        self.training_queue = Queue()
        self.results_cache = {}

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging configuration"""
        log_dir = Path("logs/training")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Setup logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        self.logger.info("\nShutdown signal received. Cleaning up...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def initialize_components(self) -> bool:
        """Initialize all components with validation"""
        try:
            # Initialize monitoring
            self.monitor = PerformanceMonitor()
            self.monitor.start_monitoring()

            # Load configurations
            self.logger.info("Loading configurations...")
            self.model_config = self._load_model_config()

            # Initialize database
            self.logger.info("Connecting to database...")
            self.db = MongoDBConnection({
                'connection_string': settings.mongodb_uri,
                'name': settings.db_name
            })

            if not self.db.connect():
                self.logger.error("Failed to connect to database")
                return False

            # Initialize components
            self.feature_eng = FeatureEngineering(self.db)
            self.trainer = ModelTrainer(self.db)

            return True

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            return False

    def _load_model_config(self) -> Dict:
        """Load model configuration with validation"""
        config_path = Path("config/model_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("Model configuration file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate configuration
        required_fields = ['training_days', 'models', 'features']
        if not all(field in config for field in required_fields):
            raise ValueError("Invalid model configuration")

        return config

    def fetch_training_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch and validate training data"""
        try:
            collection = self.db.get_collection('price_data')

            pipeline = [
                {
                    '$match': {
                        'symbol': symbol,
                        'timestamp': {
                            '$gte': start_time,
                            '$lte': end_time
                        }
                    }
                },
                {
                    '$sort': {'timestamp': 1}
                }
            ]

            data = list(collection.aggregate(pipeline))

            if not data:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Validate required columns
            required_columns = ['timestamp', 'price', 'quantity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Ensure numeric types
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

            # Remove any invalid data
            df = df.dropna(subset=['price', 'quantity'])

            # Validate data quality
            df = self._validate_data_quality(df)

            self.logger.info(f"Fetched {len(df)} data points with columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data quality"""
        try:
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Check for gaps
            timestamps = pd.to_datetime(df['timestamp'])
            gaps = timestamps.diff() > timedelta(minutes=5)  # Adjust threshold as needed
            if gaps.any():
                self.logger.warning(f"Found {gaps.sum()} gaps in data")

            # Check for outliers in price and quantity
            for col in ['price', 'quantity']:
                if col in df.columns:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = z_scores > 3
                    if outliers.any():
                        self.logger.warning(f"Found {outliers.sum()} outliers in {col}")
                        # Optionally remove extreme outliers
                        df = df[z_scores < 5]  # Remove extreme outliers (z-score > 5)

            return df

        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return df

    def process_symbol(self, symbol: str) -> Dict:
        """Process individual symbol"""
        try:
            self.logger.info(f"\nProcessing {symbol}")

            # Get training period
            end_time = datetime.now()
            days_back = self.model_config.get('training_days', 30)
            start_time = end_time - timedelta(days=days_back)

            self.logger.info(f"Training period: {start_time} to {end_time}")

            # Fetch and validate data
            df = self.fetch_training_data(symbol, start_time, end_time)
            if df.empty:
                return {'status': 'error', 'message': 'No data available'}

            self.logger.info(f"Fetched {len(df)} data points")

            # Generate features
            features_df = self.feature_eng.generate_features(df)
            if features_df.empty:
                return {'status': 'error', 'message': 'Feature generation failed'}

            self.logger.info(f"Generated features shape: {features_df.shape}")

            # Train/test split
            split_point = int(len(features_df) * 0.8)
            train_df = features_df[:split_point].copy()
            test_df = features_df[split_point:].copy()

            self.logger.info(f"Training set: {len(train_df)}, Test set: {len(test_df)}")

            # Train models
            self.logger.info("Training models...")
            models = self.trainer.train(symbol, train_df, test_df)

            # Save models
            self.logger.info("Saving models...")
            self.trainer.save_models(models, symbol)

            # Get metrics
            metrics = self.trainer.get_model_metrics(symbol)

            return {
                'status': 'success',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self.monitor:
            self.monitor.stop_monitoring()
        if self.db:
            self.db.close()
        self.logger.info("Cleanup complete")

    def run(self):
        """Main execution pipeline"""
        try:
            if not self.initialize_components():
                return

            results = {}
            with ThreadPoolExecutor(max_workers=len(settings.trading_symbols)) as executor:
                future_to_symbol = {
                    executor.submit(self.process_symbol, symbol): symbol
                    for symbol in settings.trading_symbols
                }

                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")

            # Save results
            self._save_results(results)

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
        finally:
            self.cleanup()

    def _save_results(self, results: Dict):
        """Save training results"""
        try:
            results_dir = Path("logs/results")
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = results_dir / f"training_results_{timestamp}.json"

            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)

            self.logger.info(f"Results saved to {results_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

def main():
    """Main entry point"""
    pipeline = MLPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()