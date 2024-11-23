#!/usr/bin/env python3

import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
import yaml
from typing import Dict
import pandas as pd

from src.config import settings
from src.data.database.connection import MongoDBConnection
from src.features.engineering import FeatureEngineering
from src.models.training import ModelTrainer
from src.monitoring.metrics import PerformanceMonitor

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def load_model_config() -> Dict:
    """Load model configuration"""
    config_path = Path("config/model_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Model configuration file not found")

    with open(config_path) as f:
        return yaml.safe_load(f)

def fetch_training_data(db: MongoDBConnection, symbol: str,
                        start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Fetch historical data for training"""
    try:
        collection = db.get_collection('price_data')

        # Fetch data with MongoDB aggregation
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
            },
            {
                '$project': {
                    '_id': 0,
                    'timestamp': 1,
                    'price': 1,
                    'quantity': 1,
                    'is_buyer_maker': 1
                }
            }
        ]

        data = list(collection.aggregate(pipeline))
        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame()

        # Convert to OHLCV format
        df.set_index('timestamp', inplace=True)
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df['price'].resample('1Min').first()
        ohlcv['high'] = df['price'].resample('1Min').max()
        ohlcv['low'] = df['price'].resample('1Min').min()
        ohlcv['close'] = df['price'].resample('1Min').last()
        ohlcv['volume'] = df['quantity'].resample('1Min').sum()

        return ohlcv.dropna()

    except Exception as e:
        logging.getLogger(__name__).error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def main():
    logger = setup_logging()
    db = None
    monitor = None

    try:
        # Initialize monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()

        # Load configurations
        logger.info("Loading configurations...")
        model_config = load_model_config()

        # Initialize database
        logger.info("Connecting to database...")
        db = MongoDBConnection({
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        })

        if not db.connect():
            logger.error("Failed to connect to database. Exiting...")
            return

        # Initialize components
        feature_eng = FeatureEngineering(db)
        trainer = ModelTrainer(db)

        # Process each symbol
        for symbol in settings.trading_symbols:
            logger.info(f"\nProcessing {symbol}")

            try:
                # Get training period
                end_time = datetime.now()
                days_back = model_config.get('training_days', 30)
                start_time = end_time - timedelta(days=days_back)

                logger.info(f"Training period: {start_time} to {end_time}")

                # Fetch training data
                df = fetch_training_data(db, symbol, start_time, end_time)

                if df.empty:
                    logger.warning(f"No historical data found for {symbol}")
                    continue

                logger.info(f"Fetched {len(df)} data points")

                # Generate features
                features_df = feature_eng.generate_features(df)

                if features_df.empty:
                    logger.warning(f"No features generated for {symbol}")
                    continue

                logger.info(f"Generated features shape: {features_df.shape}")

                # Train/test split
                split_point = int(len(features_df) * 0.8)
                train_df = features_df[:split_point].copy()
                test_df = features_df[split_point:].copy()

                logger.info(f"Training set: {len(train_df)}, Test set: {len(test_df)}")

                # Train models
                logger.info("Training models...")

                try:
                    models = trainer.train(
                        symbol=symbol,
                        train_data=train_df,
                        test_data=test_df
                    )

                    # Create models directory
                    models_dir = Path("models")
                    models_dir.mkdir(exist_ok=True)

                    # Save models
                    logger.info("Saving models...")
                    trainer.save_models(models, symbol)

                    # Log performance metrics
                    for name, metrics in trainer.get_model_metrics(symbol).items():
                        logger.info(f"\nMetrics for {name}:")
                        for metric, value in metrics.items():
                            logger.info(f"{metric}: {value:.4f}")

                except Exception as e:
                    logger.error(f"Error in model training: {str(e)}")
                    continue

                logger.info(f"Successfully completed processing for {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
    finally:
        if monitor:
            monitor.stop_monitoring()
        if db:
            db.close()
        logger.info("Training complete")

if __name__ == "__main__":
    main()