import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.data.database.connection import MongoDBConnection
from src.features.processor import FeatureProcessor
from src.models.ensemble import EnsembleModel
from src.utils.metrics import calculate_trading_metrics
from src.utils.visualization import TradingVisualizer
from src.config import settings

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    return logging.getLogger(__name__)

def convert_trades_to_ohlcv(trades_df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
    """Convert trade data to OHLCV format"""
    try:
        # Ensure the timestamp is the index
        if 'trade_time' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['trade_time'])
            df = trades_df.set_index('timestamp')
        elif 'timestamp' in trades_df.columns:
            df = trades_df.set_index('timestamp')
        else:
            raise ValueError("No timestamp column found")

        # Create OHLCV data
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df['price'].resample(timeframe).first()
        ohlcv['high'] = df['price'].resample(timeframe).max()
        ohlcv['low'] = df['price'].resample(timeframe).min()
        ohlcv['close'] = df['price'].resample(timeframe).last()
        ohlcv['volume'] = df['quantity'].resample(timeframe).sum()

        # Remove rows with any missing data
        ohlcv = ohlcv.dropna()

        # Log data summary
        logger.info(f"OHLCV data summary:")
        logger.info(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
        logger.info(f"Number of periods: {len(ohlcv)}")
        logger.info(f"Missing values: {ohlcv.isnull().sum().sum()}")

        return ohlcv

    except Exception as e:
        logger.error(f"Error converting trades to OHLCV: {e}")
        return pd.DataFrame()

def load_training_data(db: MongoDBConnection, symbol: str, start_date: datetime, end_date: datetime):
    """Load and prepare training data"""
    try:
        logger.info(f"Querying data from {start_date} to {end_date}")

        # Debug query
        query = {
            'symbol': symbol,
            'trade_time': {
                '$gte': start_date,
                '$lt': end_date
            }
        }
        logger.info(f"MongoDB query: {query}")

        # Count total available records
        total_records = db.get_collection('price_data').count_documents(query)
        logger.info(f"Total records found in database: {total_records}")

        # Load price data with expanded time range
        price_data = pd.DataFrame(list(db.get_collection('price_data').find(query).sort('trade_time', 1)))

        if price_data.empty:
            logger.error("No price data found")
            return None, None

        logger.info(f"Raw price data shape before OHLCV conversion: {price_data.shape}")

        # Convert time columns to datetime
        if 'trade_time' in price_data.columns:
            price_data['trade_time'] = pd.to_datetime(price_data['trade_time'])
        if 'timestamp' in price_data.columns:
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

        # Convert trades to OHLCV
        ohlcv_data = convert_trades_to_ohlcv(price_data)

        # Load order book data
        orderbook_query = {
            'symbol': symbol,
            'timestamp': {
                '$gte': start_date,
                '$lt': end_date
            }
        }
        orderbook_data = pd.DataFrame(list(db.get_collection('order_book').find(orderbook_query).sort('timestamp', 1)))

        # Log data summary
        logger.info("\nData Summary:")
        logger.info(f"Raw price records: {len(price_data)}")
        logger.info(f"OHLCV periods: {len(ohlcv_data)}")
        logger.info(f"Orderbook records: {len(orderbook_data)}")

        if len(ohlcv_data) < 100:
            logger.error(f"Insufficient OHLCV periods ({len(ohlcv_data)}). Need at least 100.")
            return None, None

        return ohlcv_data, orderbook_data

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        logger.exception("Detailed error:")
        return None, None

def train_models(price_data: pd.DataFrame, orderbook_data: pd.DataFrame, symbol: str):
    """Train and evaluate models"""
    try:
        if price_data.empty:
            raise ValueError("Empty price data")

        logger.info("Processing features...")
        processor = FeatureProcessor()
        features, target = processor.prepare_features(
            price_data=price_data,
            orderbook_data=orderbook_data,
            target_minutes=5
        )

        if features.empty or target.empty:
            raise ValueError("Feature generation failed")

        # Split data into train/test
        train_end = int(len(features) * 0.8)
        X_train = features[:train_end]
        y_train = target[:train_end]
        X_test = features[train_end:]
        y_test = target[train_end:]

        logger.info("Training ensemble model...")
        model = EnsembleModel()
        model.train(X_train, y_train)

        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_trading_metrics(
            y_train.values,
            train_predictions,
            price_data['close'][:train_end].values
        )

        test_metrics = calculate_trading_metrics(
            y_test.values,
            test_predictions,
            price_data['close'][train_end:].values
        )

        # Save metrics
        logger.info("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("\nTest Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Create visualizations
        viz = TradingVisualizer()
        figures = viz.plot_model_performance(
            y_test.values,
            test_predictions,
            price_data['close'][train_end:].values,
            X_test  # Pass the test features
        )

        # Save model
        model_dir = Path("models/trained")
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / f"{symbol}_model"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save visualizations
        fig_dir = Path("models/figures")
        fig_dir.mkdir(exist_ok=True, parents=True)
        for name, fig in figures.items():
            if fig is not None:  # Add null check
                fig.write_html(str(fig_dir / f"{name}.html"))
        logger.info(f"Figures saved to {fig_dir}")

        return model, figures, (train_metrics, test_metrics)

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        logger.exception("Detailed error:")  # Add stack trace
        return None, None, None

def main():
    logger.info("Starting model training...")

    # Connect to database
    db_config = {
        'connection_string': settings.mongodb_uri,
        'name': settings.db_name
    }
    db = MongoDBConnection(db_config)

    if not db.connect():
        logger.error("Failed to connect to database")
        return

    try:
        # Training parameters
        symbol = "BTCUSDT"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days of data

        logger.info(f"Training Configuration:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Start Date: {start_date}")
        logger.info(f"End Date: {end_date}")

        # Load data
        price_data, orderbook_data = load_training_data(db, symbol, start_date, end_date)

        if price_data is None or price_data.empty:
            logger.error("No valid price data available")
            return

        # Train models
        model, figures, metrics = train_models(price_data, orderbook_data, symbol)

        if model is None:
            logger.error("Model training failed")
            return

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.exception("Detailed error:")
    finally:
        db.close()

if __name__ == "__main__":
    logger = setup_logging()
    main()