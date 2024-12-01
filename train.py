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
            df = trades_df.set_index('trade_time')
        elif 'timestamp' in trades_df.columns:
            df = trades_df.set_index('timestamp')
        else:
            raise ValueError("No timestamp column found")

        # Resample and calculate OHLCV
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df['price'].resample(timeframe).first()
        ohlcv['high'] = df['price'].resample(timeframe).max()
        ohlcv['low'] = df['price'].resample(timeframe).min()
        ohlcv['close'] = df['price'].resample(timeframe).last()
        ohlcv['volume'] = df['quantity'].resample(timeframe).sum()

        # Forward fill any missing values
        ohlcv = ohlcv.ffill()

        return ohlcv

    except Exception as e:
        logger.error(f"Error converting trades to OHLCV: {e}")
        return pd.DataFrame()

def load_training_data(db: MongoDBConnection, symbol: str, start_date: datetime, end_date: datetime):
    """Load and prepare training data"""
    try:
        # Load price data
        price_data = pd.DataFrame(list(db.get_collection('price_data').find({
            'symbol': symbol,
            'trade_time': {
                '$gte': start_date,
                '$lt': end_date
            }
        }).sort('trade_time', 1)))

        if price_data.empty:
            logger.error("No price data found")
            return None, None

        # Convert trades to OHLCV
        price_data = convert_trades_to_ohlcv(price_data)

        # Load order book data
        orderbook_data = pd.DataFrame(list(db.get_collection('order_book').find({
            'symbol': symbol,
            'timestamp': {
                '$gte': start_date,
                '$lt': end_date
            }
        }).sort('timestamp', 1)))

        return price_data, orderbook_data

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
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
            price_data['close'][train_end:],
            features[train_end:]
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
            fig.write_html(str(fig_dir / f"{name}.html"))
        logger.info(f"Figures saved to {fig_dir}")

        return model, figures, (train_metrics, test_metrics)

    except Exception as e:
        logger.error(f"Error in model training: {e}")
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
        start_date = end_date - timedelta(days=30)  # Use last 30 days of data

        # Load data
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        price_data, orderbook_data = load_training_data(db, symbol, start_date, end_date)

        if price_data is None or price_data.empty:
            logger.error("No valid price data available")
            return

        logger.info(f"Loaded {len(price_data)} OHLCV records and {len(orderbook_data) if orderbook_data is not None else 0} orderbook records")

        # Train models
        model, figures, metrics = train_models(price_data, orderbook_data, symbol)

        if model is None:
            logger.error("Model training failed")
            return

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    logger = setup_logging()
    main()

if __name__ == "__main__":
    logger = setup_logging()
    main()