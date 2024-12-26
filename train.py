import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict

from src.data.database.connection import MongoDBConnection
from src.features.processor import FeatureProcessor
from src.features.selector import FeatureSelector
from src.models.ensemble import EnhancedEnsemble
from src.utils.metrics import calculate_trading_metrics
from src.utils.visualization import TradingVisualizer
from src.config import settings
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='best_model.weights.h5', save_weights_only=True)

class ModelTrainer:
    """Comprehensive model training system"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_processor = FeatureProcessor()
        self.feature_selector = FeatureSelector()
        self.visualizer = TradingVisualizer()

        # Create necessary directories
        for dir_name in ['logs', 'models/trained', 'models/figures']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

        return logging.getLogger(__name__)

    def load_training_data(
            self,
            db: MongoDBConnection,
            symbol: str,
            start_date: datetime,
            end_date: datetime,
            batch_size: timedelta = timedelta(days=7)
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load and prepare training data in batches"""
        try:
            self.logger.info(f"Loading data from {start_date} to {end_date}")

            price_data_list = []
            orderbook_data_list = []

            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + batch_size, end_date)
                self.logger.info(f"Processing batch from {current_start} to {current_end}")

                # Load price data
                price_batch = self._load_data_batch(db, 'price_data', symbol, current_start, current_end)
                if price_batch is not None:
                    price_data_list.append(price_batch)

                # Load orderbook data
                orderbook_batch = self._load_data_batch(db, 'order_book', symbol, current_start, current_end)
                if orderbook_batch is not None:
                    orderbook_data_list.append(orderbook_batch)

                current_start = current_end

            if not price_data_list:
                self.logger.error("No price data found")
                return None, None

            # Concatenate all batches
            price_data = pd.concat(price_data_list)
            orderbook_data = pd.concat(orderbook_data_list) if orderbook_data_list else pd.DataFrame()

            self.logger.info(f"Total loaded price records: {len(price_data)}")
            self.logger.info(f"Total loaded orderbook records: {len(orderbook_data)}")

            # Convert to OHLCV
            ohlcv_data = self._convert_to_ohlcv(price_data)

            return ohlcv_data, orderbook_data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.exception("Detailed error:")
            return None, None

    def _load_data_batch(
            self,
            db: MongoDBConnection,
            collection_name: str,
            symbol: str,
            start_date: datetime,
            end_date: datetime
    ) -> Optional[pd.DataFrame]:
        try:
            query = {
                'symbol': symbol,
                'trade_time' if collection_name == 'price_data' else 'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            self.logger.info(f"MongoDB query for {collection_name}: {query}")

            data = pd.DataFrame(
                list(db.get_collection(collection_name)
                     .find(query, allow_disk_use=True)  # Enable disk use for sorting
                     .sort('trade_time' if collection_name == 'price_data' else 'timestamp', 1))
            )

            if data.empty:
                self.logger.warning(f"No data found in {collection_name} for the given range")
                return None

            self.logger.info(f"Loaded {len(data)} records from {collection_name}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading batch from {collection_name}: {e}")
            return None

    def _convert_to_ohlcv(self, trades_df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """Convert trade data to OHLCV format"""
        try:
            # Ensure timestamp column
            if 'trade_time' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['trade_time'])
            elif 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            else:
                raise ValueError("No timestamp column found")

            df = trades_df.set_index('timestamp')

            # Create OHLCV
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df['price'].resample(timeframe).first()
            ohlcv['high'] = df['price'].resample(timeframe).max()
            ohlcv['low'] = df['price'].resample(timeframe).min()
            ohlcv['close'] = df['price'].resample(timeframe).last()
            ohlcv['volume'] = df['quantity'].resample(timeframe).sum()

            # Remove missing data
            ohlcv = ohlcv.dropna()

            self.logger.info(f"OHLCV data summary:")
            self.logger.info(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
            self.logger.info(f"Number of periods: {len(ohlcv)}")
            self.logger.info(f"Missing values: {ohlcv.isnull().sum().sum()}")

            return ohlcv

        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {e}")
            return pd.DataFrame()

    def train_model(
            self,
            price_data: pd.DataFrame,
            orderbook_data: pd.DataFrame,
            symbol: str
    ) -> Tuple[Optional[EnhancedEnsemble], Dict, Tuple[Dict, Dict]]:
        """Train and evaluate model"""
        try:
            if price_data.empty:
                raise ValueError("Empty price data")

            self.logger.info("Processing features...")

            # Log input data
            self.logger.info(f"Price data shape: {price_data.shape}")
            self.logger.info(f"Price data columns: {price_data.columns}")
            self.logger.info(f"Orderbook data shape: {orderbook_data.shape}")
            self.logger.info(f"Orderbook data columns: {orderbook_data.columns}")

            # Generate features
            features, target = self.feature_processor.prepare_features(
                price_data=price_data,
                orderbook_data=orderbook_data,
                target_minutes=5
            )

            if features.empty or target.empty:
                raise ValueError("Feature generation failed")

            # Log features and target
            self.logger.info(f"Features shape: {features.shape}")
            self.logger.info(f"Target shape: {target.shape}")

            # Split data
            train_end = int(len(features) * 0.8)
            X_train = features[:train_end]
            y_train = target[:train_end]
            X_test = features[train_end:]
            y_test = target[train_end:]

            self.logger.info("Training ensemble model...")

            # Initialize and train model
            model = EnhancedEnsemble()
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

            # Log metrics
            self.logger.info("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

            self.logger.info("\nTest Metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

            # Create visualizations
            figures = self.visualizer.plot_model_performance(
                y_test.values,
                test_predictions,
                price_data['close'][train_end:].values,
                X_test
            )

            # Save model
            model_dir = Path("models/trained")
            model_path = model_dir / f"{symbol}_model.weights.h5"  # Ensure filepath ends with .weights.h5
            model.save(str(model_path))
            self.logger.info(f"Model saved to {model_path}")

            # Save visualizations
            fig_dir = Path("models/figures")
            for name, fig in figures.items():
                if fig is not None:
                    fig.write_html(str(fig_dir / f"{name}.html"))
            self.logger.info(f"Figures saved to {fig_dir}")

            return model, figures, (train_metrics, test_metrics)

        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            self.logger.exception("Detailed error:")
            return None, None, None

    def run_training(self, symbol: str, days: int = 60):
        """Run complete training pipeline"""
        self.logger.info("Starting model training...")

        # Connect to database
        db_config = {
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        }
        db = MongoDBConnection(db_config)

        if not db.connect():
            self.logger.error("Failed to connect to database")
            return

        try:
            # Training parameters
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            self.logger.info(f"Training Configuration:")
            self.logger.info(f"Symbol: {symbol}")
            self.logger.info(f"Start Date: {start_date}")
            self.logger.info(f"End Date: {end_date}")

            # Load data
            price_data, orderbook_data = self.load_training_data(
                db, symbol, start_date, end_date
            )

            if price_data is None or price_data.empty:
                self.logger.error("No valid price data available")
                return

            # Train model
            model, figures, metrics = self.train_model(
                price_data, orderbook_data, symbol
            )

            if model is None:
                self.logger.error("Model training failed")
                return

            self.logger.info("Training completed successfully")

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {e}")
            self.logger.exception("Detailed error:")
        finally:
            db.close()

def main():
    trainer = ModelTrainer()
    trainer.run_training("BTCUSDT", days=60)

if __name__ == "__main__":
    main()