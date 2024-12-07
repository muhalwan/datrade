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
            end_date: datetime
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load and prepare training data"""
        try:
            self.logger.info(f"Loading data from {start_date} to {end_date}")

            # Construct query
            query = {
                'symbol': symbol,
                'trade_time': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            self.logger.info(f"MongoDB query: {query}")

            # Load price data
            price_data = pd.DataFrame(
                list(db.get_collection('price_data')
                     .find(query)
                     .sort('trade_time', 1))
            )

            if price_data.empty:
                self.logger.error("No price data found")
                return None, None

            self.logger.info(f"Loaded {len(price_data)} price records")

            # Convert to OHLCV
            ohlcv_data = self._convert_to_ohlcv(price_data)

            # Load orderbook data
            orderbook_query = {
                'symbol': symbol,
                'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            orderbook_data = pd.DataFrame(
                list(db.get_collection('order_book')
                     .find(orderbook_query)
                     .sort('timestamp', 1))
            )

            self.logger.info(f"Loaded {len(orderbook_data)} orderbook records")

            return ohlcv_data, orderbook_data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.exception("Detailed error:")
            return None, None

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

            # Generate features
            features, target = self.feature_processor.prepare_features(
                price_data=price_data,
                orderbook_data=orderbook_data,
                target_minutes=5
            )

            if features.empty or target.empty:
                raise ValueError("Feature generation failed")

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
            model_path = model_dir / f"{symbol}_model"
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