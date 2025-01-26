import logging
from datetime import datetime, timedelta
from typing import Optional
from src.data.database.connection import MongoDBConnection
from src.features.processor import FeatureProcessor
from src.models.ensemble import EnsembleModel
from src.utils.metrics import calculate_trading_metrics
from src.config import settings
from src.utils.visualization import TradingVisualizer
import sys
import os

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

class Trainer:
    def __init__(self, db: MongoDBConnection, config: dict):
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.config = config
        self.feature_processor = FeatureProcessor()
        self.ensemble = EnsembleModel(config=self.config['ensemble'])

    def run_training(self, symbol: str, days: int):
        try:
            self.logger.info("Starting model training...")
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()

            self.logger.info("Training Configuration:")
            self.logger.info(f"Symbol: {symbol}")
            self.logger.info(f"Start Date: {start_date}")
            self.logger.info(f"End Date: {end_date}")

            price_data = self.db.fetch_price_data(symbol, start_date, end_date)
            orderbook_data = self.db.fetch_orderbook_data(symbol, start_date, end_date)
            self.ensemble.price_data = price_data.iloc[:-1]

            if price_data is None or orderbook_data is None:
                self.logger.error("Failed to fetch necessary data for training.")
                return

            if price_data.empty or orderbook_data.empty:
                self.logger.error("Fetched data is empty. Aborting training.")
                return

            features, target = self.feature_processor.process(price_data, orderbook_data)

            if features.empty or target.empty:
                self.logger.error("Processed features or target are empty. Aborting training.")
                return

            self.ensemble.train(features, target)

            # Get LSTM sequence length from config
            lstm_seq_len = self.config['ensemble']['lstm']['sequence_length']

            # Align predictions with data starting from sequence boundary
            predictions = self.ensemble.predict(features)

            # Calculate valid start index for metrics
            start_idx = lstm_seq_len - 1  # Predictions start after sequence window
            min_length = min(len(predictions), len(target) - start_idx, len(price_data) - start_idx)

            metrics = calculate_trading_metrics(
                y_true=target.values[start_idx:start_idx + min_length],
                y_pred=predictions[:min_length],
                prices=price_data['close'].values[start_idx:start_idx + min_length]
            )

            self.logger.info("Training Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}")

            # Save models
            os.makedirs("models/trained", exist_ok=True)
            self.ensemble.save("models/trained/BTCUSDT_model")

            # --- Save visualizations ---
            trading_visualizer = TradingVisualizer()
            figures = trading_visualizer.plot_model_performance(
                y_true=target.values[start_idx:start_idx + min_length],
                y_pred=predictions[:min_length],
                prices=price_data['close'].values[start_idx:start_idx + min_length],
                features=features
            )

            report_path = f"reports/training_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            trading_visualizer.save_visualization(figures, metrics, report_path)

        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)


def main():
    logger = setup_logging()
    logger.info("Initializing MongoDB connection...")
    db_config = {
        'connection_string': settings.mongodb_uri,
        'name': settings.db_name
    }
    db = MongoDBConnection(db_config)

    if not db.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return

    config = {
        'ensemble': {
            'lstm': {
                'sequence_length': 30,
                'n_features': None,
                'lstm_units': [32, 16],
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.0,
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'xgboost': {
                'params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                }
            },
            'feature_selector': {
                'n_features': 50
            },
            'ensemble': {
                'weights': {
                    'lstm': 0.5,
                    'xgboost': 0.5
                }
            },
            'cross_validation': {
                'n_splits': 5
            }
        }
    }

    trainer = Trainer(db, config)
    trainer.run_training(symbol="BTCUSDT", days=60)

    db.close()

if __name__ == "__main__":
    main()
