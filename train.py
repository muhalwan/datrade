import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict
from keras.callbacks import ModelCheckpoint
import talib

from src.data.database.connection import MongoDBConnection
from src.features.processor import FeatureProcessor
from src.features.selector import FeatureSelector
from src.models.ensemble import EnhancedEnsemble
from src.utils.metrics import calculate_trading_metrics
from src.utils.visualization import TradingVisualizer
from src.config import settings

class ModelTrainer:
    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_processor = FeatureProcessor()
        self.feature_selector = FeatureSelector()
        self.visualizer = TradingVisualizer()
        self.scaler = None

        # Create necessary directories
        for dir_name in ['logs', 'models/trained', 'models/figures']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        self.checkpoint = ModelCheckpoint(
            filepath='models/trained/best_model.weights.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

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

    def _load_data_batch(
            self,
            db: MongoDBConnection,
            collection_name: str,
            symbol: str,
            start_date: datetime,
            end_date: datetime
    ) -> Optional[pd.DataFrame]:
        try:
            time_field = 'trade_time' if collection_name == 'price_data' else 'timestamp'
            query = {
                'symbol': symbol,
                time_field: {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            self.logger.info(f"MongoDB query for {collection_name}: {query}")

            # Use cursor to handle large datasets and sort in memory
            cursor = db.get_collection(collection_name).find(query)
            data = list(cursor)

            if not data:
                self.logger.warning(f"No data found in {collection_name} for the given range")
                return None

            df = pd.DataFrame(data)
            # Sort in pandas instead of MongoDB to avoid memory issues
            df = df.sort_values(time_field)

            self.logger.info(f"Loaded {len(df)} records from {collection_name}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading batch from {collection_name}: {e}")
            return None

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

                price_batch = self._load_data_batch(db, 'price_data', symbol, current_start, current_end)
                if price_batch is not None:
                    price_data_list.append(price_batch)

                orderbook_batch = self._load_data_batch(db, 'order_book', symbol, current_start, current_end)
                if orderbook_batch is not None:
                    orderbook_data_list.append(orderbook_batch)

                current_start = current_end

            if not price_data_list:
                self.logger.error("No price data found")
                return None, None

            price_data = pd.concat(price_data_list, ignore_index=True)
            orderbook_data = pd.concat(orderbook_data_list, ignore_index=True) if orderbook_data_list else pd.DataFrame()

            self.logger.info(f"Total loaded price records: {len(price_data)}")
            self.logger.info(f"Total loaded orderbook records: {len(orderbook_data)}")

            ohlcv_data = self._convert_to_ohlcv(price_data)

            return ohlcv_data, orderbook_data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.exception("Detailed error:")
            return None, None

    def _convert_to_ohlcv(self, trades_df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """Convert trade data to OHLCV format"""
        try:
            if 'trade_time' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['trade_time'])
            elif 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            else:
                raise ValueError("No timestamp column found")

            df = trades_df.set_index('timestamp')

            ohlcv = pd.DataFrame()
            ohlcv['open'] = df['price'].resample(timeframe).first()
            ohlcv['high'] = df['price'].resample(timeframe).max()
            ohlcv['low'] = df['price'].resample(timeframe).min()
            ohlcv['close'] = df['price'].resample(timeframe).last()
            ohlcv['volume'] = df['quantity'].resample(timeframe).sum()

            ohlcv = ohlcv.dropna()

            self.logger.info(f"OHLCV data summary:")
            self.logger.info(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
            self.logger.info(f"Number of periods: {len(ohlcv)}")
            self.logger.info(f"Missing values: {ohlcv.isnull().sum().sum()}")

            return ohlcv

        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {e}")
            return pd.DataFrame()

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA-Lib"""
        try:
            features = pd.DataFrame(index=df.index)

            # Moving averages
            features['sma_5'] = talib.SMA(df['close'].values, timeperiod=5)
            features['sma_10'] = talib.SMA(df['close'].values, timeperiod=10)
            features['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)

            # Volume indicators
            features['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            features['ad'] = talib.AD(df['high'].values, df['low'].values,
                                      df['close'].values, df['volume'].values)

            # Momentum indicators
            features['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist

            # Volatility indicators
            features['atr'] = talib.ATR(df['high'].values, df['low'].values,
                                        df['close'].values, timeperiod=14)
            features['bbands_upper'], features['bbands_middle'], features['bbands_lower'] = \
                talib.BBANDS(df['close'].values, timeperiod=20)

            # Use more modern pandas methods to handle NaN values
            features = features.ffill().bfill()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return pd.DataFrame()

    def train_model(self, price_data, orderbook_data, symbol):
        try:
            if price_data.empty:
                raise ValueError("Empty price data")

            self.logger.info("Processing features...")

            # Calculate technical features
            tech_features = self.calculate_technical_features(price_data)

            # Merge price data with technical features
            combined_features = pd.concat([price_data, tech_features], axis=1)

            # Use the feature processor with the combined features
            features, target = self.feature_processor.prepare_features(
                price_data=combined_features,
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

            # Fit scaler on training data
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

            # Initialize and train model
            model = EnhancedEnsemble(
                config={
                    'lstm': {
                        'sequence_length': 30,
                        'lstm_units': [32, 16],
                        'dropout_rate': 0.3,
                        'recurrent_dropout': 0.3,
                        'learning_rate': 0.001
                    },
                    'xgboost': {
                        'max_depth': 4,
                        'learning_rate': 0.05,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'min_child_weight': 3,
                        'gamma': 0.1
                    }
                }
            )
            model.train(X_train, y_train)

            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Calculate metrics
            train_metrics = calculate_trading_metrics(
                y_train, train_predictions, price_data['close'][:train_end].values
            )
            test_metrics = calculate_trading_metrics(
                y_test, test_predictions, price_data['close'][train_end:].values
            )

            # Save model
            model_path = Path("models/trained") / f"{symbol}_model.weights.h5"
            model.save(str(model_path))
            self.logger.info(f"Model saved to {model_path}")

            return model, None, (train_metrics, test_metrics)

        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            self.logger.exception("Detailed error:")
            return None, None, None

    def run_training(self, symbol: str, days: int = 60):
        """Run complete training pipeline"""
        self.logger.info("Starting model training...")

        db_config = {
            'connection_string': settings.mongodb_uri,
            'name': settings.db_name
        }
        db = MongoDBConnection(db_config)

        if not db.connect():
            self.logger.error("Failed to connect to database")
            return

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            self.logger.info(f"Training Configuration:")
            self.logger.info(f"Symbol: {symbol}")
            self.logger.info(f"Start Date: {start_date}")
            self.logger.info(f"End Date: {end_date}")

            price_data, orderbook_data = self.load_training_data(
                db, symbol, start_date, end_date
            )

            if price_data is None or price_data.empty:
                self.logger.error("No valid price data available")
                return

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