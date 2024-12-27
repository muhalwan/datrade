import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from src.data.database.connection import MongoDBConnection
from src.features.processor import FeatureProcessor
from src.models.ensemble import EnhancedEnsemble
from src.utils.metrics import calculate_trading_metrics
from src.utils.visualization import TradingVisualizer
from src.config import settings
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_processor = FeatureProcessor()
        self.visualizer = TradingVisualizer()
        self.ensemble_model = EnhancedEnsemble()

        for d in ["logs", "models/trained", "models/figures"]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"training_{stamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
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
            time_field = "trade_time" if collection_name == "price_data" else "timestamp"
            query = {
                "symbol": symbol,
                time_field: {"$gte": start_date, "$lt": end_date}
            }
            self.logger.info(f"MongoDB query for {collection_name}: {query}")
            cursor = db.get_collection(collection_name).find(query)
            data = list(cursor)
            if not data:
                self.logger.warning(f"No data found in {collection_name} for the given range")
                return None
            df = pd.DataFrame(data)
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
        try:
            self.logger.info(f"Loading data from {start_date} to {end_date}")
            price_data_list = []
            orderbook_data_list = []

            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + batch_size, end_date)
                self.logger.info(f"Processing batch from {current_start} to {current_end}")

                p_batch = self._load_data_batch(db, "price_data", symbol, current_start, current_end)
                if p_batch is not None:
                    price_data_list.append(p_batch)

                o_batch = self._load_data_batch(db, "order_book", symbol, current_start, current_end)
                if o_batch is not None:
                    orderbook_data_list.append(o_batch)

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

    def _convert_to_ohlcv(self, trades_df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
        try:
            if "trade_time" in trades_df.columns:
                trades_df["timestamp"] = pd.to_datetime(trades_df["trade_time"])
            elif "timestamp" in trades_df.columns:
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            else:
                raise ValueError("No timestamp column found")

            df = trades_df.set_index("timestamp")
            ohlcv = pd.DataFrame()
            ohlcv["open"] = df["price"].resample(timeframe).first()
            ohlcv["high"] = df["price"].resample(timeframe).max()
            ohlcv["low"] = df["price"].resample(timeframe).min()
            ohlcv["close"] = df["price"].resample(timeframe).last()
            ohlcv["volume"] = df["quantity"].resample(timeframe).sum()
            ohlcv = ohlcv.dropna()

            self.logger.info("OHLCV data summary:")
            self.logger.info(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
            self.logger.info(f"Number of periods: {len(ohlcv)}")
            self.logger.info(f"Missing values: {ohlcv.isnull().sum().sum()}")
            return ohlcv
        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {e}")
            return pd.DataFrame()

    def train_model(self, price_data: pd.DataFrame, orderbook_data: pd.DataFrame, symbol: str):
        try:
            if price_data.empty:
                raise ValueError("Empty price data")

            self.logger.info("Processing features...")
            features, target = self.feature_processor.prepare_features(
                price_data=price_data,
                orderbook_data=orderbook_data,
                target_minutes=5,
                include_sentiment=True
            )
            if features.empty or target.empty:
                raise ValueError("Feature generation failed")

            split_idx = int(len(features) * 0.8)
            X_train = features.iloc[:split_idx].copy()
            y_train = target.iloc[:split_idx].copy()
            X_test = features.iloc[split_idx:].copy()
            y_test = target.iloc[split_idx:].copy()

            self.logger.info("Fitting scaler and feature selector on training data...")
            self.ensemble_model.train(X_train, y_train)

            self.logger.info("Making predictions on training data...")
            train_predictions = self.ensemble_model.predict(X_train)

            self.logger.info("Making predictions on testing data...")
            test_predictions = self.ensemble_model.predict(X_test)

            train_metrics = calculate_trading_metrics(
                y_true=y_train.values,
                y_pred=train_predictions,
                prices=price_data["close"].iloc[:split_idx].values
            )
            test_metrics = calculate_trading_metrics(
                y_true=y_test.values,
                y_pred=test_predictions,
                prices=price_data["close"].iloc[split_idx:].values
            )

            self.logger.info(f"Training Metrics: {train_metrics}")
            self.logger.info(f"Testing Metrics: {test_metrics}")

            # Plot LSTM training history if available
            lstm_hist = self.ensemble_model.models["lstm"].training_history
            if "loss" in lstm_hist and "val_loss" in lstm_hist:
                plt.figure(figsize=(6, 4))
                plt.plot(lstm_hist["loss"], label="loss")
                plt.plot(lstm_hist["val_loss"], label="val_loss")
                plt.title("LSTM Training History")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                fig_path = Path("models/figures") / f"{symbol}_lstm_history.png"
                plt.savefig(fig_path)
                self.logger.info(f"Saved LSTM training history plot to {fig_path}")
                plt.close()

            model_path = Path("models/trained") / f"{symbol}_model"
            saved_ok = self.ensemble_model.save(model_path)
            if saved_ok:
                self.logger.info(f"Model saved to {model_path}.model and {model_path}.meta")
            else:
                self.logger.error("Failed to save the ensemble model.")

            return self.ensemble_model, None, (train_metrics, test_metrics)
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            self.logger.exception("Detailed error:")
            return None, None, None

    def run_training(self, symbol: str, days: int = 60):
        self.logger.info("Starting model training...")

        db_conf = {
            "connection_string": settings.mongodb_uri,
            "name": settings.db_name
        }
        db = MongoDBConnection(db_conf)

        if not db.connect():
            self.logger.error("Failed to connect to database")
            return

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            self.logger.info("Training Configuration:")
            self.logger.info(f"Symbol: {symbol}")
            self.logger.info(f"Start Date: {start_date}")
            self.logger.info(f"End Date: {end_date}")

            price_data, orderbook_data = self.load_training_data(db, symbol, start_date, end_date)
            if price_data is None or price_data.empty:
                self.logger.error("No valid price data available")
                return

            model, figs, metrics = self.train_model(price_data, orderbook_data, symbol)
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
