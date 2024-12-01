import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from pathlib import Path

from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from .prophet_model import ProphetModel
from .ensemble import EnsembleModel
from ..features.pipeline import FeaturePipelineManager

class ModelTrainer:
    """Manages model training and evaluation"""

    def __init__(self, feature_pipeline: FeaturePipelineManager, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.feature_pipeline = feature_pipeline
        self.config = config or {}

        # Ensure config structure
        if not isinstance(self.config, dict):
            self.config = {}

        if 'models' not in self.config:
            self.config['models'] = {}

        # Initialize models
        self.models = self._initialize_models()

        # Training queue
        self.training_queue = queue.Queue()

        # Start training thread
        self._start_training_thread()

    def _initialize_models(self) -> Dict[str, BaseModel]:
        """Initialize all models"""
        try:
            model_configs = self.config.get('models', {})

            return {
                'lstm': LSTMModel(model_configs.get('lstm')),
                'xgboost': XGBoostModel(model_configs.get('xgboost')),
                'prophet': ProphetModel(model_configs.get('prophet')),
                'ensemble': EnsembleModel(model_configs.get('ensemble'))
            }

        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
            raise

    def _start_training_thread(self):
        """Start background training thread"""
        def training_worker():
            while True:
                try:
                    # Get training task from queue
                    task = self.training_queue.get()
                    if task is None:
                        continue

                    model_name = task['model']
                    training_data = task['data']

                    # Train model
                    self._train_model(model_name, training_data)

                except Exception as e:
                    self.logger.error(f"Training worker error: {str(e)}")

        threading.Thread(target=training_worker, daemon=True).start()

    def _train_model(self, model_name: str, data: pd.DataFrame):
        """Train specific model"""
        try:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Unknown model: {model_name}")

            # Preprocess data
            if model_name == 'prophet':
                train_data, val_data = model.preprocess(data)
                metrics = model.train(train_data, val_data)
            else:
                X, y = model.preprocess(data)
                metrics = model.train(X, y)

            # Save trained model
            self._save_model(model, model_name, metrics)

            self.logger.info(f"Trained {model_name} model. Metrics: {metrics}")

        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")

    def _save_model(self, model: BaseModel, name: str, metrics: Dict):
        """Save trained model and metrics"""
        try:
            # Create model directory
            model_dir = Path(f"models/{name}")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"{name}_{timestamp}.joblib"
            model.save(model_path)

            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_path = model_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, mode='a', header=not metrics_path.exists())

        except Exception as e:
            self.logger.error(f"Model saving error: {str(e)}")

    def schedule_training(self, model_name: str = 'all'):
        """Schedule model training"""
        try:
            # Get latest feature data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Use last 30 days

            data = self.feature_pipeline.get_historical_features(
                start_time, end_time)

            if data.empty:
                raise ValueError("No training data available")

            # Schedule training for specified models
            if model_name == 'all':
                for name in self.models.keys():
                    self.training_queue.put({
                        'model': name,
                        'data': data
                    })
            else:
                self.training_queue.put({
                    'model': model_name,
                    'data': data
                })

        except Exception as e:
            self.logger.error(f"Training scheduling error: {str(e)}")

    def get_predictions(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all trained models"""
        try:
            predictions = {}

            for name, model in self.models.items():
                if not model.is_trained:
                    continue

                if name == 'prophet':
                    pred_df = model.predict()
                    predictions[name] = pred_df['yhat'].values
                else:
                    X, _ = model.preprocess(data)
                    predictions[name] = model.predict(X)

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return {}

    def get_model_metrics(self, model_name: str) -> pd.DataFrame:
        """Get historical training metrics for model"""
        try:
            metrics_path = Path(f"models/{model_name}/metrics.csv")
            if not metrics_path.exists():
                return pd.DataFrame()

            return pd.read_csv(metrics_path)

        except Exception as e:
            self.logger.error(f"Metrics retrieval error: {str(e)}")
            return pd.DataFrame()