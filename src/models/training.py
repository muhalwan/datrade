import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import TimeSeriesSplit

from src.features.engineering import FeatureEngineering
from .base import BaseModel, ModelConfig, ModelType, ModelMetrics
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel

class ModelTrainer:
    """Enhanced model training coordinator"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection
        self.feature_eng = FeatureEngineering(self.db)
        self.models: Dict[str, BaseModel] = {}
        self.training_history: Dict[str, Dict] = {}
        self.model_configs = self._load_model_configs()

        # Create necessary directories
        Path("models/temp").mkdir(parents=True, exist_ok=True)
        Path("logs/training").mkdir(parents=True, exist_ok=True)

    def _load_model_configs(self) -> Dict:
        """Load and validate model configurations"""
        try:
            config_path = Path("config/model_config.yaml")
            if not config_path.exists():
                self.logger.warning("Model config not found, using defaults")
                return self._get_default_configs()

            with open(config_path) as f:
                config = yaml.safe_load(f)
                return self._process_configs(config)

        except Exception as e:
            self.logger.error(f"Error loading model configs: {str(e)}")
            return self._get_default_configs()

    def create_model(self, config: ModelConfig) -> BaseModel:
        """Create model instance with proper initialization"""
        try:
            if config.type == ModelType.LSTM:
                return LSTMModel(config)
            elif config.type == ModelType.LIGHTGBM:
                return LightGBMModel(config)
            elif config.type == ModelType.ENSEMBLE:
                return EnsembleModel(config)
            else:
                raise ValueError(f"Unknown model type: {config.type}")
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def train_model_with_cv(self, model: BaseModel,
                            train_data: pd.DataFrame,
                            n_splits: int = 5) -> Dict[str, List[float]]:
        """Train model with time series cross-validation"""
        try:
            cv_results = {
                'train_metrics': [],
                'val_metrics': [],
                'fold_times': []
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)

            for fold, (train_idx, val_idx) in enumerate(tscv.split(train_data), 1):
                self.logger.info(f"Training fold {fold}/{n_splits}")

                # Split data
                fold_train = train_data.iloc[train_idx]
                fold_val = train_data.iloc[val_idx]

                # Train and validate
                start_time = datetime.now()
                model.train(fold_train)
                train_metrics = model.validate(fold_train)
                val_metrics = model.validate(fold_val)
                fold_time = (datetime.now() - start_time).total_seconds()

                # Store results
                cv_results['train_metrics'].append(train_metrics)
                cv_results['val_metrics'].append(val_metrics)
                cv_results['fold_times'].append(fold_time)

                self.logger.info(f"Fold {fold} - "
                                 f"Train RMSE: {train_metrics.rmse:.4f}, "
                                 f"Val RMSE: {val_metrics.rmse:.4f}")

            return cv_results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def train(self, symbol: str, train_data: pd.DataFrame,
              test_data: pd.DataFrame = None) -> Dict[str, BaseModel]:
        """Enhanced training pipeline with parallel processing"""
        try:
            self.logger.info(f"Starting model training for {symbol}")
            trained_models = {}

            # Train models in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_model = {}

                for name, config in self.model_configs.items():
                    if not config.enabled:
                        continue

                    model = self.create_model(config)

                    if config.cross_validation:
                        future = executor.submit(
                            self.train_model_with_cv,
                            model,
                            train_data,
                            config.cv_folds
                        )
                    else:
                        future = executor.submit(model.train, train_data)

                    future_to_model[future] = (name, model)

                # Collect results
                for future in future_to_model:
                    name, model = future_to_model[future]
                    try:
                        future.result()  # Get result or raise exception
                        trained_models[name] = model

                        # Evaluate if test data provided
                        if test_data is not None:
                            metrics = model.validate(test_data)
                            self._log_model_metrics(name, metrics)

                    except Exception as e:
                        self.logger.error(f"Error training {name} model: {str(e)}")
                        continue

            # Create and train ensemble if multiple models trained
            if len(trained_models) > 1:
                ensemble = self.create_ensemble(trained_models)
                if test_data is not None:
                    metrics = ensemble.validate(test_data)
                    self._log_model_metrics('ensemble', metrics)
                trained_models['ensemble'] = ensemble

            return trained_models

        except Exception as e:
            self.logger.error(f"Error in training process: {str(e)}")
            raise

    def _log_model_metrics(self, model_name: str, metrics: ModelMetrics):
        """Log comprehensive model metrics"""
        self.logger.info(f"\nMetrics for {model_name}:")
        self.logger.info(f"MSE: {metrics.mse:.4f}")
        self.logger.info(f"RMSE: {metrics.rmse:.4f}")
        self.logger.info(f"MAE: {metrics.mae:.4f}")
        self.logger.info(f"MAPE: {metrics.mape:.4f}")
        self.logger.info(f"Directional Accuracy: {metrics.directional_accuracy:.4f}")
        self.logger.info(f"Training Time: {metrics.training_time:.2f}s")

    def create_ensemble(self, models: Dict[str, BaseModel]) -> EnsembleModel:
        """Create enhanced ensemble from trained models"""
        try:
            config = self.model_configs.get('ensemble', ModelConfig(
                name='ensemble',
                type=ModelType.ENSEMBLE,
                params={'weights': {}},
                features=[],
                target='close'
            ))

            ensemble = EnsembleModel(config)

            # Add models with initial weights
            for name, model in models.items():
                if name != 'ensemble':
                    ensemble.add_model(name, model)

            return ensemble

        except Exception as e:
            self.logger.error(f"Error creating ensemble: {str(e)}")
            raise

    def save_models(self, models: Dict[str, BaseModel], symbol: str) -> None:
        """Save models with enhanced metadata"""
        try:
            base_path = Path(f"models/{symbol}")
            base_path.mkdir(parents=True, exist_ok=True)

            for name, model in models.items():
                model_path = base_path / name
                model.save(str(model_path))
                self.logger.info(f"Saved {name} model for {symbol}")

        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, symbol: str) -> Dict[str, BaseModel]:
        """Load models with validation"""
        try:
            base_path = Path(f"models/{symbol}")
            if not base_path.exists():
                raise FileNotFoundError(f"No models found for {symbol}")

            loaded_models = {}

            for model_dir in base_path.iterdir():
                if model_dir.is_dir():
                    name = model_dir.name
                    config_path = model_dir / "config.json"

                    if config_path.exists():
                        with open(config_path) as f:
                            config = ModelConfig(**yaml.safe_load(f))
                            model = self.create_model(config)
                            model.load(str(model_dir))
                            loaded_models[name] = model

            return loaded_models

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def get_model_metrics(self, symbol: str) -> Dict:
        """Get comprehensive model metrics"""
        try:
            metrics = {}
            for name, history in self.training_history.items():
                if 'metrics' in history:
                    metrics[name] = history['metrics']
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {}