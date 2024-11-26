import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import json
import torch

from src.features.engineering import FeatureEngineering
from .base import BaseModel, ModelConfig, ModelType, ModelMetrics
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel

class ModelTrainer:
    """Enhanced model training coordinator with advanced features"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection
        self.feature_eng = FeatureEngineering(self.db)
        self.models: Dict[str, BaseModel] = {}
        self.training_history: Dict[str, Dict] = {}

        # Create necessary directories
        for directory in ["models", "models/temp", "logs/training", "logs/tensorboard"]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Load configurations
        self.model_configs = self._load_model_configs()

        # Initialize GPU settings
        self._setup_gpu()

    def _setup_gpu(self):
        """Setup GPU configurations for different frameworks"""
        try:
            # PyTorch GPU setup
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.backends.cudnn.benchmark = True
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU for PyTorch models")

        except Exception as e:
            self.logger.error(f"Error setting up GPU: {e}")
            self.device = torch.device("cpu")

    def _get_default_configs(self) -> Dict:
        """Get default model configurations"""
        base_features = [
            'close', 'open', 'high', 'low', 'volume',
            'bb_high_20', 'bb_low_20',
            'rsi', 'macd', 'macd_signal',
            'mfi', 'adx', 'vortex_pos', 'vortex_neg',
            'volume_sma_30', 'volatility_30'
        ]

        return {
            'lstm': ModelConfig(
                name='lstm',
                type=ModelType.LSTM,
                params={
                    'sequence_length': 10,
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'weight_decay': 0.004,
                    'validation_split': 0.2
                },
                features=base_features,
                target='close',
                cross_validation=False,
                cv_folds=5
            ),
            'lightgbm': ModelConfig(
                name='lightgbm',
                type=ModelType.LIGHTGBM,
                params={
                    'objective': 'huber',
                    'metric': 'rmse',
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'early_stopping_rounds': 50,
                    'device': 'gpu' if torch.cuda.is_available() else 'cpu'
                },
                features=base_features + [
                    'ultimate_oscillator',
                    'awesome_oscillator',
                    'mass_index',
                    'stoch_k',
                    'stoch_d',
                    'williams_r'
                ],
                target='close',
                cross_validation=True,
                cv_folds=5
            )
        }

    def _process_configs(self, config: Dict) -> Dict:
        """Process raw config into ModelConfig objects"""
        processed_configs = {}

        if 'models' not in config:
            self.logger.warning("No models section in config, using defaults")
            return self._get_default_configs()

        for name, model_config in config['models'].items():
            if model_config.get('enabled', True):
                try:
                    # Create base config
                    base_config = ModelConfig(
                        name=name,
                        type=ModelType[model_config['type'].upper()],
                        params=model_config.get('params', {}),
                        features=model_config.get('features', []),
                        target='close',
                        enabled=True,
                        cross_validation=model_config.get('cross_validation', False),
                        cv_folds=model_config.get('cv_folds', 5)
                    )

                    # Add GPU settings if applicable
                    if base_config.type == ModelType.LIGHTGBM and torch.cuda.is_available():
                        base_config.params.update({
                            'device': 'gpu',
                            'gpu_platform_id': 0,
                            'gpu_device_id': 0
                        })

                    processed_configs[name] = base_config

                except Exception as e:
                    self.logger.error(f"Error processing config for {name}: {str(e)}")
                    continue

        if not processed_configs:
            self.logger.warning("No valid model configs found, using defaults")
            return self._get_default_configs()

        return processed_configs

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

                self.logger.info(
                    f"Fold {fold} - "
                    f"Train RMSE: {train_metrics.rmse:.4f}, "
                    f"Val RMSE: {val_metrics.rmse:.4f}"
                )

            return cv_results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def train(self, symbol: str, train_data: pd.DataFrame,
              test_data: pd.DataFrame = None) -> Dict[str, BaseModel]:
        """Train all models with parallel processing"""
        try:
            self.logger.info(f"Starting model training for {symbol}")
            trained_models = {}

            # Convert data types to float
            numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
            train_data[numeric_cols] = train_data[numeric_cols].astype(np.float32)
            if test_data is not None:
                test_data[numeric_cols] = test_data[numeric_cols].astype(np.float32)

            # Train models in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_model = {}

                for name, config in self.model_configs.items():
                    if not config.enabled:
                        continue

                    try:
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
                    except Exception as e:
                        self.logger.error(f"Error initializing {name} model: {str(e)}")
                        continue

                # Collect results
                for future in future_to_model:
                    name, model = future_to_model[future]
                    try:
                        future.result()
                        trained_models[name] = model

                        # Evaluate if test data provided
                        if test_data is not None:
                            metrics = model.validate(test_data)
                            self._log_model_metrics(name, metrics)

                    except Exception as e:
                        self.logger.error(f"Error training {name} model: {str(e)}")
                        continue

            # Create ensemble if multiple models trained successfully
            if len(trained_models) > 1:
                try:
                    ensemble = self.create_ensemble(trained_models)
                    if test_data is not None:
                        metrics = ensemble.validate(test_data)
                        self._log_model_metrics('ensemble', metrics)
                    trained_models['ensemble'] = ensemble
                except Exception as e:
                    self.logger.error(f"Error creating ensemble: {str(e)}")

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

    def create_ensemble(self, models: Dict[str, BaseModel]) -> Optional[EnsembleModel]:
        """Create enhanced ensemble from trained models"""
        try:
            if len(models) < 1:
                raise ValueError("No models available for ensemble")

            config = ModelConfig(
                name='ensemble',
                type=ModelType.ENSEMBLE,
                params={
                    'weights': {},
                    'performance_window': 20,
                    'weight_update_frequency': 10,
                    'min_weight': 0.1,
                    'adaptive_weighting': True
                },
                features=list(set().union(*[m.config.features for m in models.values()])),
                target='close'
            )

            ensemble = EnsembleModel(config)

            # Add successful models
            for name, model in models.items():
                if name != 'ensemble':
                    ensemble.add_model(name, model, weight=1.0/len(models))

            return ensemble

        except Exception as e:
            self.logger.error(f"Error creating ensemble: {str(e)}")
            return None

    def save_models(self, models: Dict[str, BaseModel], symbol: str) -> None:
        """Save all models with metadata"""
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
        """Load all models for a symbol"""
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
        """Get all model metrics"""
        try:
            metrics = {}
            for name, history in self.training_history.items():
                if 'metrics' in history:
                    metrics[name] = history['metrics']
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {}

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