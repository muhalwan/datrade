import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path

from src.features.engineering import FeatureEngineering
from .base import BaseModel, ModelConfig, ModelType
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel

class ModelTrainer:
    """Coordinates model training and evaluation"""

    def __init__(self, db_connection):
        """Initialize model trainer with database connection"""
        self.logger = logging.getLogger(__name__)
        self.db = db_connection
        self.feature_eng = FeatureEngineering(self.db)
        self.models: Dict[str, BaseModel] = {}
        self.training_history: Dict[str, Dict] = {}

        # Load model configurations
        self.model_configs = self._load_model_configs()

    def _load_model_configs(self) -> Dict:
        """Load model configurations from YAML"""
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

    def _get_default_configs(self) -> Dict:
        """Get default model configurations"""
        return {
            'lstm': ModelConfig(
                name='lstm',
                type=ModelType.LSTM,
                params={
                    'sequence_length': 10,
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.2
                },
                features=['close', 'volume', 'rsi', 'macd', 'bb_high', 'bb_low'],
                target='close'
            ),
            'lightgbm': ModelConfig(
                name='lightgbm',
                type=ModelType.LIGHTGBM,
                params={
                    'objective': 'regression',
                    'metric': 'mse',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                },
                features=['close', 'volume', 'rsi', 'macd', 'bb_high', 'bb_low'],
                target='close'
            )
        }


    def _process_configs(self, config: Dict) -> Dict:
        """Process raw config into ModelConfig objects"""
        processed_configs = {}

        if 'models' in config:
            for name, model_config in config['models'].items():
                if model_config.get('enabled', True):
                    processed_configs[name] = ModelConfig(
                        name=name,
                        type=ModelType[name.upper()],
                        params=model_config['params'],
                        features=model_config['features'],
                        target='close',
                        enabled=True
                    )

        return processed_configs

    def create_model(self, config: ModelConfig) -> BaseModel:
        """Create model instance based on configuration"""
        if config.type == ModelType.LSTM:
            return LSTMModel(config)
        elif config.type == ModelType.LIGHTGBM:
            return LightGBMModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.type}")

    def train(self, symbol: str, train_data: pd.DataFrame,
              test_data: pd.DataFrame = None) -> Dict[str, BaseModel]:
        """Train all models for a symbol"""
        try:
            self.logger.info(f"Starting model training for {symbol}")
            trained_models = {}

            # Train individual models
            for name, config in self.model_configs.items():
                self.logger.info(f"Training {name} model...")

                try:
                    # Create and train model
                    model = self.create_model(config)
                    train_start = datetime.now()
                    model.train(train_data)
                    train_time = (datetime.now() - train_start).total_seconds()

                    # Evaluate if test data provided
                    metrics = None
                    if test_data is not None:
                        metrics = self.evaluate_model(model, test_data)

                    # Store results
                    trained_models[name] = model
                    self.training_history[name] = {
                        'train_time': train_time,
                        'metrics': metrics,
                        'timestamp': datetime.now()
                    }

                    self.logger.info(f"Successfully trained {name} model")
                    if metrics:
                        self.logger.info(f"Test metrics for {name}: {metrics}")

                except Exception as e:
                    self.logger.error(f"Error training {name} model: {str(e)}")
                    continue

            # Create and train ensemble if multiple models trained
            if len(trained_models) > 1:
                try:
                    self.logger.info("Training ensemble model...")
                    ensemble = self.create_ensemble(trained_models)
                    trained_models['ensemble'] = ensemble

                    if test_data is not None:
                        metrics = self.evaluate_model(ensemble, test_data)
                        self.logger.info(f"Ensemble test metrics: {metrics}")

                except Exception as e:
                    self.logger.error(f"Error training ensemble model: {str(e)}")

            return trained_models

        except Exception as e:
            self.logger.error(f"Error in training process: {str(e)}")
            raise

    def create_ensemble(self, models: Dict[str, BaseModel]) -> EnsembleModel:
        """Create ensemble from trained models"""
        try:
            # Get ensemble configuration
            config = ModelConfig(
                name='ensemble',
                type=ModelType.ENSEMBLE,
                params={'weights': self.model_configs.get('ensemble', {}).get('weights', {})},
                features=[],  # Ensemble uses individual model features
                target='close'
            )

            # Create and setup ensemble
            ensemble = EnsembleModel(config)

            # Add models with weights
            default_weight = 1.0 / len(models)
            for name, model in models.items():
                weight = config.params['weights'].get(name, default_weight)
                ensemble.add_model(name, model, weight)

            return ensemble

        except Exception as e:
            self.logger.error(f"Error creating ensemble: {str(e)}")
            raise

    def evaluate_model(self, model: BaseModel, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        try:
            predictions = model.predict(test_data)
            actual = test_data[model.config.target].values

            # Remove sequence padding from predictions
            if isinstance(model, LSTMModel):
                seq_len = model.sequence_length
                predictions = predictions[seq_len:]
                actual = actual[seq_len:]

            if len(predictions) == 0 or len(actual) == 0:
                raise ValueError("No valid predictions or actual values for evaluation")

            # Calculate metrics
            mse = np.mean((predictions - actual) ** 2)
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(mse)

            # Calculate directional accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actual) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)

            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy)
            }

        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                'mse': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'directional_accuracy': np.nan
            }

    def save_models(self, models: Dict[str, BaseModel], symbol: str) -> None:
        """Save all models to disk"""
        try:
            base_path = Path(f"models/{symbol}")
            base_path.mkdir(parents=True, exist_ok=True)

            for name, model in models.items():
                model_path = base_path / name
                model.save(str(model_path))

                # Save training history
                if name in self.training_history:
                    history_path = model_path / "training_history.yaml"
                    with open(history_path, 'w') as f:
                        yaml.dump(self.training_history[name], f)

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
                    config_path = model_dir / "config.yaml"

                    # Load model configuration
                    if config_path.exists():
                        with open(config_path) as f:
                            config = ModelConfig(**yaml.safe_load(f))
                    else:
                        config = self.model_configs.get(name)

                    if config:
                        # Create and load model
                        model = self.create_model(config)
                        model.load(str(model_dir))
                        loaded_models[name] = model

                        # Load training history
                        history_path = model_dir / "training_history.yaml"
                        if history_path.exists():
                            with open(history_path) as f:
                                self.training_history[name] = yaml.safe_load(f)

            return loaded_models

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def get_training_history(self) -> Dict:
        """Get training history for all models"""
        return self.training_history

    def get_model_metrics(self, symbol: str) -> Dict:
        """Get current model metrics"""
        try:
            metrics = {}
            for name, history in self.training_history.items():
                if 'metrics' in history:
                    metrics[name] = history['metrics']
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {}