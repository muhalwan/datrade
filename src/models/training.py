import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
import os

from src.features.engineering import FeatureEngineering
from .base import BaseModel, ModelConfig, ModelType
from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel

class ModelTrainer:
    """Coordinates model training and evaluation"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection
        self.feature_eng = FeatureEngineering(db_connection)

        # Default model configurations
        self.default_configs = {
            'lstm': ModelConfig(
                name='lstm',
                type=ModelType.LSTM,
                params={
                    'sequence_length': 3,  # Reduced sequence length
                    'batch_size': 2,       # Smaller batch size
                    'epochs': 20,          # Fewer epochs
                    'validation_split': 0.1 # Smaller validation split
                },
                features=['close', 'volume', 'returns'],  # Simplified feature set
                target='close'
            ),
            'lightgbm': ModelConfig(
                name='lightgbm',
                type=ModelType.LIGHTGBM,
                params={
                    'lgb_params': {
                        'objective': 'regression',
                        'metric': 'mse',
                        'num_leaves': 7,  # Reduced complexity
                        'learning_rate': 0.1,
                        'feature_fraction': 0.8
                    }
                },
                features=['close', 'volume', 'returns'],
                target='close'
            )
        }

    def create_model(self, config: ModelConfig) -> BaseModel:
        """Create model instance based on configuration"""
        if config.type == ModelType.LSTM:
            return LSTMModel(config)
        elif config.type == ModelType.LIGHTGBM:
            return LightGBMModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.type}")

    def create_ensemble(self, models: Dict[str, BaseModel],
                        weights: Dict[str, float]) -> EnsembleModel:
        """Create ensemble from multiple models"""
        config = ModelConfig(
            name='ensemble',
            type=ModelType.ENSEMBLE,
            params={'weights': weights},
            features=[],  # Ensemble uses individual model features
            target='close'
        )

        ensemble = EnsembleModel(config)
        for name, model in models.items():
            ensemble.add_model(name, model, weights.get(name, 1.0))

        return ensemble

    def train(self, symbol: str, start_time: datetime,
              end_time: datetime, models_config: Dict[str, ModelConfig] = None) -> Dict[str, BaseModel]:
        """Train multiple models"""
        try:
            # Use default configs if none provided
            models_config = models_config or self.default_configs

            # Generate features
            features_df = self.feature_eng.generate_features(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

            if features_df.empty:
                raise ValueError("No features generated for training")

            self.logger.info(f"Generated features shape: {features_df.shape}")
            self.logger.info(f"Feature columns: {features_df.columns.tolist()}")

            trained_models = {}
            for name, config in models_config.items():
                if not config.enabled:
                    continue

                self.logger.info(f"Training {name} model...")
                model = self.create_model(config)
                model.train(features_df)
                trained_models[name] = model

            return trained_models

        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def evaluate_models(self, models: Dict[str, BaseModel],
                        test_data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate model performance"""
        try:
            results = {}
            for name, model in models.items():
                predictions = model.predict(test_data)
                actual = test_data[model.config.target].values

                # Calculate metrics
                mse = np.mean((predictions - actual) ** 2)
                mae = np.mean(np.abs(predictions - actual))
                rmse = np.sqrt(mse)

                # Calculate directional accuracy
                pred_direction = np.diff(predictions) > 0
                actual_direction = np.diff(actual) > 0
                directional_accuracy = np.mean(pred_direction == actual_direction)

                results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy
                }

                self.logger.info(f"\nResults for {name}:")
                for metric, value in results[name].items():
                    self.logger.info(f"{metric.upper()}: {value:.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise

    def save_models(self, models: Dict[str, BaseModel], symbol: str) -> None:
        """Save all models to disk"""
        try:
            for name, model in models.items():
                model_path = f"models/{symbol}/{name}"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)
                self.logger.info(f"Saved {name} model for {symbol}")

        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise