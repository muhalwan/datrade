from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import joblib
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelType(Enum):
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"

@dataclass
class ModelConfig:
    name: str
    type: ModelType
    params: Dict
    features: List[str]
    target: str
    enabled: bool = True

class BaseModel:
    """Base class for all models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model"""
        raise NotImplementedError

    def train(self, df: pd.DataFrame) -> None:
        """Train the model"""
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model to disk"""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load model from disk"""
        raise NotImplementedError

class LSTMModel(BaseModel):
    """LSTM model for time series prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sequence_length = config.params.get('sequence_length', 60)
        self.batch_size = config.params.get('batch_size', 32)
        self.epochs = config.params.get('epochs', 50)
        self.validation_split = config.params.get('validation_split', 0.2)
        self.scaler = None

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM"""
        # Select features and scale
        features = df[self.config.features].values

        # Initialize and fit scaler on first call
        if self.scaler is None:
            self.scaler = tf.keras.preprocessing.MinMaxScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # Create sequences
        X, y = self._create_sequences(features_scaled)
        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model"""
        try:
            # Preprocess data
            X, y = self.preprocess(df)

            # Build model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.config.features))),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32),
                Dense(1)
            ])

            # Compile and train
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=1
            )

            self.model = model

        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            X, _ = self.preprocess(df)
            predictions_scaled = self.model.predict(X)

            # Inverse transform predictions
            predictions = np.zeros((len(predictions_scaled), len(self.config.features)))
            predictions[:, 0] = predictions_scaled.flatten()
            predictions = self.scaler.inverse_transform(predictions)[:, 0]

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save LSTM model"""
        try:
            model_path = f"{path}/lstm_model"
            scaler_path = f"{path}/lstm_scaler.pkl"

            save_model(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load LSTM model"""
        try:
            model_path = f"{path}/lstm_model"
            scaler_path = f"{path}/lstm_scaler.pkl"

            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            raise

class LightGBMModel(BaseModel):
    """LightGBM model for prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.params = config.params.get('lgb_params', {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        })

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LightGBM"""
        X = df[self.config.features].values
        y = df[self.config.target].values
        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM model"""
        try:
            X, y = self.preprocess(df)

            # Create training dataset
            train_data = lgb.Dataset(X, label=y)

            # Train model
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                verbose_eval=10
            )

            self.model = model

        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            X = df[self.config.features].values
            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save LightGBM model"""
        try:
            model_path = f"{path}/lgb_model.txt"
            self.model.save_model(model_path)

        except Exception as e:
            self.logger.error(f"Error saving LightGBM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load LightGBM model"""
        try:
            model_path = f"{path}/lgb_model.txt"
            self.model = lgb.Booster(model_file=model_path)

        except Exception as e:
            self.logger.error(f"Error loading LightGBM model: {str(e)}")
            raise

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models = {}
        self.weights = config.params.get('weights', {})

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight

    def train(self, df: pd.DataFrame) -> None:
        """Train all models in the ensemble"""
        try:
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model...")
                model.train(df)

        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with ensemble"""
        try:
            if not self.models:
                raise ValueError("No models in ensemble")

            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(df)

            # Normalize weights
            total_weight = sum(self.weights.values())
            normalized_weights = {
                name: weight/total_weight
                for name, weight in self.weights.items()
            }

            # Calculate weighted average
            weighted_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
            for name, pred in predictions.items():
                weighted_pred += pred * normalized_weights[name]

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save ensemble model"""
        try:
            # Save individual models
            for name, model in self.models.items():
                model_path = f"{path}/{name}"
                model.save(model_path)

            # Save weights
            weights_path = f"{path}/ensemble_weights.pkl"
            joblib.dump(self.weights, weights_path)

        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load ensemble model"""
        try:
            # Load individual models
            for name, model in self.models.items():
                model_path = f"{path}/{name}"
                model.load(model_path)

            # Load weights
            weights_path = f"{path}/ensemble_weights.pkl"
            self.weights = joblib.dump(weights_path)

        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            raise

class ModelTrainer:
    """Coordinates model training and evaluation"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection

        # Import the FeatureEngineering class
        from features.engineering import FeatureEngineering
        self.feature_eng = FeatureEngineering(db_connection)

        # Default model configurations
        self.default_configs = {
            'lstm': ModelConfig(
                name='lstm',
                type=ModelType.LSTM,
                params={
                    'sequence_length': 60,
                    'batch_size': 32,
                    'epochs': 50,
                    'validation_split': 0.2
                },
                features=['close', 'volume', 'rsi', 'macd', 'bb_high', 'bb_low'],
                target='close'
            ),
            'lightgbm': ModelConfig(
                name='lightgbm',
                type=ModelType.LIGHTGBM,
                params={
                    'lgb_params': {
                        'objective': 'regression',
                        'metric': 'mse',
                        'num_leaves': 31,
                        'learning_rate': 0.05,
                        'feature_fraction': 0.9
                    }
                },
                features=['close', 'volume', 'rsi', 'macd', 'bb_high', 'bb_low'],
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

    def train_models(self, symbol: str, start_time: datetime,
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

                # Calculate R-squared
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                ss_res = np.sum((actual - predictions) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'directional_accuracy': directional_accuracy
                }

                self.logger.info(f"\nResults for {name}:")
                for metric, value in results[name].items():
                    self.logger.info(f"{metric.upper()}: {value:.4f}")

            return results

        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            raise

    def cross_validate(self, symbol: str, start_time: datetime, end_time: datetime,
                       n_splits: int = 5) -> Dict[str, Dict]:
        """Perform time series cross validation"""
        try:
            # Generate full dataset
            features_df = self.feature_eng.generate_features(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

            if features_df.empty:
                raise ValueError("No features generated for cross validation")

            # Calculate fold size
            total_periods = len(features_df)
            fold_size = total_periods // n_splits

            cv_results = {}

            # Perform walk-forward validation
            for fold in range(n_splits - 1):
                self.logger.info(f"\nFold {fold + 1}/{n_splits - 1}")

                # Calculate split indices
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = test_start + fold_size

                # Split data
                train_data = features_df.iloc[:train_end]
                test_data = features_df.iloc[test_start:test_end]

                # Train models
                trained_models = self.train_models(
                    symbol=symbol,
                    start_time=train_data.index[0],
                    end_time=train_data.index[-1]
                )

                # Evaluate models
                fold_results = self.evaluate_models(trained_models, test_data)

                # Store results
                for model_name, metrics in fold_results.items():
                    if model_name not in cv_results:
                        cv_results[model_name] = {
                            metric: [] for metric in metrics.keys()
                        }
                    for metric, value in metrics.items():
                        cv_results[model_name][metric].append(value)

            # Calculate mean and std of metrics across folds
            summary = {}
            for model_name, metrics in cv_results.items():
                summary[model_name] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                    for metric, values in metrics.items()
                }

                self.logger.info(f"\nCross-validation results for {model_name}:")
                for metric, stats in summary[model_name].items():
                    self.logger.info(
                        f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}"
                    )

            return summary

        except Exception as e:
            self.logger.error(f"Error in cross validation: {str(e)}")
            raise