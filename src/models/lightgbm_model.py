import logging
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime

from .base import BaseModel, ModelConfig

class LightGBMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'num_threads': 4
        }
        # Update with provided params
        self.params.update(config.params.get('lgb_params', {}))

        self.model = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
        self.training_time = None

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data"""
        try:
            feature_data = df[self.config.features].copy()
            feature_data = feature_data.ffill().bfill()
            target_data = df[self.config.target]

            return feature_data.values, target_data.values

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM model"""
        try:
            self.logger.info("Starting LightGBM training...")
            start_time = datetime.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Create training and validation splits
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # Create datasets
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=self.config.features
            )
            valid_sets = []
            if len(X_val) > 0:
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val,
                    reference=train_data
                )
                valid_sets = [val_data]

            # Train model
            evals_result = {}
            callbacks = [
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(100)
            ]

            num_boost_round = 1000
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )

            # Calculate feature importance
            self.feature_importance = pd.Series(
                self.model.feature_importance(),
                index=self.config.features
            ).sort_values(ascending=False)

            # Store training time
            self.training_time = (datetime.now() - start_time).total_seconds()

            # Log results
            self.logger.info(f"\nTraining completed in {self.training_time:.2f} seconds")
            self.logger.info("\nFeature Importance:")
            for feat, imp in self.feature_importance.items():
                self.logger.info(f"{feat}: {imp}")

        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            # Preprocess test data
            X, _ = self.preprocess(df)

            # Make predictions
            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Save model and artifacts"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save model
            model_path = f"{path}/model.txt"
            self.model.save_model(model_path)

            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv(f"{path}/feature_importance.csv")

            # Save metadata
            metadata = {
                'params': self.params,
                'features': self.config.features,
                'training_time': self.training_time
            }
            with open(f"{path}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving LightGBM model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load model and artifacts"""
        try:
            model_path = f"{path}/model.txt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Load model
            self.model = lgb.Booster(model_file=model_path)

            # Load feature importance if available
            importance_path = f"{path}/feature_importance.csv"
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)

            # Load metadata if available
            metadata_path = f"{path}/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.params = metadata['params']
                self.config.features = metadata['features']
                self.training_time = metadata['training_time']

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading LightGBM model: {str(e)}")
            raise

    def get_feature_importance(self, importance_type: str = 'split') -> pd.Series:
        """Get feature importance with different metrics"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            importance = self.model.feature_importance(importance_type=importance_type)
            return pd.Series(
                importance,
                index=self.config.features
            ).sort_values(ascending=False)

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def cross_validate(self, df: pd.DataFrame, num_folds: int = 5) -> Dict:
        """Perform cross-validation"""
        try:
            X, y = self.preprocess(df)

            # Setup CV splitter
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=num_folds)

            # Store results
            cv_scores = {
                'train_score': [],
                'val_score': [],
                'feature_importance': []
            }

            # Perform CV
            for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                # Train model
                model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False
                )

                # Store scores
                cv_scores['train_score'].append(model.best_score['training']['rmse'])
                cv_scores['val_score'].append(model.best_score['valid_1']['rmse'])
                cv_scores['feature_importance'].append(
                    pd.Series(model.feature_importance(), index=self.config.features)
                )

                self.logger.info(f"Fold {fold} - Train RMSE: {cv_scores['train_score'][-1]:.4f}, "
                                 f"Val RMSE: {cv_scores['val_score'][-1]:.4f}")

            # Calculate aggregate metrics
            cv_results = {
                'mean_train_score': np.mean(cv_scores['train_score']),
                'std_train_score': np.std(cv_scores['train_score']),
                'mean_val_score': np.mean(cv_scores['val_score']),
                'std_val_score': np.std(cv_scores['val_score']),
                'feature_importance': pd.concat(cv_scores['feature_importance'], axis=1).mean(axis=1)
            }

            self.logger.info("\nCross-validation results:")
            self.logger.info(f"Mean train RMSE: {cv_results['mean_train_score']:.4f} ± {cv_results['std_train_score']:.4f}")
            self.logger.info(f"Mean val RMSE: {cv_results['mean_val_score']:.4f} ± {cv_results['std_val_score']:.4f}")

            return cv_results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def optimize_hyperparameters(self, df: pd.DataFrame, num_trials: int = 100) -> Dict:
        """Optimize hyperparameters using Optuna"""
        try:
            import optuna

            X, y = self.preprocess(df)
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            def objective(trial):
                param = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                    'max_depth': trial.suggest_int('max_depth', 3, 8)
                }

                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                model = lgb.train(
                    param,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False
                )

                return model.best_score['valid_0']['rmse']

            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=num_trials)

            # Log results
            self.logger.info("\nHyperparameter optimization results:")
            self.logger.info(f"Best RMSE: {study.best_value:.4f}")
            self.logger.info("Best hyperparameters:")
            for key, value in study.best_params.items():
                self.logger.info(f"{key}: {value}")

            return study.best_params

        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise