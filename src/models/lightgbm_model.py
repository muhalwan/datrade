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
        # Default parameters with advanced settings
        self.params = {
            'objective': 'huber',
            'metric': 'rmse',
            'boosting_type': 'dart',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'num_threads': 4,
            'deterministic': True,
            'verbosity': -1
        }
        # Update with provided params
        self.params.update(config.params.get('lgb_params', {}))

        self.model = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
        self.training_time = None

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM model with advanced features"""
        try:
            self.logger.info("Starting LightGBM training...")
            start_time = datetime.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Setup cross-validation folds
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(
                n_splits=self.config.params.get('n_splits', 5),
                gap=self.config.params.get('cv_gap', 0)
            )

            # Create training and validation splits
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # Create datasets
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=self.config.features,
                free_raw_data=False
            )
            valid_sets = []
            if len(X_val) > 0:
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val,
                    reference=train_data,
                    free_raw_data=False
                )
                valid_sets = [val_data]

            # Setup advanced parameters
            self.params.update({
                'objective': 'huber',  # More robust to outliers
                'boosting_type': 'dart',  # Use DART boosting
                'drop_rate': 0.1,
                'max_drop': 50,
                'skip_drop': 0.5,
                'xgboost_dart_mode': True,
                'uniform_drop': True,
                'feature_fraction_bynode': 0.8,
                'pos_bagging_fraction': 0.9,
                'neg_bagging_fraction': 0.9,
                'lambda_l1': 0.1,  # L1 regularization
                'lambda_l2': 0.1,  # L2 regularization
                'path_smooth': 0.1,  # Smoothing
                'extra_trees': True,  # Use extremely randomized trees
                'deterministic': True  # For reproducibility
            })

            # Train model with callbacks
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=50,
                    verbose=True
                ),
                lgb.log_evaluation(100),
                self._create_custom_callback()
            ]

            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=valid_sets,
                num_boost_round=self.config.params.get('num_boost_round', 1000),
                callbacks=callbacks,
                feature_name=self.config.features
            )

            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.config.features,
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split')
            }).sort_values('importance_gain', ascending=False)

            # Store training time
            self.training_time = (datetime.now() - start_time).total_seconds()

            # Compute and log validation metrics
            if valid_sets:
                val_pred = self.model.predict(X_val)
                metrics = self._calculate_metrics(y_val, val_pred)
                self._log_training_results(metrics)

        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

    def _create_custom_callback(self):
        """Create custom callback for training monitoring"""
        def callback(env):
            if env.iteration % 100 == 0:
                self.logger.info(f"\nIteration {env.iteration}")
                if env.evaluation_result_list:
                    for metric in env.evaluation_result_list:
                        self.logger.info(f"{metric[0]}: {metric[2]:.6f}")
        return callback

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        try:
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error,
                r2_score, mean_absolute_percentage_error
            )

            # Basic metrics
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }

            # Directional accuracy
            pred_direction = np.diff(y_pred) > 0
            true_direction = np.diff(y_true) > 0
            metrics['directional_accuracy'] = np.mean(pred_direction == true_direction)

            # Custom metrics
            metrics['bias'] = np.mean(y_pred - y_true)
            metrics['variance'] = np.var(y_pred - y_true)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _log_training_results(self, metrics: Dict):
        """Log comprehensive training results"""
        try:
            self.logger.info("\nTraining Results:")
            self.logger.info(f"Training Time: {self.training_time:.2f} seconds")

            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.6f}")

            self.logger.info("\nTop 10 Features by Importance:")
            for _, row in self.feature_importance.head(10).iterrows():
                self.logger.info(
                    f"{row['feature']}: "
                    f"(gain={row['importance_gain']:.2f}, "
                    f"split={row['importance_split']:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Error logging results: {str(e)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced prediction with uncertainty estimation"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            # Preprocess test data
            X, _ = self.preprocess(df)

            # Make predictions with num_iteration
            predictions = self.model.predict(
                X,
                num_iteration=self.model.best_iteration
            )

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """Enhanced model saving with metadata"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save model
            model_path = f"{path}/model.txt"
            self.model.save_model(model_path)

            # Save feature importance
            importance_path = f"{path}/feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)

            # Save configuration
            config_path = f"{path}/config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'params': self.params,
                    'features': self.config.features,
                    'training_time': self.training_time,
                    'best_iteration': self.model.best_iteration
                }, f, indent=4)

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
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