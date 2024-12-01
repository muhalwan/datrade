import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from optuna.trial import Trial
from sklearn.metrics import mean_squared_error, mean_absolute_error
import threading
import queue

class ModelOptimizer:
    """Optimize ML model hyperparameters"""

    def __init__(self, model_trainer, feature_pipeline):
        self.logger = logging.getLogger(__name__)
        self.model_trainer = model_trainer
        self.feature_pipeline = feature_pipeline

        # Optimization results storage
        self.results = {}
        self.best_params = {}

    def optimize_model(self, model_name: str, n_trials: int = 100) -> Dict:
        """Run hyperparameter optimization for specified model"""
        try:
            # Create study
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{model_name}_optimization"
            )

            # Get training data
            data = self._get_training_data()

            # Define objective
            def objective(trial: Trial) -> float:
                params = self._get_model_params(trial, model_name)
                return self._evaluate_model(model_name, params, data)

            # Run optimization
            study.optimize(objective, n_trials=n_trials)

            # Store results
            self.best_params[model_name] = study.best_params
            self.results[model_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'trials': study.trials_dataframe()
            }

            return study.best_params

        except Exception as e:
            self.logger.error(f"Optimization error for {model_name}: {str(e)}")
            raise

    def _get_training_data(self) -> pd.DataFrame:
        """Get data for optimization"""
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(days=30)
        return self.feature_pipeline.get_historical_features(start_time, end_time)

    def _get_model_params(self, trial: Trial, model_name: str) -> Dict:
        """Get parameter space for model"""
        if model_name == 'lstm':
            return {
                'layers': [
                    trial.suggest_int(f'lstm_units_{i}', 16, 256)
                    for i in range(trial.suggest_int('n_layers', 1, 3))
                ],
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'sequence_length': trial.suggest_int('sequence_length', 30, 100)
            }
        elif model_name == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        elif model_name == 'prophet':
            return {
                'changepoint_prior_scale': trial.suggest_loguniform(
                    'changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_loguniform(
                    'seasonality_prior_scale', 0.01, 10),
                'seasonality_mode': trial.suggest_categorical(
                    'seasonality_mode', ['multiplicative', 'additive'])
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _evaluate_model(self, model_name: str, params: Dict,
                        data: pd.DataFrame) -> float:
        """Evaluate model with given parameters"""
        try:
            # Update model config
            model = self.model_trainer.models[model_name]
            model.config.update(params)

            # Train and evaluate
            if model_name == 'prophet':
                train_data, val_data = model.preprocess(data)
                metrics = model.train(train_data, val_data)
                error = metrics.get('rmse', float('inf'))
            else:
                X, y = model.preprocess(data)
                metrics = model.train(X, y)
                error = metrics.get('val_loss', float('inf'))

            return error

        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            return float('inf')

    def get_optimization_results(self, model_name: str) -> pd.DataFrame:
        """Get optimization results for model"""
        return self.results.get(model_name, {}).get('trials', pd.DataFrame())

    def apply_best_parameters(self, model_name: str):
        """Apply best parameters to model"""
        try:
            if model_name not in self.best_params:
                raise ValueError(f"No optimization results for {model_name}")

            self.model_trainer.models[model_name].config.update(
                self.best_params[model_name])

        except Exception as e:
            self.logger.error(f"Parameter application error: {str(e)}")