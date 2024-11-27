import logging
from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from .base import BaseModel, ModelConfig, ModelMetrics

class LightGBMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Default GPU parameters
        self.params = {
            'objective': 'huber',
            'metric': 'rmse',
            'verbosity': -1,
            'force_row_wise': True,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'max_bin': 255,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.0,
            'max_cat_threshold': 32,
            'num_threads': 4
        }
        # Update with provided params
        self.params.update(config.params)

        self.num_boost_round = config.params.get('num_boost_round', 1000)
        self.early_stopping_rounds = config.params.get('early_stopping_rounds', 50)

        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Initialize feature tracking
        self.feature_names = None
        self.categorical_features = []

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data efficiently"""
        try:
            self.logger.info(f"Preprocessing data of shape: {df.shape}")

            # Verify features
            missing_features = [f for f in self.config.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Select features and target
            feature_data = df[self.config.features].copy()
            target_data = df[self.config.target].copy()

            # Handle missing values
            feature_data = feature_data.ffill().bfill()
            target_data = target_data.ffill().bfill()

            # Scale features if not using GPU histogram
            if self.params.get('device', 'gpu') != 'gpu':
                feature_data = pd.DataFrame(
                    self.feature_scaler.fit_transform(feature_data),
                    columns=feature_data.columns,
                    index=feature_data.index
                )

            # Store feature names
            self.feature_names = feature_data.columns.tolist()

            return feature_data.values, target_data.values

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train(self, df: pd.DataFrame) -> None:
        """Train LightGBM model with enhanced features"""
        try:
            self.logger.info("Starting LightGBM training...")
            start_time = datetime.now()

            # Preprocess data
            X, y = self.preprocess(df)

            # Create validation split
            train_size = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # Create datasets
            train_data = lgb.Dataset(
                X_train, label=y_train,
                feature_name=self.feature_names,
                categorical_feature=self.categorical_features,
                free_raw_data=False
            )

            valid_sets = []
            if len(X_val) > 0:
                val_data = lgb.Dataset(
                    X_val, label=y_val,
                    reference=train_data,
                    feature_name=self.feature_names,
                    categorical_feature=self.categorical_features,
                    free_raw_data=False
                )
                valid_sets = [train_data, val_data]
                valid_names = ['train', 'valid']
            else:
                valid_sets = [train_data]
                valid_names = ['train']

            # Train with callbacks
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=True
                    ),
                    lgb.log_evaluation(period=100),
                    self._create_training_callback()
                ]
            )

            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance_gain': self.model.feature_importance(importance_type='gain'),
                'importance_split': self.model.feature_importance(importance_type='split')
            }).sort_values('importance_gain', ascending=False)

            self.training_time = (datetime.now() - start_time).total_seconds()
            self.last_training = datetime.now()

            # Log results
            self._log_training_results()

        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise

    def _create_training_callback(self):
        def callback(env):
            try:
                if env.iteration % 100 == 0:
                    results = {}
                    for name, value, _, _ in env.evaluation_result_list:
                        results[name] = value

                    self.training_history.append(
                        ModelMetrics(
                            mse=results.get('train', 0.0),
                            rmse=np.sqrt(results.get('train', 0.0)),
                            mae=0.0,  # LightGBM doesn't provide MAE
                            mape=0.0,  # LightGBM doesn't provide MAPE
                            directional_accuracy=0.0,
                            training_time=0.0,  # Can't get training time from callback
                            timestamp=datetime.now()
                        )
                    )
            except Exception as e:
                self.logger.error(f"Error in training callback: {e}")
        return callback

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with proper error handling"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            # Preprocess features
            X, _ = self.preprocess(df)

            # Make predictions
            predictions = self.model.predict(
                X,
                num_iteration=self.model.best_iteration
            )

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def _log_training_results(self):
        """Log comprehensive training results"""
        if not hasattr(self.model, 'best_score'):
            return

        self.logger.info("\nTraining Results:")
        self.logger.info(f"Training Time: {self.training_time:.2f} seconds")

        # Log best scores
        for dataset in self.model.best_score:
            metrics = self.model.best_score[dataset]
            self.logger.info(f"\n{dataset.title()} Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.6f}")

        # Log feature importance
        self.logger.info("\nTop 10 Features by Importance:")
        for _, row in self.feature_importance.head(10).iterrows():
            self.logger.info(
                f"{row['feature']}: "
                f"(gain={row['importance_gain']:.2f}, "
                f"split={row['importance_split']:.2f})"
            )

    def save(self, path: str) -> None:
        """Save model and artifacts"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save base model info
            super().save(path)

            # Save LightGBM model
            model_path = os.path.join(path, 'model.txt')
            self.model.save_model(model_path)

            # Save scalers
            scalers_path = os.path.join(path, 'scalers.joblib')
            joblib.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, scalers_path)

            # Save additional metadata
            metadata = {
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'params': self.params,
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score
            }

            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Load model and artifacts"""
        try:
            # Load base model info
            super().load(path)

            # Check paths
            model_path = os.path.join(path, 'model.txt')
            scalers_path = os.path.join(path, 'scalers.joblib')
            metadata_path = os.path.join(path, 'metadata.json')

            if not all(os.path.exists(p) for p in [model_path, metadata_path]):
                raise FileNotFoundError(f"Model files not found in {path}")

            # Load model
            self.model = lgb.Booster(model_file=model_path)

            # Load scalers if they exist
            if os.path.exists(scalers_path):
                scalers = joblib.load(scalers_path)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                self.categorical_features = metadata['categorical_features']
                self.params = metadata['params']

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def cross_validate(self, df: pd.DataFrame, num_folds: int = 5) -> Dict:
        """Perform time series cross-validation"""
        try:
            X, y = self.preprocess(df)
            cv = TimeSeriesSplit(n_splits=num_folds)

            cv_scores = {
                'train_score': [],
                'val_score': [],
                'feature_importance': []
            }

            for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                train_data = lgb.Dataset(
                    X_train, label=y_train,
                    feature_name=self.feature_names,
                    categorical_feature=self.categorical_features
                )
                val_data = lgb.Dataset(
                    X_val, label=y_val,
                    reference=train_data
                )

                model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False
                )

                cv_scores['train_score'].append(model.best_score['training']['rmse'])
                cv_scores['val_score'].append(model.best_score['valid_1']['rmse'])
                cv_scores['feature_importance'].append(
                    pd.Series(
                        model.feature_importance(importance_type='gain'),
                        index=self.feature_names
                    )
                )

                self.logger.info(
                    f"Fold {fold} - "
                    f"Train RMSE: {cv_scores['train_score'][-1]:.4f}, "
                    f"Val RMSE: {cv_scores['val_score'][-1]:.4f}"
                )

            # Calculate aggregate metrics
            cv_results = {
                'mean_train_score': np.mean(cv_scores['train_score']),
                'std_train_score': np.std(cv_scores['train_score']),
                'mean_val_score': np.mean(cv_scores['val_score']),
                'std_val_score': np.std(cv_scores['val_score']),
                'feature_importance': pd.concat(cv_scores['feature_importance'], axis=1).mean(axis=1)
            }

            self.logger.info(
                f"\nCV Results - "
                f"Mean Train RMSE: {cv_results['mean_train_score']:.4f} "
                f"± {cv_results['std_train_score']:.4f}, "
                f"Mean Val RMSE: {cv_results['mean_val_score']:.4f} "
                f"± {cv_results['std_val_score']:.4f}"
            )

            return cv_results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise