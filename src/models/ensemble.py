from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import logging
from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from ..features.selector import FeatureSelector

class EnhancedEnsemble(BaseModel):
    """Enhanced ensemble model with adaptive features"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("enhanced_ensemble")

        self.config = config or {
            'lstm': {
                'sequence_length': 30,
                'lstm_units': [32, 16],
                'dropout_rate': 0.3,
                'recurrent_dropout': 0.3,
                'learning_rate': 0.001
            },
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1.0
            }
        }

        # Initialize models with corrected parameter passing
        self.models = {
            'lstm': LSTMModel(**self.config['lstm']),
            'xgboost': XGBoostModel(params=self.config['xgboost'])  # Corrected
        }

        # Initialize components
        self.feature_selector = FeatureSelector(method='mutual_info', k='all')
        self.scaler = StandardScaler()

        # Model weights and performance tracking
        self.weights = {'lstm': 0.4, 'xgboost': 0.6}
        self.model_performance = {}
        self.validation_metrics = {}

    def _prepare_data_lstm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        try:
            # Select features
            self.feature_selector.fit(X, y)
            X_selected = self.feature_selector.transform(X)
            feature_importance = self.feature_selector.get_feature_importance()

            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)

            # Create sequences
            sequence_length = self.config['lstm']['sequence_length']
            sequences = []
            targets = []
            for i in range(len(X_scaled) - sequence_length):
                sequences.append(X_scaled[i:i + sequence_length])
                targets.append(y.iloc[i + sequence_length])

            X_seq = np.array(sequences)
            y_seq = np.array(targets)

            return X_seq, y_seq

        except Exception as e:
            self.logger.error(f"Error preparing data for LSTM: {e}")
            return np.array([]), np.array([])

    def _prepare_data_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for XGBoost model"""
        try:
            # Select features (assuming feature_selector is already fitted)
            X_selected = self.feature_selector.transform(X)
            feature_importance = self.feature_selector.get_feature_importance()

            # Scale features
            X_scaled = self.scaler.transform(X_selected)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)

            return X_scaled_df, y

        except Exception as e:
            self.logger.error(f"Error preparing data for XGBoost: {e}")
            return pd.DataFrame(), pd.Series()

    def _validate_model(
            self,
            model_name: str,
            model: BaseModel,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int = 5
    ) -> Dict[str, float]:
        """Perform time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'sharpe': [],
                'sortino': [],
                'max_drawdown': []
            }

            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # Prepare data based on model type
                if model_name == 'lstm':
                    X_train_seq, y_train_seq = self._prepare_data_lstm(X_train, y_train)
                    X_val_seq, y_val_seq = self._prepare_data_lstm(X_val, y_val)
                    if X_train_seq.size == 0 or X_val_seq.size == 0:
                        self.logger.warning(f"Insufficient data for training/validation for {model_name}")
                        continue
                    model.train(X_train_seq, y_train_seq)
                    val_pred = model.predict(X_val_seq)
                elif model_name == 'xgboost':
                    X_train_prepared, y_train_prepared = self._prepare_data_xgboost(X_train, y_train)
                    X_val_prepared, y_val_prepared = self._prepare_data_xgboost(X_val, y_val)
                    if X_train_prepared.empty or X_val_prepared.empty:
                        self.logger.warning(f"Insufficient data for training/validation for {model_name}")
                        continue
                    model.train(X_train_prepared, y_train_prepared)
                    val_pred = model.predict(X_val_prepared)
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue

                # Calculate metrics
                metrics['accuracy'].append(self._calculate_accuracy(y_val, val_pred))
                metrics['precision'].append(self._calculate_precision(y_val, val_pred))
                metrics['recall'].append(self._calculate_recall(y_val, val_pred))
                metrics['f1'].append(self._calculate_f1(y_val, val_pred))
                metrics['sharpe'].append(self._calculate_sharpe(val_pred, X_val['returns']))
                metrics['sortino'].append(self._calculate_sortino(val_pred, X_val['returns']))
                metrics['max_drawdown'].append(self._calculate_max_drawdown(val_pred, X_val['returns']))

            # Average metrics
            return {k: np.mean(v) for k, v in metrics.items()}

        except Exception as e:
            self.logger.error(f"Error validating {model_name}: {e}")
            return {}

    def _adjust_weights(self, performance: Dict[str, Dict[str, float]]) -> None:
        """Adjust model weights based on performance"""
        try:
            scores = {}
            for model_name, metrics in performance.items():
                # Calculate composite score
                score = (
                        metrics.get('accuracy', 0) * 0.3 +
                        metrics.get('sharpe', 0) * 0.3 +
                        metrics.get('f1', 0) * 0.2 +
                        (1 - metrics.get('max_drawdown', 0)) * 0.2
                )
                scores[model_name] = max(0, score)

            # Normalize weights
            total_score = sum(scores.values())
            if total_score > 0:
                self.weights = {
                    name: score / total_score
                    for name, score in scores.items()
                }

        except Exception as e:
            self.logger.error(f"Error adjusting weights: {e}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train ensemble with validation"""
        try:
            self.logger.info("Starting ensemble training...")

            # Validate that 'returns' column exists if needed for metrics
            if 'returns' not in X.columns:
                self.logger.error("'returns' column is missing in the feature set.")
                return

            # Validate and train models
            performance = {}
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")

                # Validate model
                metrics = self._validate_model(name, model, X, y)
                performance[name] = metrics

            # Adjust weights
            self._adjust_weights(performance)

            # Store metadata
            self.model_performance = performance

            self.logger.info("Ensemble training completed")

        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with confidence scores"""
        try:
            # Ensure 'returns' column exists if needed
            if 'returns' not in X.columns:
                self.logger.error("'returns' column is missing in the feature set.")
                return np.zeros(len(X))

            # Select features
            X_selected = self.feature_selector.transform(X)

            # Scale features
            X_scaled = self.scaler.transform(X_selected)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)

            predictions = {}
            working_models = 0

            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    if name == 'lstm':
                        # Prepare sequences for LSTM
                        sequence_length = self.config['lstm']['sequence_length']
                        sequences = []
                        for i in range(len(X_scaled_df) - sequence_length):
                            sequences.append(X_scaled_df.iloc[i:i + sequence_length].values)
                        if not sequences:
                            self.logger.warning(f"Insufficient data for LSTM predictions.")
                            continue
                        X_seq = np.array(sequences)
                        pred = model.predict(X_seq)
                        # Align predictions with original data
                        pred_full = np.zeros(len(X))
                        pred_full[sequence_length:] = pred
                        predictions[name] = pred_full
                    elif name == 'xgboost':
                        pred = model.predict(X_scaled_df)
                        predictions[name] = pred
                    else:
                        self.logger.warning(f"Unknown model type: {name}")
                        continue
                    working_models += 1
                except Exception as e:
                    self.logger.warning(f"Model {name} failed to predict: {e}")

            if working_models == 0:
                return np.zeros(len(X))

            # Calculate weighted predictions
            weighted_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                weighted_pred += pred * self.weights.get(name, 0)

            return weighted_pred

        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance"""
        try:
            importance = {}
            for name, model in self.models.items():
                model_importance = model.get_feature_importance()
                for feature, score in model_importance.items():
                    importance[feature] = importance.get(feature, 0) + score * self.weights.get(name, 0)

            # Normalize importance
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            return importance

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}

    # Placeholder methods for metric calculations
    def _calculate_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    def _calculate_precision(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive > 0 else 0.0

    def _calculate_recall(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0.0

    def _calculate_f1(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _calculate_sharpe(self, predictions: np.ndarray, returns: pd.Series) -> float:
        # Placeholder implementation
        return 0.0

    def _calculate_sortino(self, predictions: np.ndarray, returns: pd.Series) -> float:
        # Placeholder implementation
        return 0.0

    def _calculate_max_drawdown(self, predictions: np.ndarray, returns: pd.Series) -> float:
        # Placeholder implementation
        return 0.0
