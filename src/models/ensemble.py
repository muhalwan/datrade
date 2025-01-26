import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from ..features.selector import FeatureSelector


class EnsembleModel:
    def __init__(self, config: Dict):
        self.price_data = None
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Remove n_features from LSTM config before initialization
        lstm_config = self.config['lstm'].copy()
        lstm_config.pop('n_features', None)

        self.models: Dict[str, BaseModel] = {
            'lstm': LSTMModel(**lstm_config),
            'xgboost': XGBoostModel(params=self.config['xgboost']['params'])
        }
        self.feature_selector = None
        self.scaler = None

    def train(self, X: pd.DataFrame, y: pd.Series, class_weights: Optional[Dict[int, int]] = None):
        """
        Trains the ensemble model using time-series cross-validation.
        """
        try:
            if len(X) != len(y):
                self.logger.error("Feature/target length mismatch before training")
                return

            # Trim features to match target length
            X_trimmed = X.iloc[:len(y)]

            self.logger.info("Starting ensemble training...")

            # Step 1: Remove constant features
            variances = X_trimmed.var()
            non_constant_mask = variances > 1e-8
            X_filtered = X_trimmed.loc[:, non_constant_mask]
            if X_filtered.empty:
                self.logger.error("All features have zero variance after filtering. Aborting training.")
                return
            self.logger.info(f"Removed {len(X_trimmed.columns)-len(X_filtered.columns)} constant features.")

            # Step 2: Feature scaling
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_filtered)
            X_scaled_df = pd.DataFrame(X_scaled, index=X_filtered.index, columns=X_filtered.columns)

            # Step 3: Feature selection
            self.feature_selector = FeatureSelector(n_features=self.config['feature_selector']['n_features'])
            X_selected = self.feature_selector.fit_transform(X_scaled_df, y)

            if X_selected.empty or X_selected.shape[1] == 0:
                self.logger.error("No features selected after feature selection. Aborting training.")
                return

            # Update LSTM with actual feature count
            self.models['lstm'].n_features = X_selected.shape[1]

            # Time-series cross-validation
            self._time_series_cv(X_selected.values, y.values)

            # Train on full data
            self.logger.info("Training XGBoost on full data...")
            self.models['xgboost'].train(X_selected, y, class_weights=class_weights)

            self.logger.info("Training LSTM on full data...")
            self._train_lstm_full_data(X_selected.values, y.values, class_weights)

            self.logger.info("Ensemble training completed")
        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")

    def _time_series_cv(self, X: np.ndarray, y: np.ndarray):
        """Performs time-series cross-validation with proper sequence handling."""
        try:
            n_splits = self.config['cross_validation']['n_splits']
            self.logger.info(f"Performing time-series cross-validation with {n_splits} splits...")
            split_size = int(len(X) / (n_splits + 1))

            for i in range(n_splits):
                self.logger.info(f"Processing split {i+1}/{n_splits}")
                X_train_cv = X[:(i+1)*split_size]
                y_train_cv = y[:(i+1)*split_size]
                X_val_cv = X[(i+1)*split_size:(i+2)*split_size]
                y_val_cv = y[(i+1)*split_size:(i+2)*split_size]

                # Skip if no features
                if X_train_cv.shape[1] == 0:
                    self.logger.warning("Skipping split due to no features")
                    continue

                # Train XGBoost
                temp_xgb = XGBoostModel(params=self.config['xgboost']['params'])
                temp_xgb.train(pd.DataFrame(X_train_cv, columns=self.feature_selector.selected_features), y_train_cv)

                # Train LSTM
                temp_lstm = LSTMModel(
                    sequence_length=self.config['lstm']['sequence_length'],
                    lstm_units=self.config['lstm']['lstm_units'],
                    dropout_rate=self.config['lstm']['dropout_rate'],
                    recurrent_dropout=self.config['lstm']['recurrent_dropout'],
                    learning_rate=self.config['lstm']['learning_rate'],
                    batch_size=self.config['lstm']['batch_size']
                )
                temp_lstm.n_features = X_train_cv.shape[1]  # Dynamic feature count
                X_train_seq, y_train_seq = self._make_lstm_sequence(X_train_cv, y_train_cv)
                temp_lstm.train(X_train_seq, y_train_seq)

            self.logger.info("Time-series CV done.")
        except Exception as e:
            self.logger.error(f"Error during time-series cross-validation: {e}")

    def _train_lstm_full_data(self, X: np.ndarray, y: np.ndarray, class_weights: Optional[Dict[int, int]] = None):
        try:
            self.logger.info("Preparing sequences for LSTM training...")
            X_seq, y_seq = self._make_lstm_sequence(X, y)
            # Remove class_weights parameter
            self.models['lstm'].train(X_seq, y_seq)
        except Exception as e:
            self.logger.error(f"Error training LSTM on full data: {e}")

    def _make_lstm_sequence(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences for LSTM training.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequences and corresponding targets.
        """
        seq_len = self.config['lstm']['sequence_length']
        X_seq = []
        y_seq = []
        # Start from index 0 to maintain alignment with XGBoost
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len-1])  # Predict next value after sequence
        return np.array(X_seq), np.array(y_seq)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates ensemble predictions by combining LSTM and XGBoost predictions.
        """
        try:
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(
                pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            )

            # LSTM Predictions with sequence alignment
            X_seq = self._create_lstm_sequences(X_selected.values)
            if X_seq.size > 0:
                lstm_preds = self.models['lstm'].predict(X_seq)
                # Account for sequence lookback
                lstm_preds = lstm_preds[self.config['lstm']['sequence_length']-1:]
            else:
                lstm_preds = np.zeros(len(X))

            # XGBoost Predictions
            xgb_preds = self.models['xgboost'].predict(
                pd.DataFrame(X_selected, columns=self.feature_selector.selected_features)
            )

            # Align predictions
            lstm_start = self.config['lstm']['sequence_length'] - 1
            valid_length = len(X) - lstm_start
            lstm_preds = lstm_preds[:valid_length]
            xgb_preds = xgb_preds[lstm_start:]

            min_length = min(len(lstm_preds), len(xgb_preds))
            lstm_preds = lstm_preds[:min_length]
            xgb_preds = xgb_preds[:min_length]

            # Ensemble weighting
            weights = self.config['ensemble']['weights']
            ensemble_pred = lstm_preds * weights['lstm'] + xgb_preds * weights['xgboost']

            return (ensemble_pred > 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return np.zeros(len(X))

    def _create_lstm_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Creates LSTM sequences from the selected features.

        Args:
            X (np.ndarray): Selected and scaled features.

        Returns:
            np.ndarray: LSTM sequences.
        """
        try:
            seq_len = self.config['lstm']['sequence_length']
            if len(X) < seq_len:
                return np.array([])
            X_seq = []
            for i in range(len(X) - seq_len + 1):
                X_seq.append(X[i:i+seq_len])
            return np.array(X_seq)
        except Exception as e:
            self.logger.error(f"Error creating LSTM sequences: {e}")
            return np.array([])

    def save(self, path: str):
        """
        Saves the ensemble model components.

        Args:
            path (str): Base file path to save the models.
        """
        try:
            for model_name, model in self.models.items():
                model.save(f"{path}_{model_name}.pkl")
            self.logger.info("Ensemble model saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Aggregates feature importance from all models.

        Returns:
            Dict[str, float]: Aggregated feature importances.
        """
        try:
            importance = {}
            for model_name, model in self.models.items():
                fi = model.get_feature_importance()
                for feature, score in fi.items():
                    importance[feature] = importance.get(feature, 0) + score
            # Normalize importance
            total = sum(importance.values())
            for feature in importance:
                importance[feature] /= total
            return importance
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
