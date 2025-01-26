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
    """
    Combines multiple models (LSTM and XGBoost) into an enhanced ensemble.
    """

    def __init__(self, config: Dict):
        self.price_data = None
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.models: Dict[str, BaseModel] = {
            'lstm': LSTMModel(**self.config['lstm']),
            'xgboost': XGBoostModel(params=self.config['xgboost']['params'])
        }
        self.feature_selector = None
        self.scaler = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the ensemble model using time-series cross-validation.
        """
        try:
            if len(X) != len(y):
                self.logger.error("Feature/target length mismatch before training")
                return

            # Trim features to match target length after sequence creation
            X_trimmed = X.iloc[:len(y)]

            self.logger.info("Starting ensemble training...")
            # Feature selection and scaling
            self.logger.info("Fitting scaler and feature selector on training data...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.feature_selector = FeatureSelector(n_features=self.config['feature_selector']['n_features'])
            X_selected = self.feature_selector.fit_transform(pd.DataFrame(X_scaled, index=X.index, columns=X.columns), y)

            # Time-series cross-validation
            self._time_series_cv(X_selected.values, y.values)

            # Train on full data
            self.logger.info("Training XGBoost on full data...")
            self.models['xgboost'].train(pd.DataFrame(X_selected, columns=self.feature_selector.selected_features), y)

            self.logger.info("Training LSTM on full data...")
            self._train_lstm_full_data(X_selected.values, y.values)

            self.logger.info("Ensemble training completed")
        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")

    def _time_series_cv(self, X: np.ndarray, y: np.ndarray):
        """
        Performs optional time-series cross-validation.

        Args:
            X (np.ndarray): Scaled and selected features.
            y (np.ndarray): Target labels.
        """
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

                # Train XGBoost on CV split
                temp_xgb = XGBoostModel(params=self.config['xgboost']['params'])
                temp_xgb.train(pd.DataFrame(X_train_cv, columns=self.feature_selector.selected_features), y_train_cv)
                val_pred_xgb = temp_xgb.predict(pd.DataFrame(X_val_cv, columns=self.feature_selector.selected_features))

                # Train LSTM on CV split
                temp_lstm = LSTMModel(**self.config['lstm'])
                X_train_seq, y_train_seq = self._make_lstm_sequence(X_train_cv, y_train_cv)
                temp_lstm.train(X_train_seq, y_train_seq)
                if len(X_val_cv) < self.config['lstm']['sequence_length']:
                    val_pred_lstm = np.zeros(len(X_val_cv))
                else:
                    X_val_seq, y_val_seq = self._make_lstm_sequence(X_val_cv, y_val_cv)
                    val_pred_lstm = temp_lstm.predict(X_val_seq)

            self.logger.info("Time-series CV done.")
        except Exception as e:
            self.logger.error(f"Error during time-series cross-validation: {e}")

    def _train_lstm_full_data(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the LSTM model on the full dataset.

        Args:
            X (np.ndarray): Scaled and selected features.
            y (np.ndarray): Target labels.
        """
        try:
            self.logger.info("Preparing sequences for LSTM training...")
            X_seq, y_seq = self._make_lstm_sequence(X, y)
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

        Args:
            X (pd.DataFrame): Feature DataFrame.

        Returns:
            np.ndarray: Binary ensemble predictions.
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
            min_length = min(len(lstm_preds), len(xgb_preds))
            lstm_start = self.config['lstm']['sequence_length'] - 1
            lstm_preds_padded = np.zeros(len(X))
            lstm_preds_padded[lstm_start:lstm_start + len(lstm_preds)] = lstm_preds.flatten()

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
            self.logger.error(f"Error aggregating feature importance: {e}")
            return {}
