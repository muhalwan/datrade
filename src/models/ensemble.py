from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .lstm import LSTMModel
from .xgboost_model import XGBoostModel
from ..features.selector import FeatureSelector


class EnhancedEnsemble(BaseModel):
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

        self.models = {
            'lstm': LSTMModel(**self.config['lstm']),
            'xgboost': XGBoostModel(params=self.config['xgboost'])
        }

        self.feature_selector = FeatureSelector(method='mutual_info', k='all')
        self.scaler = StandardScaler()
        self.weights = {'lstm': 0.4, 'xgboost': 0.6}

        self.logger = logging.getLogger(__name__)
        self.trained_ = False


    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train ensemble model pada full dataset.
        """
        try:
            self.logger.info("Starting ensemble training...")

            # 1) Feature Selection
            self.feature_selector.fit(X, y)
            X_selected = self.feature_selector.transform(X)

            # Hapus kolom duplikat jika ada
            if len(X_selected.columns) != len(set(X_selected.columns)):
                self.logger.warning("Duplicate feature names found. Dropping duplicates.")
                X_selected = X_selected.loc[:, ~X_selected.columns.duplicated()].copy()

            # 2) Scaling
            self.scaler.fit(X_selected)
            X_scaled_array = self.scaler.transform(X_selected)
            X_scaled = pd.DataFrame(X_scaled_array, columns=X_selected.columns, index=X_selected.index)

            # 3) (Opsional) TimeSeries Cross-Validation untuk menilai kinerja
            self._time_series_cv(X_scaled, y)

            # 4) Final train sub-model pada full dataset
            X_seq, y_seq = self._make_lstm_sequence(X_scaled, y)
            self.logger.info("Training LSTM on full data...")
            self.models['lstm'].train(X_seq, y_seq)

            self.logger.info("Training XGBoost on full data...")
            self.models['xgboost'].train(X_scaled, y)

            self.trained_ = True
            self.logger.info("Ensemble training completed")

            return self

        except Exception as e:
            self.logger.error(f"Error in ensemble training: {e}")
            raise

    def _time_series_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        self.logger.info("Performing optional time-series cross-validation...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]

            # LSTM
            X_train_seq, y_train_seq = self._make_lstm_sequence(X_train_cv, y_train_cv)
            X_val_seq, y_val_seq = self._make_lstm_sequence(X_val_cv, y_val_cv)
            if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                self.logger.warning(f"Skipping fold {fold} due to insufficient data for LSTM.")
                continue
            # Train + predict (sementara)
            temp_lstm = LSTMModel(**self.config['lstm'])
            temp_lstm.train(X_train_seq, y_train_seq)
            val_pred_lstm = temp_lstm.predict(X_val_seq)

            # XGBoost
            # Perlu model XGBoost sementara
            temp_xgb = XGBoostModel(params=self.config['xgboost'])
            # Pastikan tidak ada duplikat
            if len(X_train_cv.columns) != len(set(X_train_cv.columns)):
                X_train_cv = X_train_cv.loc[:, ~X_train_cv.columns.duplicated()].copy()
            if len(X_val_cv.columns) != len(set(X_val_cv.columns)):
                X_val_cv = X_val_cv.loc[:, ~X_val_cv.columns.duplicated()].copy()

            temp_xgb.train(X_train_cv, y_train_cv)
            val_pred_xgb = temp_xgb.predict(X_val_cv)


        self.logger.info("Time-series CV done.")

    # -------------------------------------------------------------------------
    # Bagian Prediksi
    # -------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.trained_:
            self.logger.warning("Ensemble model has not been trained. Returning zeros.")
            return np.zeros(len(X))

        try:
            # 1) Transform fitur
            X_sel = self.feature_selector.transform(X)

            # Pastikan tidak ada kolom duplikat
            if len(X_sel.columns) != len(set(X_sel.columns)):
                X_sel = X_sel.loc[:, ~X_sel.columns.duplicated()].copy()

            X_scl_array = self.scaler.transform(X_sel)
            X_scl = pd.DataFrame(X_scl_array, columns=X_sel.columns, index=X_sel.index)

            # 2) Prediksi sub-model
            # LSTM
            pred_lstm = self._predict_lstm(X_scl)

            # XGBoost
            pred_xgb = self._predict_xgb(X_scl)

            # 3) Kalkulasi weighting
            if pred_lstm is None and pred_xgb is None:
                return np.zeros(len(X_scl))
            elif pred_lstm is None:
                return pred_xgb
            elif pred_xgb is None:
                return pred_lstm
            else:
                ensemble_pred = pred_lstm * self.weights['lstm'] + pred_xgb * self.weights['xgboost']
                # Biner threshold 0.5
                return (ensemble_pred > 0.5).astype(int)

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return np.zeros(len(X))

    def _predict_lstm(self, X_scl: pd.DataFrame) -> Optional[np.ndarray]:
        if self.models['lstm'].model is None:
            self.logger.warning("LSTM model not trained.")
            return None

        seq_length = self.config['lstm']['sequence_length']
        X_arr = X_scl.values
        sequences = []
        for i in range(len(X_arr) - seq_length):
            sequences.append(X_arr[i : i + seq_length])

        if not sequences:
            self.logger.warning("No enough data for LSTM sequences in predict().")
            return None

        X_seq = np.array(sequences)
        pred_seq = self.models['lstm'].predict(X_seq)  # float [0..1]
        # Selaraskan panjang
        pred_full = np.zeros(len(X_scl))
        pred_full[seq_length:] = pred_seq
        return pred_full

    def _predict_xgb(self, X_scl: pd.DataFrame) -> Optional[np.ndarray]:
        if self.models['xgboost'].model is None:
            self.logger.warning("XGBoost model not trained.")
            return None
        pred = self.models['xgboost'].predict(X_scl)  # biner 0/1
        return pred

    def _make_lstm_sequence(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.config['lstm']['sequence_length']
        X_arr = X.values
        sequences = []
        targets = []
        for i in range(len(X_arr) - seq_len):
            sequences.append(X_arr[i : i + seq_len])
            targets.append(y.iloc[i + seq_len])
        return np.array(sequences), np.array(targets)
