import numpy as np
import pandas as pd
from typing import List, Dict
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.exceptions import NotFittedError


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'mutual_info', k: int = 'all'):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            if self.method == 'mutual_info':
                score_func = mutual_info_classif
            else:
                raise ValueError(f"Unknown selection method: {self.method}")

            self.selector = SelectKBest(score_func=score_func, k=self.k)
            self.selector.fit(X, y)

            scores = self.selector.scores_
            self.feature_importance = dict(zip(X.columns, scores))
            mean_score = np.mean(scores)
            mask = scores > mean_score

            self.selected_features = X.columns[mask].tolist()

            # Pastikan kolom 'returns' jika ada, tetap dipertahankan
            if 'returns' in X.columns and 'returns' not in self.selected_features:
                self.selected_features.append('returns')

            # Buang duplikasi
            self.selected_features = list(dict.fromkeys(self.selected_features))
            self.logger.info(f"Selected features: {self.selected_features}")
            return self
        except Exception as e:
            self.logger.error(f"Error in fitting feature selector: {e}")
            raise e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.selector is None:
                raise NotFittedError("FeatureSelector instance is not fitted yet.")
            needed = [f for f in self.selected_features if f in X.columns]
            return X[needed]
        except Exception as e:
            self.logger.error(f"Error in transforming data: {e}")
            raise e

    def get_feature_importance(self) -> Dict[str, float]:
        return self.feature_importance

    def get_selected_features(self) -> List[str]:
        return self.selected_features