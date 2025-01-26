from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import logging
from typing import Optional, List


class FeatureSelector:
    """
    Selects the most relevant features based on mutual information.
    """

    def __init__(self, n_features: int = 50):
        self.n_features = n_features
        self.selected_features: Optional[List[str]] = None
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            self.logger.info("Fitting FeatureSelector using mutual information.")
            mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
            mi_series = pd.Series(mi, index=X.columns)
            self.selected_features = mi_series.sort_values(ascending=False).head(self.n_features).index.tolist()
            self.logger.info(f"Selected top {self.n_features} features based on mutual information.")
        except Exception as e:
            self.logger.error(f"Error during feature selection: {e}")
            self.selected_features = []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.selected_features is None:
                self.logger.warning("FeatureSelector has not been fitted yet. Returning original features.")
                return X
            self.logger.info("Transforming features to selected subset.")
            return X[self.selected_features]
        except Exception as e:
            self.logger.error(f"Error during feature transformation: {e}")
            return pd.DataFrame()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
