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
            self.logger.info("Starting feature selection")

            # Final NaN check with fallback
            if X.isnull().any().any():
                self.logger.warning("NaN values detected - using mean imputation")
                X = X.fillna(X.mean())

            # Variance threshold
            variances = X.var()
            valid_features = variances[variances > 1e-8].index.tolist()

            if not valid_features:
                self.logger.error("No valid features after variance threshold")
                self.selected_features = []
                return

            # Mutual information calculation
            mi = mutual_info_classif(X[valid_features], y, random_state=42)
            mi_series = pd.Series(mi, index=valid_features)

            # Select top features with positive MI
            positive_mi = mi_series[mi_series > 0]
            if not positive_mi.empty:
                self.selected_features = positive_mi.nlargest(
                    min(self.n_features, len(positive_mi))
                ).index.tolist()
            else:
                self.logger.warning("No features with positive MI - using top variance features")
                self.selected_features = variances.nlargest(self.n_features).index.tolist()

            self.logger.info(f"Selected {len(self.selected_features)} features")

        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            self.selected_features = X.columns.tolist()[:self.n_features]

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
