import pandas as pd
import numpy as np
from typing import List
import logging
from sklearn.feature_selection import mutual_info_classif

class FeatureSelector:
    """
    Selects the most relevant features based on mutual information.
    """

    def __init__(self, n_features: int = 50):
        self.logger = logging.getLogger(__name__)
        self.n_features = n_features
        self.selected_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the feature selector to the data.
        
        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target labels.
        """
        try:
            self.logger.info("Fitting feature selector using mutual information...")
            mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
            mi_series = pd.Series(mi, index=X.columns)
            mi_series = mi_series.sort_values(ascending=False)
            self.selected_features = mi_series.head(self.n_features).index.tolist()
            self.logger.info(f"Selected features: {self.selected_features}")
        except Exception as e:
            self.logger.error(f"Error fitting feature selector: {e}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by selecting the top features.
        
        Args:
            X (pd.DataFrame): Feature DataFrame.
        
        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        try:
            if not self.selected_features:
                self.logger.warning("No features selected. Returning original DataFrame.")
                return X
            return X[self.selected_features]
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits the feature selector and transforms the data.
        
        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target labels.
        
        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
