import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

class FeatureSelector:
    """Feature selection and importance analysis"""

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.selector = None
        self.selected_features = None
        self.feature_importance = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = 'rf'):
        """Fit feature selector"""
        try:
            if method == 'kbest':
                k = self.config.get('k_best', 20)
                self.selector = SelectKBest(score_func=f_regression, k=k)
                self.selector.fit(X, y)

                scores = pd.Series(
                    self.selector.scores_,
                    index=X.columns
                ).sort_values(ascending=False)

                self.selected_features = scores.head(k).index.tolist()
                self.feature_importance = scores

            else:  # Random Forest
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.selector = SelectFromModel(rf, prefit=False)
                self.selector.fit(X, y)

                importance = pd.Series(
                    self.selector.estimator_.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)

                self.selected_features = importance[
                    importance > importance.mean()].index.tolist()
                self.feature_importance = importance

            self.fitted = True

        except Exception as e:
            self.logger.error(f"Feature selection error: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features using fitted selector"""
        if not self.fitted:
            raise ValueError("Feature selector not fitted")

        try:
            if self.selected_features is None:
                return X

            return X[self.selected_features]

        except Exception as e:
            self.logger.error(f"Transform error: {str(e)}")
            return X

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if not self.fitted:
            raise ValueError("Feature selector not fitted")

        return self.feature_importance

    def save(self, path: str):
        """Save fitted selector"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'selector': self.selector,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance
            }

            joblib.dump(save_dict, save_path)
            self.logger.info(f"Feature selector saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving selector: {str(e)}")

    def load(self, path: str):
        """Load fitted selector"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Selector file not found: {load_path}")

            load_dict = joblib.load(load_path)

            self.selector = load_dict['selector']
            self.selected_features = load_dict['selected_features']
            self.feature_importance = load_dict['feature_importance']
            self.fitted = True

            self.logger.info(f"Feature selector loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading selector: {str(e)}")