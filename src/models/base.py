from abc import ABC, abstractmethod
from typing import Optional, Dict
import logging
import pickle

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def train(self, X, y):
        """
        Trains the model.

        Args:
            X: Features.
            y: Targets.
        """
        pass

    @abstractmethod
    def predict(self, X) -> Optional[np.ndarray]:
        """
        Makes predictions using the trained model.

        Args:
            X: Features.

        Returns:
            Optional[np.ndarray]: Prediction probabilities or classes.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retrieves feature importance scores.

        Returns:
            Dict[str, float]: Feature importances.
        """
        pass

    def save(self, path: str):
        """
        Saves the trained model to the specified path.

        Args:
            path (str): File path to save the model.
        """
        try:
            if self.model:
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                self.logger.info(f"Model saved to {path}")
            else:
                self.logger.warning("No model to save.")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def load(self, path: str):
        """
        Loads a trained model from the specified path.

        Args:
            path (str): File path to load the model from.
        """
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
