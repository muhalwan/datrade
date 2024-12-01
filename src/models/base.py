from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import joblib
import os

class BaseModel(ABC):
    """Base class for all trading models"""

    def __init__(self, name: str):
        self.name = name
        self.model: Any = None
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    def save(self, path: str) -> bool:
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load model from disk"""
        try:
            self.model = joblib.load(path)
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False