import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class for all ML models"""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"model.{name}")
        self.model = None
        self.is_trained = False

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model"""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def save(self, path: str):
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("No model to save")

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.model, save_path)
            self.logger.info(f"Model saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path: str):
        """Load model from disk"""
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            self.model = joblib.load(load_path)
            self.is_trained = True
            self.logger.info(f"Model loaded from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise