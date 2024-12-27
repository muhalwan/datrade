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
        self.training_history: Dict = {}
        self.model_config: Dict = {}

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

            # Save model
            model_path = f"{path}.model"
            joblib.dump(self.model, model_path)

            # Save metadata
            metadata = {
                'name': self.name,
                'training_history': self.training_history,
                'model_config': self.model_config,
                'timestamp': datetime.now().isoformat()
            }
            metadata_path = f"{path}.meta"
            joblib.dump(metadata, metadata_path)

            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load model from disk"""
        try:
            # Load model
            model_path = f"{path}.model"
            self.model = joblib.load(model_path)

            # Load metadata
            metadata_path = f"{path}.meta"
            metadata = joblib.load(metadata_path)
            self.training_history = metadata.get('training_history', {})
            self.model_config = metadata.get('model_config', {})

            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        return {}
