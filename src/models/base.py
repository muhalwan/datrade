import logging
import pickle
from pathlib import Path
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name: str):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.training_history = {}

    @abstractmethod
    def train(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_feature_importance(self):
        return {}

    def save(self, model_path: Path) -> bool:
        """Serialize model + metadata."""
        try:
            # Asumsi: sub-class punya .models (dict) atau setidaknya .model
            if not hasattr(self, 'models') and not hasattr(self, 'model'):
                raise ValueError("No model to save")

            with open(f"{model_path}.model", "wb") as f:
                pickle.dump(self, f)

            meta_data = {
                'name': self.name,
                'training_history': self.training_history
            }
            with open(f"{model_path}.meta", "wb") as f:
                pickle.dump(meta_data, f)

            self.logger.info(f"Model {self.name} saved successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
