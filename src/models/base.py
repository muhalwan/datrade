from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    type: ModelType
    params: Dict
    features: List[str]
    target: str = 'close'
    enabled: bool = True

class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config: ModelConfig):
        """Initialize model with configuration"""
        self.config = config
        self.model = None

    @abstractmethod
    def preprocess(self, df: pd.DataFrame):
        """Preprocess data for model"""
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass