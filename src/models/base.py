from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import json
import os
import logging
from pathlib import Path

class ModelType(Enum):
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"
    PROPHET = "prophet"
    XGB = "xgboost"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mse: float
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    training_time: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            'mse': float(self.mse),
            'rmse': float(self.rmse),
            'mae': float(self.mae),
            'mape': float(self.mape),
            'directional_accuracy': float(self.directional_accuracy),
            'training_time': float(self.training_time),
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetrics':
        return cls(
            mse=data['mse'],
            rmse=data['rmse'],
            mae=data['mae'],
            mape=data['mape'],
            directional_accuracy=data['directional_accuracy'],
            training_time=data['training_time'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class ModelConfig:
    """Enhanced model configuration"""
    name: str
    type: ModelType
    params: Dict
    features: List[str]
    target: str = 'close'
    enabled: bool = True
    version: str = "1.0"

    # Training configuration
    validation_split: float = 0.2
    shuffle: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 10

    # Feature engineering settings
    feature_engineering: Dict = None

    # Validation settings
    cross_validation: bool = False
    cv_folds: int = 5
    cv_gap: int = 0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['type'] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        data = data.copy()
        data['type'] = ModelType[data['type'].upper()]
        return cls(**data)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

class BaseModel(ABC):
    """Enhanced abstract base class for all models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.training_history: List[ModelMetrics] = []
        self.validation_history: List[ModelMetrics] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.best_metrics: Optional[ModelMetrics] = None
        self.training_time: Optional[float] = None
        self.last_training: Optional[datetime] = None

        # Create model directory
        self.model_dir = Path(f"models/{config.name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Union[Tuple[np.ndarray, np.ndarray], pd.DataFrame]:
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

    def validate(self, df: pd.DataFrame) -> ModelMetrics:
        """Validate model performance"""
        try:
            predictions = self.predict(df)
            actuals = df[self.config.target].values

            metrics = self._calculate_metrics(actuals, predictions)
            self.validation_history.append(metrics)

            if self.best_metrics is None or metrics.rmse < self.best_metrics.rmse:
                self.best_metrics = metrics

            return metrics

        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        try:
            # Remove NaN values for metric calculation
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                raise ValueError("No valid data points for metric calculation")

            # Basic metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))

            # Handle zeros in MAPE calculation
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-7, None))) * 100

            # Directional accuracy
            pred_direction = np.diff(y_pred) > 0
            true_direction = np.diff(y_true) > 0
            directional_accuracy = np.mean(pred_direction == true_direction)

            return ModelMetrics(
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                mape=float(mape),
                directional_accuracy=float(directional_accuracy),
                training_time=self.training_time or 0.0,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance"""
        return self.feature_importance

    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return [m.to_dict() for m in self.training_history]

    def get_validation_history(self) -> List[Dict]:
        """Get validation history"""
        return [m.to_dict() for m in self.validation_history]

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model and metadata"""
        try:
            os.makedirs(path, exist_ok=True)

            # Save model config
            config_path = os.path.join(path, 'config.json')
            self.config.save(config_path)

            # Save metrics history
            metrics_path = os.path.join(path, 'metrics.json')
            metrics_data = {
                'training_history': self.get_training_history(),
                'validation_history': self.get_validation_history(),
                'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
                'last_training': self.last_training.isoformat() if self.last_training else None,
                'training_time': self.training_time
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            # Save feature importance
            if self.feature_importance is not None:
                self.feature_importance.to_csv(
                    os.path.join(path, 'feature_importance.csv'),
                    index=True
                )

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model and metadata"""
        try:
            # Load config
            config_path = os.path.join(path, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

            self.config = ModelConfig.load(config_path)

            # Load metrics history
            metrics_path = os.path.join(path, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    self.training_history = [
                        ModelMetrics.from_dict(m) for m in metrics_data['training_history']
                    ]
                    self.validation_history = [
                        ModelMetrics.from_dict(m) for m in metrics_data['validation_history']
                    ]
                    if metrics_data['best_metrics']:
                        self.best_metrics = ModelMetrics.from_dict(metrics_data['best_metrics'])
                    if metrics_data['last_training']:
                        self.last_training = datetime.fromisoformat(metrics_data['last_training'])
                    self.training_time = metrics_data['training_time']

            # Load feature importance
            importance_path = os.path.join(path, 'feature_importance.csv')
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path, index_col=0)

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise