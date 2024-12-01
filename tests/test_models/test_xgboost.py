import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_model import XGBoostModel

@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_xgboost_model_initialization():
    model = XGBoostModel()
    assert model.model is None
    assert isinstance(model.params, dict)

def test_xgboost_model_training(sample_data):
    X, y = sample_data
    model = XGBoostModel()
    model.train(X, y)
    assert model.model is not None

def test_xgboost_model_prediction(sample_data):
    X, y = sample_data
    model = XGBoostModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(p, (float, np.float32, np.float64)) for p in predictions)