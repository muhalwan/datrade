import pytest
import pandas as pd
import numpy as np
from src.models.prophet_model import ProphetModel

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    X = pd.DataFrame(np.random.randn(100, 5), index=dates)
    y = pd.Series(np.random.randint(0, 2, 100), index=dates)
    return X, y

def test_prophet_model_initialization():
    model = ProphetModel()
    assert model.model is None
    assert isinstance(model.params, dict)

def test_prophet_model_training(sample_data):
    X, y = sample_data
    model = ProphetModel()
    model.train(X, y)
    assert model.model is not None

def test_prophet_model_prediction(sample_data):
    X, y = sample_data
    model = ProphetModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(p, (int, np.integer)) for p in predictions)