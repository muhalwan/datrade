import pytest
import pandas as pd
import numpy as np
from src.models.lstm import LSTMModel

@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_lstm_model_initialization():
    model = LSTMModel(sequence_length=30)
    assert model.sequence_length == 30
    assert model.n_features is None

def test_lstm_model_training(sample_data):
    X, y = sample_data
    model = LSTMModel(sequence_length=10)
    model.train(X, y, epochs=1)
    assert model.model is not None

def test_lstm_model_prediction(sample_data):
    X, y = sample_data
    model = LSTMModel(sequence_length=10)
    model.train(X, y, epochs=1)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(p, (float, np.float32, np.float64)) for p in predictions)
