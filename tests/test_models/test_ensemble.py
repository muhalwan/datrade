import pytest
import pandas as pd
import numpy as np
from src.models.ensemble import EnsembleModel

def test_ensemble_model():
    # Create larger dataset
    size = 200
    X = pd.DataFrame(np.random.randn(size, 10))
    y = pd.Series(np.random.randint(0, 2, size))

    model = EnsembleModel()
    model.train(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert all(isinstance(p, (int, np.integer)) for p in predictions)