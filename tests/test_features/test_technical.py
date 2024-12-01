import pytest
import pandas as pd
import numpy as np
from src.features.technical import TechnicalFeatureCalculator

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    return pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

def test_technical_features():
    calculator = TechnicalFeatureCalculator()
    data = sample_data()
    result = calculator.calculate_features(data)
    assert not result.empty
    assert 'rsi' in result.columns
    assert 'macd' in result.columns