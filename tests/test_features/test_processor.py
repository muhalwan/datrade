import pytest
import pandas as pd
import numpy as np
from src.features.processor import FeatureProcessor

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    price_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    orderbook_data = pd.DataFrame({
        'timestamp': dates.repeat(10),
        'side': ['bid', 'ask'] * 500,
        'price': np.random.randn(1000) + 100,
        'quantity': np.random.rand(1000) * 10
    })

    return price_data, orderbook_data

def test_feature_processor_initialization():
    processor = FeatureProcessor()
    assert processor is not None
    assert processor.technical_calculator is not None
    assert processor.sentiment_analyzer is not None

def test_prepare_features(sample_data):
    price_data, orderbook_data = sample_data
    processor = FeatureProcessor()

    features, target = processor.prepare_features(
        price_data=price_data,
        orderbook_data=orderbook_data,
        target_minutes=5
    )

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert len(features) == len(target)
    assert not features.empty
    assert not target.empty