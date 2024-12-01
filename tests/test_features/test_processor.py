import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.processor import FeatureProcessor

@pytest.fixture
def sample_data():
    """Create realistic sample data for testing"""
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')

    # Create price data with coherent OHLCV data
    base_price = 100 + np.random.randn(100).cumsum()
    price_data = pd.DataFrame({
        'open': base_price,
        'high': base_price + abs(np.random.randn(100)),
        'low': base_price - abs(np.random.randn(100)),
        'close': base_price + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure high is always highest and low is always lowest
    price_data['high'] = price_data[['open', 'high', 'low', 'close']].max(axis=1)
    price_data['low'] = price_data[['open', 'high', 'low', 'close']].min(axis=1)

    # Create order book data
    orderbook_records = []
    for date in dates:
        current_price = price_data.loc[date, 'close']
        # Add multiple bids and asks around the current price
        for i in range(5):
            # Bids slightly below current price
            orderbook_records.append({
                'timestamp': date,
                'side': 'bid',
                'price': current_price * (1 - 0.001 * i),
                'quantity': np.random.uniform(0.1, 2.0)
            })
            # Asks slightly above current price
            orderbook_records.append({
                'timestamp': date,
                'side': 'ask',
                'price': current_price * (1 + 0.001 * i),
                'quantity': np.random.uniform(0.1, 2.0)
            })

    orderbook_data = pd.DataFrame(orderbook_records)

    return price_data, orderbook_data

def test_prepare_features(sample_data):
    price_data, orderbook_data = sample_data
    processor = FeatureProcessor()

    # Verify input data
    print("\nPrice Data Sample:")
    print(price_data.head())
    print("\nOrderbook Data Sample:")
    print(orderbook_data.head())

    features, target = processor.prepare_features(
        price_data=price_data,
        orderbook_data=orderbook_data,
        target_minutes=5
    )

    # Basic assertions
    assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
    assert isinstance(target, pd.Series), "Target should be a Series"
    assert not features.empty, "Features DataFrame should not be empty"

    # Check specific features
    expected_features = [
        'price_change', 'returns', 'rsi', 'macd',
        'bb_high', 'bb_low', 'price_momentum',
        'volume_pressure', 'order_book_imbalance'
    ]

    for feature in expected_features:
        assert feature in features.columns, f"Missing feature: {feature}"

    # Check data quality
    assert not features.isnull().all().any(), "Some features are all null"
    assert not target.isnull().all(), "Target is all null"
    assert len(features) > 0, "Features length should be greater than 0"
    assert len(features) == len(target), "Features and target should have same length"

    # Check target values
    assert target.isin([0, 1]).all(), "Target values should be binary (0 or 1)"