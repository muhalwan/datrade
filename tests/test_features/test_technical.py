import pytest
import pandas as pd
import numpy as np
from src.features.technical import TechnicalFeatureCalculator

@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')

    # Create coherent OHLCV data
    base_price = 100 + np.random.randn(100).cumsum()
    df = pd.DataFrame({
        'open': base_price,
        'high': base_price + abs(np.random.randn(100)),
        'low': base_price - abs(np.random.randn(100)),
        'close': base_price + np.random.randn(100) * 0.5,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure high is always highest and low is always lowest
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df

def test_technical_calculator_initialization():
    calculator = TechnicalFeatureCalculator()
    assert calculator is not None

def test_add_price_features(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator._add_price_features(sample_data)

    assert 'price_change' in result.columns
    assert 'returns' in result.columns
    assert 'log_returns' in result.columns
    assert 'higher_high' in result.columns
    assert 'lower_low' in result.columns

    # Check data types
    assert result['price_change'].dtype == np.float64
    assert result['returns'].dtype == np.float64
    assert result['higher_high'].dtype == bool
    assert result['lower_low'].dtype == bool

def test_add_moving_averages(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator._add_moving_average(sample_data, window=20)

    assert 'sma_20' in result.columns
    assert 'ema_20' in result.columns
    assert not result['sma_20'].isna().all()
    assert not result['ema_20'].isna().all()

    # Check moving average properties
    assert result['sma_20'].std() < sample_data['close'].std()
    assert result['ema_20'].std() < sample_data['close'].std()

def test_add_momentum_indicators(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator._add_momentum_indicators(sample_data)

    # Check presence of indicators
    assert 'rsi' in result.columns
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'stoch_k' in result.columns
    assert 'stoch_d' in result.columns

    # Check RSI range
    assert result['rsi'].min() >= 0
    assert result['rsi'].max() <= 100

def test_add_volatility_indicators(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator._add_volatility_indicators(sample_data)

    # Check presence of indicators
    assert 'bb_high' in result.columns
    assert 'bb_mid' in result.columns
    assert 'bb_low' in result.columns
    assert 'bb_width' in result.columns
    assert 'atr' in result.columns
    assert 'volatility' in result.columns

    # Drop NaN values before checking relationships
    valid_data = result.dropna()

    # Check Bollinger Bands relationships
    assert (valid_data['bb_high'] >= valid_data['bb_mid']).all(), "High band should be >= middle band"
    assert (valid_data['bb_mid'] >= valid_data['bb_low']).all(), "Middle band should be >= low band"

    # Check value ranges
    assert (valid_data['bb_width'] >= 0).all(), "Bandwidth should be non-negative"
    assert (valid_data['atr'] >= 0).all(), "ATR should be non-negative"
    assert (valid_data['volatility'] >= 0).all(), "Volatility should be non-negative"

    # Check for expected NaN pattern in initial window
    assert result['bb_high'].iloc[:19].isna().all(), "First 19 BB values should be NaN"
    assert not result['bb_high'].iloc[19:].isna().all(), "Not all BB values after window should be NaN"

def test_add_volume_indicators(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator._add_volume_indicators(sample_data)

    # Check presence of indicators
    assert 'obv' in result.columns
    assert 'vwap' in result.columns
    assert 'mfi' in result.columns

    # Check MFI range
    assert result['mfi'].min() >= 0
    assert result['mfi'].max() <= 100

def test_calculate_features(sample_data):
    calculator = TechnicalFeatureCalculator()
    result = calculator.calculate_features(sample_data)

    # Check that all main feature groups are present
    feature_groups = {
        'Price': ['price_change', 'returns', 'log_returns'],
        'Moving Averages': ['sma_20', 'ema_20'],
        'Momentum': ['rsi', 'macd', 'stoch_k'],
        'Volatility': ['bb_high', 'bb_low', 'atr'],
        'Volume': ['obv', 'vwap', 'mfi']
    }

    for group, features in feature_groups.items():
        for feature in features:
            assert feature in result.columns, f"Missing {feature} from {group} group"

    # Check data quality
    assert not result.empty
    assert not result.isnull().all().any()
    assert len(result) == len(sample_data)

def test_edge_cases():
    calculator = TechnicalFeatureCalculator()

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = calculator.calculate_features(empty_df)
    assert result.empty

    # Test with single row
    single_row = pd.DataFrame({
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100.5],
        'volume': [1000]
    })
    result = calculator.calculate_features(single_row)
    assert not result.empty
    assert len(result) == 1