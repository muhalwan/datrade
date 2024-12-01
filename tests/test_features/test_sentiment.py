import pytest
import pandas as pd
import numpy as np
from src.features.sentiment import SentimentAnalyzer

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    price_data = pd.DataFrame({
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

def test_sentiment_analyzer_initialization():
    analyzer = SentimentAnalyzer()
    assert analyzer is not None

def test_analyze_news_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_news_sentiment("Bitcoin price rises amid strong market sentiment")
    assert 'polarity' in result
    assert 'subjectivity' in result
    assert -1 <= result['polarity'] <= 1
    assert 0 <= result['subjectivity'] <= 1

def test_calculate_market_sentiment(sample_data):
    price_data, orderbook_data = sample_data
    analyzer = SentimentAnalyzer()

    result = analyzer.calculate_market_sentiment(
        price_data=price_data,
        orderbook_data=orderbook_data
    )

    assert isinstance(result, pd.DataFrame)
    assert 'composite_sentiment' in result.columns
    assert -1 <= result['composite_sentiment'].max() <= 1