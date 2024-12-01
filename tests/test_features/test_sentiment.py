import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.sentiment import SentimentAnalyzer

@pytest.fixture
def sample_data():
    """Create sample data for sentiment analysis"""
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')

    # Create price data
    price_data = pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Create orderbook data
    orderbook_data = []
    for date in dates:
        for side in ['bid', 'ask']:
            orderbook_data.append({
                'timestamp': date,
                'side': side,
                'price': price_data.loc[date, 'close'] * (0.99 if side == 'bid' else 1.01),
                'quantity': np.random.uniform(0.1, 2.0)
            })

    orderbook_df = pd.DataFrame(orderbook_data)

    return price_data, orderbook_df

def test_analyze_news_sentiment():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_news_sentiment("Bitcoin price rises amid strong market sentiment")

    assert isinstance(result, dict)
    assert 'polarity' in result
    assert 'subjectivity' in result
    assert isinstance(result['polarity'], float)
    assert isinstance(result['subjectivity'], float)
    assert -1 <= result['polarity'] <= 1
    assert 0 <= result['subjectivity'] <= 1

def test_calculate_market_sentiment(sample_data):
    price_data, orderbook_data = sample_data
    analyzer = SentimentAnalyzer()

    sentiment = analyzer.calculate_market_sentiment(
        price_data=price_data,
        orderbook_data=orderbook_data
    )

    assert isinstance(sentiment, pd.DataFrame)
    assert not sentiment.empty
    assert 'price_momentum' in sentiment.columns
    assert 'volume_pressure' in sentiment.columns
    assert 'order_book_imbalance' in sentiment.columns
    assert 'composite_sentiment' in sentiment.columns

    # Check value ranges (should be normalized to [-1, 1])
    for column in sentiment.columns:
        assert sentiment[column].min() >= -1
        assert sentiment[column].max() <= 1

def test_calculate_market_sentiment_with_news(sample_data):
    price_data, orderbook_data = sample_data
    analyzer = SentimentAnalyzer()

    # Create sample news data
    news_data = [
        {
            'timestamp': price_data.index[0],
            'text': "Bitcoin reaches new high as institutional adoption grows"
        },
        {
            'timestamp': price_data.index[50],
            'text': "Market concerns over cryptocurrency regulations"
        }
    ]

    sentiment = analyzer.calculate_market_sentiment(
        price_data=price_data,
        orderbook_data=orderbook_data,
        news_data=news_data
    )

    assert isinstance(sentiment, pd.DataFrame)
    assert not sentiment.empty
    assert 'news_sentiment' in sentiment.columns
    assert sentiment['news_sentiment'].notna().any()