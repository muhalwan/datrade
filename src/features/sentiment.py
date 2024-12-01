import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from textblob import TextBlob

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_news_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)

            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}

    def calculate_market_sentiment(self,
                                   price_data: pd.DataFrame,
                                   orderbook_data: pd.DataFrame,
                                   news_data: Optional[List[Dict]] = None) -> pd.DataFrame:
        """Calculate overall market sentiment"""
        try:
            sentiment_df = pd.DataFrame(index=price_data.index)

            # Calculate price momentum first to avoid dependency issues
            sentiment_df['price_momentum'] = self._calculate_price_momentum(price_data)

            # Add price-based sentiment
            sentiment_df['returns'] = price_data['close'].pct_change()
            sentiment_df['volume_pressure'] = self._calculate_volume_pressure(
                price_data,
                sentiment_df['returns']
            )

            # Add order book sentiment
            sentiment_df['order_book_imbalance'] = self._calculate_orderbook_imbalance(
                orderbook_data
            )

            # Add news sentiment if available
            if news_data:
                news_sentiment = self._aggregate_news_sentiment(news_data)
                if not news_sentiment.empty:
                    sentiment_df['news_sentiment'] = news_sentiment

            # Normalize all features
            for column in sentiment_df.columns:
                sentiment_df[column] = self._normalize_feature(sentiment_df[column])

            # Calculate composite sentiment
            feature_columns = [col for col in sentiment_df.columns if col != 'returns']
            if feature_columns:
                sentiment_df['composite_sentiment'] = sentiment_df[feature_columns].mean(axis=1)

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating market sentiment: {e}")
            return pd.DataFrame()

    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price momentum indicator"""
        try:
            # Use exponential moving averages
            ema_short = df['close'].ewm(span=5).mean()
            ema_long = df['close'].ewm(span=20).mean()

            # Calculate momentum
            momentum = (ema_short - ema_long) / ema_long

            return momentum
        except Exception as e:
            self.logger.error(f"Error calculating price momentum: {e}")
            return pd.Series()

    def _calculate_volume_pressure(self, df: pd.DataFrame, returns: pd.Series) -> pd.Series:
        """Calculate buying/selling pressure based on volume"""
        try:
            # Calculate volume-weighted returns
            volume_returns = returns * df['volume']

            # Calculate pressure using rolling window
            pressure = (
                    volume_returns.rolling(window=20).sum() /
                    df['volume'].rolling(window=20).sum()
            )

            return pressure
        except Exception as e:
            self.logger.error(f"Error calculating volume pressure: {e}")
            return pd.Series()

    def _calculate_orderbook_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order book imbalance"""
        try:
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                self.logger.error("Timestamp column missing in orderbook data")
                return pd.Series()

            # Group by timestamp and calculate bid/ask volumes
            grouped = df.groupby(['timestamp', 'side'])['quantity'].sum().unstack()

            # Calculate imbalance ratio
            if 'bid' in grouped and 'ask' in grouped:
                imbalance = (grouped['bid'] - grouped['ask']) / (grouped['bid'] + grouped['ask'])
                imbalance.index = pd.to_datetime(imbalance.index)
                return imbalance
            else:
                self.logger.error("Missing bid or ask data in orderbook")
                return pd.Series()

        except Exception as e:
            self.logger.error(f"Error calculating orderbook imbalance: {e}")
            return pd.Series()

    def _aggregate_news_sentiment(self, news_data: List[Dict]) -> pd.Series:
        """Aggregate sentiment from news data"""
        try:
            sentiments = []
            timestamps = []

            for news in news_data:
                sentiment = self.analyze_news_sentiment(news['text'])
                sentiments.append(sentiment['polarity'])
                timestamps.append(pd.to_datetime(news['timestamp']))

            return pd.Series(sentiments, index=timestamps)
        except Exception as e:
            self.logger.error(f"Error aggregating news sentiment: {e}")
            return pd.Series()

    def _normalize_feature(self, series: pd.Series) -> pd.Series:
        """Normalize feature to [-1, 1] range"""
        try:
            if series.empty or series.isna().all():
                return series

            min_val = series.min()
            max_val = series.max()

            if min_val == max_val:
                return pd.Series(0, index=series.index)

            return 2 * (series - min_val) / (max_val - min_val) - 1

        except Exception as e:
            self.logger.error(f"Error normalizing feature: {e}")
            return series