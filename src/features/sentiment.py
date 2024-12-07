# src/features/sentiment.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from textblob import TextBlob
from dataclasses import dataclass

@dataclass
class MarketSentiment:
    timestamp: datetime
    price_sentiment: float
    volume_sentiment: float
    momentum_sentiment: float
    overall_sentiment: float
    confidence: float

class SentimentAnalyzer:
    """Advanced market sentiment analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_window = 20
        self.lookback_periods = 50

    def analyze_market_sentiment(
            self,
            price_data: pd.DataFrame,
            orderbook_data: Optional[pd.DataFrame] = None,
            news_data: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """Calculate comprehensive market sentiment"""
        try:
            sentiment_df = pd.DataFrame(index=price_data.index)

            # Technical sentiment
            sentiment_df = self._calculate_price_sentiment(price_data, sentiment_df)
            sentiment_df = self._calculate_momentum_sentiment(price_data, sentiment_df)

            # Volume sentiment
            if 'volume' in price_data.columns:
                sentiment_df = self._calculate_volume_sentiment(price_data, sentiment_df)

            # Order book sentiment
            if orderbook_data is not None:
                sentiment_df = self._calculate_orderbook_sentiment(orderbook_data, sentiment_df)

            # News sentiment
            if news_data:
                sentiment_df = self._add_news_sentiment(news_data, sentiment_df)

            # Normalize features
            for column in sentiment_df.columns:
                sentiment_df[column] = self._normalize_feature(sentiment_df[column])

            # Calculate composite sentiment
            sentiment_df['composite_sentiment'] = self._calculate_composite_sentiment(sentiment_df)

            # Add confidence scores
            sentiment_df['confidence'] = self._calculate_confidence_scores(sentiment_df)

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating market sentiment: {e}")

    def _calculate_price_sentiment(self, price_data: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based sentiment indicators"""
        try:
            # Price momentum sentiment
            returns = price_data['close'].pct_change()
            sentiment_df['price_momentum'] = returns.rolling(window=self.sentiment_window).mean()

            # Price trend strength
            for window in [5, 10, 20]:
                ema = price_data['close'].ewm(span=window).mean()
                sentiment_df[f'trend_strength_{window}'] = (price_data['close'] - ema) / ema

            # Price action sentiment
            sentiment_df['price_action'] = np.where(
                (price_data['close'] > price_data['open']) &
                (price_data['high'] - price_data['low'] > price_data['close'] - price_data['open']),
                1,  # Bullish candle
                np.where(
                    (price_data['close'] < price_data['open']) &
                    (price_data['high'] - price_data['low'] > price_data['open'] - price_data['close']),
                    -1,  # Bearish candle
                    0   # Neutral
                )
            )

            # Support/Resistance sentiment
            for window in [20, 50]:
                support = price_data['low'].rolling(window=window).min()
                resistance = price_data['high'].rolling(window=window).max()
                mid_point = (support + resistance) / 2

                sentiment_df[f'sr_position_{window}'] = (price_data['close'] - mid_point) / (resistance - support)

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating price sentiment: {e}")
            return sentiment_df

    def _calculate_momentum_sentiment(self, price_data: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based sentiment"""
        try:
            # RSI sentiment
            from ta.momentum import RSIIndicator
            rsi = RSIIndicator(close=price_data['close']).rsi()
            sentiment_df['rsi_sentiment'] = (rsi - 50) / 50  # Normalize to [-1, 1]

            # MACD sentiment
            from ta.trend import MACD
            macd = MACD(close=price_data['close'])
            sentiment_df['macd_sentiment'] = np.where(
                macd.macd() > macd.macd_signal(),
                macd.macd_diff() / price_data['close'],
                -macd.macd_diff() / price_data['close']
            )

            # Rate of Change sentiment
            for period in [5, 10, 20]:
                roc = (price_data['close'] - price_data['close'].shift(period)) / price_data['close'].shift(period)
                sentiment_df[f'roc_sentiment_{period}'] = roc

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating momentum sentiment: {e}")
            return sentiment_df

    def _calculate_volume_sentiment(self, price_data: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based sentiment"""
        try:
            # Volume trend
            volume_sma = price_data['volume'].rolling(window=20).mean()
            sentiment_df['volume_trend'] = (price_data['volume'] - volume_sma) / volume_sma

            # Volume Force Index
            sentiment_df['force_index'] = price_data['close'].diff() * price_data['volume']
            sentiment_df['force_index'] = sentiment_df['force_index'].ewm(span=13).mean()

            # Volume Price Trend
            close_change = price_data['close'].diff()
            sentiment_df['vpt'] = np.where(
                close_change > 0,
                price_data['volume'],
                np.where(
                    close_change < 0,
                    -price_data['volume'],
                    0
                )
            ).cumsum()

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating volume sentiment: {e}")
            return sentiment_df

    def _calculate_orderbook_sentiment(self, orderbook_data: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order book sentiment"""
        try:
            # Group by timestamp
            grouped = orderbook_data.groupby(['timestamp', 'side'])['quantity'].sum().unstack()

            if 'bid' in grouped and 'ask' in grouped:
                # Calculate bid-ask imbalance
                sentiment_df['orderbook_imbalance'] = (
                        (grouped['bid'] - grouped['ask']) /
                        (grouped['bid'] + grouped['ask'])
                )

                # Calculate spread
                bid_prices = orderbook_data[orderbook_data['side'] == 'bid']['price'].max()
                ask_prices = orderbook_data[orderbook_data['side'] == 'ask']['price'].min()
                sentiment_df['spread'] = (ask_prices - bid_prices) / bid_prices

                # Calculate order book depth
                sentiment_df['depth_ratio'] = grouped['bid'].rolling(window=20).mean() / grouped['ask'].rolling(window=20).mean()

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error calculating orderbook sentiment: {e}")
            return sentiment_df

    def _add_news_sentiment(self, news_data: List[Dict], sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Add news sentiment analysis"""
        try:
            # Process news data
            news_sentiments = []
            for news in news_data:
                sentiment = self._analyze_news_text(news['text'])
                news_sentiments.append({
                    'timestamp': pd.to_datetime(news['timestamp']),
                    'sentiment': sentiment['polarity'],
                    'subjectivity': sentiment['subjectivity']
                })

            # Convert to DataFrame
            news_df = pd.DataFrame(news_sentiments)
            news_df.set_index('timestamp', inplace=True)

            # Resample to match price data frequency
            resampled_sentiment = news_df['sentiment'].resample('1min').mean()
            resampled_subjectivity = news_df['subjectivity'].resample('1min').mean()

            # Forward fill missing values
            sentiment_df['news_sentiment'] = resampled_sentiment
            sentiment_df['news_subjectivity'] = resampled_subjectivity

            sentiment_df.fillna(method='ffill', inplace=True)

            return sentiment_df

        except Exception as e:
            self.logger.error(f"Error adding news sentiment: {e}")
            return sentiment_df

    def _analyze_news_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.error(f"Error analyzing news text: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}

    def _calculate_composite_sentiment(self, sentiment_df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite sentiment"""
        try:
            weights = {
                'price_momentum': 0.3,
                'rsi_sentiment': 0.2,
                'macd_sentiment': 0.2,
                'volume_trend': 0.15,
                'orderbook_imbalance': 0.1,
                'news_sentiment': 0.05
            }

            composite = pd.Series(0, index=sentiment_df.index)
            for column, weight in weights.items():
                if column in sentiment_df.columns:
                    composite += sentiment_df[column] * weight

            return composite

        except Exception as e:
            self.logger.error(f"Error calculating composite sentiment: {e}")
            return pd.Series(0, index=sentiment_df.index)

    def _calculate_confidence_scores(self, sentiment_df: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for sentiment signals"""
        try:
            # Calculate signal strength
            signal_strength = abs(sentiment_df['composite_sentiment'])

            # Calculate signal consistency
            signal_consistency = sentiment_df['composite_sentiment'].rolling(window=5).std()
            signal_consistency = 1 - signal_consistency.clip(0, 1)

            # Calculate volume confidence
            if 'volume_trend' in sentiment_df.columns:
                volume_confidence = (
                                            sentiment_df['volume_trend'].rolling(window=20).mean().clip(-1, 1) + 1
                                    ) / 2
            else:
                volume_confidence = pd.Series(0.5, index=sentiment_df.index)

            # Combine confidence metrics
            confidence = (
                    signal_strength * 0.4 +
                    signal_consistency * 0.4 +
                    volume_confidence * 0.2
            )

            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {e}")
            return pd.Series(0.5, index=sentiment_df.index)

    def _normalize_feature(self, series: pd.Series) -> pd.Series:
        """Normalize feature to [-1, 1] range"""
        try:
            if series.empty or series.isna().all():
                return series

            min_val = series.rolling(window=self.lookback_periods).min()
            max_val = series.rolling(window=self.lookback_periods).max()

            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)

            return 2 * (series - min_val) / range_val - 1

        except Exception as e:
            self.logger.error(f"Error normalizing feature: {e}")
            return series