import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
import threading
import queue
import time

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    news_sources: List[str] = None
    social_platforms: List[str] = None
    update_interval: int = 300  # seconds
    batch_size: int = 100
    cache_duration: int = 3600  # seconds

    def __post_init__(self):
        self.news_sources = self.news_sources or ["cryptopanic", "reddit"]
        self.social_platforms = self.social_platforms or ["twitter", "reddit"]

class MarketSentimentAnalyzer:
    """Real-time market sentiment analyzer"""

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize sentiment analyzers
        self.initialize_nlp()

        # Setup queues for async processing
        self.news_queue = queue.Queue()
        self.social_queue = queue.Queue()

        # Cache for sentiment scores
        self.sentiment_cache = {}

        # Start processing threads
        self._start_processors()

    def initialize_nlp(self):
        """Initialize NLP models and downloads required data"""
        try:
            # Initialize NLTK
            nltk.download('vader_lexicon', quiet=True)
            self.vader = SentimentIntensityAnalyzer()

            # Initialize FinBERT for financial sentiment
            self.finbert = pipeline("sentiment-analysis",
                                    model="ProsusAI/finbert",
                                    max_length=512)

            self.logger.info("NLP models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {str(e)}")
            raise

    def _start_processors(self):
        """Start async processing threads"""
        # News processing thread
        threading.Thread(target=self._process_news_queue,
                         daemon=True).start()

        # Social media processing thread
        threading.Thread(target=self._process_social_queue,
                         daemon=True).start()

    def _process_news_queue(self):
        """Process news items from queue"""
        while True:
            try:
                batch = []
                while not self.news_queue.empty() and len(batch) < self.config.batch_size:
                    batch.append(self.news_queue.get_nowait())

                if batch:
                    self._analyze_news_batch(batch)

                time.sleep(1)
            except Exception as e:
                self.logger.error(f"News processing error: {str(e)}")
                time.sleep(5)

    def _process_social_queue(self):
        """Process social media items from queue"""
        while True:
            try:
                batch = []
                while not self.social_queue.empty() and len(batch) < self.config.batch_size:
                    batch.append(self.social_queue.get_nowait())

                if batch:
                    self._analyze_social_batch(batch)

                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Social processing error: {str(e)}")
                time.sleep(5)

    def _analyze_news_batch(self, news_items: List[Dict]):
        """Analyze batch of news items"""
        try:
            for item in news_items:
                # Extract text
                text = item.get('title', '') + ' ' + item.get('content', '')

                # Get VADER sentiment
                vader_scores = self.vader.polarity_scores(text)

                # Get FinBERT sentiment
                finbert_result = self.finbert(text[:512])[0]

                # Combine scores
                sentiment_score = self._combine_sentiment_scores(
                    vader_scores, finbert_result)

                # Cache result
                self._cache_sentiment(
                    item['id'], 'news', sentiment_score, item.get('timestamp'))

        except Exception as e:
            self.logger.error(f"News analysis error: {str(e)}")

    def _analyze_social_batch(self, social_items: List[Dict]):
        """Analyze batch of social media items"""
        try:
            for item in social_items:
                # Extract text
                text = item.get('text', '')

                # Get VADER sentiment
                vader_scores = self.vader.polarity_scores(text)

                # Calculate engagement weight
                engagement_weight = self._calculate_engagement_weight(item)

                # Combine scores with engagement
                sentiment_score = self._combine_social_scores(
                    vader_scores, engagement_weight)

                # Cache result
                self._cache_sentiment(
                    item['id'], 'social', sentiment_score, item.get('timestamp'))

        except Exception as e:
            self.logger.error(f"Social analysis error: {str(e)}")

    def _combine_sentiment_scores(self, vader_scores: Dict,
                                  finbert_result: Dict) -> float:
        """Combine different sentiment scores into single value"""
        try:
            # Weight VADER and FinBERT scores
            vader_compound = vader_scores['compound']
            finbert_score = self._convert_finbert_score(finbert_result)

            # Weighted average (giving more weight to FinBERT for financial text)
            return 0.4 * vader_compound + 0.6 * finbert_score

        except Exception as e:
            self.logger.error(f"Error combining scores: {str(e)}")
            return 0.0

    def _convert_finbert_score(self, finbert_result: Dict) -> float:
        """Convert FinBERT classification to numeric score"""
        try:
            label = finbert_result['label']
            score = finbert_result['score']

            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"FinBERT conversion error: {str(e)}")
            return 0.0

    def _calculate_engagement_weight(self, item: Dict) -> float:
        """Calculate engagement weight for social media items"""
        try:
            # Get engagement metrics
            likes = item.get('likes', 0)
            replies = item.get('replies', 0)
            reposts = item.get('reposts', 0)

            # Calculate engagement score
            engagement = (likes + replies * 2 + reposts * 3) / 100

            # Apply log scaling and normalize
            return np.log1p(engagement) / 10

        except Exception as e:
            self.logger.error(f"Engagement calculation error: {str(e)}")
            return 0.0

    def _cache_sentiment(self, item_id: str, source: str,
                         score: float, timestamp: datetime):
        """Cache sentiment score"""
        try:
            self.sentiment_cache[f"{source}_{item_id}"] = {
                'score': score,
                'timestamp': timestamp,
                'source': source
            }

            # Cleanup old cache entries
            self._cleanup_cache()

        except Exception as e:
            self.logger.error(f"Cache error: {str(e)}")

    def _cleanup_cache(self):
        """Remove old entries from cache"""
        try:
            current_time = datetime.now()
            expired_keys = [
                k for k, v in self.sentiment_cache.items()
                if (current_time - v['timestamp']).total_seconds() >
                   self.config.cache_duration
            ]

            for k in expired_keys:
                del self.sentiment_cache[k]

        except Exception as e:
            self.logger.error(f"Cache cleanup error: {str(e)}")

    def get_aggregate_sentiment(self, timeframe: str = '1h') -> Dict:
        """Get aggregate sentiment scores for given timeframe"""
        try:
            current_time = datetime.now()

            # Convert timeframe to seconds
            seconds = self._timeframe_to_seconds(timeframe)

            # Filter relevant scores
            recent_scores = [
                v['score'] for v in self.sentiment_cache.values()
                if (current_time - v['timestamp']).total_seconds() <= seconds
            ]

            if not recent_scores:
                return {
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'sample_size': 0
                }

            return {
                'sentiment': np.mean(recent_scores),
                'confidence': 1.0 - (1.0 / len(recent_scores)),
                'sample_size': len(recent_scores)
            }

        except Exception as e:
            self.logger.error(f"Aggregate calculation error: {str(e)}")
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'sample_size': 0
            }

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        try:
            value = int(timeframe[:-1])
            unit = timeframe[-1].lower()

            if unit == 'h':
                return value * 3600
            elif unit == 'm':
                return value * 60
            else:
                return 3600  # default to 1h

        except Exception:
            return 3600