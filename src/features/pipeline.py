import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import time

from src.features.technical.indicators import TechnicalAnalysis, IndicatorConfig
from src.features.sentiment.analyzer import MarketSentimentAnalyzer, SentimentConfig
from src.data.database.connection import MongoDBConnection

class FeaturePipelineManager:
    """Manages feature generation pipeline"""

    def __init__(self,
                 db: MongoDBConnection,
                 indicator_config: Optional[IndicatorConfig] = None,
                 sentiment_config: Optional[SentimentConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.db = db

        # Initialize analyzers
        self.technical = TechnicalAnalysis(indicator_config)
        self.sentiment = MarketSentimentAnalyzer(sentiment_config)

        # Setup processing queue
        self.feature_queue = queue.Queue()

        # Feature cache
        self.feature_cache = {}
        self.cache_lock = threading.Lock()

        # Start processing
        self._start_pipeline()

    def _start_pipeline(self):
        """Start feature processing pipeline"""
        def pipeline_worker():
            while True:
                try:
                    # Get latest price data
                    price_data = self._get_latest_price_data()

                    if price_data is not None:
                        # Generate features
                        features = self._generate_features(price_data)

                        # Cache features
                        self._cache_features(features)

                    time.sleep(1)  # Prevent CPU overuse

                except Exception as e:
                    self.logger.error(f"Pipeline error: {str(e)}")
                    time.sleep(5)

        # Start pipeline thread
        threading.Thread(target=pipeline_worker, daemon=True).start()
        self.logger.info("Feature pipeline started")

    def _get_latest_price_data(self) -> Optional[pd.DataFrame]:
        """Get latest price data from MongoDB"""
        try:
            collection = self.db.get_collection('price_data')
            if collection is None:
                return None

            # Get last 1000 price points
            cursor = collection.find(
                {},
                sort=[('timestamp', -1)],
                limit=1000
            )

            data = list(cursor)
            if not data:
                self.logger.warning("No data found in database")
                return None

            df = pd.DataFrame(data)

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"Missing required columns: {missing}")
                return None

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error getting price data: {str(e)}")
            return None

    def _generate_features(self, price_data: pd.DataFrame) -> Dict:
        """Generate all features"""
        try:
            features = {}

            # Generate technical features
            technical_features = self.technical.calculate_all_indicators(price_data)

            # Get latest technical signals
            technical_signals = self.technical.get_latest_signals(technical_features)

            # Get sentiment features
            sentiment_features = self.sentiment.get_aggregate_sentiment('1h')

            # Combine features
            features['technical'] = technical_signals
            features['sentiment'] = sentiment_features
            features['timestamp'] = datetime.now()

            return features

        except Exception as e:
            self.logger.error(f"Feature generation error: {str(e)}")
            return {}

    def _cache_features(self, features: Dict):
        """Cache generated features"""
        try:
            with self.cache_lock:
                self.feature_cache = features

        except Exception as e:
            self.logger.error(f"Cache error: {str(e)}")

    def get_latest_features(self) -> Dict:
        """Get latest generated features"""
        try:
            with self.cache_lock:
                return self.feature_cache.copy()

        except Exception as e:
            self.logger.error(f"Error getting features: {str(e)}")
            return {}

    def get_historical_features(self,
                                start_time: datetime,
                                end_time: datetime = None) -> pd.DataFrame:
        """Get historical features for backtesting"""
        try:
            # Get price data
            collection = self.db.get_collection('price_data')
            if collection is None:
                return pd.DataFrame()

            query = {
                'timestamp': {
                    '$gte': start_time,
                    '$lte': end_time or datetime.now()
                }
            }

            cursor = collection.find(query).sort('timestamp', 1)
            df = pd.DataFrame(list(cursor))

            if df.empty:
                return df

            # Set index
            df.set_index('timestamp', inplace=True)

            # Generate technical features
            df = self.technical.calculate_all_indicators(df)

            # Normalize features
            df = self.technical.normalize_features(df)

            return df

        except Exception as e:
            self.logger.error(f"Error getting historical features: {str(e)}")
            return pd.DataFrame()