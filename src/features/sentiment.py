import pandas as pd
import logging

class SentimentAnalyzer:
    """
    Analyzes orderbook data to derive sentiment features.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, orderbook_data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Analyzing orderbook data for sentiment features.")
            sentiment_df = pd.DataFrame(index=orderbook_data.index)

            # Directly use pre-aggregated orderbook features
            sentiment_df['bid_ask_spread'] = orderbook_data['best_ask'] - orderbook_data['best_bid']
            sentiment_df['volume_imbalance'] = orderbook_data['bid_volume'] - orderbook_data['ask_volume']

            # Handle potential missing columns
            for col in ['bid_ask_spread', 'volume_imbalance']:
                if col in sentiment_df.columns:
                    sentiment_df[f'{col}_norm'] = (
                            (sentiment_df[col] - sentiment_df[col].mean())
                            / sentiment_df[col].std()
                    )

            self.logger.info("Sentiment features calculated successfully.")
            return sentiment_df

        except KeyError as e:
            missing = str(e).strip("'")
            self.logger.error(f"Missing required column in orderbook data: {missing}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
            return pd.DataFrame()