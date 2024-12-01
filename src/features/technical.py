import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta
import logging
from datetime import datetime, timedelta

class TechnicalFeatureCalculator:
    """Calculates technical indicators for trading data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features"""
        try:
            # Make copy to avoid modifying original
            data = df.copy()

            # Basic price features
            data = self._add_price_features(data)

            # Moving averages
            data = self._add_moving_averages(data)

            # Momentum indicators
            data = self._add_momentum_indicators(data)

            # Volatility indicators
            data = self._add_volatility_indicators(data)

            # Volume indicators
            data = self._add_volume_indicators(data)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        try:
            # Price changes
            df['price_change'] = df['close'].diff()
            df['returns'] = df['close'].pct_change()

            # Log returns
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))

            # Price trends
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['lower_low'] = df['low'] < df['low'].shift(1)

            return df
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators"""
        try:
            periods = [5, 10, 20, 50, 200]

            for period in periods:
                # Simple Moving Average
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)

                # Exponential Moving Average
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)

                # Moving Average Crossovers
                if period < 50:
                    df[f'ma_crossover_{period}_50'] = (
                            df[f'sma_{period}'] > df['sma_50']
                    ).astype(int)

            return df
        except Exception as e:
            self.logger.error(f"Error adding moving averages: {e}")
            return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                df['high'],
                df['low'],
                df['close']
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            return df
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {e}")
            return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        try:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_width'] = (
                    (df['bb_high'] - df['bb_low']) / df['bb_mid']
            )

            # Average True Range
            df['atr'] = ta.volatility.average_true_range(
                df['high'],
                df['low'],
                df['close']
            )

            # Historical Volatility
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            return df
        except Exception as e:
            self.logger.error(f"Error adding volatility indicators: {e}")
            return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            # On-Balance Volume
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            # Volume Weighted Average Price
            df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()

            # Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(
                df['high'],
                df['low'],
                df['close'],
                df['volume']
            )

            return df
        except Exception as e:
            self.logger.error(f"Error adding volume indicators: {e}")
            return df