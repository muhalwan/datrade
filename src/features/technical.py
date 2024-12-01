import pandas as pd
import numpy as np
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

            # Add basic price features first (includes returns calculation)
            data = self._add_price_features(data)

            # Add moving averages
            for window in [5, 10, 20]:
                data = self._add_moving_average(data, window)

            # Add momentum indicators
            data = self._add_momentum_indicators(data)

            # Add volatility indicators
            data = self._add_volatility_indicators(data)

            # Add volume indicators
            data = self._add_volume_indicators(data)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return pd.DataFrame()

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        try:
            df = df.copy()
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

    def _add_moving_average(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add moving average for specific window"""
        try:
            # Simple Moving Average
            df[f'sma_{window}'] = ta.trend.sma_indicator(df['close'], window=window)

            # Exponential Moving Average
            df[f'ema_{window}'] = ta.trend.ema_indicator(df['close'], window=window)

            return df
        except Exception as e:
            self.logger.error(f"Error adding moving average for window {window}: {e}")
            return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'])

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            return df
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {e}")
            return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()

            # Calculate returns first if not present
            if 'returns' not in df.columns:
                df['returns'] = df['close'].pct_change()

            # Bollinger Bands with default window=20, window_dev=2
            window = 20
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()

            df['bb_mid'] = rolling_mean
            df['bb_high'] = rolling_mean + (rolling_std * 2)
            df['bb_low'] = rolling_mean - (rolling_std * 2)
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

            # Average True Range
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

            # Historical Volatility (20-day)
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
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])

            return df
        except Exception as e:
            self.logger.error(f"Error adding volume indicators: {e}")
            return df