# src/features/technical.py

import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class TechnicalPattern:
    name: str
    confidence: float
    signal: int  # 1 for bullish, -1 for bearish, 0 for neutral

class TechnicalFeatureCalculator:
    """Advanced technical analysis system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features."""
        try:
            # Make copy to avoid modifying original
            data = df.copy()

            # Add basic price features
            data = self._add_price_features(data)

            # Add volume features
            data = self._add_volume_features(data)

            # Add moving averages for multiple timeframes
            for window in [5, 10, 20, 40, 50, 100]:
                data = self._add_moving_averages(data, window)

            # Add momentum indicators
            data = self._add_momentum_indicators(data)

            # Add volatility indicators
            data = self._add_volatility_indicators(data)

            # Add pattern recognition features
            data = self._add_pattern_features(data)

            # Drop any remaining NaN values
            data.dropna(inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return pd.DataFrame()

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        try:
            df = df.copy()

            # Price changes
            df['price_change'] = df['close'].diff()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))

            # Price trends
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['lower_low'] = df['low'] < df['low'].shift(1)

            # Gap analysis
            df['gap_up'] = df['low'] > df['high'].shift(1)
            df['gap_down'] = df['high'] < df['low'].shift(1)

            # Price levels
            for window in [20, 50]:
                df[f'highest_{window}'] = df['high'].rolling(window=window).max()
                df[f'lowest_{window}'] = df['low'].rolling(window=window).min()
                df[f'price_position_{window}'] = (df['close'] - df[f'lowest_{window}']) / (df[f'highest_{window}'] - df[f'lowest_{window}']).replace(0, np.nan)

            return df
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return df

    def _add_moving_averages(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Add moving averages and related features."""
        try:
            df = df.copy()

            # Simple Moving Average
            df[f'sma_{window}'] = ta.SMA(df['close'], timeperiod=window)

            # Exponential Moving Average
            df[f'ema_{window}'] = ta.EMA(df['close'], timeperiod=window)

            # Moving Average Position
            df[f'ma_position_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}'].replace(0, np.nan)

            # Moving Average Slopes
            df[f'sma_slope_{window}'] = (df[f'sma_{window}'] - df[f'sma_{window}'].shift(5)) / df[f'sma_{window}'].shift(5).replace(0, np.nan)
            df[f'ema_slope_{window}'] = (df[f'ema_{window}'] - df[f'ema_{window}'].shift(5)) / df[f'ema_{window}'].shift(5).replace(0, np.nan)

            # Moving Average Crossovers
            # Ensure that the next_period moving averages exist
            next_period = window * 2
            if next_period in [5, 10, 20, 40, 50, 100]:
                if f'sma_{next_period}' in df.columns and f'ema_{next_period}' in df.columns:
                    df[f'ma_cross_{window}_{next_period}'] = (
                            (df[f'sma_{window}'] > df[f'sma_{next_period}']) &
                            (df[f'sma_{window}'].shift(1) <= df[f'sma_{next_period}'].shift(1))
                    ).astype(int)

            return df
        except Exception as e:
            self.logger.error(f"Error adding moving averages for window {window}: {e}")
            return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        try:
            df = df.copy()

            # RSI
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_diff'] = macd_hist

            # Stochastic Oscillator
            stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = ta.ROC(df['close'], timeperiod=period)

            # Average Directional Index
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

            return df
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {e}")
            return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        try:
            df = df.copy()

            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle.replace(0, np.nan)
            df['bb_position'] = (df['close'] - lower) / (upper - lower).replace(0, np.nan)

            # Average True Range
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['atr_percent'] = df['atr'] / df['close'].replace(0, np.nan)

            # Keltner Channels
            # Calculate EMA and ATR for Keltner Channels
            ema = ta.EMA(df['close'], timeperiod=20)
            atr_kc = ta.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            df['kc_upper'] = ema + (atr_kc * 1.5)
            df['kc_middle'] = ema
            df['kc_lower'] = ema - (atr_kc * 1.5)

            # Volatility regimes
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volatility_ma'] = df['volatility'].rolling(window=50).mean()
            df['high_volatility'] = (df['volatility'] > df['volatility_ma']).astype(int)

            return df
        except Exception as e:
            self.logger.error(f"Error adding volatility indicators: {e}")
            return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        try:
            df = df.copy()

            # Volume trends
            df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)

            # On-Balance Volume
            df['obv'] = ta.OBV(df['close'], df['volume'])
            df['obv_sma'] = ta.SMA(df['obv'], timeperiod=20)

            # Money Flow Index
            df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

            # Volume-weighted Average Price
            vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            df['vwap'] = vwap

            return df
        except Exception as e:
            self.logger.error(f"Error adding volume features: {e}")
            return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition."""
        try:
            df = df.copy()

            # Doji patterns
            df['doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])

            # Hammer patterns
            df['hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['inverted_hammer'] = ta.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])

            # Engulfing patterns
            df['bullish_engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['bearish_engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

            # Star patterns
            df['morning_star'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

            return df
        except Exception as e:
            self.logger.error(f"Error adding pattern features: {e}")
            return df
