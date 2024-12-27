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
    """Advanced technical analysis system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features"""
        try:
            # Make copy to avoid modifying original
            data = df.copy()

            # Add basic price features
            data = self._add_price_features(data)

            # Add volume features
            data = self._add_volume_features(data)

            # Add moving averages for multiple timeframes
            for window in [5, 10, 20, 50, 100]:
                data = self._add_moving_averages(data, window)

            # Add momentum indicators
            data = self._add_momentum_indicators(data)

            # Add volatility indicators
            data = self._add_volatility_indicators(data)

            # Add pattern recognition features
            data = self._add_pattern_features(data)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating technical features: {e}")
            return pd.DataFrame()

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
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
                df[f'price_position_{window}'] = (df['close'] - df[f'lowest_{window}']) / (df[f'highest_{window}'] - df[f'lowest_{window}'])

            return df
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return df

    def _add_moving_averages(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        try:
            df = df.copy()

            # Simple Moving Average
            df[f'sma_{window}'] = ta.SMA(df['close'], timeperiod=window)

            # Exponential Moving Average
            df[f'ema_{window}'] = ta.EMA(df['close'], timeperiod=window)

            # Moving Average Position
            df[f'ma_position_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

            # Moving Average Slopes
            df[f'sma_slope_{window}'] = (df[f'sma_{window}'] - df[f'sma_{window}'].shift(5)) / df[f'sma_{window}'].shift(5)
            df[f'ema_slope_{window}'] = (df[f'ema_{window}'] - df[f'ema_{window}'].shift(5)) / df[f'ema_{window}'].shift(5)

            # Moving Average Crossovers
            if window < 100:
                next_period = window * 2
                df[f'ma_cross_{window}_{next_period}'] = (
                        (df[f'sma_{window}'] > df[f'sma_{next_period}']) &
                        (df[f'sma_{window}'].shift(1) <= df[f'sma_{next_period}'].shift(1))
                ).astype(int)

            return df
        except Exception as e:
            self.logger.error(f"Error adding moving averages for window {window}: {e}")
            return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            df = df.copy()

            # RSI
            df['rsi'] = ta.RSI(df['close'])
            df['rsi_ma'] = ta.SMA(df['rsi'], timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = ta.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_diff'] = macd_hist
            df['macd_crossing'] = (
                    (df['macd'] > df['macd_signal']) &
                    (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            ).astype(int)

            # Stochastic
            stoch = ta.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Rate of Change
            for period in [9, 21]:
                df[f'roc_{period}'] = ta.ROC(df['close'], period)

            # Average Directional Index
            adx = ta.ADX(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()

            return df
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {e}")
            return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(df['close'])
            df['bb_high'] = upper
            df['bb_mid'] = middle
            df['bb_low'] = lower
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

            # Average True Range
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
            df['atr_percent'] = df['atr'] / df['close']

            # Keltner Channels
            # Keltner Channels are not directly available in TA-Lib, so we calculate them manually
            atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            ema = ta.EMA(df['close'], timeperiod=20)
            df['kc_high'] = ema + 2 * atr
            df['kc_mid'] = ema
            df['kc_low'] = ema - 2 * atr

            # Volatility regimes
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volatility_ma'] = df['volatility'].rolling(window=50).mean()
            df['high_volatility'] = (df['volatility'] > df['volatility_ma']).astype(int)

            return df
        except Exception as e:
            self.logger.error(f"Error adding volatility indicators: {e}")
            return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            df = df.copy()

            # Volume trends
            df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # On-Balance Volume
            df['obv'] = ta.OBV(df['close'], df['volume'])
            df['obv_sma'] = ta.SMA(df['obv'], timeperiod=20)

            # Volume Force Index
            df['force_index'] = ta.FORCE(df['close'], df['volume'])

            # Money Flow Index
            df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'])

            # Ease of Movement
            df['eom'] = ta.EOM(df['high'], df['low'], df['volume'])

            # Volume-weighted Average Price
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

            return df
        except Exception as e:
            self.logger.error(f"Error adding volume features: {e}")
            return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        try:
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

    def identify_patterns(self, df: pd.DataFrame) -> List[TechnicalPattern]:
        """Identify technical patterns in price data"""
        patterns = []

        try:
            # Head and Shoulders
            if self._is_head_and_shoulders(df):
                patterns.append(TechnicalPattern(
                    name="Head and Shoulders",
                    confidence=0.8,
                    signal=-1
                ))

            # Double Top/Bottom
            if self._is_double_top(df):
                patterns.append(TechnicalPattern(
                    name="Double Top",
                    confidence=0.7,
                    signal=-1
                ))
            elif self._is_double_bottom(df):
                patterns.append(TechnicalPattern(
                    name="Double Bottom",
                    confidence=0.7,
                    signal=1
                ))

            # Flag/Pennant
            if self._is_bull_flag(df):
                patterns.append(TechnicalPattern(
                    name="Bull Flag",
                    confidence=0.6,
                    signal=1
                ))

            return patterns

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {e}")
            return []

    def _is_head_and_shoulders(self, df: pd.DataFrame, lookback: int = 100) -> bool:
        """Identify head and shoulders pattern"""
        try:
            data = df.tail(lookback)
            peaks = self._find_peaks(data['high'].values)

            if len(peaks) >= 3:
                # Check for characteristic peak heights
                left_shoulder = peaks[-3]
                head = peaks[-2]
                right_shoulder = peaks[-1]

                if (head > left_shoulder and
                        head > right_shoulder and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error in head and shoulders detection: {e}")
            return False

    def _find_peaks(self, data: np.ndarray, min_dist: int = 10) -> np.ndarray:
        """Find peaks in price data"""
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(data, distance=min_dist)
            return peaks
        except Exception as e:
            self.logger.error(f"Error finding peaks: {e}")
            return np.array([])