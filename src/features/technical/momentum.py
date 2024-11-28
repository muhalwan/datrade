import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging
from dataclasses import dataclass

@dataclass
class MomentumConfig:
    """Configuration for momentum indicators"""
    rsi_periods: List[int] = None
    roc_periods: List[int] = None
    macd_params: Dict = None
    stoch_params: Dict = None

    def __post_init__(self):
        self.rsi_periods = self.rsi_periods or [14, 28]
        self.roc_periods = self.roc_periods or [12, 25]
        self.macd_params = self.macd_params or {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
        self.stoch_params = self.stoch_params or {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3
        }

class MomentumFeatures:
    """Calculate momentum-based technical indicators"""

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
        self.logger = logging.getLogger(__name__)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators"""
        try:
            result = df.copy()

            # RSI
            for period in self.config.rsi_periods:
                result[f'rsi_{period}'] = self._calculate_rsi(result['close'], period)

            # Rate of Change
            for period in self.config.roc_periods:
                result[f'roc_{period}'] = self._calculate_roc(result['close'], period)

            # MACD
            macd_data = self._calculate_macd(result['close'])
            result = pd.concat([result, macd_data], axis=1)

            # Stochastic Oscillator
            stoch_data = self._calculate_stochastic(result)
            result = pd.concat([result, stoch_data], axis=1)

            # Custom Momentum Indicators
            result = self._add_custom_momentum(result)

            return result

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"RSI calculation error: {str(e)}")
            return pd.Series(index=prices.index)

    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        try:
            return prices.pct_change(period) * 100

        except Exception as e:
            self.logger.error(f"ROC calculation error: {str(e)}")
            return pd.Series(index=prices.index)

    def _calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD indicator"""
        try:
            fast = prices.ewm(span=self.config.macd_params['fast']).mean()
            slow = prices.ewm(span=self.config.macd_params['slow']).mean()
            macd_line = fast - slow
            signal_line = macd_line.ewm(span=self.config.macd_params['signal']).mean()
            histogram = macd_line - signal_line

            return pd.DataFrame({
                'macd_line': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }, index=prices.index)

        except Exception as e:
            self.logger.error(f"MACD calculation error: {str(e)}")
            return pd.DataFrame(index=prices.index)

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        try:
            k_period = self.config.stoch_params['k_period']
            d_period = self.config.stoch_params['d_period']
            smooth_k = self.config.stoch_params['smooth_k']

            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()

            k = 100 * (df['close'] - low_min) / (high_max - low_min)
            k = k.rolling(window=smooth_k).mean()
            d = k.rolling(window=d_period).mean()

            return pd.DataFrame({
                'stoch_k': k,
                'stoch_d': d
            }, index=df.index)

        except Exception as e:
            self.logger.error(f"Stochastic calculation error: {str(e)}")
            return pd.DataFrame(index=df.index)

    def _add_custom_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom momentum indicators"""
        try:
            # Average Directional Index (ADX)
            df['adx'] = self._calculate_adx(df)

            # Price Momentum Indicator
            df['pmi'] = self._calculate_pmi(df)

            # Volume-Weighted Momentum
            df['volume_momentum'] = self._calculate_volume_momentum(df)

            return df

        except Exception as e:
            self.logger.error(f"Custom momentum calculation error: {str(e)}")
            return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(period).mean()

            # Calculate Directional Movement
            plus_dm = high - high.shift(1)
            minus_dm = low.shift(1) - low
            plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
            minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

            # Calculate Directional Indicators
            plus_di = 100 * plus_dm.rolling(period).mean() / atr
            minus_di = 100 * minus_dm.rolling(period).mean() / atr

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()

            return adx

        except Exception as e:
            self.logger.error(f"ADX calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _calculate_pmi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Price Momentum Indicator"""
        try:
            close = df['close']
            volume = df['volume']

            # Calculate price change and volume change
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Calculate momentum
            momentum = price_change * np.where(volume_change > 0, 1 + volume_change, 1)

            return pd.Series(momentum, index=df.index)

        except Exception as e:
            self.logger.error(f"PMI calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _calculate_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Weighted Momentum"""
        try:
            close = df['close']
            volume = df['volume']

            # Calculate normalized volume
            norm_volume = volume / volume.rolling(20).mean()

            # Calculate momentum with volume weighting
            momentum = close.pct_change() * norm_volume

            return pd.Series(momentum, index=df.index)

        except Exception as e:
            self.logger.error(f"Volume momentum calculation error: {str(e)}")
            return pd.Series(index=df.index)