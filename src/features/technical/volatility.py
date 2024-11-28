import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging
from dataclasses import dataclass

@dataclass
class VolatilityConfig:
    """Configuration for volatility indicators"""
    bb_periods: List[int] = None
    atr_periods: List[int] = None

    def __post_init__(self):
        self.bb_periods = self.bb_periods or [20, 50]
        self.atr_periods = self.atr_periods or [14, 28]

class VolatilityFeatures:
    """Calculate volatility-based technical indicators"""

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
        self.logger = logging.getLogger(__name__)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility indicators"""
        try:
            result = df.copy()

            # Bollinger Bands
            for period in self.config.bb_periods:
                bb_data = self._calculate_bollinger_bands(result['close'], period)
                result = pd.concat([result, bb_data], axis=1)

            # Average True Range
            for period in self.config.atr_periods:
                result[f'atr_{period}'] = self._calculate_atr(result, period)

            # Add additional volatility indicators
            result = self._add_custom_volatility(result)

            return result

        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {str(e)}")
            return df

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                   period: int, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            width = (upper - lower) / sma * 100

            return pd.DataFrame({
                f'bb_upper_{period}': upper,
                f'bb_lower_{period}': lower,
                f'bb_middle_{period}': sma,
                f'bb_width_{period}': width
            }, index=prices.index)

        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {str(e)}")
            return pd.DataFrame(index=prices.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

            return tr.rolling(window=period).mean()

        except Exception as e:
            self.logger.error(f"ATR calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _add_custom_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom volatility indicators"""
        try:
            # Garman-Klass Volatility
            df['gk_volatility'] = self._calculate_garman_klass(df)

            # Parkinson Volatility
            df['parkinson_volatility'] = self._calculate_parkinson(df)

            # Volatility Ratio
            df['volatility_ratio'] = self._calculate_volatility_ratio(df)

            return df

        except Exception as e:
            self.logger.error(f"Custom volatility calculation error: {str(e)}")
            return df

    def _calculate_garman_klass(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Garman-Klass Volatility"""
        try:
            log_hl = np.log(df['high'] / df['low']) ** 2
            log_co = np.log(df['close'] / df['open']) ** 2

            gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
            return np.sqrt(gk.rolling(window=period).mean() * 252)

        except Exception as e:
            self.logger.error(f"Garman-Klass calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _calculate_parkinson(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Parkinson Volatility"""
        try:
            log_hl = np.log(df['high'] / df['low']) ** 2
            return np.sqrt(log_hl.rolling(window=period).mean() * 252 / (4 * np.log(2)))

        except Exception as e:
            self.logger.error(f"Parkinson calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _calculate_volatility_ratio(self, df: pd.DataFrame,
                                    short_period: int = 5,
                                    long_period: int = 20) -> pd.Series:
        """Calculate Volatility Ratio"""
        try:
            returns = df['close'].pct_change()
            short_vol = returns.rolling(window=short_period).std()
            long_vol = returns.rolling(window=long_period).std()

            return short_vol / long_vol

        except Exception as e:
            self.logger.error(f"Volatility ratio calculation error: {str(e)}")
            return pd.Series(index=df.index)