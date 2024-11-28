import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from ta import momentum, trend, volatility, volume
from datetime import datetime, timedelta

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    timeframes: List[str] = None  # e.g., ["1m", "5m", "15m", "1h"]
    ma_periods: List[int] = None  # Moving average periods
    rsi_periods: List[int] = None  # RSI periods
    bb_periods: List[int] = None  # Bollinger Bands periods
    macd_params: Dict = None      # MACD parameters

    def __post_init__(self):
        # Default values
        self.timeframes = self.timeframes or ["5m", "15m", "1h"]
        self.ma_periods = self.ma_periods or [20, 50, 200]
        self.rsi_periods = self.rsi_periods or [14, 28]
        self.bb_periods = self.bb_periods or [20]
        self.macd_params = self.macd_params or {
            "fast": 12,
            "slow": 26,
            "signal": 9
        }

class TechnicalAnalysis:
    """Technical Analysis Calculator optimized for Python 3.9"""

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.logger = logging.getLogger(__name__)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for given DataFrame"""
        try:
            # Ensure DataFrame has required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame missing required columns: {required_columns}")

            # Create copy to avoid modifying original
            result = df.copy()

            # Add Moving Averages
            for period in self.config.ma_periods:
                result[f'sma_{period}'] = trend.sma_indicator(result['close'], period)
                result[f'ema_{period}'] = trend.ema_indicator(result['close'], period)

            # Add RSI
            for period in self.config.rsi_periods:
                result[f'rsi_{period}'] = momentum.rsi(result['close'], period)

            # Add Bollinger Bands
            for period in self.config.bb_periods:
                result[f'bb_upper_{period}'] = volatility.bollinger_hband(
                    result['close'], period)
                result[f'bb_lower_{period}'] = volatility.bollinger_lband(
                    result['close'], period)
                result[f'bb_mavg_{period}'] = volatility.bollinger_mavg(
                    result['close'], period)

            # Add MACD
            macd = trend.MACD(
                close=result['close'],
                window_fast=self.config.macd_params['fast'],
                window_slow=self.config.macd_params['slow'],
                window_sign=self.config.macd_params['signal']
            )
            result['macd_line'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_histogram'] = macd.macd_diff()

            # Add Volume Indicators
            result['obv'] = volume.on_balance_volume(result['close'], result['volume'])
            result['vwap'] = self._calculate_vwap(result)

            # Add Volatility Indicators
            result['atr'] = volatility.average_true_range(
                high=result['high'],
                low=result['low'],
                close=result['close'],
                window=14
            )

            # Calculate price momentum
            result['price_momentum'] = self._calculate_momentum(result['close'])

            # Add custom indicators
            result = self._add_custom_indicators(result)

            return result

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        except Exception as e:
            self.logger.error(f"VWAP calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def _calculate_momentum(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate price momentum"""
        try:
            return prices.diff(period)
        except Exception as e:
            self.logger.error(f"Momentum calculation error: {str(e)}")
            return pd.Series(index=prices.index)

    def _add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom technical indicators"""
        try:
            # Relative Volume
            df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()

            # Price Change Rate
            df['price_change_rate'] = df['close'].pct_change()

            # Volatility Ratio
            df['volatility_ratio'] = (df['high'] - df['low']) / df['close']

            # Custom RSI Divergence
            df['rsi_divergence'] = self._calculate_rsi_divergence(df)

            return df

        except Exception as e:
            self.logger.error(f"Custom indicator calculation error: {str(e)}")
            return df

    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI divergence signal"""
        try:
            rsi = momentum.rsi(df['close'], window=14)
            price_diff = df['close'].diff()
            rsi_diff = rsi.diff()

            # Bullish divergence: Price making lower lows but RSI making higher lows
            bullish = (price_diff < 0) & (rsi_diff > 0)

            # Bearish divergence: Price making higher highs but RSI making lower highs
            bearish = (price_diff > 0) & (rsi_diff < 0)

            return pd.Series(data=np.where(bullish, 1, np.where(bearish, -1, 0)),
                             index=df.index)

        except Exception as e:
            self.logger.error(f"RSI divergence calculation error: {str(e)}")
            return pd.Series(index=df.index)

    def get_latest_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get latest trading signals based on indicators"""
        try:
            latest = df.iloc[-1]
            signals = {}

            # RSI signals
            for period in self.config.rsi_periods:
                rsi_value = latest[f'rsi_{period}']
                signals[f'rsi_{period}'] = {
                    'value': rsi_value,
                    'oversold': rsi_value < 30,
                    'overbought': rsi_value > 70
                }

            # MACD signals
            signals['macd'] = {
                'line': latest['macd_line'],
                'signal': latest['macd_signal'],
                'histogram': latest['macd_histogram'],
                'crossover': latest['macd_line'] > latest['macd_signal']
            }

            # Bollinger Bands signals
            for period in self.config.bb_periods:
                price = latest['close']
                upper = latest[f'bb_upper_{period}']
                lower = latest[f'bb_lower_{period}']
                signals[f'bb_{period}'] = {
                    'upper': upper,
                    'lower': lower,
                    'above_upper': price > upper,
                    'below_lower': price < lower
                }

            return signals

        except Exception as e:
            self.logger.error(f"Error getting latest signals: {str(e)}")
            return {}

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize technical indicators for ML model input"""
        try:
            # Create copy to avoid modifying original
            normalized = df.copy()

            # List of columns to normalize
            columns_to_normalize = [
                col for col in normalized.columns
                if any(indicator in col for indicator in
                       ['sma', 'ema', 'rsi', 'macd', 'bb', 'obv', 'vwap'])
            ]

            # Apply min-max normalization
            for column in columns_to_normalize:
                min_val = normalized[column].rolling(window=100).min()
                max_val = normalized[column].rolling(window=100).max()
                normalized[column] = (normalized[column] - min_val) / (max_val - min_val)

            return normalized

        except Exception as e:
            self.logger.error(f"Normalization error: {str(e)}")
            return df