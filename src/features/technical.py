import pandas as pd
import numpy as np
from typing import Optional
import logging
import talib

class TechnicalIndicators:
    """
    Calculates a wide range of technical indicators using TA-Lib.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators and returns a DataFrame.

        Args:
            price_data (pd.DataFrame): OHLCV price data.

        Returns:
            pd.DataFrame: DataFrame containing technical indicators.
        """
        try:
            indicators = {}
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            volume = price_data['volume'].values

            # Moving Averages
            indicators['sma_5'] = talib.SMA(close, timeperiod=5)
            indicators['ema_5'] = talib.EMA(close, timeperiod=5)
            indicators['sma_10'] = talib.SMA(close, timeperiod=10)
            indicators['ema_10'] = talib.EMA(close, timeperiod=10)
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            indicators['sma_100'] = talib.SMA(close, timeperiod=100)
            indicators['ema_100'] = talib.EMA(close, timeperiod=100)

            # Momentum Indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

            # Volatility Indicators
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
            indicators['kc_middle'], indicators['kc_upper'], indicators['kc_lower'] = talib.KC(close, high, low, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

            # Volume Indicators
            indicators['obv'] = talib.OBV(close, volume)
            indicators['obv_sma'] = talib.SMA(indicators['obv'], timeperiod=20)

            # Candlestick Patterns
            indicators['cdl_engulfing'] = talib.CDLENGULFING(open=price_data['open'].values, high=high, low=low, close=close)
            indicators['cdl_hammer'] = talib.CDLHAMMER(open=price_data['open'].values, high=high, low=low, close=close)

            # Additional Features
            indicators['vwap'] = self.calculate_vwap(price_data)
            indicators['volatility'] = indicators['atr']
            indicators['volatility_ma'] = talib.SMA(indicators['volatility'], timeperiod=20)
            indicators['trend_strength_5'] = self.calculate_trend_strength(close, window=5)
            indicators['trend_strength_20'] = self.calculate_trend_strength(close, window=20)
            indicators['returns'] = self.calculate_returns(close)

            # Convert to DataFrame
            tech_df = pd.DataFrame(indicators, index=price_data.index)
            self.logger.info("Technical indicators calculated successfully.")
            return tech_df
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return pd.DataFrame()

    def calculate_vwap(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Calculates Volume Weighted Average Price (VWAP).

        Args:
            price_data (pd.DataFrame): OHLCV price data.

        Returns:
            np.ndarray: VWAP values.
        """
        try:
            vwap = (price_data['volume'] * (price_data['high'] + price_data['low'] + price_data['close']) / 3).cumsum() / price_data['volume'].cumsum()
            return vwap.values
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return np.full(len(price_data), np.nan)

    def calculate_trend_strength(self, close: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Calculates trend strength based on moving averages.

        Args:
            close (np.ndarray): Closing prices.
            window (int): Window size for moving average.

        Returns:
            np.ndarray: Trend strength values.
        """
        try:
            sma = talib.SMA(close, timeperiod=window)
            ema = talib.EMA(close, timeperiod=window)
            trend_strength = np.abs(ema - sma) / sma
            return trend_strength
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return np.full(len(close), 0.0)

    def calculate_returns(self, close: np.ndarray) -> np.ndarray:
        """
        Calculates daily returns.

        Args:
            close (np.ndarray): Closing prices.

        Returns:
            np.ndarray: Daily returns.
        """
        try:
            returns = np.diff(close) / close[:-1]
            returns = np.append(returns, 0)  # Append 0 for the last entry
            return returns
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return np.zeros(len(close))
