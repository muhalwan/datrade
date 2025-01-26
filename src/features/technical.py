import pandas as pd
import numpy as np
import talib
import logging

class TechnicalIndicators:
    """
    Calculates a wide range of technical indicators using TA-Lib.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate(self, price_data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Calculating technical indicators.")
            tech_df = pd.DataFrame({'close': price_data['close']}, index=price_data.index)
            required_columns = ['close', 'high', 'low', 'open', 'volume']  # Added all required OHLCV columns

            # Changed df to price_data
            missing_columns = [col for col in required_columns if col not in price_data.columns]

            if missing_columns:
                self.logger.error(f"Missing columns for technical indicators: {missing_columns}")
                raise KeyError(f"Missing columns: {missing_columns}")

            # Moving Averages
            tech_df['sma_20'] = talib.SMA(price_data['close'], timeperiod=20)
            tech_df['ema_20'] = talib.EMA(price_data['close'], timeperiod=20)

            # Momentum Indicators
            tech_df['rsi_14'] = talib.RSI(price_data['close'], timeperiod=14)
            tech_df['stochastic_k'], tech_df['stochastic_d'] = talib.STOCH(
                price_data['high'], price_data['low'], price_data['close'],
                fastk_period=14, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            tech_df['macd'], tech_df['macdsignal'], tech_df['macdhist'] = talib.MACD(
                price_data['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )

            # Volatility Indicators
            tech_df['atr_14'] = talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
            tech_df['bollinger_upper'], tech_df['bollinger_middle'], tech_df['bollinger_lower'] = talib.BBANDS(
                price_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            tech_df['keltner_upper'], tech_df['keltner_middle'], tech_df['keltner_lower'] = self._calculate_keltner_channels(price_data)

            # Volume Indicators
            tech_df['obv'] = talib.OBV(price_data['close'], price_data['volume'])

            # Candlestick Patterns
            tech_df['cdl_pattern'] = self._identify_candlestick_patterns(price_data)

            # Additional Features
            tech_df['vwap'] = self.calculate_vwap(price_data)
            tech_df['trend_strength_20'] = self.calculate_trend_strength(price_data['close'].values, window=20)
            tech_df['returns'] = self.calculate_returns(price_data['close'].values)

            self.logger.info("Technical indicators calculated successfully.")
            return tech_df
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return pd.DataFrame()

    def _calculate_keltner_channels(self, price_data: pd.DataFrame) -> tuple:
        """Calculate Keltner Channels"""
        try:
            ema = talib.EMA(price_data['close'], timeperiod=20)
            atr = talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=10)
            upper = ema + (1.5 * atr)
            lower = ema - (1.5 * atr)
            return upper, ema, lower
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {e}")
            return pd.Series([np.nan]*len(price_data)), pd.Series([np.nan]*len(price_data)), pd.Series([np.nan]*len(price_data))

    def _identify_candlestick_patterns(self, price_data: pd.DataFrame) -> pd.Series:
        """Identify candlestick patterns"""
        try:
            patterns = talib.CDLDRAGONFLYDOJI(price_data['open'], price_data['high'], price_data['low'], price_data['close'])
            return patterns
        except Exception as e:
            self.logger.error(f"Error identifying candlestick patterns: {e}")
            return pd.Series([0]*len(price_data))

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
