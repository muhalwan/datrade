import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import ta
import logging

class FeatureSelector:
    """Advanced feature selection and engineering"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.scaler = StandardScaler()

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following features"""
        try:
            result = df.copy()

            # Price momentum
            result['returns'] = df['close'].pct_change()
            result['returns_volatility'] = result['returns'].rolling(window=20).std()

            # Trend Indicators
            for period in [5, 10, 20, 50]:
                # Simple Moving Average
                result[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)

                # EMA and trend direction
                ema = ta.trend.ema_indicator(df['close'], window=period)
                result[f'ema_{period}'] = ema
                result[f'trend_direction_{period}'] = (ema > ema.shift(1)).astype(int)

                # Price relative to moving averages
                result[f'price_to_sma_{period}'] = df['close'] / result[f'sma_{period}']

                # Moving average slopes
                result[f'sma_slope_{period}'] = (
                                                        result[f'sma_{period}'] - result[f'sma_{period}'].shift(5)
                                                ) / result[f'sma_{period}'].shift(5)

                # Crossovers
                if period < 50:
                    next_period = period * 2
                    result[f'ma_crossover_{period}_{next_period}'] = (
                            result[f'ema_{period}'] > result[f'ema_{next_period}']
                    ).astype(int)

            # MACD
            macd = ta.trend.MACD(df['close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()

            # RSI and conditions
            result['rsi'] = ta.momentum.rsi(df['close'])
            result['rsi_oversold'] = (result['rsi'] < 30).astype(int)
            result['rsi_overbought'] = (result['rsi'] > 70).astype(int)

            # Volume analysis
            result['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            result['volume_trend'] = (df['volume'] > result['volume_sma']).astype(int)
            result['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            result['obv_slope'] = (result['obv'] - result['obv'].shift(5)) / result['obv'].shift(5)

            # Price patterns
            result['higher_high'] = (
                    (df['high'] > df['high'].shift(1)) &
                    (df['high'].shift(1) > df['high'].shift(2))
            ).astype(int)

            result['lower_low'] = (
                    (df['low'] < df['low'].shift(1)) &
                    (df['low'].shift(1) < df['low'].shift(2))
            ).astype(int)

            # Support and Resistance
            result['support_level'] = df['low'].rolling(20).min()
            result['resistance_level'] = df['high'].rolling(20).max()
            result['price_to_support'] = df['close'] / result['support_level']
            result['price_to_resistance'] = df['close'] / result['resistance_level']

            return result

        except Exception as e:
            self.logger.error(f"Error adding trend features: {e}")
            return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            result = df.copy()

            # Bollinger Bands
            for period in [20, 50]:
                bb = ta.volatility.BollingerBands(df['close'], window=period)
                result[f'bb_width_{period}'] = bb.bollinger_wband()
                result[f'bb_position_{period}'] = (
                        (df['close'] - bb.bollinger_lband()) /
                        (bb.bollinger_hband() - bb.bollinger_lband())
                )

            # ATR and volatility
            for period in [14, 30]:
                result[f'atr_{period}'] = ta.volatility.average_true_range(
                    df['high'], df['low'], df['close'], window=period
                )
                result[f'atr_percent_{period}'] = result[f'atr_{period}'] / df['close']

            # Historical volatility
            for period in [5, 10, 20]:
                result[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)

            # Price range features
            result['daily_range'] = (df['high'] - df['low']) / df['close']
            result['range_ma'] = result['daily_range'].rolling(10).mean()
            result['range_expansion'] = result['daily_range'] > result['range_ma']

            # Keltner Channels
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            result['kc_width'] = (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / df['close']

            # Volatility regime
            result['high_volatility'] = (
                    result['volatility_20'] > result['volatility_20'].rolling(50).mean()
            ).astype(int)

            return result

        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            result = df.copy()

            # Stochastic Oscillator
            for period in [14, 28]:
                stoch = ta.momentum.StochasticOscillator(
                    df['high'], df['low'], df['close'], window=period
                )
                result[f'stoch_{period}'] = stoch.stoch()
                result[f'stoch_signal_{period}'] = stoch.stoch_signal()

            # Rate of Change
            for period in [5, 10, 20]:
                result[f'roc_{period}'] = ta.momentum.roc(df['close'], window=period)

            # Money Flow Index
            result['mfi'] = ta.volume.money_flow_index(
                df['high'], df['low'], df['close'], df['volume']
            )

            # Awesome Oscillator
            result['ao'] = ta.momentum.awesome_oscillator(df['high'], df['low'])

            # TSI - True Strength Index
            result['tsi'] = ta.momentum.tsi(df['close'])

            return result

        except Exception as e:
            self.logger.error(f"Error adding momentum features: {e}")
            return df

    def select_features(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            method: str = 'mutual_info',
            n_features: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select most important features"""
        try:
            # Add technical features
            X = self.add_trend_features(X)
            X = self.add_volatility_features(X)
            X = self.add_momentum_features(X)

            # Remove any infinite or missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(method='ffill').fillna(method='bfill')

            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            if method == 'mutual_info':
                # Mutual information based selection
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k='all' if n_features is None else n_features
                )
                selector.fit(X_scaled, y)
                scores = dict(zip(X.columns, selector.scores_))

                # Select features above mean importance
                mean_score = np.mean(list(scores.values()))
                selected_features = [
                    col for col, score in scores.items()
                    if score > mean_score
                ]

            else:
                raise ValueError(f"Unknown selection method: {method}")

            self.selected_features = selected_features
            self.feature_importance = scores

            return X[selected_features], scores

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return X, {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            # Add technical features
            X = self.add_trend_features(X)
            X = self.add_volatility_features(X)
            X = self.add_momentum_features(X)

            # Select only previously selected features
            if self.selected_features:
                X = X[self.selected_features]

            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill()  # Replace fillna with ffill/bfill

            # Scale features
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return X