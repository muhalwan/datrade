import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class FeatureEngineering:
    """Advanced feature engineering pipeline"""

    def __init__(self, db_connection):
        self.logger = logging.getLogger(__name__)
        self.db = db_connection

        # Technical features configuration
        self.tech_features = {
            'sma': [7, 14, 21, 50, 200],
            'ema': [7, 14, 21, 50, 200],
            'bb_window': 20,
            'bb_std': 2,
            'rsi_window': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_window': 20,
            'stoch_window': 14,
            'stoch_smooth': 3,
            'atr_window': 14,
            'williams_r_window': 14,
            'donchian_window': 20,
            'keltner_window': 20,
            'keltner_atr_multiplier': 2
        }

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data preparation and cleaning"""
        try:
            df = df.copy()
            self.logger.info(f"Initial data shape: {df.shape}")

            # Ensure timestamp is index and sorted
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Convert price data to OHLCV format
            if 'price' in df.columns and 'quantity' in df.columns:
                df_ohlcv = self._convert_to_ohlcv(df)
                if df_ohlcv is not None:
                    df = df_ohlcv

            # Forward fill limited gaps
            df = df.ffill(limit=5)
            df = df.bfill(limit=5)

            # Remove remaining NaN values
            df = df.dropna()

            self.logger.info(f"Prepared data shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            return pd.DataFrame()

    def _convert_to_ohlcv(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Convert tick data to OHLCV format"""
        try:
            # Create OHLCV dataframe
            ohlc = df['price'].resample('1min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            volume = df['quantity'].resample('1min').sum().rename('volume')

            # Combine OHLCV data
            df_ohlcv = pd.concat([ohlc, volume], axis=1)

            # Add additional base price info
            df_ohlcv['typical_price'] = (df_ohlcv['high'] + df_ohlcv['low'] + df_ohlcv['close']) / 3
            df_ohlcv['price_change'] = df_ohlcv['close'].diff()
            df_ohlcv['returns'] = df_ohlcv['close'].pct_change(fill_method=None)

            return df_ohlcv

        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {str(e)}")
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features with parallel processing"""
        try:
            self.logger.info(f"Starting feature generation for data shape: {df.shape}")
            features = self.prepare_data(df)

            if features.empty:
                self.logger.warning("No data available for feature generation")
                return pd.DataFrame()

            # Generate features in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                feature_jobs = {
                    executor.submit(self._calculate_price_features, features): 'price',
                    executor.submit(self._calculate_volume_features, features): 'volume',
                    executor.submit(self._calculate_volatility_features, features): 'volatility',
                    executor.submit(self._calculate_trend_features, features): 'trend',
                    executor.submit(self._calculate_momentum_features, features): 'momentum',
                    executor.submit(self._calculate_pattern_features, features): 'pattern'
                }

                # Collect results
                results = {}
                for future in feature_jobs:
                    name = feature_jobs[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[name] = result
                            self.logger.info(f"Generated {name} features")
                    except Exception as e:
                        self.logger.error(f"Error generating {name} features: {str(e)}")

            # Combine all features
            if results:
                # Start with original features
                final_features = features.copy()

                # Add generated features
                for feature_set in results.values():
                    final_features = pd.concat([final_features, feature_set], axis=1)

                # Remove duplicate columns
                final_features = final_features.loc[:, ~final_features.columns.duplicated()]

                # Handle missing values
                final_features = final_features.ffill().bfill()

                # Log all available features
                self.logger.info("Available features:")
                self.logger.info(", ".join(sorted(final_features.columns)))

                return final_features
            else:
                self.logger.error("No features were generated")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error in feature generation: {str(e)}")
            return pd.DataFrame()

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive price-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Calculate returns first
            if 'returns' not in df.columns:
                df['returns'] = df['close'].pct_change(fill_method=None)

            # Basic price features
            features['log_returns'] = np.log1p(df['returns'].fillna(0))
            features['log_price'] = np.log(df['close'])

            # Price spreads and ratios
            features['hl_spread'] = df['high'] - df['low']
            features['oc_spread'] = df['open'] - df['close']
            features['hl_to_oc'] = features['hl_spread'] / features['oc_spread'].abs()
            features['close_to_open'] = df['close'] / df['open'] - 1
            features['close_to_high'] = df['close'] / df['high'] - 1
            features['close_to_low'] = df['close'] / df['low'] - 1

            # Price momentum
            for window in [5, 15, 30]:
                features[f'momentum_{window}'] = df['close'].pct_change(window)
                features[f'log_momentum_{window}'] = np.log(df['close']).diff(window)

            # Price acceleration
            features['price_acceleration'] = features['returns'].diff()
            for window in [5, 15, 30]:
                features[f'acceleration_{window}'] = features['returns'].diff(window)

            # Normalized prices
            for window in [5, 15, 30]:
                rolling_mean = df['close'].rolling(window=window).mean()
                rolling_std = df['close'].rolling(window=window).std()
                features[f'normalized_price_{window}'] = (df['close'] - rolling_mean) / rolling_std

            return features

        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return None

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Basic volume features
            features['volume_momentum'] = df['volume'].pct_change()
            features['log_volume'] = np.log1p(df['volume'])

            # Volume moving averages
            for window in [5, 15, 30]:
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
                features[f'volume_zscore_{window}'] = (
                        (df['volume'] - features[f'volume_sma_{window}']) /
                        features[f'volume_std_{window}']
                )

            # Volume price correlation
            for window in [5, 15, 30]:
                features[f'volume_price_corr_{window}'] = (
                    df[['close', 'volume']]
                    .rolling(window)
                    .corr()
                    .unstack()
                    .iloc[:, 1]
                )

            # Money Flow Index
            features['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=self.tech_features['rsi_window'],
                fillna=True
            )

            # On-Balance Volume (OBV)
            features['obv'] = ta.volume.on_balance_volume(
                close=df['close'],
                volume=df['volume'],
                fillna=True
            )

            # Volume Force Index
            features['force_index'] = ta.volume.force_index(
                close=df['close'],
                volume=df['volume'],
                window=13,
                fillna=True
            )

            # Ease of Movement
            features['eom'] = ta.volume.ease_of_movement(
                high=df['high'],
                low=df['low'],
                volume=df['volume'],
                window=14,
                fillna=True
            )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return None

    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Rolling volatility
            for window in [5, 15, 30]:
                features[f'volatility_{window}'] = df['returns'].rolling(window).std()
                features[f'log_volatility_{window}'] = np.log1p(features[f'volatility_{window}'])

            # Parkinson Volatility
            features['parkinson_volatility'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                np.power(np.log(df['high'] / df['low']), 2)
            )

            # Bollinger Bands
            for window in [20, 50]:
                bb = ta.volatility.BollingerBands(
                    close=df['close'],
                    window=window,
                    window_dev=2,
                    fillna=True
                )
                features[f'bb_high_{window}'] = bb.bollinger_hband()
                features[f'bb_low_{window}'] = bb.bollinger_lband()
                features[f'bb_middle_{window}'] = bb.bollinger_mavg()
                features[f'bb_width_{window}'] = (
                        (features[f'bb_high_{window}'] - features[f'bb_low_{window}']) /
                        features[f'bb_middle_{window}']
                )

            # Average True Range
            atr = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.tech_features['atr_window'],
                fillna=True
            )
            features['atr'] = atr.average_true_range()
            features['atr_pct'] = features['atr'] / df['close']

            # Donchian Channels
            dc = ta.volatility.DonchianChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.tech_features['donchian_window'],
                fillna=True
            )
            features['dc_high'] = dc.donchian_channel_hband()
            features['dc_low'] = dc.donchian_channel_lband()
            features['dc_width'] = (features['dc_high'] - features['dc_low']) / df['close']

            # Keltner Channels
            kc = ta.volatility.KeltnerChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.tech_features['keltner_window'],
                fillna=True
            )
            features['kc_high'] = kc.keltner_channel_hband()
            features['kc_low'] = kc.keltner_channel_lband()
            features['kc_width'] = (features['kc_high'] - features['kc_low']) / df['close']

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return None

    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Moving Averages
            for window in self.tech_features['sma']:
                features[f'sma_{window}'] = ta.trend.sma_indicator(
                    df['close'], window=window, fillna=True
                )
                features[f'ema_{window}'] = ta.trend.ema_indicator(
                    df['close'], window=window, fillna=True
                )

                # Distance from MA
                features[f'dist_to_sma_{window}'] = df['close'] / features[f'sma_{window}'] - 1
                features[f'dist_to_ema_{window}'] = df['close'] / features[f'ema_{window}'] - 1

            # MACD
            macd = ta.trend.MACD(
                close=df['close'],
                window_slow=self.tech_features['macd_slow'],
                window_fast=self.tech_features['macd_fast'],
                window_sign=self.tech_features['macd_signal'],
                fillna=True
            )
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            features['macd_pct'] = features['macd'] / df['close']

            # Trend Direction
            features['adx'] = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                fillna=True
            ).adx()

            # Mass Index
            features['mass_index'] = ta.trend.MassIndex(
                high=df['high'],
                low=df['low'],
                window_fast=9,
                window_slow=25,
                fillna=True
            ).mass_index()

            # Vortex Indicator
            vortex = ta.trend.VortexIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                fillna=True
            )
            features['vortex_pos'] = vortex.vortex_indicator_pos()
            features['vortex_neg'] = vortex.vortex_indicator_neg()
            features['vortex_diff'] = features['vortex_pos'] - features['vortex_neg']

            return features

        except Exception as e:
            self.logger.error(f"Error calculating trend features: {str(e)}")
            return None

    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features"""
        try:
            features = pd.DataFrame(index=df.index)

            # RSI
            features['rsi'] = ta.momentum.RSIIndicator(
                close=df['close'],
                window=self.tech_features['rsi_window'],
                fillna=True
            ).rsi()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.tech_features['stoch_window'],
                smooth_window=self.tech_features['stoch_smooth'],
                fillna=True
            )
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()

            # Williams %R
            features['williams_r'] = ta.momentum.WilliamsRIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                lbp=self.tech_features['williams_r_window'],
                fillna=True
            ).williams_r()

            # Ultimate Oscillator
            features['ultimate_oscillator'] = ta.momentum.UltimateOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window1=7,
                window2=14,
                window3=28,
                weight1=4.0,
                weight2=2.0,
                weight3=1.0,
                fillna=True
            ).ultimate_oscillator()

            # ROC
            for window in [5, 10, 20]:
                features[f'roc_{window}'] = ta.momentum.ROCIndicator(
                    close=df['close'],
                    window=window,
                    fillna=True
                ).roc()

            # Awesome Oscillator
            features['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(
                high=df['high'],
                low=df['low'],
                window1=5,
                window2=34,
                fillna=True
            ).awesome_oscillator()

            return features

        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {str(e)}")
            return None

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Check for patterns using ta-lib
            try:
                import talib

                # Candlestick Patterns
                patterns = {
                    'doji': talib.CDLDOJI,
                    'hammer': talib.CDLHAMMER,
                    'shooting_star': talib.CDLSHOOTINGSTAR,
                    'engulfing': talib.CDLENGULFING,
                    'morning_star': talib.CDLMORNINGSTAR,
                    'evening_star': talib.CDLEVENINGSTAR
                }

                for pattern_name, pattern_func in patterns.items():
                    features[f'pattern_{pattern_name}'] = pattern_func(
                        df['open'].values,
                        df['high'].values,
                        df['low'].values,
                        df['close'].values
                    )

            except ImportError:
                self.logger.warning("ta-lib not installed, skipping pattern features")

            # Custom pattern features
            features['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])

            # Inside bars
            features['inside_bar'] = (
                    (df['high'] <= df['high'].shift(1)) &
                    (df['low'] >= df['low'].shift(1))
            ).astype(int)

            # Outside bars
            features['outside_bar'] = (
                    (df['high'] >= df['high'].shift(1)) &
                    (df['low'] <= df['low'].shift(1))
            ).astype(int)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating pattern features: {str(e)}")
            return None

    def _validate_features(self, features: pd.DataFrame, required_features: List[str]) -> bool:
        """Validate that all required features are present"""
        missing_features = [f for f in required_features if f not in features.columns]
        if missing_features:
            self.logger.error(f"Missing required features: {missing_features}")
            self.logger.info(f"Available features: {features.columns.tolist()}")
            return False
        return True

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features with parallel processing"""
        try:
            self.logger.info(f"Starting feature generation for data shape: {df.shape}")
            features = self.prepare_data(df)

            if features.empty:
                self.logger.warning("No data available for feature generation")
                return pd.DataFrame()

            # Generate features in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                feature_jobs = {
                    executor.submit(self._calculate_price_features, features): 'price',
                    executor.submit(self._calculate_volume_features, features): 'volume',
                    executor.submit(self._calculate_volatility_features, features): 'volatility',
                    executor.submit(self._calculate_trend_features, features): 'trend',
                    executor.submit(self._calculate_momentum_features, features): 'momentum',
                    executor.submit(self._calculate_pattern_features, features): 'pattern'
                }

                # Collect results
                results = {}
                for future in feature_jobs:
                    name = feature_jobs[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[name] = result
                            self.logger.info(f"Generated {name} features")
                    except Exception as e:
                        self.logger.error(f"Error generating {name} features: {str(e)}")

            # Combine all features
            if results:
                # Start with original features
                final_features = features.copy()

                # Add generated features
                for feature_set in results.values():
                    final_features = pd.concat([final_features, feature_set], axis=1)

                # Remove duplicate columns
                final_features = final_features.loc[:, ~final_features.columns.duplicated()]

                # Handle missing values
                final_features = final_features.ffill().bfill()

                # Log available features
                self.logger.info(f"Available features: {final_features.columns.tolist()}")

                # Return only if we have all basic features
                basic_features = ['open', 'high', 'low', 'close', 'volume']
                if self._validate_features(final_features, basic_features):
                    self.logger.info(f"Final features shape: {final_features.shape}")
                    return final_features

            self.logger.error("Feature generation failed")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error in feature generation: {str(e)}")
            return pd.DataFrame()