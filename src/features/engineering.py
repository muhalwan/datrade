import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import joblib
import os
from pathlib import Path

@dataclass
class FeatureSet:
    """Feature set configuration"""
    name: str
    features: pd.DataFrame
    metadata: Dict = None

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

        # Initialize cache
        self._feature_cache = {}
        self.cache_expiry = timedelta(hours=1)
        self.cache_dir = Path("cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, start_time: datetime, end_time: datetime) -> Path:
        """Get cache file path for given parameters"""
        cache_key = f"{symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load features from cache if valid"""
        try:
            if cache_path.exists():
                cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - cache_time < self.cache_expiry:
                    return joblib.load(cache_path)
            return None
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return None

    def _save_to_cache(self, features: pd.DataFrame, cache_path: Path):
        """Save features to cache"""
        try:
            joblib.dump(features, cache_path)
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")

    def prepare_data(self, df: pd.DataFrame, freq='1min') -> pd.DataFrame:
        """Comprehensive data preparation and cleaning"""
        try:
            df = df.copy()
            self.logger.info(f"Initial data shape: {df.shape}")

            # Ensure timestamp is index and sorted
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Convert tick data to OHLCV if needed
            if 'price' in df.columns and 'quantity' in df.columns:
                df_ohlcv = self._convert_to_ohlcv(df, freq)
                if df_ohlcv is not None:
                    df = df_ohlcv

            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()

            # Clean numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df[col] = self._clean_series(df[col])

            # Convert to float32 for memory efficiency
            for col in numeric_cols:
                df[col] = df[col].astype('float32')

            self.logger.info(f"Prepared data shape: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            return pd.DataFrame()

    def _clean_series(self, series: pd.Series) -> pd.Series:
        """Clean time series data without warnings"""
        if series.isna().any():
            series = series.ffill()
            series = series.bfill()
            series = series.fillna(series.mean())
        return series

    def _convert_to_ohlcv(self, df: pd.DataFrame, freq: str = '1min') -> Optional[pd.DataFrame]:
        """Convert tick data to OHLCV format"""
        try:
            # Create OHLCV dataframe
            ohlc = df['price'].resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            volume = df['quantity'].resample(freq).sum().rename('volume')

            # Combine OHLCV data
            df_ohlcv = pd.concat([ohlc, volume], axis=1)

            # Add additional base price info
            df_ohlcv['typical_price'] = (df_ohlcv['high'] + df_ohlcv['low'] + df_ohlcv['close']) / 3
            df_ohlcv['price_change'] = df_ohlcv['close'].diff()
            df_ohlcv['returns'] = df_ohlcv['close'].pct_change()

            # Add time-based features
            df_ohlcv['hour'] = df_ohlcv.index.hour
            df_ohlcv['minute'] = df_ohlcv.index.minute
            df_ohlcv['day_of_week'] = df_ohlcv.index.dayofweek

            return df_ohlcv

        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {str(e)}")
            return None

    def generate_features(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """Generate all features with parallel processing and caching"""
        try:
            self.logger.info(f"Starting feature generation for data shape: {df.shape}")

            # Prepare data first
            features = self.prepare_data(df)
            if features.empty:
                self.logger.warning("No data available for feature generation")
                return pd.DataFrame()

            # Check cache
            if use_cache:
                cache_path = self._get_cache_path(
                    df.get('symbol', ['UNKNOWN'])[0],
                    df.index[0],
                    df.index[-1]
                )
                cached_features = self._load_from_cache(cache_path)
                if cached_features is not None:
                    self.logger.info("Using cached features")
                    return cached_features

            # Generate features in parallel
            with ThreadPoolExecutor(max_workers=6) as executor:
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
                # Start with base features
                final_features = features.copy()

                # Add generated features
                for feature_set in results.values():
                    final_features = pd.concat([final_features, feature_set], axis=1)

                # Remove duplicate columns
                final_features = final_features.loc[:, ~final_features.columns.duplicated()]

                # Handle missing values
                final_features = final_features.ffill().bfill()

                # Convert to float32 for memory efficiency
                numeric_cols = final_features.select_dtypes(
                    include=['float64', 'int64']
                ).columns
                for col in numeric_cols:
                    final_features[col] = final_features[col].astype('float32')

                # Cache results
                if use_cache:
                    self._save_to_cache(final_features, cache_path)

                self.logger.info(f"Final features shape: {final_features.shape}")
                return final_features

            self.logger.error("No features were generated")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error in feature generation: {str(e)}")
            return pd.DataFrame()

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'].fillna(0))
            features['log_price'] = np.log(df['close'])

            # Price spreads
            features['hl_spread'] = df['high'] - df['low']
            features['oc_spread'] = df['open'] - df['close']
            features['hl_to_oc'] = features['hl_spread'] / features['oc_spread'].abs()

            # Price ratios
            features['close_to_open'] = df['close'] / df['open'] - 1
            features['close_to_high'] = df['close'] / df['high'] - 1
            features['close_to_low'] = df['close'] / df['low'] - 1

            # Momentum and acceleration
            for window in [5, 15, 30]:
                features[f'momentum_{window}'] = df['close'].pct_change(window)
                features[f'log_momentum_{window}'] = np.log(df['close']).diff(window)
                features[f'acceleration_{window}'] = features['returns'].diff(window)

            # Normalized prices
            for window in [5, 15, 30]:
                rolling_mean = df['close'].rolling(window=window).mean()
                rolling_std = df['close'].rolling(window=window).std()
                features[f'normalized_price_{window}'] = (
                                                                 df['close'] - rolling_mean
                                                         ) / rolling_std

            # Price channels
            for window in [5, 15, 30]:
                features[f'upper_channel_{window}'] = df['high'].rolling(window).max()
                features[f'lower_channel_{window}'] = df['low'].rolling(window).min()
                features[f'channel_width_{window}'] = (
                                                              features[f'upper_channel_{window}'] -
                                                              features[f'lower_channel_{window}']
                                                      ) / df['close']

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return pd.DataFrame()

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Basic volume features
            features['volume_momentum'] = df['volume'].pct_change()
            features['log_volume'] = np.log1p(df['volume'])

            # Relative volume
            for window in [5, 15, 30]:
                features[f'relative_volume_{window}'] = (
                        df['volume'] / df['volume'].rolling(window).mean()
                )

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

            # Advanced volume indicators
            features['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=self.tech_features['rsi_window']
            )

            features['obv'] = ta.volume.on_balance_volume(
                close=df['close'],
                volume=df['volume']
            )

            features['force_index'] = ta.volume.force_index(
                close=df['close'],
                volume=df['volume'],
                window=13
            )

            features['ease_of_movement'] = ta.volume.ease_of_movement(
                high=df['high'],
                low=df['low'],
                volume=df['volume'],
                window=14
            )

            # Volume-weighted metrics
            features['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
            features['volume_weighted_volatility'] = (
                                                             df['volume'] * features['returns'].rolling(window=30).std()
                                                     ).rolling(window=30).mean() / df['volume'].rolling(window=30).mean()

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return pd.DataFrame()

    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Basic volatility
            returns = df['close'].pct_change()
            log_returns = np.log(df['close']).diff()

            # Rolling volatility
            for window in [5, 15, 30]:
                features[f'volatility_{window}'] = returns.rolling(window).std()
                features[f'log_volatility_{window}'] = np.log1p(
                    features[f'volatility_{window}']
                )

            # Parkinson volatility
            features['parkinson_volatility'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                np.power(np.log(df['high'] / df['low']), 2)
            )

            # Bollinger bands
            for window in [20, 50]:
                bb = ta.volatility.BollingerBands(
                    close=df['close'],
                    window=window,
                    window_dev=self.tech_features['bb_std'],
                    fillna=True
                )
                features[f'bb_high_{window}'] = bb.bollinger_hband()
                features[f'bb_low_{window}'] = bb.bollinger_lband()
                features[f'bb_middle_{window}'] = bb.bollinger_mavg()
                features[f'bb_width_{window}'] = (
                        (features[f'bb_high_{window}'] - features[f'bb_low_{window}']) /
                        features[f'bb_middle_{window}']
                )
                features[f'bb_percent_{window}'] = bb.bollinger_pband()

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
            features['dc_middle'] = dc.donchian_channel_mband()
            features['dc_width'] = (
                                           features['dc_high'] - features['dc_low']
                                   ) / df['close']
            features['dc_percent'] = (
                                             df['close'] - features['dc_low']
                                     ) / (features['dc_high'] - features['dc_low'])

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
            features['kc_middle'] = kc.keltner_channel_mband()
            features['kc_width'] = (
                                           features['kc_high'] - features['kc_low']
                                   ) / df['close']
            features['kc_percent'] = (
                                             df['close'] - features['kc_low']
                                     ) / (features['kc_high'] - features['kc_low'])

            # Volatility ratio features
            features['true_range'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=1,
                fillna=True
            ).average_true_range()

            features['volatility_ratio'] = (
                    features['true_range'] / df['close']
            ).rolling(window=14).mean()

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return pd.DataFrame()

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
                features[f'dist_to_sma_{window}'] = (
                        df['close'] / features[f'sma_{window}'] - 1
                )
                features[f'dist_to_ema_{window}'] = (
                        df['close'] / features[f'ema_{window}'] - 1
                )

                # MA Crossovers
                if window > 20:
                    features[f'sma_cross_20_{window}'] = np.where(
                        features[f'sma_20'] > features[f'sma_{window}'], 1,
                        np.where(features[f'sma_20'] < features[f'sma_{window}'], -1, 0)
                    )

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
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                fillna=True
            )
            features['adx'] = adx.adx()
            features['adx_pos'] = adx.adx_pos()
            features['adx_neg'] = adx.adx_neg()

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

            # Trend Strength
            features['trix'] = ta.trend.TRIXIndicator(
                close=df['close'], window=14, fillna=True
            ).trix()

            features['cci'] = ta.trend.CCIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=20,
                constant=0.015,
                fillna=True
            ).cci()

            # Detrended Price Oscillator
            features['dpo'] = ta.trend.DPOIndicator(
                close=df['close'], window=20, fillna=True
            ).dpo()

            # Ichimoku Indicators
            ichimoku = ta.trend.IchimokuIndicator(
                high=df['high'],
                low=df['low'],
                window1=9,
                window2=26,
                window3=52,
                fillna=True
            )
            features['ichimoku_a'] = ichimoku.ichimoku_a()
            features['ichimoku_b'] = ichimoku.ichimoku_b()

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating trend features: {str(e)}")
            return pd.DataFrame()

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
            features['stoch_diff'] = features['stoch_k'] - features['stoch_d']

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

            # PPO
            features['ppo'] = ta.momentum.PercentagePriceOscillator(
                close=df['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9,
                fillna=True
            ).ppo()

            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(
                close=df['close'],
                window=14,
                smooth1=3,
                smooth2=3,
                fillna=True
            )
            features['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            features['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {str(e)}")
            return pd.DataFrame()

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Basic candlestick features
            features['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])

            # Candlestick patterns using ta-lib if available
            try:
                import talib

                # Single candlestick patterns
                patterns = {
                    'doji': talib.CDLDOJI,
                    'hammer': talib.CDLHAMMER,
                    'shooting_star': talib.CDLSHOOTINGSTAR,
                    'spinning_top': talib.CDLSPINNINGTOP,
                    'marubozu': talib.CDLMARUBOZU
                }

                # Double candlestick patterns
                patterns.update({
                    'engulfing': talib.CDLENGULFING,
                    'harami': talib.CDLHARAMI,
                    'piercing_line': talib.CDLPIERCING,
                    'dark_cloud_cover': talib.CDLDARKCLOUDCOVER
                })

                # Triple candlestick patterns
                patterns.update({
                    'morning_star': talib.CDLMORNINGSTAR,
                    'evening_star': talib.CDLEVENINGSTAR,
                    'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                    'three_black_crows': talib.CDL3BLACKCROWS
                })

                for pattern_name, pattern_func in patterns.items():
                    features[f'pattern_{pattern_name}'] = pattern_func(
                        df['open'].values,
                        df['high'].values,
                        df['low'].values,
                        df['close'].values
                    )

            except ImportError:
                self.logger.warning("ta-lib not installed, using basic patterns only")
                # Calculate basic patterns without ta-lib
                features['is_doji'] = (
                    (abs(df['close'] - df['open']) <= 0.1 * (df['high'] - df['low']))
                ).astype(int)

                features['is_hammer'] = (
                        (features['lower_shadow'] > 2 * features['body_size']) &
                        (features['upper_shadow'] < 0.2)
                ).astype(int)

            # Other pattern features
            features['inside_bar'] = (
                    (df['high'] <= df['high'].shift(1)) &
                    (df['low'] >= df['low'].shift(1))
            ).astype(int)

            features['outside_bar'] = (
                    (df['high'] >= df['high'].shift(1)) &
                    (df['low'] <= df['low'].shift(1))
            ).astype(int)

            # Gap features
            features['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            features['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)

            return features.fillna(method='ffill').fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating pattern features: {str(e)}")
            return pd.DataFrame()

    def optimize_features(self, df: pd.DataFrame, target_col: str = 'close',
                          correlation_threshold: float = 0.95,
                          importance_threshold: float = 0.1) -> List[str]:
        """Optimize feature set and remove highly correlated features"""
        try:
            # Generate all features
            features_df = self.generate_features(df)
            if features_df.empty:
                return []

            # Calculate correlations with target
            correlations = features_df.corrwith(features_df[target_col]).abs()

            # Sort features by importance
            sorted_features = correlations.sort_values(ascending=False)

            # Select features
            selected_features = []
            for feature in sorted_features.index:
                if feature == target_col:
                    continue

                # Check importance threshold
                if correlations[feature] < importance_threshold:
                    continue

                # Check correlation with already selected features
                if selected_features:
                    corr_matrix = features_df[selected_features + [feature]].corr()
                    if (corr_matrix[feature].abs() > correlation_threshold).any():
                        continue

                selected_features.append(feature)

            self.logger.info(
                f"Selected {len(selected_features)} features from "
                f"{len(features_df.columns)} original features"
            )
            return selected_features

        except Exception as e:
            self.logger.error(f"Error optimizing features: {str(e)}")
            return []

    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive feature statistics"""
        try:
            features = self.generate_features(df)
            if features.empty:
                return {}

            stats = {
                'feature_count': len(features.columns),
                'groups': {
                    'price': len([col for col in features.columns if 'price' in col]),
                    'volume': len([col for col in features.columns if 'volume' in col]),
                    'volatility': len([col for col in features.columns if 'volatility' in col]),
                    'trend': len([col for col in features.columns if 'trend' in col]),
                    'momentum': len([col for col in features.columns if 'momentum' in col]),
                    'pattern': len([col for col in features.columns if 'pattern' in col])
                },
                'null_counts': features.isnull().sum().to_dict(),
                'memory_usage': features.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'correlation_stats': {
                    'high_correlation_pairs': self._get_high_correlation_pairs(features),
                    'target_correlations': features.corrwith(features['close']).sort_values(ascending=False).to_dict()
                }
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting feature stats: {str(e)}")
            return {}

    def _get_high_correlation_pairs(self, df: pd.DataFrame,
                                    threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Get highly correlated feature pairs"""
        corr_matrix = df.corr()
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        return high_corr

    def save_feature_metadata(self, path: str = "metadata/features"):
        """Save feature engineering metadata"""
        try:
            metadata_dir = Path(path)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                'technical_params': self.tech_features,
                'feature_groups': {
                    'price': self._get_price_feature_names(),
                    'volume': self._get_volume_feature_names(),
                    'volatility': self._get_volatility_feature_names(),
                    'trend': self._get_trend_feature_names(),
                    'momentum': self._get_momentum_feature_names(),
                    'pattern': self._get_pattern_feature_names()
                },
                'cache_settings': {
                    'expiry_hours': self.cache_expiry.total_seconds() / 3600
                }
            }

            with open(metadata_dir / 'feature_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            self.logger.error(f"Error saving feature metadata: {str(e)}")

    def _get_price_feature_names(self) -> List[str]:
        """Get list of price feature names"""
        base_features = [
            'returns', 'log_returns', 'log_price', 'hl_spread',
            'oc_spread', 'hl_to_oc', 'close_to_open', 'close_to_high',
            'close_to_low'
        ]

        window_features = []
        for window in [5, 15, 30]:
            window_features.extend([
                f'momentum_{window}',
                f'log_momentum_{window}',
                f'acceleration_{window}',
                f'normalized_price_{window}',
                f'upper_channel_{window}',
                f'lower_channel_{window}',
                f'channel_width_{window}'
            ])

        return base_features + window_features

    def _get_volume_feature_names(self) -> List[str]:
        """Get list of volume feature names"""
        base_features = [
            'volume_momentum', 'log_volume', 'mfi', 'obv', 'force_index',
            'ease_of_movement', 'vwap', 'volume_weighted_volatility'
        ]

        window_features = []
        for window in [5, 15, 30]:
            window_features.extend([
                f'relative_volume_{window}',
                f'volume_sma_{window}',
                f'volume_std_{window}',
                f'volume_zscore_{window}',
                f'volume_price_corr_{window}'
            ])

        return base_features + window_features

    def _get_volatility_feature_names(self) -> List[str]:
        """Get list of volatility feature names"""
        base_features = [
            'parkinson_volatility', 'atr', 'atr_pct', 'true_range',
            'volatility_ratio'
        ]

        window_features = []
        for window in [5, 15, 30]:
            window_features.extend([
                f'volatility_{window}',
                f'log_volatility_{window}'
            ])

        for window in [20, 50]:
            window_features.extend([
                f'bb_high_{window}',
                f'bb_low_{window}',
                f'bb_middle_{window}',
                f'bb_width_{window}',
                f'bb_percent_{window}'
            ])

        channel_features = [
            'dc_high', 'dc_low', 'dc_middle', 'dc_width', 'dc_percent',
            'kc_high', 'kc_low', 'kc_middle', 'kc_width', 'kc_percent'
        ]

        return base_features + window_features + channel_features

    def _get_trend_feature_names(self) -> List[str]:
        """Get list of trend feature names"""
        ma_features = []
        for window in self.tech_features['sma']:
            ma_features.extend([
                f'sma_{window}',
                f'ema_{window}',
                f'dist_to_sma_{window}',
                f'dist_to_ema_{window}'
            ])
            if window > 20:
                ma_features.append(f'sma_cross_20_{window}')

        macd_features = ['macd', 'macd_signal', 'macd_diff', 'macd_pct']

        adx_features = ['adx', 'adx_pos', 'adx_neg']

        other_features = [
            'mass_index', 'vortex_pos', 'vortex_neg', 'vortex_diff',
            'trix', 'cci', 'dpo', 'ichimoku_a', 'ichimoku_b'
        ]

        return ma_features + macd_features + adx_features + other_features

    def _get_momentum_feature_names(self) -> List[str]:
        """Get list of momentum feature names"""
        base_features = [
            'rsi', 'stoch_k', 'stoch_d', 'stoch_diff', 'williams_r',
            'ultimate_oscillator', 'awesome_oscillator', 'ppo',
            'stoch_rsi_k', 'stoch_rsi_d'
        ]

        roc_features = [f'roc_{window}' for window in [5, 10, 20]]

        return base_features + roc_features

    def _get_pattern_feature_names(self) -> List[str]:
        """Get list of pattern feature names"""
        base_features = [
            'body_size', 'upper_shadow', 'lower_shadow', 'is_doji',
            'is_hammer', 'inside_bar', 'outside_bar', 'gap_up', 'gap_down'
        ]

        pattern_prefixes = [
            'doji', 'hammer', 'shooting_star', 'spinning_top', 'marubozu',
            'engulfing', 'harami', 'piercing_line', 'dark_cloud_cover',
            'morning_star', 'evening_star', 'three_white_soldiers',
            'three_black_crows'
        ]
        pattern_features = [f'pattern_{pattern}' for pattern in pattern_prefixes]

        return base_features + pattern_features