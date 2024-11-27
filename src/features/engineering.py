import json
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

    def _get_cache_path(self, symbol: str, start_time: pd.Timestamp,
                        end_time: pd.Timestamp) -> Path:
        """Get cache file path for given parameters"""
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

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

            # Ensure timestamp is index and sorted
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Convert tick data to OHLCV if needed
            if 'price' in df.columns and 'quantity' in df.columns:
                df_ohlcv = self._convert_to_ohlcv(df, freq)
                if df_ohlcv is not None and not df_ohlcv.empty:
                    df = df_ohlcv
                else:
                    self.logger.error("Failed to convert tick data to OHLCV format")
                    return pd.DataFrame()

            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
                return pd.DataFrame()

            # Convert numeric columns and clean
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = self._clean_series(df[col])

            # Remove rows with any NaN values
            df = df.dropna()

            # Convert everything to float32 for efficiency
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
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
        try:
            price_data = df['price'].resample(freq)
            volume_data = df['quantity'].resample(freq)

            df_ohlcv = pd.DataFrame()
            df_ohlcv['open'] = price_data.first()
            df_ohlcv['high'] = price_data.max()
            df_ohlcv['low'] = price_data.min()
            df_ohlcv['close'] = price_data.last()
            df_ohlcv['volume'] = volume_data.sum()

            # Add some basic derived columns
            df_ohlcv['typical_price'] = (df_ohlcv['high'] + df_ohlcv['low'] + df_ohlcv['close']) / 3
            df_ohlcv['price_change'] = df_ohlcv['close'].diff()
            df_ohlcv['returns'] = df_ohlcv['close'].pct_change()

            # Handle NaN values
            df_ohlcv = df_ohlcv.dropna()

            # Verify we got all the data
            if df_ohlcv.empty:
                raise ValueError("No valid OHLCV data after conversion")

            return df_ohlcv

        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {str(e)}")
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # First prepare the data
            df = self.prepare_data(df)
            if df.empty:
                return pd.DataFrame()

            # Create features DataFrame starting with OHLCV data
            features = pd.DataFrame(index=df.index)

            # Keep original OHLCV columns
            features['open'] = df['open']
            features['high'] = df['high']
            features['low'] = df['low']
            features['close'] = df['close']
            features['volume'] = df['volume']

            # Price features
            features['returns'] = df['close'].pct_change(fill_method=None)
            features['log_returns'] = np.log1p(features['returns'].fillna(0))

            # Moving averages
            for window in [7, 14, 21, 50, 200]:
                features[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window).mean()

            # Volatility
            features['volatility_30'] = df['close'].pct_change(fill_method=None).rolling(30).std()

            # Technical indicators
            features['rsi'] = ta.momentum.rsi(close=df['close'], window=14)

            macd = ta.trend.MACD(close=df['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()

            bb = ta.volatility.BollingerBands(close=df['close'])
            features['bb_high_20'] = bb.bollinger_hband()
            features['bb_low_20'] = bb.bollinger_lband()

            # Volume features
            features['volume_sma_30'] = df['volume'].rolling(30).mean()
            features['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )

            # Trend indicators
            features['adx'] = ta.trend.adx(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )

            vortex = ta.trend.VortexIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            features['vortex_pos'] = vortex.vortex_indicator_pos()
            features['vortex_neg'] = vortex.vortex_indicator_neg()

            # Additional indicators
            features['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )

            features['awesome_oscillator'] = ta.momentum.awesome_oscillator(
                high=df['high'],
                low=df['low']
            )

            features['mass_index'] = ta.trend.mass_index(
                high=df['high'],
                low=df['low']
            )

            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()

            features['williams_r'] = ta.momentum.williams_r(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )

            # Fill any remaining NaN values with 0
            features = features.fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
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

            return features.ffill().fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating price features: {str(e)}")
            return pd.DataFrame()

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features with proper error handling"""
        try:
            features = pd.DataFrame(index=df.index)

            # Basic volume features
            features['volume_momentum'] = df['volume'].pct_change()
            features['log_volume'] = np.log1p(df['volume'])

            # Volume SMAs
            for window in [5, 15, 30]:
                features[f'volume_sma_{window}'] = (
                    df['volume'].rolling(window=window, min_periods=1)
                    .mean()
                    .ffill()
                    .fillna(0)
                )

            # Calculate MFI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']

            pos_flow = pd.Series(0, index=df.index)
            neg_flow = pd.Series(0, index=df.index)

            price_diff = typical_price.diff()
            pos_flow[price_diff > 0] = money_flow[price_diff > 0]
            neg_flow[price_diff < 0] = money_flow[price_diff < 0]

            pos_mf = pos_flow.rolling(window=14, min_periods=1).sum()
            neg_mf = neg_flow.rolling(window=14, min_periods=1).sum()

            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
            features['mfi'] = mfi.ffill().fillna(50)

            return features.fillna(0)

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

            return features.ffill().fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return pd.DataFrame()

    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Calculate SMA 20 first as it's needed for crossovers
            sma_20 = ta.trend.sma_indicator(
                df['close'], window=20, fillna=True
            )
            features['sma_20'] = sma_20

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
                    df['close'] / features[f'sma_{window}']).fillna(1) - 1
                features[f'dist_to_ema_{window}'] = (
                    df['close'] / features[f'ema_{window}']).fillna(1) - 1

                # MA Crossovers
                if window > 20:
                    features[f'sma_cross_20_{window}'] = np.where(
                        sma_20 > features[f'sma_{window}'], 1,
                        np.where(sma_20 < features[f'sma_{window}'], -1, 0)
                    )

            # MACD
            macd_ind = ta.trend.MACD(
                close=df['close'],
                window_slow=self.tech_features['macd_slow'],
                window_fast=self.tech_features['macd_fast'],
                window_sign=self.tech_features['macd_signal']
            )
            features['macd'] = macd_ind.macd()
            features['macd_signal'] = macd_ind.macd_signal()
            features['macd_diff'] = macd_ind.macd_diff()
            features['macd_pct'] = features['macd'].fillna(0) / df['close']

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

            return features.fillna(0)

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

            return features.ffill().fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {str(e)}")
            return pd.DataFrame()

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition features"""
        try:
            features = pd.DataFrame(index=df.index)

            # Ensure all inputs are float64 for talib compatibility
            ohlc = df[['open', 'high', 'low', 'close']].astype('float64')

            # Basic candlestick features
            features['body_size'] = (
                    abs(ohlc['close'] - ohlc['open']) / (ohlc['high'] - ohlc['low'])
            ).fillna(0)

            features['upper_shadow'] = (
                    (ohlc['high'] - ohlc[['open', 'close']].max(axis=1)) /
                    (ohlc['high'] - ohlc['low'])
            ).fillna(0)

            features['lower_shadow'] = (
                    (ohlc[['open', 'close']].min(axis=1) - ohlc['low']) /
                    (ohlc['high'] - ohlc['low'])
            ).fillna(0)

            try:
                import talib
                # Candlestick patterns
                pattern_funcs = {
                    'doji': talib.CDLDOJI,
                    'hammer': talib.CDLHAMMER,
                    'shooting_star': talib.CDLSHOOTINGSTAR,
                    'engulfing': talib.CDLENGULFING,
                    'morning_star': talib.CDLMORNINGSTAR,
                    'evening_star': talib.CDLEVENINGSTAR
                }

                for pattern_name, pattern_func in pattern_funcs.items():
                    features[f'pattern_{pattern_name}'] = pattern_func(
                        ohlc['open'].values,
                        ohlc['high'].values,
                        ohlc['low'].values,
                        ohlc['close'].values
                    )

            except ImportError:
                self.logger.warning("ta-lib not installed, skipping pattern features")

            # Other pattern features
            features['inside_bar'] = (
                    (ohlc['high'] <= ohlc['high'].shift(1)) &
                    (ohlc['low'] >= ohlc['low'].shift(1))
            ).astype(int)

            features['outside_bar'] = (
                    (ohlc['high'] >= ohlc['high'].shift(1)) &
                    (ohlc['low'] <= ohlc['low'].shift(1))
            ).astype(int)

            return features.fillna(0)

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