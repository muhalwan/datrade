# src/backtesting/metrics.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: pd.Timedelta
    profit_factor: float
    total_pnl: float
    return_pct: float

@dataclass
class RiskMetrics:
    """Risk and volatility metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: pd.Timedelta
    calmar_ratio: float
    var_95: float
    var_99: float
    volatility: float
    beta: float

class BacktestMetrics:
    """Calculate and analyze backtesting performance metrics"""

    def __init__(self,
                 risk_free_rate: float = 0.02,
                 trading_days: int = 252):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_metrics(self,
                          equity_curve: pd.Series,
                          trades: List[Dict],
                          benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate all performance metrics"""
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)

            # Calculate returns
            returns = equity_curve.pct_change().fillna(0)

            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(trades_df)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                returns, benchmark_returns)

            # Combine metrics
            return {
                'trade_metrics': trade_metrics.__dict__,
                'risk_metrics': risk_metrics.__dict__,
                'additional_metrics': self._calculate_additional_metrics(
                    returns, trades_df)
            }

        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            raise

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> TradeMetrics:
        """Calculate trade-based performance metrics"""
        try:
            if trades.empty:
                return TradeMetrics(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    avg_trade_duration=pd.Timedelta(0),
                    profit_factor=0.0,
                    total_pnl=0.0,
                    return_pct=0.0
                )

            # Basic trade metrics
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]

            total_trades = len(trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)

            # PnL metrics
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            largest_win = winning_trades['pnl'].max() if not winning_trades.empty else 0
            largest_loss = losing_trades['pnl'].min() if not losing_trades.empty else 0

            # Duration metrics
            trades['duration'] = pd.to_datetime(trades['exit_time']) - \
                                 pd.to_datetime(trades['entry_time'])
            avg_duration = trades['duration'].mean()

            # Profit metrics
            total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

            return TradeMetrics(
                total_trades=total_trades,
                winning_trades=win_count,
                losing_trades=loss_count,
                win_rate=win_count / total_trades if total_trades > 0 else 0,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_duration,
                profit_factor=profit_factor,
                total_pnl=trades['pnl'].sum(),
                return_pct=(trades['pnl'].sum() / trades['pnl'].iloc[0]) * 100 \
                    if not trades.empty else 0
            )

        except Exception as e:
            self.logger.error(f"Trade metrics calculation error: {str(e)}")
            raise

    def _calculate_risk_metrics(self,
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series]) -> RiskMetrics:
        """Calculate risk and volatility metrics"""
        try:
            # Annualized metrics
            ann_factor = np.sqrt(self.trading_days)
            excess_returns = returns - self.risk_free_rate / self.trading_days

            # Volatility
            volatility = returns.std() * ann_factor

            # Sharpe Ratio
            sharpe = (excess_returns.mean() * self.trading_days) / volatility \
                if volatility != 0 else 0

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * ann_factor
            sortino = (excess_returns.mean() * self.trading_days) / downside_std \
                if downside_std != 0 else 0

            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Drawdown Duration
            drawdown_series = (cum_returns / rolling_max - 1)
            is_drawdown = drawdown_series < 0
            drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
            drawdown_duration = is_drawdown.groupby(drawdown_start).cumsum()
            max_drawdown_duration = pd.Timedelta(
                seconds=int(drawdown_duration.max() * 24 * 60 * 60))

            # Calmar Ratio
            calmar = -(returns.mean() * self.trading_days) / max_drawdown \
                if max_drawdown != 0 else 0

            # Value at Risk
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)

            # Beta (if benchmark provided)
            beta = 0.0
            if benchmark_returns is not None:
                common_idx = returns.index.intersection(benchmark_returns.index)
                if len(common_idx) > 0:
                    returns_aligned = returns[common_idx]
                    benchmark_aligned = benchmark_returns[common_idx]
                    covariance = returns_aligned.cov(benchmark_aligned)
                    variance = benchmark_aligned.var()
                    beta = covariance / variance if variance != 0 else 0

            return RiskMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                calmar_ratio=calmar,
                var_95=var_95,
                var_99=var_99,
                volatility=volatility,
                beta=beta
            )

        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            raise

    def _calculate_additional_metrics(self,
                                      returns: pd.Series,
                                      trades: pd.DataFrame) -> Dict:
        """Calculate additional performance metrics"""
        try:
            # Monthly returns
            monthly_returns = returns.resample('M').agg(
                lambda x: (1 + x).prod() - 1)

            # Trading activity metrics
            trades_by_month = trades.set_index('entry_time') \
                .resample('M')['pnl'].count()

            # Win streak analysis
            trades['win'] = trades['pnl'] > 0
            win_streak = trades['win'].groupby(
                (trades['win'] != trades['win'].shift()).cumsum()).cumsum()

            return {
                'monthly_returns': monthly_returns.to_dict(),
                'trades_per_month': trades_by_month.mean(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'avg_monthly_return': monthly_returns.mean(),
                'max_win_streak': win_streak.max(),
                'max_loss_streak': (~trades['win']).groupby(
                    (trades['win'] != trades['win'].shift()).cumsum()).cumsum().max()
            }

        except Exception as e:
            self.logger.error(f"Additional metrics calculation error: {str(e)}")
            return {}