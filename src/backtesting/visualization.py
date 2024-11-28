import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

@dataclass
class TradePosition:
    """Represents a trade position"""
    entry_time: datetime
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = 'open'

class BacktestEngine:
    """Event-driven backtesting engine"""

    def __init__(self,
                 initial_balance: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage

        # Performance tracking
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = []

    def run_backtest(self,
                     data: pd.DataFrame,
                     strategy,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Dict:
        """Run backtest simulation"""
        try:
            # Reset state
            self.balance = self.initial_balance
            self.positions = []
            self.trades = []
            self.equity_curve = []

            # Filter data by date range
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= data.index >= start_date
            if end_date:
                mask &= data.index <= end_date
            data = data[mask]

            # Run simulation
            for timestamp, candle in data.iterrows():
                # Update positions
                self._update_positions(timestamp, candle)

                # Get strategy signals
                signals = strategy.generate_signals(candle)

                # Execute trades
                self._execute_trades(timestamp, candle, signals)

                # Record equity
                self._record_equity(timestamp)

            # Close all positions at end
            self._close_all_positions(data.index[-1], data.iloc[-1])

            # Calculate performance metrics
            return self._calculate_performance()

        except Exception as e:
            self.logger.error(f"Backtest error: {str(e)}")
            raise

    def _update_positions(self, timestamp: datetime, candle: pd.Series):
        """Update open positions"""
        for position in self.positions:
            if position.status != 'open':
                continue

            # Check stop loss
            if position.stop_loss:
                if (position.side == 'long' and candle['low'] <= position.stop_loss) or \
                        (position.side == 'short' and candle['high'] >= position.stop_loss):
                    self._close_position(position, timestamp, position.stop_loss)
                    continue

            # Check take profit
            if position.take_profit:
                if (position.side == 'long' and candle['high'] >= position.take_profit) or \
                        (position.side == 'short' and candle['low'] <= position.take_profit):
                    self._close_position(position, timestamp, position.take_profit)

    def _execute_trades(self, timestamp: datetime,
                        candle: pd.Series, signals: Dict):
        """Execute trading signals"""
        for signal in signals:
            if signal['action'] == 'buy':
                self._open_position(
                    timestamp,
                    candle['close'] * (1 + self.slippage),
                    signal['size'],
                    'long',
                    signal.get('stop_loss'),
                    signal.get('take_profit')
                )

            elif signal['action'] == 'sell':
                self._open_position(
                    timestamp,
                    candle['close'] * (1 - self.slippage),
                    signal['size'],
                    'short',
                    signal.get('stop_loss'),
                    signal.get('take_profit')
                )

            elif signal['action'] == 'close':
                self._close_all_positions(timestamp, candle)

    def _open_position(self, timestamp: datetime, price: float,
                       size: float, side: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None):
        """Open new position"""
        # Calculate position cost with commission
        cost = price * size * (1 + self.commission)

        if cost > self.balance:
            self.logger.warning(f"Insufficient balance for trade at {timestamp}")
            return

        position = TradePosition(
            entry_time=timestamp,
            entry_price=price,
            size=size,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions.append(position)
        self.balance -= cost

    def _close_position(self, position: TradePosition,
                        timestamp: datetime, price: float):
        """Close existing position"""
        if position.status != 'open':
            return

        # Calculate PnL
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * position.size

        # Subtract commission
        pnl -= price * position.size * self.commission

        # Update position
        position.exit_time = timestamp
        position.exit_price = price
        position.pnl = pnl
        position.status = 'closed'

        # Update balance
        self.balance += (price * position.size) + pnl

        # Record trade
        self.trades.append({
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': price,
            'size': position.size,
            'pnl': pnl
        })

    def _close_all_positions(self, timestamp: datetime, candle: pd.Series):
        """Close all open positions"""
        for position in self.positions:
            if position.status == 'open':
                price = candle['close']
                if position.side == 'long':
                    price *= (1 - self.slippage)
                else:
                    price *= (1 + self.slippage)
                self._close_position(position, timestamp, price)

    def _record_equity(self, timestamp: datetime):
        """Record equity curve point"""
        unrealized_pnl = 0
        for position in self.positions:
            if position.status == 'open':
                # Calculate unrealized PnL
                pass

        equity = self.balance + unrealized_pnl
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })

    def _calculate_performance(self) -> Dict:
        """Calculate backtest performance metrics"""
        try:
            if not self.trades:
                return {}

            df_trades = pd.DataFrame(self.trades)
            df_equity = pd.DataFrame(self.equity_curve)

            # Basic metrics
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['pnl'] > 0])
            total_pnl = df_trades['pnl'].sum()

            # Returns
            returns = df_equity['equity'].pct_change().dropna()

            # Calculate metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'return': (self.balance - self.initial_balance) / self.initial_balance,
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(df_equity['equity']),
                'avg_trade_pnl': df_trades['pnl'].mean(),
                'best_trade': df_trades['pnl'].max(),
                'worst_trade': df_trades['pnl'].min(),
                'avg_trade_duration': (df_trades['exit_time'] -
                                       df_trades['entry_time']).mean()
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Performance calculation error: {str(e)}")
            return {}

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio"""
        if returns.empty:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return drawdown.min()

# src/backtesting/visualization.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List

class BacktestVisualizer:
    """Visualize backtest results"""

    def create_equity_curve(self, equity_data: pd.DataFrame) -> go.Figure:
        """Create equity curve visualization"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_data['timestamp'],
            y=equity_data['equity'],
            mode='lines',
            name='Portfolio Value'
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template='plotly_dark'
        )

        return fig

    def create_trades_analysis(self, trades: List[Dict]) -> go.Figure:
        """Create trades analysis visualization"""
        df = pd.DataFrame(trades)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PnL Distribution', 'Cumulative PnL',
                            'Trade Duration', 'Win/Loss by Hour')
        )

        # PnL Distribution
        fig.add_trace(
            go.Histogram(x=df['pnl'], name='PnL Distribution'),
            row=1, col=1
        )

        # Cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['pnl'].cumsum(),
                mode='lines',
                name='Cumulative PnL'
            ),
            row=1, col=2
        )

        # Trade Duration
        durations = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
        fig.add_trace(
            go.Histogram(x=durations, name='Trade Duration (hours)'),
            row=2, col=1
        )

        # Win/Loss by Hour
        df['hour'] = df['entry_time'].dt.hour
        hourly_wins = df[df['pnl'] > 0]['hour'].value_counts()
        hourly_losses = df[df['pnl'] < 0]['hour'].value_counts()

        fig.add_trace(
            go.Bar(
                x=hourly_wins.index,
                y=hourly_wins.values,
                name='Winning Trades'
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Bar(
                x=hourly_losses.index,
                y=-hourly_losses.values,
                name='Losing Trades'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text='Trading Performance Analysis',
            template='plotly_dark'
        )

        return fig

    def create_drawdown_chart(self, equity_data: pd.DataFrame) -> go.Figure:
        """Create drawdown visualization"""
        fig = go.Figure()

        # Calculate drawdown
        peak = equity_data['equity'].expanding(min_periods=1).max()
        drawdown = (equity_data['equity'] - peak) / peak * 100

        fig.add_trace(go.Scatter(
            x=equity_data['timestamp'],
            y=drawdown,
            fill='tozeroy',
            name='Drawdown'
        ))

        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark'
        )

        return fig