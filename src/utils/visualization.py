import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os

class TradingVisualizer:
    """Advanced trading visualization system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.default_height = 800
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'neutral': '#7f7f7f'
        }

    def plot_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           prices: np.ndarray, features: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create comprehensive performance visualization"""
        try:
            figures = {}

            # Ensure all inputs are numpy arrays
            prices = np.asarray(prices)
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Use features index for alignment
            valid_index = features.index[:-1]  # Account for returns being 1 shorter
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = returns * y_pred[:-1]

            # 1. Equity Curve
            figures['equity'] = self._plot_equity_curve(returns, strategy_returns, valid_index)

            # 2. Feature Importance
            figures['features'] = self._plot_feature_importance(features, returns)

            # 3. Rolling Metrics
            figures['metrics'] = self._plot_rolling_metrics(strategy_returns, features.index)

            # 4. Trade Analysis
            figures['trades'] = self._plot_trade_analysis(
                prices, returns, strategy_returns, y_pred, features.index
            )

            # 5. Risk Analysis
            figures['risk'] = self._plot_risk_analysis(returns, strategy_returns, features.index)

            return figures

        except Exception as e:
            self.logger.error(f"Error in plot_model_performance: {e}")
            return {'error': self._create_error_figure(str(e))}

    def _plot_equity_curve(
            self,
            returns: np.ndarray,
            strategy_returns: np.ndarray,
            index: pd.DatetimeIndex
    ) -> go.Figure:
        """Plot cumulative returns comparison"""
        try:
            equity_curve = pd.DataFrame({
                'Buy & Hold': (1 + returns).cumprod(),
                'Strategy': (1 + strategy_returns).cumprod()
            }, index=index)

            fig = go.Figure()
            for col in equity_curve.columns:
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve[col],
                    name=col,
                    mode='lines'
                ))

            fig.update_layout(
                title='Strategy Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                height=self.default_height,
                hovermode='x unified',
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
            return self._create_error_figure("Error plotting equity curve")

    def _plot_feature_importance(
            self,
            features: pd.DataFrame,
            returns: np.ndarray
    ) -> go.Figure:
        """Plot feature importance analysis"""
        try:
            # Calculate correlation-based importance
            importance = {}
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    correlation = np.corrcoef(features[col].values[:-1], returns)[0, 1]
                    importance[col] = abs(correlation) if not np.isnan(correlation) else 0

            # Sort features by importance
            importance = pd.Series(importance).sort_values(ascending=True)

            # Create bar plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance.values,
                y=importance.index,
                orientation='h',
                marker_color=self.color_scheme['primary']
            ))

            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Absolute Correlation with Returns',
                yaxis_title='Feature',
                height=max(400, len(importance) * 20),
                showlegend=False
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
            return self._create_error_figure("Error plotting feature importance")

    def _plot_rolling_metrics(
            self,
            strategy_returns: np.ndarray,
            index: pd.DatetimeIndex,
            window: int = 50
    ) -> go.Figure:
        """Plot rolling performance metrics"""
        try:
            metrics_df = pd.DataFrame(index=index)

            # Calculate rolling metrics
            metrics_df['Win Rate'] = pd.Series(strategy_returns > 0).rolling(window).mean()
            metrics_df['Volatility'] = pd.Series(strategy_returns).rolling(window).std() * np.sqrt(252)
            metrics_df['Sharpe'] = (pd.Series(strategy_returns).rolling(window).mean() * 252) / (
                    pd.Series(strategy_returns).rolling(window).std() * np.sqrt(252)
            )

            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=('Win Rate', 'Volatility', 'Sharpe Ratio')
            )

            # Add traces
            for i, col in enumerate(metrics_df.columns, 1):
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df.index,
                        y=metrics_df[col],
                        name=col,
                        line=dict(color=list(self.color_scheme.values())[i-1])
                    ),
                    row=i, col=1
                )

            fig.update_layout(
                height=900,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting rolling metrics: {e}")
            return self._create_error_figure("Error plotting rolling metrics")

    def _plot_trade_analysis(
            self,
            prices: np.ndarray,
            returns: np.ndarray,
            strategy_returns: np.ndarray,
            predictions: np.ndarray,
            index: pd.DatetimeIndex
    ) -> go.Figure:
        """Plot trade analysis dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Price and Signals',
                    'Return Distribution',
                    'Trade Duration',
                    'Win/Loss Analysis'
                )
            )

            # 1. Price and Signals
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=prices,
                    name='Price',
                    line=dict(color=self.color_scheme['primary'])
                ),
                row=1, col=1
            )

            # Add buy/sell signals
            signals = np.diff((predictions > 0.5).astype(int)) != 0
            signal_dates = index[1:][signals]
            signal_prices = prices[1:][signals]

            fig.add_trace(
                go.Scatter(
                    x=signal_dates,
                    y=signal_prices,
                    mode='markers',
                    name='Trades',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=self.color_scheme['secondary']
                    )
                ),
                row=1, col=1
            )

            # 2. Return Distribution
            fig.add_trace(
                go.Histogram(
                    x=strategy_returns[strategy_returns != 0],
                    name='Strategy Returns',
                    nbinsx=50,
                    marker_color=self.color_scheme['primary']
                ),
                row=1, col=2
            )

            # 3. Trade Duration Analysis
            trade_durations = self._calculate_trade_durations(predictions)
            fig.add_trace(
                go.Histogram(
                    x=trade_durations,
                    name='Trade Duration',
                    nbinsx=30,
                    marker_color=self.color_scheme['secondary']
                ),
                row=2, col=1
            )

            # 4. Win/Loss Analysis
            win_returns = strategy_returns[strategy_returns > 0]
            loss_returns = strategy_returns[strategy_returns < 0]

            fig.add_trace(
                go.Box(
                    y=win_returns,
                    name='Winning Trades',
                    marker_color=self.color_scheme['success']
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Box(
                    y=loss_returns,
                    name='Losing Trades',
                    marker_color=self.color_scheme['danger']
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=1000,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting trade analysis: {e}")
            return self._create_error_figure("Error plotting trade analysis")

    def _plot_risk_analysis(
            self,
            returns: np.ndarray,
            strategy_returns: np.ndarray,
            index: pd.DatetimeIndex
    ) -> go.Figure:
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Drawdown Analysis',
                    'Rolling Volatility',
                    'Value at Risk',
                    'Risk Contribution'
                ),
                specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]]
            )

            # 1. Drawdown Analysis
            drawdown = self._calculate_drawdown(strategy_returns)
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=drawdown,
                    name='Strategy Drawdown',
                    fill='tozeroy',
                    line=dict(color=self.color_scheme['danger'])
                ),
                row=1, col=1
            )

            # 2. Rolling Volatility
            vol_window = min(60, len(returns) // 4)
            rolling_vol = pd.Series(strategy_returns).rolling(vol_window).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=rolling_vol,
                    name='Rolling Volatility',
                    line=dict(color=self.color_scheme['secondary'])
                ),
                row=1, col=2
            )

            # 3. Value at Risk
            var_levels = [0.99, 0.95, 0.90]
            vars = [np.percentile(strategy_returns, (1-p)*100) for p in var_levels]
            fig.add_trace(
                go.Bar(
                    x=[f"{int(p*100)}% VaR" for p in var_levels],
                    y=[-v for v in vars],
                    name='Value at Risk',
                    marker_color=self.color_scheme['primary']
                ),
                row=2, col=1
            )

            # 4. Risk Contribution
            risk_metrics = {
                'Market': np.std(returns),
                'Strategy': np.std(strategy_returns),
                'Active': np.std(strategy_returns - returns)
            }
            fig.add_trace(
                go.Pie(
                    labels=list(risk_metrics.keys()),
                    values=list(risk_metrics.values()),
                    name='Risk Contribution'
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                showlegend=True,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error plotting risk analysis: {e}")
            return self._create_error_figure("Error plotting risk analysis")

    def _calculate_trade_durations(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate durations of trades"""
        try:
            trade_starts = np.where(np.diff((predictions > 0.5).astype(int)) != 0)[0] + 1
            trade_ends = np.append(trade_starts[1:], len(predictions))
            return trade_ends - trade_starts
        except Exception as e:
            self.logger.error(f"Error calculating trade durations: {e}")
            return np.array([])

    def _calculate_drawdown(self, returns: np.ndarray) -> np.ndarray:
        """Calculate drawdown series"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            return (cumulative - running_max) / running_max
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return np.zeros_like(returns)

    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    def save_html_report(
            self,
            figures: Dict[str, go.Figure],
            metrics: Dict[str, float],
            output_path: str
    ) -> None:
        """Save interactive HTML report"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("<html><head>")
                f.write("<title>Trading Strategy Analysis</title>")
                f.write("<style>")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
                f.write(".metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }")
                f.write("</style>")
                f.write("</head><body>")

                # Add metrics summary
                f.write("<h2>Performance Metrics</h2>")
                f.write("<div>")
                for name, value in metrics.items():
                    f.write(f'<div class="metric"><b>{name.replace("_", " ").title()}:</b> {value:.4f}</div>')
                f.write("</div>")

                # Add figures
                for name, fig in figures.items():
                    f.write(f"<h2>{name.replace('_', ' ').title()}</h2>")
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

                f.write("</body></html>")

            self.logger.info(f"HTML report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving HTML report: {e}")

    def save_visualization(
            self,
            figures: Dict[str, go.Figure],
            metrics: Dict[str, float],
            output_path: str
    ) -> None:
        """Save interactive HTML report with figures and metrics"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("<html><head>")
                f.write("<title>Trading Strategy Analysis</title>")
                f.write("<style>")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }")
                f.write(".metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }")
                f.write("</style>")
                f.write("</head><body>")

                # Add metrics summary
                f.write("<h2>Performance Metrics</h2>")
                f.write("<div>")
                for name, value in metrics.items():
                    f.write(f'<div class="metric"><b>{name.replace("_", " ").title()}:</b> {value:.4f}</div>')
                f.write("</div>")

                # Add figures
                for name, fig in figures.items():
                    f.write(f"<h2>{name.replace('_', ' ').title()}</h2>")
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

                f.write("</body></html>")

            self.logger.info(f"HTML report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving HTML report: {e}")