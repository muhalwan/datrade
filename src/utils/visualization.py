import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import plotly.express as px

class TradingVisualizer:
    """Visualization tools for trading analysis"""

    @staticmethod
    def plot_price_with_signals(
            price_data: pd.DataFrame,
            predictions: np.ndarray,
            features: Optional[pd.DataFrame] = None,
            window_size: int = 100
    ) -> go.Figure:
        """Plot price chart with trading signals and indicators"""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Price and signals
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add trading signals
        signal_dates = price_data.index[predictions == 1]
        if len(signal_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=signal_dates,
                    y=price_data.loc[signal_dates, 'high'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='green'
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )

        # Add volume
        fig.add_trace(
            go.Bar(
                x=price_data.index,
                y=price_data['volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        # Add indicators if available
        if features is not None:
            if 'rsi' in features.columns:
                fig.add_trace(
                    go.Scatter(
                        x=features.index,
                        y=features['rsi'],
                        name='RSI'
                    ),
                    row=3, col=1
                )

            if 'macd' in features.columns:
                fig.add_trace(
                    go.Scatter(
                        x=features.index,
                        y=features['macd'],
                        name='MACD'
                    ),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            title='Trading Signals and Indicators',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    @staticmethod
    def plot_model_performance(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            prices: pd.Series,
            features: pd.DataFrame
    ) -> Dict[str, go.Figure]:
        """Create comprehensive performance visualization"""
        figures = {}

        try:
            # Convert to numpy arrays to avoid Series ambiguity
            if isinstance(prices, pd.Series):
                prices = prices.values

            # Ensure all inputs are numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Calculate returns properly
            returns = np.diff(prices) / prices[:-1]

            # Calculate strategy returns using numpy
            strategy_returns = returns * y_pred[:-1]  # Align lengths

            # Create equity curves - Use features.index[1:] to match returns length
            index = features.index[1:len(returns)+1]  # Align index with returns
            equity_curve = pd.DataFrame({
                'Buy & Hold': (1 + returns).cumprod(),
                'Strategy': (1 + strategy_returns).cumprod()
            }, index=index)

            figures['equity'] = px.line(
                equity_curve,
                title='Strategy Performance',
                labels={'value': 'Cumulative Return', 'variable': 'Strategy'}
            )

            # Feature importance - align features with returns
            feature_importance = {}
            features_aligned = features.iloc[1:len(returns)+1]  # Skip first row and align with returns
            for col in features_aligned:
                # Check dtype of individual series/column
                if pd.api.types.is_numeric_dtype(features_aligned[col]):
                    # Ensure same length for correlation
                    correlation = np.corrcoef(features_aligned[col].values, returns)[0, 1]
                    feature_importance[col] = abs(correlation) if not np.isnan(correlation) else 0

            feature_importance = pd.Series(feature_importance).sort_values(ascending=True)

            figures['features'] = px.bar(
                x=feature_importance.values,
                y=feature_importance.index,
                orientation='h',
                title='Feature Importance (Correlation with Returns)'
            )

            # Calculate rolling metrics using aligned data
            window = min(len(returns) // 10, 50)
            rolling_metrics = pd.DataFrame(index=index)

            # Win rate calculation
            rolling_metrics['Win Rate'] = pd.Series(strategy_returns > 0, index=index).rolling(window).mean()

            # Sharpe ratio calculation (safe division)
            def rolling_sharpe(x):
                if len(x) < 2:
                    return 0
                std = np.std(x)
                if std == 0:
                    return 0
                return np.sqrt(252) * np.mean(x) / std

            rolling_metrics['Sharpe'] = pd.Series(strategy_returns, index=index).rolling(window).apply(rolling_sharpe)

            figures['metrics'] = px.line(
                rolling_metrics,
                title=f'Rolling Metrics ({window} periods)',
                labels={'value': 'Metric Value', 'variable': 'Metric'}
            )

            # Trade distribution (non-zero returns only)
            trade_returns = strategy_returns[strategy_returns != 0]
            if len(trade_returns) > 0:
                figures['distribution'] = px.histogram(
                    trade_returns,
                    title='Trade Return Distribution',
                    labels={'value': 'Return', 'count': 'Frequency'},
                    nbins=50
                )

            # Drawdown calculation using aligned data
            cum_returns = (1 + strategy_returns).cumprod()
            rolling_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - rolling_max) / rolling_max

            figures['drawdown'] = px.line(
                pd.Series(drawdown, index=index),
                title='Strategy Drawdown',
                labels={'value': 'Drawdown', 'index': 'Date'}
            )

            return figures

        except Exception as e:
            logging.error(f"Error in plot_model_performance: {e}")
            return {'error': go.Figure().add_annotation(
                text=f"Error generating plots: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper"
            )}

    @staticmethod
    def plot_feature_analysis(features: pd.DataFrame, returns: pd.Series) -> Dict[str, go.Figure]:
        """Create feature analysis visualizations"""
        figures = {}

        # 1. Feature correlations
        corr_matrix = features.corr()
        figures['correlations'] = px.imshow(
            corr_matrix,
            title='Feature Correlations',
            color_continuous_scale='RdBu'
        )

        # 2. Feature distributions
        for col in features.columns[:5]:  # Limit to first 5 features
            figures[f'dist_{col}'] = px.histogram(
                features[col].dropna(),
                title=f'{col} Distribution',
                nbins=50
            )

        # 3. Feature vs Returns scatter plots
        for col in features.columns[:5]:
            figures[f'scatter_{col}'] = px.scatter(
                x=features[col],
                y=returns,
                title=f'{col} vs Returns',
                trendline='ols'
            )

        return figures

    @staticmethod
    def plot_orderbook_heatmap(
            orderbook_data: pd.DataFrame,
            price_data: pd.DataFrame,
            timestamp: datetime
    ) -> go.Figure:
        """Create orderbook heatmap visualization"""
        # Filter data for the specific timestamp
        current_price = price_data.loc[price_data.index <= timestamp, 'close'].iloc[-1]
        window_data = orderbook_data[
            (orderbook_data['timestamp'] >= timestamp - pd.Timedelta(minutes=5)) &
            (orderbook_data['timestamp'] <= timestamp)
            ]

        # Separate bids and asks
        bids = window_data[window_data['side'] == 'bid']
        asks = window_data[window_data['side'] == 'ask']

        fig = go.Figure()

        # Add bid volumes
        fig.add_trace(go.Bar(
            x=bids['price'],
            y=bids['quantity'],
            name='Bids',
            marker_color='green',
            opacity=0.5
        ))

        # Add ask volumes
        fig.add_trace(go.Bar(
            x=asks['price'],
            y=asks['quantity'],
            name='Asks',
            marker_color='red',
            opacity=0.5
        ))

        # Add current price line
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="blue",
            annotation_text="Current Price"
        )

        # Update layout for orderbook heatmap
        fig.update_layout(
            title=f'Order Book Depth at {timestamp}',
            xaxis_title='Price',
            yaxis_title='Quantity',
            barmode='overlay',
            showlegend=True,
            height=600,
            bargap=0
        )

        return fig

    @staticmethod
    def create_dashboard(
            price_data: pd.DataFrame,
            predictions: np.ndarray,
            features: pd.DataFrame,
            metrics: Dict[str, float],
            orderbook_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, go.Figure]:
        """Create comprehensive trading dashboard"""
        viz = TradingVisualizer()
        dashboard = {}

        # 1. Main trading chart with signals
        dashboard['main_chart'] = viz.plot_price_with_signals(
            price_data,
            predictions,
            features
        )

        # 2. Performance metrics
        returns = pd.Series(np.diff(price_data['close']) / price_data['close'][:-1])
        strategy_returns = returns * predictions[:-1]

        dashboard['performance'] = viz.plot_model_performance(
            predictions[:-1],
            predictions[:-1],
            price_data['close'],
            features
        )

        # 3. Feature analysis
        dashboard['features'] = viz.plot_feature_analysis(
            features,
            returns
        )

        # 4. Metrics summary
        metrics_fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(metrics.keys()),
                        [f"{v:.4f}" for v in metrics.values()]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )
        ])

        metrics_fig.update_layout(
            title='Performance Metrics',
            height=400
        )

        dashboard['metrics'] = metrics_fig

        # 5. Order book analysis (if available)
        if orderbook_data is not None:
            latest_timestamp = price_data.index[-1]
            dashboard['orderbook'] = viz.plot_orderbook_heatmap(
                orderbook_data,
                price_data,
                latest_timestamp
            )

        return dashboard

    @staticmethod
    def plot_backtest_results(
            portfolio_value: pd.Series,
            trades: List[Dict],
            benchmark: Optional[pd.Series] = None
    ) -> Dict[str, go.Figure]:
        """Create backtest analysis visualizations"""
        figures = {}

        # 1. Portfolio value over time
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            name='Strategy',
            line=dict(color='blue')
        ))

        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                name='Benchmark',
                line=dict(color='gray', dash='dash')
            ))

        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Value',
            height=600
        )

        figures['portfolio'] = fig

        # 2. Trade analysis
        if trades:
            trade_data = pd.DataFrame(trades)

            # Trade sizes
            figures['trade_sizes'] = px.histogram(
                trade_data['size'],
                title='Trade Size Distribution',
                labels={'value': 'Trade Size', 'count': 'Frequency'}
            )

            # Trade returns
            if 'return' in trade_data.columns:
                figures['trade_returns'] = px.histogram(
                    trade_data['return'],
                    title='Trade Return Distribution',
                    labels={'value': 'Return (%)', 'count': 'Frequency'}
                )

            # Trade durations
            if 'duration' in trade_data.columns:
                figures['trade_durations'] = px.histogram(
                    trade_data['duration'],
                    title='Trade Duration Distribution',
                    labels={'value': 'Duration (hours)', 'count': 'Frequency'}
                )

        return figures

    @staticmethod
    def create_pdf_report(
            figures: Dict[str, go.Figure],
            metrics: Dict[str, float],
            output_path: str
    ) -> None:
        """Generate PDF report with all visualizations and metrics"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Add title
            story.append(Paragraph("Trading Strategy Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))

            # Add metrics table
            metrics_data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
            metrics_table = Table([["Metric", "Value"]] + metrics_data)
            metrics_table.setStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            story.append(metrics_table)
            story.append(Spacer(1, 12))

            # Add figures
            for name, fig in figures.items():
                # Save figure as temporary image
                img_path = f"temp_{name}.png"
                fig.write_image(img_path)

                # Add to report
                story.append(Paragraph(name.replace('_', ' ').title(), styles['Heading2']))
                story.append(Image(img_path, width=500, height=300))
                story.append(Spacer(1, 12))

            # Generate PDF
            doc.build(story)

        except Exception as e:
            print(f"Error generating PDF report: {e}")

    @staticmethod
    def save_interactive_html(
            figures: Dict[str, go.Figure],
            output_path: str
    ) -> None:
        """Save interactive HTML dashboard"""
        try:
            with open(output_path, 'w') as f:
                f.write("<html><head><title>Trading Analysis Dashboard</title></head><body>")

                for name, fig in figures.items():
                    f.write(f"<h2>{name.replace('_', ' ').title()}</h2>")
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

                f.write("</body></html>")

        except Exception as e:
            print(f"Error saving interactive HTML: {e}")