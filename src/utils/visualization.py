import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_price_and_signals(price_data: pd.DataFrame, signals: pd.Series) -> go.Figure:
    """Plot price chart with trading signals"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Price candlesticks
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

    # Trading signals
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals * price_data['high'].max(),
            mode='markers',
            name='Signals',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red'
            )
        ),
        row=1, col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name='Volume'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title='Price Chart with Trading Signals',
        xaxis_rangeslider_visible=False
    )

    return fig

def plot_model_metrics(metrics_history: pd.DataFrame) -> go.Figure:
    """Plot model performance metrics over time"""
    fig = go.Figure()

    for column in metrics_history.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics_history.index,
                y=metrics_history[column],
                name=column
            )
        )

    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Date',
        yaxis_title='Value'
    )

    return fig