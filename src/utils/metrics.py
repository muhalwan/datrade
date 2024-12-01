import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict

def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray, price_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate both ML and trading performance metrics"""

    metrics = {
        # ML metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),

        # Trading metrics
        'sharpe_ratio': calculate_sharpe_ratio(y_pred, price_data),
        'max_drawdown': calculate_max_drawdown(y_pred, price_data),
        'win_rate': calculate_win_rate(y_pred, price_data),
        'profit_factor': calculate_profit_factor(y_pred, price_data)
    }

    return metrics

def calculate_sharpe_ratio(predictions: np.ndarray, price_data: pd.DataFrame) -> float:
    """Calculate Sharpe ratio of strategy"""
    returns = calculate_strategy_returns(predictions, price_data)
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return np.sqrt(252) * (returns.mean() / returns.std())

def calculate_max_drawdown(predictions: np.ndarray, price_data: pd.DataFrame) -> float:
    """Calculate maximum drawdown of strategy"""
    returns = calculate_strategy_returns(predictions, price_data)
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1
    return abs(min(drawdown))

def calculate_win_rate(predictions: np.ndarray, price_data: pd.DataFrame) -> float:
    """Calculate win rate of trades"""
    returns = calculate_strategy_returns(predictions, price_data)
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()

def calculate_profit_factor(predictions: np.ndarray, price_data: pd.DataFrame) -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    returns = calculate_strategy_returns(predictions, price_data)
    if len(returns) == 0:
        return 0.0

    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    return profits / losses

def calculate_strategy_returns(predictions: np.ndarray, price_data: pd.DataFrame) -> pd.Series:
    """Calculate strategy returns based on predictions"""
    price_returns = price_data['close'].pct_change()
    return predictions[:-1] * price_returns[1:]