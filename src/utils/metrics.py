from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001
) -> Dict[str, float]:
    """
    Calculate comprehensive trading and ML metrics
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values
        prices: Asset prices corresponding to the predictions
        transaction_cost: Trading fee as a fraction (default 0.001 = 0.1%)
    
    Returns:
        Dictionary containing various performance metrics
    """
    try:
        # ML metrics
        ml_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * y_pred[:-1]

        # Apply transaction costs
        trades = np.diff(y_pred) != 0
        strategy_returns[trades] -= transaction_cost

        # Trading metrics
        trading_metrics = {
            'total_return': np.sum(strategy_returns),
            'annualized_return': np.mean(strategy_returns) * 252,
            'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
            'max_drawdown': calculate_max_drawdown(strategy_returns),
            'win_rate': calculate_win_rate(strategy_returns),
            'profit_factor': calculate_profit_factor(strategy_returns),
            'volatility': calculate_volatility(strategy_returns),
            'sortino_ratio': calculate_sortino_ratio(strategy_returns),
            'calmar_ratio': calculate_calmar_ratio(strategy_returns),
            'trades_per_day': calculate_trades_per_day(y_pred)
        }

        return {**ml_metrics, **trading_metrics}

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0

    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return abs(np.min(drawdown))

def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate of trades"""
    if len(returns) == 0:
        return 0.0
    return np.mean(returns > 0)

def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    if len(returns) == 0:
        return 0.0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses

def calculate_volatility(returns: np.ndarray) -> float:
    """Calculate annualized volatility"""
    if len(returns) == 0:
        return 0.0
    return np.std(returns) * np.sqrt(252)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation"""
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    return np.sqrt(252) * np.mean(excess_returns) / downside_std

def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)"""
    if len(returns) == 0:
        return 0.0

    annualized_return = np.mean(returns) * 252
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    return annualized_return / max_dd

def calculate_trades_per_day(predictions: np.ndarray) -> float:
    """Calculate average number of trades per day"""
    if len(predictions) == 0:
        return 0.0

    trades = np.sum(np.diff(predictions) != 0)
    days = len(predictions) / 252  # Assuming 252 trading days per year
    return trades / days if days > 0 else 0.0

def generate_performance_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray,
        timestamps: np.ndarray
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Generate comprehensive performance report
    
    Returns:
        Tuple containing metrics dictionary and detailed performance DataFrame
    """
    # Calculate all metrics
    metrics = calculate_trading_metrics(y_true, y_pred, prices)

    # Create performance DataFrame
    performance = pd.DataFrame({
        'timestamp': timestamps[1:],
        'price': prices[1:],
        'predicted': y_pred[:-1],
        'actual': y_true[:-1],
        'returns': np.diff(prices) / prices[:-1]
    })

    # Add strategy returns
    performance['strategy_returns'] = performance['returns'] * performance['predicted']

    # Add cumulative returns
    performance['cumulative_returns'] = (1 + performance['returns']).cumprod()
    performance['strategy_cumulative'] = (1 + performance['strategy_returns']).cumprod()

    return metrics, performance