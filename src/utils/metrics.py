import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

@dataclass
class TradingMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    trades_per_day: float = 0.0
    avg_trade_return: float = 0.0
    avg_win_trade: float = 0.0
    avg_loss_trade: float = 0.0
    win_loss_ratio: float = 0.0
    kelly_fraction: float = 0.0
    ulcer_index: float = 0.0
    information_ratio: float = 0.0

def calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """Calculate comprehensive trading and ML metrics"""
    try:
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]

        # Get trading positions and ensure alignment
        positions = (y_pred > 0.5).astype(int)
        strategy_returns = price_returns * positions[:-1]  # Positions affect next day's return

        # Calculate trades between positions
        trades = np.diff(positions) != 0

        # Apply transaction costs
        strategy_returns[trades] -= transaction_cost

        # ML metrics (aligned with y_true)
        y_true_aligned = y_true[:-1]  # Match strategy_returns length
        y_pred_aligned = positions[:-1]

        ml_metrics = {
            'accuracy': accuracy_score(y_true_aligned, y_pred_aligned),
            'precision': precision_score(y_true_aligned, y_pred_aligned, zero_division=0),
            'recall': recall_score(y_true_aligned, y_pred_aligned, zero_division=0),
            'f1': f1_score(y_true_aligned, y_pred_aligned, zero_division=0),
            'roc_auc': roc_auc_score(y_true_aligned, y_pred[:-1])  # Use probabilities if available
        }

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * y_pred[:-1]

        # Apply transaction costs
        trades = np.diff((y_pred > 0.5).astype(int)) != 0
        strategy_returns[trades] -= transaction_cost

        # Trading metrics
        trading_metrics = {
            'total_return': calculate_total_return(strategy_returns),
            'annualized_return': calculate_annualized_return(strategy_returns),
            'sharpe_ratio': calculate_sharpe_ratio(strategy_returns, risk_free_rate),
            'sortino_ratio': calculate_sortino_ratio(strategy_returns, risk_free_rate),
            'max_drawdown': calculate_max_drawdown(strategy_returns),
            'win_rate': calculate_win_rate(strategy_returns),
            'profit_factor': calculate_profit_factor(strategy_returns),
            'volatility': calculate_volatility(strategy_returns),
            'calmar_ratio': calculate_calmar_ratio(strategy_returns),
            'trades_per_day': calculate_trades_per_day(trades),
            'avg_trade_return': calculate_avg_trade_return(strategy_returns[trades]),
            'kelly_fraction': calculate_kelly_fraction(strategy_returns),
            'ulcer_index': calculate_ulcer_index(strategy_returns)
        }

        return {**ml_metrics, **trading_metrics}

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}

def calculate_total_return(returns: np.ndarray) -> float:
    """Calculate total return"""
    try:
        return (1 + returns).prod() - 1
    except:
        return 0.0

def calculate_annualized_return(returns: np.ndarray) -> float:
    """Calculate annualized return"""
    try:
        total_return = calculate_total_return(returns)
        years = len(returns) / 252  # Assuming 252 trading days
        return (1 + total_return) ** (1/years) - 1
    except:
        return 0.0

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio"""
    try:
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    except:
        return 0.0

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation"""
    try:
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    except:
        return 0.0

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    try:
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns))
    except:
        return 0.0

def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate of trades"""
    try:
        return np.mean(returns > 0)
    except:
        return 0.0

def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    try:
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / losses
    except:
        return 0.0

def calculate_volatility(returns: np.ndarray) -> float:
    """Calculate annualized volatility"""
    try:
        return np.std(returns) * np.sqrt(252)
    except:
        return 0.0

def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)"""
    try:
        annualized_return = calculate_annualized_return(returns)
        max_dd = calculate_max_drawdown(returns)
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
        return annualized_return / max_dd
    except:
        return 0.0

def calculate_trades_per_day(trades: np.ndarray) -> float:
    """Calculate average number of trades per day"""
    try:
        total_trades = np.sum(trades)
        days = len(trades) / 252
        return total_trades / days if days > 0 else 0.0
    except:
        return 0.0

def calculate_avg_trade_return(trade_returns: np.ndarray) -> float:
    """Calculate average return per trade"""
    try:
        return np.mean(trade_returns) if len(trade_returns) > 0 else 0.0
    except:
        return 0.0

def calculate_kelly_fraction(returns: np.ndarray) -> float:
    """Calculate Kelly fraction for optimal position sizing"""
    try:
        win_prob = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if any(returns < 0) else 0
        if avg_loss == 0:
            return 0.0
        kelly = win_prob - ((1 - win_prob) / (avg_win/avg_loss))
        return np.clip(kelly, 0, 1)  # Limit to 100% max position size
    except:
        return 0.0

def calculate_ulcer_index(returns: np.ndarray) -> float:
    """Calculate Ulcer Index (measure of downside risk)"""
    try:
        cumulative = (1 + returns).cumprod()
        drawdowns = np.maximum.accumulate(cumulative) - cumulative
        squared_drawdowns = (drawdowns / cumulative) ** 2
        return np.sqrt(np.mean(squared_drawdowns))
    except:
        return 0.0

def calculate_information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray
) -> float:
    """Calculate Information Ratio relative to benchmark"""
    try:
        tracking_error = np.std(returns - benchmark_returns)
        if tracking_error == 0:
            return 0.0
        return np.mean(returns - benchmark_returns) / tracking_error * np.sqrt(252)
    except:
        return 0.0

def calculate_risk_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive risk metrics"""
    try:
        return {
            'var_95': calculate_var(returns, 0.95),
            'cvar_95': calculate_cvar(returns, 0.95),
            'volatility': calculate_volatility(returns),
            'downside_deviation': calculate_downside_deviation(returns),
            'max_drawdown': calculate_max_drawdown(returns),
            'ulcer_index': calculate_ulcer_index(returns)
        }
    except:
        return {}

def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Value at Risk"""
    try:
        return np.percentile(returns, (1 - confidence) * 100)
    except:
        return 0.0

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    try:
        var = calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    except:
        return 0.0

def calculate_downside_deviation(returns: np.ndarray) -> float:
    """Calculate downside deviation"""
    try:
        negative_returns = returns[returns < 0]
        return np.std(negative_returns) if len(negative_returns) > 0 else 0.0
    except:
        return 0.0
