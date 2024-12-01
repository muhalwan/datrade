"""Global variables for the trading platform"""

from typing import Optional
from src.data.collector import BinanceDataCollector

# Global collector instance
collector: Optional[BinanceDataCollector] = None

def set_collector(collector_instance: BinanceDataCollector):
    """Set the global collector instance"""
    global collector
    collector = collector_instance

def get_collector() -> Optional[BinanceDataCollector]:
    """Get the global collector instance"""
    global collector
    return collector