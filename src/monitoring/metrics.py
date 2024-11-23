import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import threading
import time
import psutil
import os

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime

@dataclass
class CollectionMetrics:
    trades_per_second: float
    orderbook_updates_per_second: float
    data_latency: float
    processing_time: float
    timestamp: datetime

class PerformanceMonitor:
    """Performance monitoring system"""

    def __init__(self, metrics_interval: int = 60):
        self.logger = logging.getLogger(__name__)
        self.metrics_interval = metrics_interval
        self.system_metrics: List[SystemMetrics] = []
        self.collection_metrics: List[CollectionMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()

    def start_monitoring(self):
        """Start performance monitoring"""
        def monitor_loop():
            while not self.should_stop.is_set():
                try:
                    # Collect metrics
                    system_metric = self._collect_system_metrics()
                    self.system_metrics.append(system_metric)

                    # Log metrics
                    self._log_metrics(system_metric)

                    # Cleanup old metrics
                    self._cleanup_old_metrics()

                    time.sleep(self.metrics_interval)

                except Exception as e:
                    self.logger.error(f"Monitoring error: {str(e)}")
                    time.sleep(5)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.should_stop.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Performance monitoring stopped")

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io={
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            timestamp=datetime.now()
        )

    def _log_metrics(self, metrics: SystemMetrics):
        """Log performance metrics"""
        self.logger.info(
            f"\nPerformance Metrics:"
            f"\nCPU Usage: {metrics.cpu_percent}%"
            f"\nMemory Usage: {metrics.memory_percent}%"
            f"\nDisk Usage: {metrics.disk_usage}%"
            f"\nNetwork IO - Sent: {metrics.network_io['bytes_sent'] / 1024:.2f}KB"
            f"\nNetwork IO - Received: {metrics.network_io['bytes_recv'] / 1024:.2f}KB"
        )

    def _cleanup_old_metrics(self):
        """Clean up metrics older than 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        self.system_metrics = [m for m in self.system_metrics
                               if m.timestamp > cutoff]
        self.collection_metrics = [m for m in self.collection_metrics
                                   if m.timestamp > cutoff]

    def get_metrics_summary(self) -> Dict:
        """Get summary of recent metrics"""
        recent_metrics = self.system_metrics[-60:]  # Last hour
        if not recent_metrics:
            return {}

        return {
            'cpu': {
                'current': recent_metrics[-1].cpu_percent,
                'average': np.mean([m.cpu_percent for m in recent_metrics])
            },
            'memory': {
                'current': recent_metrics[-1].memory_percent,
                'average': np.mean([m.memory_percent for m in recent_metrics])
            },
            'disk': recent_metrics[-1].disk_usage,
            'network': recent_metrics[-1].network_io
        }