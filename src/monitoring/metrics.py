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
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime

@dataclass
class CollectionMetrics:
    """Data collection performance metrics"""
    trades_per_second: float
    orderbook_updates_per_second: float
    data_latency: float
    processing_time: float
    timestamp: datetime

class PerformanceMonitor:
    """System performance monitoring"""

    def __init__(self, metrics_interval: int = 60):
        self.logger = logging.getLogger(__name__)
        self.metrics_interval = metrics_interval
        self.system_metrics: List[SystemMetrics] = []
        self.collection_metrics: List[CollectionMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()

        # Performance thresholds
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 80.0,
            'disk_high': 80.0,
            'latency_high': 1000.0  # ms
        }

    def start_monitoring(self):
        """Start performance monitoring"""
        def monitor_loop():
            while not self.should_stop.is_set():
                try:
                    # Collect system metrics
                    system_metric = self._collect_system_metrics()
                    self.system_metrics.append(system_metric)

                    # Log metrics
                    self._log_metrics(system_metric)

                    # Check thresholds
                    self._check_thresholds(system_metric)

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
        try:
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
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            return None

    def collect_collection_metrics(self, collector) -> CollectionMetrics:
        """Collect data collection metrics"""
        try:
            now = datetime.now()

            # Calculate rates
            trades_per_second = collector.stats['trades_processed'] / \
                                (now - collector.stats['start_time']).total_seconds() \
                if collector.stats.get('start_time') else 0

            updates_per_second = collector.stats['orderbook_updates'] / \
                                 (now - collector.stats['start_time']).total_seconds() \
                if collector.stats.get('start_time') else 0

            # Calculate latency
            last_update = collector.stats.get('last_update')
            latency = (now - last_update).total_seconds() * 1000 if last_update else 0

            return CollectionMetrics(
                trades_per_second=trades_per_second,
                orderbook_updates_per_second=updates_per_second,
                data_latency=latency,
                processing_time=0.0,  # TODO: Implement processing time tracking
                timestamp=now
            )
        except Exception as e:
            self.logger.error(f"Error collecting collection metrics: {str(e)}")
            return None

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

    def _check_thresholds(self, metrics: SystemMetrics):
        """Check performance thresholds and log warnings"""
        if metrics.cpu_percent > self.thresholds['cpu_high']:
            self.logger.warning(f"High CPU usage: {metrics.cpu_percent}%")

        if metrics.memory_percent > self.thresholds['memory_high']:
            self.logger.warning(f"High memory usage: {metrics.memory_percent}%")

        if metrics.disk_usage > self.thresholds['disk_high']:
            self.logger.warning(f"High disk usage: {metrics.disk_usage}%")

    def _cleanup_old_metrics(self):
        """Clean up metrics older than 24 hours"""
        try:
            cutoff = datetime.now() - timedelta(hours=24)

            self.system_metrics = [m for m in self.system_metrics
                                   if m.timestamp > cutoff]

            self.collection_metrics = [m for m in self.collection_metrics
                                       if m.timestamp > cutoff]
        except Exception as e:
            self.logger.error(f"Metrics cleanup error: {str(e)}")

    def get_metrics_summary(self) -> Dict:
        """Get summary of recent metrics"""
        try:
            recent_system_metrics = self.system_metrics[-60:]  # Last hour
            recent_collection_metrics = self.collection_metrics[-60:]

            if not recent_system_metrics:
                return {}

            return {
                'system': {
                    'cpu': {
                        'current': recent_system_metrics[-1].cpu_percent,
                        'average': np.mean([m.cpu_percent for m in recent_system_metrics])
                    },
                    'memory': {
                        'current': recent_system_metrics[-1].memory_percent,
                        'average': np.mean([m.memory_percent for m in recent_system_metrics])
                    },
                    'disk': recent_system_metrics[-1].disk_usage,
                    'network': recent_system_metrics[-1].network_io
                },
                'collection': {
                    'trades_rate': {
                        'current': recent_collection_metrics[-1].trades_per_second if recent_collection_metrics else 0,
                        'average': np.mean([m.trades_per_second for m in recent_collection_metrics]) if recent_collection_metrics else 0
                    },
                    'updates_rate': {
                        'current': recent_collection_metrics[-1].orderbook_updates_per_second if recent_collection_metrics else 0,
                        'average': np.mean([m.orderbook_updates_per_second for m in recent_collection_metrics]) if recent_collection_metrics else 0
                    },
                    'latency': {
                        'current': recent_collection_metrics[-1].data_latency if recent_collection_metrics else 0,
                        'average': np.mean([m.data_latency for m in recent_collection_metrics]) if recent_collection_metrics else 0
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {str(e)}")
            return {}