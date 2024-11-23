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

from src.utils.logging import LoggerConfig

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime

@dataclass
class DataMetrics:
    trades_per_second: float
    orderbook_updates_per_second: float
    data_latency: float
    processing_time: float
    timestamp: datetime

@dataclass
class ModelMetrics:
    prediction_accuracy: float
    inference_time: float
    feature_generation_time: float
    timestamp: datetime

class PerformanceMonitor:
    """Performance monitoring system"""

    def __init__(self, metrics_interval: int = 60):
        self.logger = LoggerConfig.get_logger(__name__)
        self.metrics_interval = metrics_interval
        self.system_metrics: List[SystemMetrics] = []
        self.data_metrics: List[DataMetrics] = []
        self.model_metrics: List[ModelMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()

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

                    # Cleanup old metrics
                    self._cleanup_old_metrics()

                    time.sleep(self.metrics_interval)

                except Exception as e:
                    self.logger.error(f"Error in monitoring: {str(e)}")
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

    def record_data_metrics(self, trades_count: int, orderbook_count: int,
                            latency: float, processing_time: float):
        """Record data collection metrics"""
        metric = DataMetrics(
            trades_per_second=trades_count / self.metrics_interval,
            orderbook_updates_per_second=orderbook_count / self.metrics_interval,
            data_latency=latency,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        self.data_metrics.append(metric)

    def record_model_metrics(self, accuracy: float, inference_time: float,
                             feature_time: float):
        """Record model performance metrics"""
        metric = ModelMetrics(
            prediction_accuracy=accuracy,
            inference_time=inference_time,
            feature_generation_time=feature_time,
            timestamp=datetime.now()
        )
        self.model_metrics.append(metric)

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

        # Log data metrics if available
        if self.data_metrics:
            latest_data = self.data_metrics[-1]
            self.logger.info(
                f"\nData Collection Metrics:"
                f"\nTrades/sec: {latest_data.trades_per_second:.2f}"
                f"\nOrderbook Updates/sec: {latest_data.orderbook_updates_per_second:.2f}"
                f"\nLatency: {latest_data.data_latency:.3f}ms"
                f"\nProcessing Time: {latest_data.processing_time:.3f}ms"
            )

        # Log model metrics if available
        if self.model_metrics:
            latest_model = self.model_metrics[-1]
            self.logger.info(
                f"\nModel Performance Metrics:"
                f"\nPrediction Accuracy: {latest_model.prediction_accuracy:.2f}%"
                f"\nInference Time: {latest_model.inference_time:.3f}ms"
                f"\nFeature Generation Time: {latest_model.feature_generation_time:.3f}ms"
            )

    def _cleanup_old_metrics(self):
        """Clean up metrics older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        self.system_metrics = [m for m in self.system_metrics
                               if m.timestamp > cutoff_time]
        self.data_metrics = [m for m in self.data_metrics
                             if m.timestamp > cutoff_time]
        self.model_metrics = [m for m in self.model_metrics
                              if m.timestamp > cutoff_time]

    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Filter recent metrics
        recent_system = [m for m in self.system_metrics if m.timestamp > hour_ago]
        recent_data = [m for m in self.data_metrics if m.timestamp > hour_ago]
        recent_model = [m for m in self.model_metrics if m.timestamp > hour_ago]

        return {
            'system': {
                'avg_cpu': np.mean([m.cpu_percent for m in recent_system]),
                'avg_memory': np.mean([m.memory_percent for m in recent_system]),
                'avg_disk': np.mean([m.disk_usage for m in recent_system]),
            },
            'data': {
                'avg_trades_per_sec': np.mean([m.trades_per_second for m in recent_data]) if recent_data else 0,
                'avg_latency': np.mean([m.data_latency for m in recent_data]) if recent_data else 0,
                'total_trades': sum([m.trades_per_second * self.metrics_interval for m in recent_data]) if recent_data else 0,
            },
            'model': {
                'avg_accuracy': np.mean([m.prediction_accuracy for m in recent_model]) if recent_model else 0,
                'avg_inference_time': np.mean([m.inference_time for m in recent_model]) if recent_model else 0,
            }
        }

    def save_metrics(self, filepath: str):
        """Save metrics to CSV file"""
        try:
            # Convert metrics to DataFrames
            system_df = pd.DataFrame([vars(m) for m in self.system_metrics])
            data_df = pd.DataFrame([vars(m) for m in self.data_metrics])
            model_df = pd.DataFrame([vars(m) for m in self.model_metrics])

            # Save to CSV
            system_df.to_csv(f"{filepath}/system_metrics.csv", index=False)
            data_df.to_csv(f"{filepath}/data_metrics.csv", index=False)
            model_df.to_csv(f"{filepath}/model_metrics.csv", index=False)

            self.logger.info(f"Metrics saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")