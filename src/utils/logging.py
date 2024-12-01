import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(name: str = 'main', level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create results directory for output
    results_dir = log_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create and setup module logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    return logger

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger instance"""
    return logging.getLogger(name)