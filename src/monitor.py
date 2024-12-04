import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time
import sys
from flask import Flask, render_template_string
import threading

from data.database.connection import MongoDBConnection
from config import settings

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Data Monitor</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .card {
            background-color: white;
            border: 1px solid #ddd;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning { color: red; font-weight: bold; }
        .ok { color: green; font-weight: bold; }
        .metric { margin: 10px 0; }
        .metric-label { display: inline-block; width: 200px; font-weight: bold; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading Data Monitor</h1>
        <div class="card">
            <h2>Collection Status</h2>
            {% for key, value in status.items() %}
                <div class="metric">
                    <span class="metric-label">{{ key }}:</span>
                    <span {% if key == 'status' %} class="{{ value.lower() }}" {% endif %}>
                        {{ value }}
                    </span>
                </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
'''

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"monitor_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    return logging.getLogger(__name__)

class DataCollectionMonitor:
    def __init__(self, db: MongoDBConnection):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.app = Flask(__name__)

        @self.app.route('/')
        def home():
            status = self.check_data_collection('BTCUSDT')
            return render_template_string(HTML_TEMPLATE, status=status)

    def check_data_collection(self, symbol: str, hours: int = 24) -> dict:
        """Check data collection status"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=60)  # Match train.py's 60-day window

            # Query for all data
            query = {
                'symbol': symbol,
                'trade_time': {
                    '$gte': start_time,
                    '$lt': end_time
                }
            }

            # Get total records
            total_records = self.db.get_collection('price_data').count_documents(query)
            self.logger.info(f"Total records found in database: {total_records}")

            # Get recent data for rate calculation
            recent_query = {
                'symbol': symbol,
                'trade_time': {
                    '$gte': end_time - timedelta(hours=hours),
                    '$lt': end_time
                }
            }
            recent_data = pd.DataFrame(list(self.db.get_collection('price_data').find(
                recent_query).sort('trade_time', 1)))

            # Get orderbook data
            orderbook_count = self.db.get_collection('order_book').count_documents(recent_query)

            # Calculate data rate from recent data
            if not recent_data.empty:
                data_rate = len(recent_data) / hours
                recent_data['trade_time'] = pd.to_datetime(recent_data['trade_time'])
                last_update = recent_data['trade_time'].max()
                time_since_last = datetime.now() - pd.to_datetime(last_update)
            else:
                data_rate = 0
                last_update = None
                time_since_last = timedelta(hours=hours)

            # Get all price data for OHLCV calculation
            all_price_data = pd.DataFrame(list(self.db.get_collection('price_data').find(
                query).sort('trade_time', 1)))

            # Calculate OHLCV periods
            ohlcv_info = self._get_ohlcv_info(all_price_data)

            status = {
                'Total Records (60d)': total_records,
                'Recent Price Records (24h)': len(recent_data) if not recent_data.empty else 0,
                'Recent Orderbook Records (24h)': orderbook_count,
                'Data Rate (records/hour)': round(data_rate, 2),
                'Last Update': last_update,
                'Time Since Last Update': time_since_last,
                'Total OHLCV Periods': ohlcv_info.get('total_periods', 0),
                'Trainable Periods': ohlcv_info.get('trainable_periods', 0),
                'Data Start': ohlcv_info.get('oldest_data'),
                'Data End': ohlcv_info.get('newest_data'),
                'Status': 'OK' if time_since_last < timedelta(minutes=5) else 'WARNING'
            }

            self.logger.info("\nCollection Status:")
            for key, value in status.items():
                self.logger.info(f"{key}: {value}")

            return status

        except Exception as e:
            self.logger.error(f"Error checking data collection: {e}", exc_info=True)
            return {}

    def _get_ohlcv_info(self, price_data: pd.DataFrame) -> dict:
        """Get OHLCV period information"""
        try:
            if price_data.empty:
                return {}

            # Convert trades to OHLCV
            price_data['trade_time'] = pd.to_datetime(price_data['trade_time'])
            df = price_data.set_index('trade_time')

            # Create OHLCV data with 5-minute intervals
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df['price'].resample('5min').first()
            ohlcv['high'] = df['price'].resample('5min').max()
            ohlcv['low'] = df['price'].resample('5min').min()
            ohlcv['close'] = df['price'].resample('5min').last()
            ohlcv['volume'] = df['quantity'].resample('5min').sum()

            # Remove periods with missing data
            ohlcv = ohlcv.dropna()

            total_periods = len(ohlcv)
            trainable_periods = max(0, total_periods - 100)  # Minimum 100 periods needed

            return {
                'total_periods': total_periods,
                'trainable_periods': trainable_periods,
                'oldest_data': ohlcv.index.min(),
                'newest_data': ohlcv.index.max()
            }

        except Exception as e:
            self.logger.error(f"Error getting OHLCV info: {e}")
            return {}

    def start_web_monitor(self, port: int = 5000):
        """Start web monitoring interface"""
        self.app.run(host='0.0.0.0', port=port, debug=False)

def monitor_loop(interval_seconds: int = 300):
    """Continuous monitoring loop"""
    logger.info("Starting data collection monitor...")

    # Connect to database
    db_config = {
        'connection_string': settings.mongodb_uri,
        'name': settings.db_name
    }
    db = MongoDBConnection(db_config)

    if not db.connect():
        logger.error("Failed to connect to database")
        return

    try:
        monitor = DataCollectionMonitor(db)

        # Start web monitor in a separate thread
        web_thread = threading.Thread(target=monitor.start_web_monitor)
        web_thread.daemon = True
        web_thread.start()

        while True:
            logger.info("\n" + "="*50)
            logger.info(f"Monitoring Check at {datetime.now()}")

            # Check each symbol
            for symbol in settings.trading_symbols:
                status = monitor.check_data_collection(symbol)

                if status.get('Status') == 'WARNING':
                    logger.warning(f"Data collection issues detected for {symbol}")

            logger.info("="*50 + "\n")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Monitor error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    logger = setup_logging()
    monitor_loop()