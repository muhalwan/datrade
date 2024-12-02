import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys

from data.database.connection import MongoDBConnection
from config import settings

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
        self.figures_dir = Path("monitoring/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def check_data_collection(self, symbol: str, hours: int = 24) -> dict:
        """Check data collection status for the last n hours"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Check price data
            price_query = {
                'symbol': symbol,
                'trade_time': {
                    '$gte': start_time,
                    '$lt': end_time
                }
            }
            price_count = self.db.get_collection('price_data').count_documents(price_query)

            # Check orderbook data
            orderbook_query = {
                'symbol': symbol,
                'timestamp': {
                    '$gte': start_time,
                    '$lt': end_time
                }
            }
            orderbook_count = self.db.get_collection('order_book').count_documents(orderbook_query)

            # Get all price data for OHLCV calculation
            price_data = pd.DataFrame(list(self.db.get_collection('price_data').find(
                {'symbol': symbol}).sort('trade_time', 1)))

            # Calculate OHLCV periods
            ohlcv_data = None
            total_periods = 0
            trainable_periods = 0
            if not price_data.empty:
                price_data['trade_time'] = pd.to_datetime(price_data['trade_time'])

                # Convert to OHLCV
                ohlcv_data = self._convert_trades_to_ohlcv(price_data)
                total_periods = len(ohlcv_data) if ohlcv_data is not None else 0

                # Calculate trainable periods (need at least 100 periods for training)
                trainable_periods = max(0, total_periods - 100)

            # Get data rate and status
            if not price_data.empty:
                price_data['trade_time'] = pd.to_datetime(price_data['trade_time'])
                data_rate = len(price_data) / hours
                last_update = price_data['trade_time'].max()
                time_since_last = datetime.now() - pd.to_datetime(last_update)
            else:
                data_rate = 0
                last_update = None
                time_since_last = timedelta(hours=hours)

            status = {
                'price_records': price_count,
                'orderbook_records': orderbook_count,
                'data_rate_per_hour': data_rate,
                'last_update': last_update,
                'time_since_last': time_since_last,
                'total_ohlcv_periods': total_periods,
                'trainable_periods': trainable_periods,
                'oldest_data': ohlcv_data.index.min() if ohlcv_data is not None else None,
                'newest_data': ohlcv_data.index.max() if ohlcv_data is not None else None,
                'status': 'OK' if time_since_last < timedelta(minutes=5) else 'WARNING'
            }

            self.logger.info("\nCollection Status:")
            for key, value in status.items():
                self.logger.info(f"{key}: {value}")

            return status

        except Exception as e:
            self.logger.error(f"Error checking data collection: {e}")
            return {}

    def _convert_trades_to_ohlcv(self, trades_df: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """Convert trade data to OHLCV format"""
        try:
            if trades_df.empty:
                return None

            # Set index to timestamp
            df = trades_df.set_index('trade_time')

            # Create OHLCV data
            ohlcv = pd.DataFrame()
            ohlcv['open'] = df['price'].resample(timeframe).first()
            ohlcv['high'] = df['price'].resample(timeframe).max()
            ohlcv['low'] = df['price'].resample(timeframe).min()
            ohlcv['close'] = df['price'].resample(timeframe).last()
            ohlcv['volume'] = df['quantity'].resample(timeframe).sum()

            # Remove any rows with NaN values
            ohlcv = ohlcv.dropna()

            return ohlcv

        except Exception as e:
            self.logger.error(f"Error converting trades to OHLCV: {e}")
            return None

    def plot_collection_status(self, symbol: str, hours: int = 24):
        """Plot data collection statistics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Get price data
            price_data = pd.DataFrame(list(self.db.get_collection('price_data').find({
                'symbol': symbol,
                'trade_time': {'$gte': start_time, '$lt': end_time}
            }).sort('trade_time', 1)))

            if price_data.empty:
                self.logger.error("No price data available for plotting")
                return

            price_data['trade_time'] = pd.to_datetime(price_data['trade_time'])
            price_data.set_index('trade_time', inplace=True)

            # Create figure
            fig = make_subplots(rows=3, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=('Price', 'Volume', 'Data Rate'))

            # Price plot
            fig.add_trace(
                go.Scatter(x=price_data.index, y=price_data['price'], name='Price'),
                row=1, col=1
            )

            # Volume plot
            fig.add_trace(
                go.Bar(x=price_data.index, y=price_data['quantity'], name='Volume'),
                row=2, col=1
            )

            # Data rate plot (records per minute)
            data_rate = price_data.resample('1min').count()['price']
            fig.add_trace(
                go.Scatter(x=data_rate.index, y=data_rate, name='Records/min'),
                row=3, col=1
            )

            # Update layout
            fig.update_layout(
                title=f'Data Collection Status - {symbol}',
                height=800,
                showlegend=True
            )

            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.write_html(str(self.figures_dir / f"collection_status_{timestamp}.html"))

        except Exception as e:
            self.logger.error(f"Error plotting collection status: {e}")

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

        while True:
            logger.info("\n" + "="*50)
            logger.info(f"Monitoring Check at {datetime.now()}")

            # Check each symbol
            for symbol in settings.trading_symbols:
                status = monitor.check_data_collection(symbol)

                if status.get('status') == 'WARNING':
                    logger.warning(f"Data collection issues detected for {symbol}")

                # Create plots every hour
                if datetime.now().minute == 0:
                    monitor.plot_collection_status(symbol)

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