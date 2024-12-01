from pydantic import BaseSettings, Field
from typing import List, Dict, Optional
import yaml
import os
from pathlib import Path

class Settings(BaseSettings):
    # MongoDB settings
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        env='MONGODB_URI'
    )
    db_name: str = Field(
        default="crypto_trading",
        env='DB_NAME'
    )

    # Binance settings
    binance_api_key: str = Field(
        default="test_api_key",
        env='BINANCE_API_KEY'
    )
    binance_secret_key: str = Field(
        default="test_secret_key",
        env='BINANCE_SECRET_KEY'
    )
    use_testnet: bool = Field(default=True, env='USE_TESTNET')
    trading_symbols: List[str] = Field(default=["BTCUSDT"])

    # Data Collection settings
    batch_size: int = Field(default=100)
    history_size: int = Field(default=1000)
    update_interval: int = Field(default=1)  # seconds

    # Monitoring settings
    monitoring_host: str = Field(default="0.0.0.0")
    monitoring_port: int = Field(default=8000)
    metrics_interval: int = Field(default=60)  # seconds

    class Config:
        env_file = '.env'

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Load settings from YAML file"""
        try:
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)

            return cls(
                mongodb_uri=config_data['database']['connection_string'],
                db_name=config_data['database']['name'],
                binance_api_key=config_data['exchange']['api_key'],
                binance_secret_key=config_data['exchange']['secret_key'],
                use_testnet=config_data['exchange'].get('use_testnet', True),
                trading_symbols=config_data['exchange'].get('symbols', ["BTCUSDT"]),
                batch_size=config_data.get('collection', {}).get('batch_size', 100),
                history_size=config_data.get('collection', {}).get('history_size', 1000),
                update_interval=config_data.get('collection', {}).get('update_interval', 1),
                monitoring_host=config_data.get('monitoring', {}).get('host', "0.0.0.0"),
                monitoring_port=config_data.get('monitoring', {}).get('port', 8000),
                metrics_interval=config_data.get('monitoring', {}).get('metrics_interval', 60)
            )
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()

# Load settings
yaml_path = Path('config/config.yaml')
settings = Settings.from_yaml(str(yaml_path)) if yaml_path.exists() else Settings()