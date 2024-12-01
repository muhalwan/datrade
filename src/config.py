from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import yaml
import os

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

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls(
            mongodb_uri=config_data['database']['connection_string'],
            db_name=config_data['database']['name'],
            binance_api_key=config_data['exchange']['api_key'],
            binance_secret_key=config_data['exchange']['secret_key'],
            use_testnet=config_data['exchange']['use_testnet'],
            trading_symbols=config_data['exchange']['symbols']
        )

# Load settings from YAML or environment variables
yaml_path = 'config/config.yaml'
settings = Settings.from_yaml(yaml_path) if os.path.exists(yaml_path) else Settings()