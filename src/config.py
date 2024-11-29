from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Optional
import yaml
import os
from pathlib import Path

class MLConfig:
    def __init__(self, config_path: str = 'config/ml_config.yaml'):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load ML config: {e}")
            self.config = {
                'models': {
                    'lstm': {
                        'sequence_length': 60,
                        'layers': [128, 64, 32],
                        'dropout': 0.2
                    },
                    'xgboost': {
                        'n_estimators': 1000,
                        'learning_rate': 0.1
                    },
                    'prophet': {},
                    'ensemble': {}
                }
            }

    def __getitem__(self, key):
        return self.config.get(key, {})

    def get(self, key: str, default=None):
        try:
            parts = key.split('.')
            value = self.config
            for part in parts:
                value = value.get(part, default)
            return value
        except Exception:
            return default

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

    # ML settings
    ml_config: Optional[MLConfig] = None

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)

        instance = cls(
            mongodb_uri=config_data['database']['connection_string'],
            db_name=config_data['database']['name'],
            binance_api_key=config_data['exchange']['api_key'],
            binance_secret_key=config_data['exchange']['secret_key'],
            use_testnet=config_data['exchange']['use_testnet'],
            trading_symbols=config_data['exchange']['symbols']
        )

        # Load ML config
        ml_config_path = Path('config/ml_config.yaml')
        if ml_config_path.exists():
            instance.ml_config = MLConfig(str(ml_config_path))

        return instance

# Load settings from YAML or environment variables
yaml_path = 'config/config.yaml'
settings = Settings.from_yaml(yaml_path) if os.path.exists(yaml_path) else Settings()