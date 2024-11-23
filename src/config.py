from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import json
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

    # Trading settings
    trading_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT"])

    @classmethod
    def get_symbols(cls) -> List[str]:
        symbols = os.getenv('TRADING_SYMBOLS')
        if symbols:
            return json.loads(symbols)
        return ["BTCUSDT"]

    class Config:
        env_file = '.env'
        case_sensitive = False

# Create settings instance
settings = Settings()