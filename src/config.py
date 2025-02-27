from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import json
import os

class Settings(BaseSettings):
    # MongoDB settings
    mongodb_uri: str = Field(..., env='MONGODB_URI')
    db_name: str = Field(..., env='DB_NAME')

    # Binance settings
    binance_api_key: str = Field(..., env='BINANCE_API_KEY')
    binance_secret_key: str = Field(..., env='BINANCE_SECRET_KEY')
    use_testnet: bool = Field(True, env='USE_TESTNET')

    # Trading settings
    trading_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT"], env='TRADING_SYMBOLS')

    class Config:
        env_file = '.env'
        case_sensitive = False

settings = Settings()