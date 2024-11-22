from pydantic import BaseSettings, Field
from typing import List
import json

class Settings(BaseSettings):
    # MongoDB settings
    mongodb_uri: str = Field(..., env='MONGODB_URI')
    db_name: str = Field(..., env='DB_NAME')

    # Binance settings
    binance_api_key: str = Field(..., env='BINANCE_API_KEY')
    binance_secret_key: str = Field(..., env='BINANCE_SECRET_KEY')
    use_testnet: bool = Field(True, env='USE_TESTNET')

    # Trading settings
    trading_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT"])

    @classmethod
    def get_symbols(cls) -> List[str]:
        env_symbols = json.loads(Field(..., env='TRADING_SYMBOLS'))
        return env_symbols if env_symbols else ["BTCUSDT"]

    class Config:
        env_file = '.env'
        case_sensitive = False

# Create settings instance
settings = Settings()