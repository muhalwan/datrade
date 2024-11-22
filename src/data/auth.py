from enum import Enum
from typing import Optional
import logging

class AuthType(Enum):
    HMAC = "hmac"

class BinanceAuth:
    """Handles Binance authentication"""

    def __init__(self, api_key: str, auth_type: AuthType, secret_key: Optional[str] = None):
        """
        Initialize Binance authentication

        Args:
            api_key: Binance API key
            auth_type: Authentication type
            secret_key: Secret key for HMAC authentication
        """
        self.api_key = api_key
        self.auth_type = auth_type
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)
