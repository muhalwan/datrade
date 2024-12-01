from enum import Enum
from typing import Optional
import logging

class AuthType(Enum):
    HMAC = "hmac"

class BinanceAuth:
    def __init__(self, api_key: str, auth_type: AuthType, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.auth_type = auth_type
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)