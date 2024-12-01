from enum import Enum
from typing import Optional, Dict
import logging
import hmac
import hashlib
import time

class AuthType(Enum):
    """Authentication types for exchange APIs"""
    HMAC = "hmac"
    JWT = "jwt"
    API_KEY = "api_key"

class BinanceAuth:
    """Authentication handler for Binance API"""

    def __init__(self, api_key: str, auth_type: AuthType, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.auth_type = auth_type
        self.secret_key = secret_key
        self.logger = logging.getLogger(__name__)

    def generate_signature(self, params: Dict) -> str:
        """Generate HMAC signature for API requests"""
        try:
            if self.auth_type != AuthType.HMAC or not self.secret_key:
                raise ValueError("HMAC signature requires secret key")

            # Convert params to query string
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])

            # Create signature
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            return signature

        except Exception as e:
            self.logger.error(f"Signature generation error: {str(e)}")
            raise

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        try:
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }

            return headers

        except Exception as e:
            self.logger.error(f"Header generation error: {str(e)}")
            raise

    def sign_request(self, params: Dict) -> Dict:
        """Sign request parameters"""
        try:
            # Add timestamp
            params['timestamp'] = int(time.time() * 1000)

            # Generate signature
            signature = self.generate_signature(params)
            params['signature'] = signature

            return params

        except Exception as e:
            self.logger.error(f"Request signing error: {str(e)}")
            raise

    def validate_credentials(self) -> bool:
        """Validate authentication credentials"""
        try:
            if not self.api_key:
                self.logger.error("Missing API key")
                return False

            if self.auth_type == AuthType.HMAC and not self.secret_key:
                self.logger.error("Missing secret key for HMAC auth")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Credential validation error: {str(e)}")
            return False