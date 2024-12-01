from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

class FeatureType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"

@dataclass
class FeatureConfig:
    name: str
    type: FeatureType
    params: Dict
    enabled: bool = True

# Default feature configurations
DEFAULT_FEATURES = [
    FeatureConfig(
        name="SMA",
        type=FeatureType.TREND,
        params={"windows": [7, 14, 21, 50, 200]}
    ),
    FeatureConfig(
        name="EMA",
        type=FeatureType.TREND,
        params={"windows": [7, 14, 21, 50, 200]}
    ),
    FeatureConfig(
        name="RSI",
        type=FeatureType.MOMENTUM,
        params={"window": 14}
    ),
    FeatureConfig(
        name="MACD",
        type=FeatureType.MOMENTUM,
        params={"window_slow": 26, "window_fast": 12, "window_sign": 9}
    ),
    FeatureConfig(
        name="BB",
        type=FeatureType.VOLATILITY,
        params={"window": 20, "window_dev": 2}
    ),
    FeatureConfig(
        name="OBV",
        type=FeatureType.VOLUME,
        params={}
    )
]