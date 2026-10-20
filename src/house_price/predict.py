"""Inference: validated, single-house price prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config


@dataclass(frozen=True)
class HouseFeatures:
    area_sqft: float
    bedrooms: int
    bathrooms: int
    age_years: float
    distance_to_city_km: float
    location: str

    def __post_init__(self) -> None:
        if self.location not in config.LOCATIONS:
            raise ValueError(f"location must be one of {config.LOCATIONS}, got {self.location!r}")
        for name, (lo, hi) in config.FEATURE_BOUNDS.items():
            value = getattr(self, name)
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])[config.FEATURES]


@lru_cache(maxsize=4)
def load_model(model_path: str = str(config.MODEL_PATH)):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `house-price train` first.")
    return joblib.load(path)


def predict_price(features: HouseFeatures, model_path: Path | str = config.MODEL_PATH) -> float:
    """Return the predicted price (INR) for one house."""
    model = load_model(str(model_path))
    return float(model.predict(features.to_frame())[0])
