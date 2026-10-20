"""Data generation, loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_LOCATION_PREMIUM = {"Downtown": 1.35, "Suburb": 1.0, "Rural": 0.75}


def generate_synthetic_data(
    n_samples: int = 1200, seed: int = config.RANDOM_SEED, missing_rate: float = 0.03
) -> pd.DataFrame:
    """Create a synthetic housing dataset with a known pricing structure.

    Prices follow a linear base (area, rooms, age, distance) scaled by a location
    premium plus Gaussian noise. A few values are blanked out in selected columns
    so the cleaning/imputation step has real work to do.
    """
    rng = np.random.default_rng(seed)

    location = rng.choice(config.LOCATIONS, size=n_samples, p=[0.3, 0.5, 0.2])
    area = rng.normal(1500, 500, n_samples).clip(400, 4500)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.25, 0.35, 0.2, 0.1])
    bathrooms = (bedrooms - rng.choice([0, 1], size=n_samples, p=[0.6, 0.4])).clip(1, None)
    age = rng.exponential(15, n_samples).clip(0, 80)
    distance = rng.exponential(8, n_samples).clip(0.2, 50)

    base_price = (
        area * 3800 + bedrooms * 150_000 + bathrooms * 90_000 - age * 4000 - distance * 6000
    )
    premium = np.array([_LOCATION_PREMIUM[loc] for loc in location])
    noise = rng.normal(0, 180_000, n_samples)
    price = (base_price * premium + noise).clip(500_000, None)

    df = pd.DataFrame(
        {
            "area_sqft": area.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms.astype(float),
            "age_years": age.round(1),
            "distance_to_city_km": distance.round(2),
            "location": location,
            "price": price.round(0),
        }
    )

    for col in ["area_sqft", "age_years", "bathrooms"]:
        idx = rng.choice(df.index, size=int(missing_rate * n_samples), replace=False)
        df.loc[idx, col] = np.nan
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the dataframe cannot be used for training."""
    expected = set(config.FEATURES + [config.TARGET])
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")
    if df[config.TARGET].isna().any():
        raise ValueError(f"Target column '{config.TARGET}' contains missing values")
    if (df[config.TARGET] <= 0).any():
        raise ValueError(f"Target column '{config.TARGET}' must be strictly positive")
    if len(df) < 50:
        raise ValueError(f"Need at least 50 rows to train, got {len(df)}")


def load_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV and validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `house-price generate-data` first."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def save_data(df: pd.DataFrame, path: Path | str = config.DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path
