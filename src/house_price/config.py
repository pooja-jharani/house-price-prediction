"""Central configuration: paths, feature definitions and defaults."""

from __future__ import annotations

import os
from pathlib import Path

# Project root can be overridden (useful in containers / CI).
PROJECT_ROOT = Path(os.environ.get("HOUSE_PRICE_ROOT", Path(__file__).resolve().parents[2]))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "houses.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "price"
NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
CATEGORICAL_FEATURES = ["location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LOCATIONS = ["Downtown", "Suburb", "Rural"]

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CURRENCY = "INR"

# Plausible input ranges, used to validate prediction requests.
FEATURE_BOUNDS = {
    "area_sqft": (100, 10_000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "age_years": (0, 150),
    "distance_to_city_km": (0, 200),
}
