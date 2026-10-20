import numpy as np
import pandas as pd

from house_price import config
from house_price.features import build_preprocessor


def test_preprocessor_imputes_missing_values(small_df):
    X = small_df[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    # 5 numeric columns + one-hot columns for the 3 locations
    assert out.shape == (len(X), 5 + 3)


def test_preprocessor_ignores_unseen_category(small_df):
    pre = build_preprocessor().fit(small_df[config.FEATURES])
    new = small_df[config.FEATURES].head(2).copy()
    new["location"] = "Moon"
    out = pre.transform(new)
    assert out.shape[0] == 2 and not np.isnan(out).any()


def test_preprocessor_handles_all_nan_row():
    df = pd.DataFrame({
        "area_sqft": [1000.0, 2000.0, np.nan], "bedrooms": [2, 3, np.nan],
        "bathrooms": [1.0, 2.0, np.nan], "age_years": [5.0, 10.0, np.nan],
        "distance_to_city_km": [2.0, 4.0, np.nan], "location": ["Suburb", "Rural", None],
    })
    assert not np.isnan(build_preprocessor().fit_transform(df)).any()
