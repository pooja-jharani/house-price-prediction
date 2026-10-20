import pytest

from house_price import config
from house_price.data import generate_synthetic_data, load_data, save_data, validate_schema


def test_generated_data_has_expected_schema(small_df):
    assert list(small_df.columns) == config.FEATURES + [config.TARGET]
    assert len(small_df) == 300
    assert set(small_df["location"]) <= set(config.LOCATIONS)
    assert (small_df[config.TARGET] >= 500_000).all()


def test_generation_is_reproducible():
    a = generate_synthetic_data(100, seed=7)
    b = generate_synthetic_data(100, seed=7)
    assert a.equals(b)


def test_missing_values_are_injected(small_df):
    assert small_df[["area_sqft", "age_years", "bathrooms"]].isna().any().all()
    assert not small_df[config.TARGET].isna().any()


def test_validate_rejects_missing_column(small_df):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(small_df.drop(columns=["location"]))


def test_validate_rejects_non_positive_target(small_df):
    bad = small_df.copy()
    bad.loc[0, config.TARGET] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_schema(bad)


def test_load_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate-data"):
        load_data(tmp_path / "nope.csv")


def test_save_and_load_roundtrip(small_df, tmp_path):
    path = save_data(small_df, tmp_path / "nested" / "houses.csv")
    assert len(load_data(path)) == len(small_df)
