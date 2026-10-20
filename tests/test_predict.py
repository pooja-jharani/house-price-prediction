import pytest

from house_price.predict import HouseFeatures, predict_price

BASE = dict(area_sqft=1800, bedrooms=3, bathrooms=2, age_years=5,
            distance_to_city_km=4.5, location="Suburb")


def test_prediction_is_positive_float(trained):
    _, root = trained
    price = predict_price(HouseFeatures(**BASE), root / "models" / "best_model.joblib")
    assert isinstance(price, float) and price > 0


def test_larger_house_costs_more(trained):
    _, root = trained
    path = root / "models" / "best_model.joblib"
    small = predict_price(HouseFeatures(**{**BASE, "area_sqft": 900}), path)
    large = predict_price(HouseFeatures(**{**BASE, "area_sqft": 3200}), path)
    assert large > small


def test_downtown_costs_more_than_rural(trained):
    _, root = trained
    path = root / "models" / "best_model.joblib"
    downtown = predict_price(HouseFeatures(**{**BASE, "location": "Downtown"}), path)
    rural = predict_price(HouseFeatures(**{**BASE, "location": "Rural"}), path)
    assert downtown > rural


def test_invalid_location_rejected():
    with pytest.raises(ValueError, match="location"):
        HouseFeatures(**{**BASE, "location": "Atlantis"})


def test_out_of_range_area_rejected():
    with pytest.raises(ValueError, match="area_sqft"):
        HouseFeatures(**{**BASE, "area_sqft": 5})


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        predict_price(HouseFeatures(**BASE), tmp_path / "missing.joblib")
