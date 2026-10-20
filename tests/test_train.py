import json

import joblib

from house_price import config


def test_training_produces_artifacts(trained):
    result, root = trained
    assert (root / "models" / "best_model.joblib").exists()
    assert (root / "models" / "metadata.json").exists()
    assert (root / "reports" / "model_comparison.csv").exists()
    assert result.best_name in set(result.results["model"])


def test_model_beats_trivial_baseline(trained):
    result, _ = trained
    assert result.test_metrics["r2"] > 0.8


def test_metadata_is_consistent(trained):
    result, root = trained
    meta = json.loads((root / "models" / "metadata.json").read_text())
    assert meta["model_name"] == result.best_name
    assert meta["features"] == config.FEATURES
    assert meta["n_train"] + meta["n_test"] == 300


def test_saved_pipeline_predicts_on_raw_features(trained, small_df):
    _, root = trained
    model = joblib.load(root / "models" / "best_model.joblib")
    preds = model.predict(small_df[config.FEATURES].head(5))  # includes NaNs
    assert len(preds) == 5 and (preds > 0).all()
