"""Training pipeline: load -> validate -> EDA -> cross-validate -> select -> evaluate -> save."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import __version__, config, visualize
from .data import load_data
from .evaluate import regression_metrics
from .features import build_preprocessor
from .models import SEARCH_SPACES, get_candidate_models

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_name: str
    best_pipeline: Pipeline
    results: pd.DataFrame
    test_metrics: dict[str, float]
    model_path: Path


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def train(
    data_path: Path | str = config.DATA_PATH,
    model_dir: Path | str = config.MODEL_DIR,
    reports_dir: Path | str = config.REPORTS_DIR,
    seed: int = config.RANDOM_SEED,
    test_size: float = config.TEST_SIZE,
    cv_folds: int = config.CV_FOLDS,
    tune: bool = False,
    make_plots: bool = True,
) -> TrainingResult:
    model_dir, reports_dir = Path(model_dir), Path(reports_dir)
    figures_dir = reports_dir / "figures"

    df = load_data(data_path)
    logger.info("Missing values per column:\n%s", df.isna().sum().to_string())
    if make_plots:
        visualize.generate_eda_plots(df, figures_dir)

    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    rows, fitted = [], {}
    for name, estimator in get_candidate_models(seed).items():
        pipe = _make_pipeline(estimator)
        cv = cross_validate(pipe, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        test = regression_metrics(y_test, pipe.predict(X_test))
        rows.append({
            "model": name,
            "cv_rmse": -cv["test_rmse"].mean(),
            "cv_rmse_std": cv["test_rmse"].std(),
            "cv_r2": cv["test_r2"].mean(),
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "test_mape": test["mape"],
            "test_r2": test["r2"],
            "tuned": False,
        })
        fitted[name] = pipe
        logger.info("%-18s CV RMSE=%.0f  test RMSE=%.0f  test R2=%.4f",
                    name, rows[-1]["cv_rmse"], test["rmse"], test["r2"])

    results = pd.DataFrame(rows)
    # Select on cross-validated error (training data only) so the test set stays unbiased.
    best_name = results.loc[results["cv_rmse"].idxmin(), "model"]

    if tune and best_name in SEARCH_SPACES:
        logger.info("Tuning %s with randomized search...", best_name)
        search = RandomizedSearchCV(
            _make_pipeline(get_candidate_models(seed)[best_name]),
            SEARCH_SPACES[best_name],
            n_iter=15, cv=3, scoring="neg_root_mean_squared_error",
            random_state=seed, n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuned_pipe = search.best_estimator_
        test = regression_metrics(y_test, tuned_pipe.predict(X_test))
        logger.info("Tuned params: %s", search.best_params_)
        results = pd.concat([results, pd.DataFrame([{
            "model": f"{best_name} (tuned)",
            "cv_rmse": -search.best_score_, "cv_rmse_std": float("nan"), "cv_r2": float("nan"),
            "test_rmse": test["rmse"], "test_mae": test["mae"],
            "test_mape": test["mape"], "test_r2": test["r2"], "tuned": True,
        }])], ignore_index=True)
        # Keep the tuned model only if it wins on cross-validated error (training data only).
        base_cv_rmse = results.loc[results["model"] == best_name, "cv_rmse"].iloc[0]
        if -search.best_score_ < base_cv_rmse:
            tuned_name = f"{best_name} (tuned)"
            fitted[tuned_name] = tuned_pipe
            best_name = tuned_name

    best_pipe = fitted[best_name]
    y_pred = best_pipe.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)

    reports_dir.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(reports_dir / "model_comparison.csv", index=False)

    importances = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=seed, scoring="r2"
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean": importances.importances_mean,
        "std": importances.importances_std,
    }).sort_values("mean", ascending=False)

    if make_plots:
        visualize.plot_actual_vs_predicted(y_test, y_pred, best_name, figures_dir)
        visualize.plot_residuals(y_test, y_pred, best_name, figures_dir)
        visualize.plot_model_comparison(results, figures_dir)
        visualize.plot_feature_importance(importance_df, figures_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_pipe, model_path)
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "features": config.FEATURES,
        "target": config.TARGET,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": test_metrics,
        "feature_importance": importance_df.round(5).to_dict(orient="records"),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Best model: %s (test R2=%.4f) -> %s", best_name, test_metrics["r2"], model_path)

    return TrainingResult(best_name, best_pipe, results, test_metrics, model_path)
