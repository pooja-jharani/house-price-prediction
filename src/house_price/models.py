"""Candidate models and hyper-parameter search spaces."""

from __future__ import annotations

from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def get_candidate_models(seed: int) -> dict[str, BaseEstimator]:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=seed
        ),
        "GradientBoosting": HistGradientBoostingRegressor(random_state=seed),
    }


# Search spaces use the ``model__`` prefix because the estimator sits inside a Pipeline.
SEARCH_SPACES: dict[str, dict] = {
    "Ridge": {"model__alpha": uniform(0.01, 50)},
    "RandomForest": {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(5, 20),
        "model__min_samples_leaf": randint(1, 8),
    },
    "GradientBoosting": {
        "model__learning_rate": uniform(0.03, 0.2),
        "model__max_iter": randint(100, 500),
        "model__max_leaf_nodes": randint(8, 48),
        "model__l2_regularization": uniform(0, 2),
    },
}
