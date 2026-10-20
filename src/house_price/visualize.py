"""EDA and model-evaluation figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from . import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
_millions = FuncFormatter(lambda v, _: f"{v / 1e6:.1f}")
_PRICE_LABEL = f"Price ({config.CURRENCY}, millions)"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[config.TARGET], kde=True, bins=40, ax=ax, color="#4C72B0")
    ax.xaxis.set_major_formatter(_millions)
    ax.set(title="Distribution of House Prices", xlabel=_PRICE_LABEL)
    return _save(fig, out_dir / "price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return _save(fig, out_dir / "correlation_heatmap.png")


def plot_price_by_location(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="location", y=config.TARGET, order=config.LOCATIONS, ax=ax)
    ax.yaxis.set_major_formatter(_millions)
    ax.set(title="Price by Location", ylabel=_PRICE_LABEL, xlabel="Location")
    return _save(fig, out_dir / "price_by_location.png")


def plot_area_vs_price(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="area_sqft", y=config.TARGET, hue="location",
        hue_order=config.LOCATIONS, alpha=0.6, ax=ax,
    )
    ax.yaxis.set_major_formatter(_millions)
    ax.set(title="Area vs Price", xlabel="Area (sq ft)", ylabel=_PRICE_LABEL)
    return _save(fig, out_dir / "area_vs_price.png")


def plot_actual_vs_predicted(y_true, y_pred, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.45, color="#4C72B0")
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.xaxis.set_major_formatter(_millions)
    ax.yaxis.set_major_formatter(_millions)
    ax.set(title=f"Actual vs Predicted: {name}",
           xlabel=f"Actual ({config.CURRENCY}, millions)",
           ylabel=f"Predicted ({config.CURRENCY}, millions)")
    ax.legend()
    return _save(fig, out_dir / "actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, name: str, out_dir: Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(y_pred, residuals, alpha=0.45, color="#55A868")
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].xaxis.set_major_formatter(_millions)
    axes[0].yaxis.set_major_formatter(_millions)
    axes[0].set(title="Residuals vs Predicted", xlabel=f"Predicted ({config.CURRENCY}, millions)",
                ylabel="Residual (millions)")
    sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].xaxis.set_major_formatter(_millions)
    axes[1].set(title="Residual Distribution", xlabel="Residual (millions)")
    fig.suptitle(f"Residual Analysis: {name}")
    return _save(fig, out_dir / "residuals.png")


def plot_model_comparison(results: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ordered = results.sort_values("test_rmse")
    sns.barplot(data=ordered, x="model", y="test_rmse", ax=axes[0], color="#4C72B0")
    axes[0].yaxis.set_major_formatter(_millions)
    axes[0].set(title="Test RMSE (lower is better)", ylabel=f"RMSE ({config.CURRENCY}, millions)",
                xlabel="")
    sns.barplot(data=ordered, x="model", y="test_r2", ax=axes[1], color="#55A868")
    axes[1].set(title="Test R² (higher is better)", ylabel="R²", xlabel="",
                ylim=(max(0, ordered["test_r2"].min() - 0.05), 1))
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    return _save(fig, out_dir / "model_comparison.png")


def plot_feature_importance(importances: pd.DataFrame, out_dir: Path) -> Path:
    """``importances`` has columns: feature, mean, std (permutation importance)."""
    data = importances.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(data["feature"], data["mean"], xerr=data["std"], color="#C44E52")
    ax.set(title="Permutation Feature Importance (test set)",
           xlabel="Decrease in R² when feature is shuffled")
    return _save(fig, out_dir / "feature_importance.png")


def generate_eda_plots(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    return [
        plot_price_distribution(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
        plot_price_by_location(df, out_dir),
        plot_area_vs_price(df, out_dir),
    ]
