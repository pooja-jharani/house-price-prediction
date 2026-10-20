import pytest

from house_price.data import generate_synthetic_data, save_data
from house_price.train import train


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic_data(n_samples=300, seed=0)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_df):
    """Train once on a small dataset; reused by model and prediction tests."""
    root = tmp_path_factory.mktemp("run")
    data_path = save_data(small_df, root / "houses.csv")
    result = train(
        data_path=data_path,
        model_dir=root / "models",
        reports_dir=root / "reports",
        cv_folds=3,
        make_plots=False,
    )
    return result, root
