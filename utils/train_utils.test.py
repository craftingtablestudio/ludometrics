import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from utils.train_utils import (
    build_linear_pipeline,
    top_feature_importances,
    update_results_table,
)


CONTINUOUS_COLS = ["GameWeight", "MinPlayers"]
BINARY_COLS = ["Alliances", "Cooperative", "Fantasy"]
ALL_COLS = CONTINUOUS_COLS + BINARY_COLS


def _sample_df():
    """Small fake DataFrame that mimics games_processed.csv structure."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "GameWeight": rng.uniform(1, 5, 20),
            "MinPlayers": rng.uniform(1, 8, 20),
            "Alliances": rng.integers(0, 2, 20),
            "Cooperative": rng.integers(0, 2, 20),
            "Fantasy": rng.integers(0, 2, 20),
        }
    )


# ---------------------------------------------------------------------------
# build_linear_pipeline
# ---------------------------------------------------------------------------


def test_build_linear_pipeline_returns_pipeline():
    pipeline = build_linear_pipeline(CONTINUOUS_COLS, ALL_COLS)
    assert isinstance(pipeline, Pipeline)


def test_binary_columns_pass_through_unchanged():
    df = _sample_df()
    pipeline = build_linear_pipeline(CONTINUOUS_COLS, ALL_COLS)
    transformed_df = pipeline[:-1].fit_transform(df)
    for col in BINARY_COLS:
        assert set(transformed_df[col].unique()).issubset({0, 1, 0.0, 1.0}), (
            f"Binary column '{col}' should only contain 0/1 after transform"
        )


def test_continuous_columns_are_standardised():
    df = _sample_df()
    pipeline = build_linear_pipeline(CONTINUOUS_COLS, ALL_COLS)
    transformed_df = pipeline[:-1].fit_transform(df)
    for col in CONTINUOUS_COLS:
        assert abs(transformed_df[col].mean()) < 1e-9, (
            f"Continuous column '{col}' should have mean ≈ 0 after StandardScaler"
        )
        assert abs(transformed_df[col].std() - 1.0) < 0.1, (
            f"Continuous column '{col}' should have std ≈ 1 after StandardScaler"
        )


# ---------------------------------------------------------------------------
# update_results_table
# ---------------------------------------------------------------------------


def _row(tmp_path, algorithm="LightGBM", split="80/20", r2=0.66):
    update_results_table(
        tmp_path / "quality_score.md",
        algorithm=algorithm,
        split=split,
        train_size=17540,
        test_size=4385,
        training_time="11.2s",
        rmse=2.32,
        r2=r2,
    )


def test_update_results_table_creates_file(tmp_path):
    _row(tmp_path)
    assert (tmp_path / "quality_score.md").exists()


def test_update_results_table_contains_header(tmp_path):
    _row(tmp_path)
    content = (tmp_path / "quality_score.md").read_text()
    assert "| Algorithm |" in content
    assert "| R² |" in content


def test_update_results_table_contains_row(tmp_path):
    _row(tmp_path)
    content = (tmp_path / "quality_score.md").read_text()
    assert "LightGBM" in content
    assert "80/20" in content
    assert "17,540" in content
    assert "0.6600" in content


def test_update_results_table_replaces_existing_row(tmp_path):
    _row(tmp_path, r2=0.50)
    _row(tmp_path, r2=0.66)
    rows = [
        l
        for l in (tmp_path / "quality_score.md").read_text().splitlines()
        if l.startswith("| LightGBM")
    ]
    assert len(rows) == 1
    assert "0.6600" in rows[0]


def test_update_results_table_sorted_by_r2_descending(tmp_path):
    _row(tmp_path, algorithm="ModelA", r2=0.50)
    _row(tmp_path, algorithm="ModelB", r2=0.70)
    _row(tmp_path, algorithm="ModelC", r2=0.60)
    lines = [
        l
        for l in (tmp_path / "quality_score.md").read_text().splitlines()
        if l.startswith("| Model")
    ]
    r2_values = [float(l.split("|")[-2].strip()) for l in lines]
    assert r2_values == sorted(r2_values, reverse=True)


def test_update_results_table_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "quality_score.md"
    update_results_table(
        path,
        algorithm="X",
        split="80/20",
        train_size=100,
        test_size=20,
        training_time="1s",
        rmse=1.0,
        r2=0.5,
    )
    assert path.exists()


# ---------------------------------------------------------------------------
# top_feature_importances
# ---------------------------------------------------------------------------


def _fitted_tree():
    """A tiny fitted DecisionTreeRegressor to use as a test fixture."""
    from sklearn.tree import DecisionTreeRegressor

    X = pd.DataFrame(
        {
            "mechanic_a": [1, 0, 1, 0, 1],
            "mechanic_b": [0, 1, 0, 1, 0],
            "mechanic_c": [1, 1, 0, 0, 1],
        }
    )
    y = [0.8, 0.3, 0.7, 0.2, 0.9]
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model, list(X.columns)


def test_top_feature_importances_returns_n_items():
    model, feature_names = _fitted_tree()
    result = top_feature_importances(model, feature_names, n=2)
    assert len(result) == 2


def test_top_feature_importances_sorted_descending():
    model, feature_names = _fitted_tree()
    result = top_feature_importances(model, feature_names, n=3)
    values = [v for _, v in result]
    assert values == sorted(values, reverse=True)


def test_top_feature_importances_names_match_features():
    model, feature_names = _fitted_tree()
    result = top_feature_importances(model, feature_names, n=3)
    names = [name for name, _ in result]
    assert all(name in feature_names for name in names)
