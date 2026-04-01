import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np


class HasFeatureImportances(Protocol):
    feature_importances_: np.ndarray


def top_feature_importances(
    model: HasFeatureImportances,
    feature_names: list[str],
    n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-n features sorted by importance descending.

    Works with any sklearn-compatible model that exposes `feature_importances_`
    (DecisionTree, RandomForest, LightGBM, etc.).
    """
    pairs = sorted(
        zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    return pairs[:n]


def update_results_table(
    results_path: Path,
    *,
    algorithm: str,
    split: str,
    train_size: int,
    test_size: int,
    training_time: str,
    rmse: float,
    r2: float,
) -> None:
    """Add or update a row in the shared results table, sorted by R² descending.

    Creates the file with a header if it doesn't exist. If a row for the same
    algorithm + split already exists, it is replaced.

    Args:
        results_path: Path to the shared results markdown file (e.g. results/quality_score.md).
        algorithm: Human-readable algorithm name, e.g. "LightGBM (LGBMRegressor)".
        split: Train/test split string, e.g. "80/20".
        train_size: Number of training samples.
        test_size: Number of test samples.
        training_time: Formatted duration string, e.g. "11.2s".
        rmse: Root mean squared error on the test set.
        r2: R² score on the test set.
    """
    HEADER = (
        "| Algorithm | Split | Train Size | Test Size | Training Time | RMSE | R² |"
    )
    SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- |"

    new_row = (
        f"| {algorithm} | {split} | {train_size:,} | {test_size:,} "
        f"| {training_time} | {rmse:.4f} | {r2:.4f} |"
    )

    if results_path.exists():
        lines = results_path.read_text().splitlines()
        rows = [l for l in lines if l.startswith("| ") and "---" not in l and "Algorithm" not in l]
        # Remove existing row for this algorithm + split if present
        # Parse cells by splitting on | and stripping to handle oxfmt's column padding
        def _row_cells(row: str) -> list[str]:
            return [c.strip() for c in row.split("|")[1:-1]]

        rows = [r for r in rows if (_row_cells(r)[0], _row_cells(r)[1]) != (algorithm, split)]
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []

    rows.append(new_row)
    rows.sort(key=lambda r: float(r.split("|")[-2].strip()), reverse=True)

    results_path.write_text("\n".join([HEADER, SEPARATOR] + rows) + "\n")

    try:
        subprocess.run(
            ["oxfmt", "--write", str(results_path.resolve())],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        pass  # oxfmt not installed, skip
