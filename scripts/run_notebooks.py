"""Run all training notebooks headlessly and promote the best models to results/.

Usage:
    uv run run-notebooks                          # 80/20 only
    uv run run-notebooks --splits 70_30 50_50     # specific splits
    uv run run-notebooks --splits 80_20 70_30 90_10 50_50  # all four
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


NOTEBOOKS = [
    (
        "notebooks/01_linear_regression.ipynb",
        "results/models/linear_regression",
        "Linear Regression (RidgeCV)",
    ),
    (
        "notebooks/02_regression_trees.ipynb",
        "results/models/regression_trees",
        "Regression Trees (DecisionTreeRegressor)",
    ),
    (
        "notebooks/03_random_forest.ipynb",
        "results/models/random_forest",
        "Random Forest (RandomForestRegressor)",
    ),
    (
        "notebooks/04_lightgbm.ipynb",
        "results/models/lightgbm",
        "LightGBM (LGBMRegressor)",
    ),
]


def split_label_to_ratio(split: str) -> float:
    """Convert '80_20' -> 0.80."""
    train_part = split.split("_")[0]
    return int(train_part) / 100


def run_notebook(notebook_path: Path, train_ratio: float) -> None:
    """Execute a notebook in-place with the given train ratio."""
    env = {**os.environ, "LUDOMETRICS_TRAIN_RATIO": str(train_ratio)}
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook_path),
        ],
        env=env,
        check=True,
    )


def promote_winners(root: Path) -> None:
    """Copy the best 80/20 model for each target to results/."""
    for target in ("quality_score", "commercial_score"):
        table_path = root / "results" / f"{target}.md"
        if not table_path.exists():
            print(f"  Warning: {table_path} not found -- skipping promotion")
            continue

        lines = [
            l
            for l in table_path.read_text().splitlines()
            if l.startswith("| ") and "---" not in l and "Algorithm" not in l
        ]
        if not lines:
            continue

        # Top row is the winner (table is sorted by R2 desc)
        winner_row = lines[0]
        parts = [p.strip() for p in winner_row.split("|")[1:-1]]
        algorithm, split = parts[0], parts[1]
        split_label = split.replace("/", "_")

        for _, model_dir_rel, algo_name in NOTEBOOKS:
            if algo_name == algorithm:
                src = root / model_dir_rel / f"{target}_{split_label}.pkl"
                dst = root / "results" / f"{target}.pkl"
                if src.exists():
                    shutil.copy2(src, dst)
                    print(
                        f"  Winner ({target}): {algorithm} {split} -> results/{target}.pkl"
                    )
                else:
                    print(f"  Warning: model file not found: {src}")
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all training notebooks and promote winners."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["80_20"],
        help="One or more split suffixes, e.g. --splits 80_20 70_30 50_50",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    all_timings = []

    for split in args.splits:
        train_ratio = split_label_to_ratio(split)
        print(f"\n=== Split {split.replace('_', '/')} ===")
        split_timings = []

        for notebook_rel, _, _ in NOTEBOOKS:
            notebook_path = root / notebook_rel
            print(f"  Running {notebook_path.name} ...", flush=True)
            t0 = time.monotonic()
            run_notebook(notebook_path, train_ratio)
            duration = time.monotonic() - t0
            split_timings.append((notebook_path.stem, duration))
            print(f"  Done in {duration:.1f}s")

        all_timings.append((split, split_timings))

    print("\nPromoting winners ...")
    promote_winners(root)

    print()
    print("Summary")
    print("-------")
    for split, timings in all_timings:
        print(f"  {split.replace('_', '/')}:")
        for name, duration in timings:
            print(f"    {name}: {duration:.1f}s")
        print(f"    subtotal: {sum(d for _, d in timings):.1f}s")


if __name__ == "__main__":
    main()
