"""Remove all generated results, processed data, and notebook outputs.

Leaves source data (dataset/), source code, and notebook markdown/code intact.

Usage:
    uv run clean
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def clean_notebook_outputs(notebook_path: Path) -> bool:
    """Strip execution counts and cell outputs from a notebook."""
    nb = json.loads(notebook_path.read_text())
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
    if changed:
        notebook_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return changed


def main():
    removed = []

    # Processed data (from 00_preprocessing.ipynb)
    processed = ROOT / "data" / "games_processed.csv"
    if processed.exists():
        processed.unlink()
        removed.append(str(processed.relative_to(ROOT)))

    # All model outputs and results tables
    results_dir = ROOT / "results"
    if results_dir.exists():
        # Promoted winner models
        for pkl in results_dir.glob("*.pkl"):
            pkl.unlink()
            removed.append(str(pkl.relative_to(ROOT)))

        # Results tables (quality_score.md, commercial_score.md)
        for md in results_dir.glob("*.md"):
            md.unlink()
            removed.append(str(md.relative_to(ROOT)))

        # Per-algorithm model directories
        models_dir = results_dir / "models"
        if models_dir.exists():
            for algo_dir in models_dir.iterdir():
                if algo_dir.is_dir():
                    shutil.rmtree(algo_dir)
                    removed.append(str(algo_dir.relative_to(ROOT)))

    # Notebook embedded outputs
    notebooks_dir = ROOT / "notebooks"
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        if clean_notebook_outputs(nb_path):
            removed.append(f"{nb_path.relative_to(ROOT)} (outputs cleared)")

    if removed:
        print("Cleaned:")
        for r in removed:
            print(f"  {r}")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
