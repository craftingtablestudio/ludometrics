<p align="center">
  <img src="assets/ludometrics_logo.png" alt="ludometrics" width="300" />
</p>

# ludometrics 🎲

Machine learning models that predict board game success from board game data

## What it does

Given a board game's feature profile — mechanics, complexity, categories, themes, playtime, player count, and recommended age — the models predict two independent success scores on a 0–100 scale:

- **Quality score** — how well-regarded the game is among players who played it (based on BGG's Bayesian average rating)
- **Commercial score** — how commercially successful the game is, time-normalised by years on market

The end goal is a tool that helps determine what kind of board game to build for the highest chance of success on Apple Vision Pro.

See [pipeline-plan.md](pipeline-plan.md) for the full design.

## Results

LightGBM wins on both scores.

|          | Quality Score                                                               | Commercial Score                            |
| -------- | --------------------------------------------------------------------------- | ------------------------------------------- |
| **RMSE** | 2.32 — predictions are typically within **2.3 points on a 100-point scale** | 8.78 — typically within **8.8 points**      |
| **R²**   | 0.66 — the model captures about **two-thirds of what drives quality**       | 0.64 — similar story for commercial success |

The remaining third is things features alone can't predict — timing, marketing, luck.

Full comparison across all algorithms and splits: [`results/quality_score.md`](results/quality_score.md) and [`results/commercial_score.md`](results/commercial_score.md).

## Setup

If you don't have `uv`, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```sh
uv sync
```

**macOS only:** LightGBM requires `libomp` (OpenMP), which is not bundled in its Python wheel:

```sh
brew install libomp
```

## Usage

### Preprocessing

Loads the raw CSVs, applies all pre-processing, computes success scores, and saves `data/games_processed.csv`.

```sh
uv run euporie-notebook notebooks/00_preprocessing.ipynb
```

### Training

Each notebook trains one algorithm against both targets. Results are appended to `results/quality_score.md` and `results/commercial_score.md`. Winner models are saved to `results/`.

```sh
# Run all four notebooks headlessly (default: 80/20 split)
uv run run-notebooks

# Run with specific splits
uv run run-notebooks --splits 80_20 70_30 50_50 90_10

# Open a notebook interactively
uv run euporie-notebook notebooks/01_linear_regression.ipynb
```

### Tests

```sh
uv run pytest
```

### Format

```sh
uv run format
```

## TODO

- [x] Verify BGG Bayesian average formula
- [x] Pre-process dataset
- [x] Train prediction models (`quality_score`, `commercial_score`)
- [ ] Cluster mechanics and themes (Louvain / k-means)
- [ ] Label mechanics and themes with development viability weights
- [ ] Build HTML interface
