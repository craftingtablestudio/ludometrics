<p align="center">
  <img src="assets/ludometrics_logo.png" alt="ludometrics" width="300" />
</p>

# ludometrics 🎲

Machine learning models that predict board game success from board game data

## Preamble

Based on a [dataset](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) of 22k board games. I plan to predict the *chance of succeeding* a board game has based on its *feature profile*.

A secondary *development viability* score — computed on demand from mechanic and theme clusters — estimates how feasible a game concept is to build on a mixed reality platform like Apple Vision Pro. The end goal is a tool that helps determine what kind of board game to build for the highest chance of success on that platform.

A _feature profile_ consists of: mechanics, complexity, categories & subcategories, themes, playtime (manufacturer stated + community min/max), min/max players, and recommended age.

The *chance of succeeding* is split into two independent scores:
- **Quality score** — how well-regarded is the game among players who played it (`BayesAvgRating`, Bayesian-corrected for low vote counts)
- **Commercial score** — how commercially successful is the game, time-normalised by years on market (`NumOwned / clamp(years_on_market, 1, 10)`, log-compressed)

*Development viability* will (subjectively) depend on:
- mechanics (will need to label each mechanic with a development viability rating)
- complexity (simpler is more viable)
- category and subcategories (will need to label each category with a development viability rating)
- playtime (2+ hour long sessions might not be optimal for AR)
- recommended age (under 14 years old is out of scope for target AR market)

## Plan

- [Pipeline](plan/pipeline.md)

## Scripts

### Setup

If you don't have `uv`, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```sh
uv sync
```

### preprocessing.ipynb

Loads the four source CSVs, applies all pre-processing, computes the two target labels (`quality_score`, `commercial_score`), and writes `data/games_processed.csv`.

**Input:** `dataset/`
**Output:** `data/games_processed.csv`

```sh
# open in notebook UI
uv run euporie-notebook preprocessing.ipynb

# run headless in cli
uv run jupyter execute preprocessing.ipynb

# run headless in cli — save results
uv run jupyter nbconvert --to notebook --execute --inplace preprocessing.ipynb
```

### verify_bayes_avg.py

Fits and validates the BGG Bayesian average formula (`BayesAvg = (C×m + N×AvgRating) / (C+N)`) against the stored `BayesAvgRating` column. Prints best-fit parameters, formula accuracy, and a sanity check comparing `user_ratings.csv` against `games.csv`.

**Input:** `dataset/`

```sh
uv run verify_bayes_avg.py
```

## TODO

- [x] Verify BGG Bayesian average formula (`verify_bayes_avg.py`)
- [x] Pre-process dataset (`preprocessing.ipynb`)
- [ ] Train prediction models (`quality_score`, `commercial_score`)
- [ ] Cluster mechanics and themes (Louvain / k-means)
- [ ] Label mechanics and themes with development viability weights
- [ ] Build HTML interface
