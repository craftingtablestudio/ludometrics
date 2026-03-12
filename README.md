<p align="center">
  <img src="assets/ludometrics_logo.png" alt="ludometrics" width="300" />
</p>

# ludometrics 🎲

Predicts board game success from BoardGameGeek data using machine learning.

## Preamble

Based on a [dataset](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) of 22k board games. I plan to predict the *chance of succeeding* a board game has based on its *feature profile*.
I plan to also categorise the data based on *development viability* for developing said board game for an mixed reality platform like the Apple Vision Pro, to achieve the end goal of determining what kind of board game to build for highest chance of success on that platform.

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
- [Pre-processing details](plan/pre-processing%20details.md)

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
uv run euporie-notebook preprocessing.ipynb
```

### verify_bayes_avg.py

Fits and validates the BGG Bayesian average formula (`BayesAvg = (C×m + N×AvgRating) / (C+N)`) against the stored `BayesAvgRating` column. Prints best-fit parameters, formula accuracy, and a sanity check comparing `user_ratings.csv` against `games.csv`.

**Input:** `dataset/`

```sh
uv run verify_bayes_avg.py
```
