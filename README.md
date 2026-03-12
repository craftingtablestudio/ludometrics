# ludometrics

Predicts board game success from BoardGameGeek data using machine learning.

## Setup

```sh
uv sync
```

## Scripts

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
