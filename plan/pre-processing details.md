# Files

Data source examples available at [bgg_data_documentation](../dataset/bgg_data_documentation.md)

| CSV                        | Rows       | Columns | Used in ML Plan | Reason                                                                                                                                         |
| -------------------------- | ---------- | ------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [games.csv](../dataset/games.csv)                | 21,925     | 48      | ✓               | Core source for feature profile (complexity, playtime, players, language ease, age) and success scores (bayes rating, owned, year published) |
| [mechanics.csv](../dataset/mechanics.csv)            | 21,925     | 158     | ✓               | 157 mechanic binary flags — primary training features and clustering input                                                                     |
| [themes.csv](../dataset/themes.csv)               | 21,925     | 218     | ✓               | 217 theme binary flags — primary training features and clustering input                                                                        |
| [subcategories.csv](../dataset/subcategories.csv)        | 21,925     | 11      | ✓               | 10 subcategory binary flags — training features                                                                                                |
| [user_ratings.csv](../dataset/user_ratings.csv)         | 18,942,215 | 3       | ✗               | Individual ratings already aggregated in games.csv (BayesAvgRating, NumUserRatings)                                                            |
| [ratings_distribution.csv](../dataset/ratings_distribution.csv) | 21,925     | 96      | ✗               | Rating distributions already summarised in games.csv                                                                                           |
| [artists_reduced.csv](../dataset/artists_reduced.csv)      | 21,925     | 1,690   | ✗               | Artist identity not part of feature profile or success score                                                                                   |
| [designers_reduced.csv](../dataset/designers_reduced.csv)    | 21,925     | 1,599   | ✗               | Designer identity not part of feature profile or success score                                                                                 |
| [publishers_reduced.csv](../dataset/publishers_reduced.csv)   | 21,925     | 1,956   | ✗               | Publisher identity not part of feature profile or success score                                                                                |

# Pre-processing choices

Columns from all 4 used CSVs with their role and preprocessing suggestion.

**Roles:**
- `feature` — training input
- `quality_score` — used to compute the quality target label, not a training input
- `commercial_score` — used to compute the commercial target label, not a training input
- `drop` — not used

Tree-based models (Regression Trees, Random Forest, LightGBM/XGBoost) are scale-invariant — normalization and log transforms have no effect on them. The only universally required steps are **imputation** and **joins**. Scaling/transforms only matter for Linear Regression.

| Column                     | Source        | Role             | Comments                                                                                                                                      |
| -------------------------- | ------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `BGGId`                    | games         | identifier       | Kept in output for game lookups. Not a training feature — exclude when passing data to models.                                                |
| `Name`                     | games         | drop             | String, not a feature                                                                                                                         |
| `Description`              | games         | drop             | Lemmatized text, not used                                                                                                                     |
| `YearPublished`            | games         | commercial_score | Not a training feature. Used to compute `years_active = clamp(2025 − year, 1, 10)` for time-normalising `NumOwned`                           |
| `GameWeight`               | games         | feature          | 506 games have value 0 (unrated/unknown) — treat as missing and impute with median                                                            |
| `AvgRating`                | games         | drop             | Use `BayesAvgRating` instead                                                                                                                  |
| `BayesAvgRating`           | games         | quality_score    | `quality_score = BayesAvgRating / 10` → 0–1. Bayesian correction already handles low-vote reliability; actual range is 3.6–8.5 (never 0–10) |
| `StdDev`                   | games         | drop             | Not in current formula                                                                                                                        |
| `MinPlayers`               | games         | feature          | Some 0 values (unknown) — impute with median or 1                                                                                             |
| `MaxPlayers`               | games         | feature          | Max value is 999 — likely a sentinel for "unlimited"; cap or impute above a reasonable threshold (e.g. 20)                                    |
| `ComAgeRec`                | games         | feature          | 5,530 missing (25%) — impute with `MfgAgeRec` first, then median for remaining                                                               |
| `LanguageEase`             | games         | drop             | Dropped — column is unreliably encoded: some games store a weighted average (1–5), others store a raw vote sum (up to 1,757), mixed across scraping runs. 27% missing on top. |
| `BestPlayers`              | games         | drop             | Redundant with player range                                                                                                                   |
| `GoodPlayers`              | games         | drop             | List format, complex to parse                                                                                                                 |
| `NumOwned`                 | games         | commercial_score | `commercial_score = log1p(NumOwned / years_active)`, normalised to 0–1 against dataset max. log1p because heavily right-skewed (median 320, max 166,497) |
| `NumWant`                  | games         | drop             | Similar signal to `NumWish`; both dropped                                                                                                     |
| `NumWish`                  | games         | drop             | Dropped — represents intent to buy, not actual purchase or quality; too ambiguous a signal                                                    |
| `NumWeightVotes`           | games         | drop             | Meta, not a game attribute                                                                                                                    |
| `MfgPlaytime`              | games         | feature          | Extreme outliers (max = 60,000 min); cap at a reasonable ceiling (e.g. 600 min) or winsorise                                                  |
| `ComMinPlaytime`           | games         | feature          | Same outlier issue as `MfgPlaytime`; cap or winsorise                                                                                         |
| `ComMaxPlaytime`           | games         | feature          | Same outlier issue as `MfgPlaytime`; cap or winsorise                                                                                         |
| `MfgAgeRec`                | games         | feature          | Some 0 values (unknown) — impute with median                                                                                                  |
| `NumUserRatings`           | games         | drop             | Already factored into `BayesAvgRating`                                                                                                        |
| `NumComments`              | games         | drop             | Not in feature profile                                                                                                                        |
| `NumAlternates`            | games         | drop             | Not in feature profile                                                                                                                        |
| `NumExpansions`            | games         | drop             | Removed from plan                                                                                                                             |
| `NumImplementations`       | games         | drop             | Not in feature profile                                                                                                                        |
| `IsReimplementation`       | games         | drop             | Not in feature profile                                                                                                                        |
| `Family`                   | games         | drop             | Free-text string, not usable                                                                                                                  |
| `Kickstarted`              | games         | drop             | Not in feature profile                                                                                                                        |
| `ImagePath`                | games         | drop             | URL, not useful                                                                                                                               |
| `Rank:boardgame`           | games         | drop             | Dropped — rank is just the sorted order of `BayesAvgRating`; including both would double-count the same signal                                |
| `Rank:strategygames`       | games         | drop             | Too many unranked                                                                                                                             |
| `Rank:abstracts`           | games         | drop             | Same reason                                                                                                                                   |
| `Rank:familygames`         | games         | drop             | Same reason                                                                                                                                   |
| `Rank:thematic`            | games         | drop             | Same reason                                                                                                                                   |
| `Rank:cgs`                 | games         | drop             | Same reason                                                                                                                                   |
| `Rank:wargames`            | games         | drop             | Same reason                                                                                                                                   |
| `Rank:partygames`          | games         | drop             | Same reason                                                                                                                                   |
| `Rank:childrensgames`      | games         | drop             | Same reason                                                                                                                                   |
| `Cat:*` (8 columns)        | games         | feature          | Already binary (0/1), keep as-is                                                                                                              |
| *(160 mechanic columns)*   | mechanics     | feature          | Already binary (0/1); join on `BGGId`                                                                                                         |
| *(219 theme columns)*      | themes        | feature          | Already binary (0/1); join on `BGGId`                                                                                                         |
| *(10 subcategory columns)* | subcategories | feature          | Already binary (0/1); join on `BGGId`                                                                                                         |

## Features

| Column                     | Format     | Min | Max    | Median | p95  | Missing             | Pre-processing (all models)                                    | Pre-processing (Linear Regression only) |
| -------------------------- | ---------- | --- | ------ | ------ | ---- | ------------------- | -------------------------------------------------------------- | --------------------------------------- |
| `GameWeight`               | float      | 0   | 5      | 2.0    | 3.52 | 506 zeros (unrated) | Impute 0 with median (~2.0)                                    | Then normalize to 0–1 (divide by 5)     |
| `MinPlayers`               | int        | 0   | 10     | 2      | 3    | 50 zeros            | Impute 0 with 1 or median                                      | Then normalize to 0–1                   |
| `MaxPlayers`               | int        | 0   | 999    | 4      | 10   | 173 zeros           | Cap at 20 (or winsorise); impute 0 with median                 | Then normalize to 0–1                   |
| `ComAgeRec`                | float      | 2   | 21     | 10     | 16   | 5,530 NaN (25%)     | Impute with `MfgAgeRec` first, then median for remaining       | Then normalize to 0–1                   |
| `MfgPlaytime`              | int (min)  | 0   | 60,000 | 45     | 240  | 780 zeros           | Cap at 600 min (10 hrs) or winsorise; impute 0 with median     | Log-transform, then normalize           |
| `ComMinPlaytime`           | int (min)  | 0   | 60,000 | 30     | 180  | 652 zeros           | Same as `MfgPlaytime`                                          | Log-transform, then normalize           |
| `ComMaxPlaytime`           | int (min)  | 0   | 60,000 | 45     | 240  | 780 zeros           | Same as `MfgPlaytime`                                          | Log-transform, then normalize           |
| `MfgAgeRec`                | int        | 0   | 25     | 10     | 14   | 1,325 zeros         | Impute 0 with median                                           | Then normalize to 0–1                   |
| `Cat:*` (8 columns)        | binary 0/1 | 0   | 1      | —      | —    | none                | Keep as-is                                                     | —                                       |
| *(157 mechanic columns)*   | binary 0/1 | 0   | 1      | —      | —    | none                | Keep as-is; join on `BGGId`                                    | —                                       |
| *(217 theme columns)*      | binary 0/1 | 0   | 1      | —      | —    | none                | Keep as-is; join on `BGGId`                                    | —                                       |
| *(10 subcategory columns)* | binary 0/1 | 0   | 1      | —      | —    | none                | Keep as-is; join on `BGGId`                                    | —                                       |

## Success Scores

Two separate target labels are computed during pre-processing, then used to train two independent models.

| Score              | Formula                                                                        | Output range | Reason                                                                                              |
| ------------------ | ------------------------------------------------------------------------------ | ------------ | --------------------------------------------------------------------------------------------------- |
| `quality_score`    | `BayesAvgRating / 10`                                                          | 0.36–0.85    | How well-regarded is this game among players who played it? Bayesian correction already handles low-vote reliability. |
| `commercial_score` | `log1p(NumOwned / clamp(2025 − YearPublished, 1, 10))`, normalised to 0–1     | 0–1          | How commercially successful is this game, accounting for time on market? log1p compresses the heavy right skew. |

Both scores are normalised to the same 0–1 range so they are directly comparable in the interface output.

| Score type         | Column           | Format | Min    | Max     | Median | p95   | Missing   | Pre-processing                                                                                                                |
| ------------------ | ---------------- | ------ | ------ | ------- | ------ | ----- | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `quality_score`    | `BayesAvgRating` | float  | 3.57   | 8.51    | 5.55   | 6.50  | none      | Divide by 10 → 0–1                                                                                                            |
| `commercial_score` | `NumOwned`       | int    | 0      | 166,497 | 320    | 5,490 | 1 zero    | Time-normalise: divide by `years_active`; apply `log1p`; normalise to 0–1 against dataset max                                 |
| `commercial_score` | `YearPublished`  | int    | −3,500 | 2021    | 2011   | 2020  | 193 zeros | Compute `years_active = clamp(2025 − year, 1, 10)`. Ancient games (pre-2015) all get years_active=10. Not a training feature. |
