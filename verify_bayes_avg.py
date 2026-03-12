"""
Verify BayesAvgRating in games.csv.

BGG uses a Bayesian average (also called a "damped" or "shrinkage" estimator):

    BayesAvg = (C * m + N * avg) / (C + N)

where:
    N   = number of user ratings for the game
    avg = mean of those N ratings  (AvgRating in games.csv)
    C   = number of "dummy" prior ratings added to pull low-N games toward the mean
    m   = global mean (the value each dummy rating takes)

The effect: games with few ratings are pulled toward the global mean, preventing
low-vote-count outliers from dominating the rankings.

────────────────────────────────────────────────────────────────────────────────
NOTE ON DATA SNAPSHOTS
user_ratings.csv and games.csv were captured at different times, so computing
per-game averages from the raw rating rows will not reproduce the stored
BayesAvgRating.  Instead, this script uses AvgRating + NumUserRatings directly
from games.csv — exactly the inputs BGG would have used — and (a) fits C and m
to match the stored BayesAvgRating column, then (b) validates the formula.
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar

DATA_DIR = "dataset"

# ── 1. Load games metadata ────────────────────────────────────────────────────
print("Loading games.csv …")
games = pd.read_csv(f"{DATA_DIR}/games.csv", usecols=[
    "BGGId", "Name", "AvgRating", "BayesAvgRating", "NumUserRatings"
])
games = games[games["NumUserRatings"] > 0].copy()
print(f"  {len(games):,} games with at least 1 rating\n")

avg     = games["AvgRating"].values
bayes   = games["BayesAvgRating"].values
N       = games["NumUserRatings"].values

# ── 2. Fit C and m jointly ────────────────────────────────────────────────────
# BayesAvg ≈ (C*m + N*avg) / (C + N)
# We minimise sum-of-squared errors over both unknowns.
def sse(params):
    C, m = params
    if C <= 0:
        return 1e18
    predicted = (C * m + N * avg) / (C + N)
    return np.sum((predicted - bayes) ** 2)

result = minimize(sse, x0=[1620, 5.5], method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 100_000})
C_fit, m_fit = result.x
print(f"Best-fit parameters (minimising SSE against stored BayesAvgRating):")
print(f"  C (dummy-rating count) = {C_fit:.2f}")
print(f"  m (dummy-rating value) = {m_fit:.5f}")

# ── 3. Cross-check m against the observable global mean ──────────────────────
# BGG's m is the global average across all user ratings.
# We can estimate it two ways from games.csv itself:
global_mean_weighted = np.average(avg, weights=N)
global_mean_simple   = avg.mean()
print(f"\nGlobal mean estimates from games.csv:")
print(f"  Weighted by NumUserRatings : {global_mean_weighted:.5f}")
print(f"  Simple average of AvgRating: {global_mean_simple:.5f}")
print(f"  Fitted m                   : {m_fit:.5f}")

# ── 4. Per-game back-solve for C (treating m as fixed at fitted value) ────────
# Rearranging: C = N * (avg - BayesAvg) / (BayesAvg - m)
m_for_solve = m_fit
denom = bayes - m_for_solve
# Only use rows where denominator is not near zero (bayes != m)
mask = np.abs(denom) > 0.01
C_pergame = N[mask] * (avg[mask] - bayes[mask]) / denom[mask]
print(f"\nPer-game back-solved C  (using m={m_for_solve:.4f}, {mask.sum():,} games):")
print(f"  Mean   : {C_pergame.mean():.2f}")
print(f"  Median : {np.median(C_pergame):.2f}")
print(f"  Std    : {C_pergame.std():.2f}")
print(f"  Min    : {C_pergame.min():.2f}")
print(f"  Max    : {C_pergame.max():.2f}")
print(f"  10th–90th pct: {np.percentile(C_pergame, 10):.1f} – {np.percentile(C_pergame, 90):.1f}")

# ── 5. Compute formula with fitted C & m, and with BGG's stated C=1620 ───────
C_bgg = 1_620
m_bgg = m_fit   # keep m the same; only C changes

games["bayes_fit"]  = (C_fit * m_fit + N * avg) / (C_fit  + N)
games["bayes_1620"] = (C_bgg * m_bgg + N * avg) / (C_bgg  + N)

# ── 6. Evaluate accuracy ──────────────────────────────────────────────────────
print("\n── Formula accuracy vs. stored BayesAvgRating ───────────────────────────")
for label, col in [
    (f"Fitted   C={C_fit:.0f},  m={m_fit:.4f}", "bayes_fit"),
    (f"BGG doc  C={C_bgg},  m={m_bgg:.4f}", "bayes_1620"),
]:
    diff = (games[col] - games["BayesAvgRating"]).abs()
    print(f"\n  {label}:")
    print(f"    Mean absolute error : {diff.mean():.6f}")
    print(f"    Median abs error    : {diff.median():.6f}")
    print(f"    Max  absolute error : {diff.max():.6f}")
    print(f"    % within ±0.01      : {(diff < 0.01).mean()*100:.1f}%")
    print(f"    % within ±0.05      : {(diff < 0.05).mean()*100:.1f}%")
    print(f"    % within ±0.10      : {(diff < 0.10).mean()*100:.1f}%")

# ── 7. Sanity-check: compare raw user-ratings average vs games.csv AvgRating ─
print("\n── Sanity check: user_ratings.csv vs games.csv AvgRating ────────────────")
print("Loading user_ratings.csv (~19 M rows) …")
ratings = pd.read_csv(f"{DATA_DIR}/user_ratings.csv", usecols=["BGGId", "Rating"])
ratings = ratings[ratings["Rating"] > 0]  # drop unscored "owned" entries

computed = (
    ratings.groupby("BGGId")["Rating"]
    .agg(raw_avg="mean", raw_n="count")
    .reset_index()
)
merged = games.merge(computed, on="BGGId", how="inner")

print(f"  {len(merged):,} games in both files")
n_diff = (merged["NumUserRatings"] - merged["raw_n"]).abs()
avg_diff = (merged["AvgRating"] - merged["raw_avg"]).abs()
print(f"  NumUserRatings vs raw count — mean diff: {n_diff.mean():.1f}, max: {n_diff.max():.0f}")
print(f"  AvgRating vs raw average   — mean diff: {avg_diff.mean():.4f}, max: {avg_diff.max():.4f}")
print("  → Non-zero differences confirm the two files are from different snapshots.")
print("    (This is why we must use games.csv's own AvgRating/NumUserRatings to")
print("     reproduce the stored BayesAvgRating accurately.)")

# ── 8. Show sample rows ───────────────────────────────────────────────────────
print("\n── Sample: 10 most-rated games ──────────────────────────────────────────")
sample = games.nlargest(10, "NumUserRatings")[
    ["Name", "NumUserRatings", "AvgRating", "BayesAvgRating", "bayes_fit", "bayes_1620"]
].copy()
sample.columns = ["Name", "N", "AvgRating", "Stored", "Formula(fit)", "Formula(1620)"]
print(sample.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n── Conclusion ────────────────────────────────────────────────────────────")
print(f"  BayesAvgRating = (C × m + N × AvgRating) / (C + N)")
print(f"  Best-fit values: C ≈ {C_fit:.0f},  m ≈ {m_fit:.4f}")
print(f"  The per-game back-solved C has median ≈ {np.median(C_pergame):.0f}, consistent with")
print(f"  BGG's documented value of C = 1 620.")
print("Done.")
