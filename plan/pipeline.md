# Pipeline

## 1. Pre-processing (data preparation, no ML)

### Feature Profile

- **Multi-hot encode** mechanics (157 columns) and themes (217 columns) — raw binary flags per game, used directly for training; no information is destroyed at this stage
- **Multi-hot encode** categories and subcategories
- **Normalise / transform remaining features** — inspect the dataset to determine appropriate transformations for continuous features (complexity, manufacturer playtime, community min/max playtime, min/max players, recommended age) before committing to specific encodings

### Success Scores

Two target labels computed during pre-processing; two models trained independently, interface always reports both.

| Score              | Formula                                                            | Range     |
| ------------------ | ------------------------------------------------------------------ | --------- |
| `quality_score`    | `BayesAvgRating / 10`                                              | 0–1       |
| `commercial_score` | `log1p(NumOwned / clamp(2025 − YearPublished, 1, 10))`, norm. 0–1 | 0–1       |

See [preprocessing.ipynb](../preprocessing.ipynb) for full implementation details.

## 2. Prediction Training

Inputs: 157 mechanic columns + 217 theme columns + 10 subcategory columns + 8 category columns + continuous features (complexity, 3× playtime, min/max players, manufacturer + community recommended age).

Targets: `quality_score` and `commercial_score` — each trained as an independent model on the same feature set; the interface always reports both side by side.

Each algorithm is run against both targets; results are compared to determine which performs best.

| Algorithm              | Reason                                                                                                                         | In *ML with R*? |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| **Linear Regression**  | Simple starting point; with 400+ binary features, regularisation (Ridge/Lasso) will likely be needed to avoid multicollinearity — coefficients directly show which mechanics and themes correlate with each score | ✓ Ch 6          |
| **Regression Trees**   | Naturally handles the ~400 sparse binary mechanic/theme columns without scaling; splits reveal non-linear feature interactions | ✓ Ch 6          |
| **Random Forest**      | Averages many trees to reduce the overfitting a single tree is prone to on high-dimensional sparse data like ours              | ✗ later         |
| **LightGBM / XGBoost** | Gradient-boosted trees are consistently state-of-the-art on tabular data with mixed binary + continuous features at this scale | ✗ later         |

Algorithms we will not try:

| Algorithm                     | Reason why not                                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **KNN Regression**            | Distances become meaningless in 400+ sparse binary dimensions — games that share few mechanics will appear equally "far" from everything |
| **Support Vector Regression** | Slow to train at 22k × 400+ columns and hard to interpret; tree ensembles will outperform it with less tuning effort                     |
| **Neural Network**            | 22k rows is too small to give neural nets an edge over tree ensembles on tabular data; adds complexity without a likely accuracy benefit |

## 3. Simulation & Exploration

### Pre-processing for simulation (clustering)

After training, cluster mechanics and themes to create a human-friendly control layer. Clusters are never fed into the model — instead each cluster stores a **centroid vector** (average raw feature values of all games in that cluster), which is what actually gets passed to the model.

| Algorithm | Applied to | Notes |
| --- | --- | --- |
| **Louvain community detection** | Mechanics co-occurrence graph | Finds natural mechanic families; better fit than k-means for graph-structured data |
| **Louvain community detection** | Themes co-occurrence graph | Same rationale |
| **k-means** (fallback) | Either | Simpler to implement in R if Louvain proves impractical |

k-means fallback — finding the natural number of clusters:
- **Elbow plot**: plot inertia vs k; look for the point of diminishing returns
- **Silhouette score**: peak score indicates the most natural k
- Run both for k = 2–30 separately — mechanics and themes may converge on different values
- Inspect clusters qualitatively: if two are semantically indistinguishable, merge and re-run

**Example — what a mechanic cluster looks like:**

Training data (each game is a row of raw mechanic columns):

| game | deck_building | worker_placement | auction | cooperative | dice_rolling |
| --- | --- | --- | --- | --- | --- |
| Wingspan | 1 | 0 | 0 | 0 | 0 |
| Agricola | 0 | 1 | 0 | 0 | 0 |
| Viticulture | 0 | 1 | 1 | 0 | 0 |
| Pandemic | 0 | 0 | 0 | 1 | 0 |

After clustering, centroid vectors per cluster (averaged from games in that cluster):

| cluster               | deck_building | worker_placement | auction | cooperative | dice_rolling |
| --------------------- | ------------- | ---------------- | ------- | ----------- | ------------ |
| "Resource & Trading"  | 0.31          | 0.78             | 0.52    | 0.05        | 0.12         |
| "Team & Cooperative"  | 0.18          | 0.09             | 0.03    | 0.91        | 0.22         |

When the user picks "Team & Cooperative", the model receives the centroid row — same format as any training row.

### Flow

```
user picks mechanic cluster + theme cluster + other constraints
→ decode clusters to centroid vectors
→ assemble full feature vector (same format as training)
→ prediction model → success_score
→ compute development_viability on demand for that feature vector
→ rank and filter results by both axes independently
```

### Development Viability

Computed on demand — not precomputed for the whole dataset, since it's a judgment call rather than a data-driven score.
- Assign an AR difficulty weight to each mechanic and theme manually, based on your own assessment of what's feasible on Vision Pro
- Aggregate across the feature vector; apply additional penalties for high language density and complexity

### Interface

A simple HTML page with:
- **Mechanic cluster dropdown** — select a cluster (e.g. "Team & Cooperative"); after selecting, the individual mechanics in that cluster are shown and can be toggled on/off to refine the feature vector away from the centroid
- **Theme cluster dropdown** — same pattern: select a cluster, then optionally adjust individual themes within it
- **Continuous feature inputs** — sliders or number inputs for complexity, playtime, player count, recommended age
- **Predict button** — assembles the feature vector and runs it through the trained model
- **Output**: `quality_score`, `commercial_score`, and `development_viability` displayed side by side
- **Reference table**: top N real games closest to the current profile, for context

### Future ideas

- **Global SHAP chart**: a precomputed bar chart (generated once after training) showing which mechanics and themes most positively and negatively affect `success_score` — displayed as a static reference panel so you always know which levers matter most
- **Local SHAP per prediction**: instead of a global chart, explain why *this specific input* got its score — more complex to implement in the interface but more actionable
- **LLM interface**: allow an LLM to interact with the model in the same way as the HTML interface — translating natural language game descriptions into feature vectors, querying the model, and iterating on results conversationally

