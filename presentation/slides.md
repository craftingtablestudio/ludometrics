---
theme: seriph
title: Ludometrics
info: |
  Machine learning models that predict board game success from BoardGameGeek data.
class: text-left
background: https://images.unsplash.com/photo-1633506079263-7029b4f46762?q=80&w=3072&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
drawings:
  persist: false
transition: fade
duration: 12min
---

<div class="absolute inset-0 bg-black/50"></div>

<div class="relative z-10">

# Ludometrics

Predicting board game success from BoardGameGeek (BGG) data

<div class="mt-4 h-1 w-42 bg-[#D8A33A] rounded-full"></div>

<br>

**Luca Ban** · Special Topic Assessment 1

<span class="opacity-50">Machine learning models for quality and commercial potential</span>

<img src="/ludometrics_logo.png" class="absolute right-12 bottom-12 w-56 drop-shadow-xl" />

</div>

<!--
Introduce the topic as a practical ML question: before building a board game, can we estimate whether the design profile points toward success?
-->

---

# Research Question

<div class="text-4xl leading-tight mt-10 max-w-220">
Can we predict whether a board game will be successful before it is even built?
</div>

<div class="mt-12 grid grid-cols-3 gap-5 text-xl">
  <div class="border-l-4 border-[#59788E] pl-5">
    Use real BoardGameGeek data
  </div>
  <div class="border-l-4 border-[#6A8F5F] pl-5">
    Train four regression approaches
  </div>
  <div class="border-l-4 border-[#D8A33A] pl-5">
    Compare prediction quality
  </div>
</div>

<!--
Speaker line: I wanted to know if there is enough signal in the design profile of a board game to predict success before release.
-->

---

# Dataset Choice

I compared several board game datasets on Kaggle and chose the largest, most complete BoardGameGeek dataset I could find.

<div class="grid grid-cols-5 gap-7 mt-7">
  <div class="col-span-3 text-sm leading-tight">

| Source file         |   Rows | Columns | Column kinds                                                              |
| ------------------- | -----: | ------: | ------------------------------------------------------------------------- |
| `games.csv`         | 21,925 |      48 | IDs/text; numeric ratings, counts and ranks; lists/images; category flags |
| `mechanics.csv`     | 21,925 |     158 | `BGGId` plus 157 binary mechanic flags                                    |
| `themes.csv`        | 21,925 |     218 | `BGGId` plus 217 binary theme flags                                       |
| `subcategories.csv` | 21,925 |      11 | `BGGId` plus 10 binary subcategory flags                                  |

  </div>
  <div class="col-span-2 text-lg leading-relaxed">
    <p><b>Why this dataset?</b></p>
    <ul>
      <li>Collected from BGG's documented Application Programming Interface (API)</li>
      <li>Reflects real user activity</li>
      <li>Includes both design features and outcome signals</li>
      <li>Bayesian rating values were checked before use</li>
    </ul>
  </div>
</div>

<!--
Avoid overselling the dataset as perfect. Say it is strong because it has both inputs and outcomes, and because the rating target was verified.
-->

---

# Defining Success

In this project, "success" has two meanings. Both became model targets, but they started from different raw signals.

<div class="mt-5 text-sm leading-snug">

| Target             | Raw values found                     | Cleaning and final scale                                                   |
| ------------------ | ------------------------------------ | -------------------------------------------------------------------------- |
| `quality_score`    | `BayesAvgRating` complete; 3.57-8.51 | No imputation; multiply by 10 -> 35.7-85.1 observed                        |
| `commercial_score` | `NumOwned` 0-166,497; one zero       | No target imputation; years-active adjustment + log + min-max -> 0.0-100.0 |

</div>

<div class="grid grid-cols-2 gap-5 mt-5 text-base leading-snug">
  <div class="border-l-4 border-[#59788E] pl-4">
    <b>Quality</b><br />
    Are players who rated it positive about it? Bayesian averaging keeps low-vote games near the site-wide average until enough real ratings pull them away.
  </div>
  <div class="border-l-4 border-[#D8A33A] pl-4">
    <b>Commercial</b><br />
    Did many people own it, accounting for time? Years active was clamped from 1 to 10 before log-scaling the ownership count.
  </div>
</div>

<div class="mt-6 text-2xl leading-relaxed border-l-4 border-[#C26B4E] pl-5">
Quality and commercial success are related, but they are not the same thing.
</div>

<!--
Use "success" as the umbrella word, but say every chart is labelled as quality or commercial.
-->

---

# Preprocessing Problems

The raw data needed cleaning before it was safe to model.

<div class="mt-5">

| Raw issue                            | Why it matters                        | Fix                                             |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------- |
| Age fields had gaps                  | Age is an accessibility signal        | Use manufacturer age to fill community age gaps |
| `MaxPlayers` max was 999; p95 was 10 | 999 behaves like a sentinel value     | Cap at 20                                       |
| Playtime max was 60,000 minutes      | Extreme values distort learning       | Cap at 600                                      |
| `GameWeight` had 506 zeros           | Zero means missing, not no complexity | Replace with median                             |
| Mechanics and themes were 0/1        | Already model-ready                   | Keep as binary flags                            |

</div>

<div class="mt-4 text-sm opacity-70 leading-snug">
Example: `ComAgeRec` was missing for 5,530 games, so missing values were filled from non-zero `MfgAgeRec`, then the median. `MfgAgeRec` zeros were also replaced with its median, and both age columns stayed in the model.
</div>

<!--
This slide is evidence for the preprocessing rubric. It shows actual discovered issues, not just generic cleaning.
-->

---

# Processed Dataset

<div class="grid grid-cols-4 gap-5 mt-10 text-center">
  <div>
    <div class="text-5xl font-bold text-[#59788E]">21,925</div>
    <div class="mt-2 opacity-70">games</div>
  </div>
  <div>
    <div class="text-5xl font-bold text-[#6A8F5F]">400</div>
    <div class="mt-2 opacity-70">input features</div>
  </div>
  <div>
    <div class="text-5xl font-bold text-[#D8A33A]">2</div>
    <div class="mt-2 opacity-70">target scores</div>
  </div>
  <div>
    <div class="text-5xl font-bold text-[#C26B4E]">0</div>
    <div class="mt-2 opacity-70">missing values</div>
  </div>
</div>

<div class="mt-14 text-xl leading-relaxed">

| Feature group                                                 | Count |
| ------------------------------------------------------------- | ----: |
| Continuous features: complexity, players, playtime, age       |     8 |
| Binary features: categories, mechanics, themes, subcategories |   392 |
| Targets: `quality_score`, `commercial_score`                  |     2 |

</div>

<!--
The final CSV is data/games_processed.csv. It has 403 columns total: BGGId, 400 features, and 2 targets.
-->

---

# Four Algorithms

Each algorithm was trained separately for `quality_score` and `commercial_score`.

<div class="mt-6">

| Algorithm                                  | Why try it?                                                   |
| ------------------------------------------ | ------------------------------------------------------------- |
| Linear Regression                          | Simple interpretable baseline: one learned weight per feature |
| Decision Tree                              | Learns yes/no rules and captures non-linear splits            |
| Random Forest                              | Averages many trees to reduce single-tree instability         |
| Light Gradient Boosting Machine (LightGBM) | Builds boosted trees that correct previous errors             |

</div>

<div class="mt-8 text-xl border-l-4 border-[#59788E] pl-5">
Same 400 features, same train/test split, same two target scores.
</div>

<div class="mt-5 text-base opacity-70">
Written in Python with pandas, scikit-learn, LightGBM, and matplotlib.
</div>

<!--
This gives the audience the map before going one slide deeper on each algorithm.
-->

---

# Linear Regression

<div class="algorithm-detail mt-2">
  <div class="algorithm-top">
    <div>
      <img src="/diagram_linear_regression.png" />
    </div>
    <div class="text-lg leading-snug">
      <p><b>Runtime idea</b></p>
      <p>Each of the 400 features gets a <b>weight</b> — a number that says how much that feature pushes the predicted score up or down. The final prediction is just all 400 weights multiplied by the game's features, added together.</p>
    </div>
  </div>
  <div class="text-lg leading-snug">
    <p>A weight of +3.8 on <code>TableauBuilding</code> means having that mechanic adds 3.8 points to the predicted quality score. We use <b>Ridge regression</b>, which prevents any single feature from getting an unreasonably large weight — important when you have 400 features and many are correlated.</p>
    <p class="opacity-70 text-base mt-2">Limitation: one fixed effect per feature. It can't say "this mechanic helps only in strategy games but not party games."</p>
  </div>
</div>

<!--
Limitation: one fixed effect per feature. It cannot easily say "this mechanic helps only in this context."
-->

---

# Decision Tree

<div class="algorithm-detail mt-2">
  <div class="algorithm-top">
    <div>
      <img src="/diagram_decision_tree.png" />
    </div>
    <div class="text-lg leading-snug">
      <p><b>Runtime idea</b></p>
      <p>The model works like a flowchart: "Is the complexity above 3?" If yes, go left; if no, go right. "Does it have worker placement?" Left or right again. After up to 10 of these yes/no questions, you land on a <b>leaf</b> — the average score of all training games that answered the same way.</p>
    </div>
  </div>
  <div class="text-lg leading-snug">
    <p>A depth-10 tree could have up to 1,024 leaves, but ours ended up with ~400 because many branches get cut short when splitting further didn't help separate scores.</p>
    <p class="opacity-70 text-base mt-2">Limitation: all games that answer the same 10 questions land in one leaf and get the same prediction, even if their actual scores differ. One tree is readable but coarse.</p>
  </div>
</div>

<!--
Limitation: many different games can land in the same leaf, so one tree is readable but coarse.
-->

---

# Random Forest

<div class="algorithm-detail mt-2">
  <div class="algorithm-top">
    <div>
      <img src="/diagram_random_forest.png" />
    </div>
    <div class="text-lg leading-snug">
      <p><b>Runtime idea</b></p>
      <p>Instead of one fragile tree, build 200 — each trained on a random sample of games and a random subset of features. Every tree makes its own prediction, and the forest averages them all together.</p>
    </div>
  </div>
  <div class="text-lg leading-snug">
    <p>This "wisdom of the crowd" approach means that even if individual trees make bad predictions, their errors cancel out across 200 trees. Each split only considers ~20 of the 400 features (√400 ≈ 20), so the forest collectively covers far more of the feature space than any single tree could.</p>
    <p class="opacity-70 text-base mt-2">Limitation: averaging smooths out extremes. A game with an unusually high score gets pulled toward the middle because most trees haven't seen enough similar examples.</p>
  </div>
</div>

<!--
The notebook shows the first five trees can disagree strongly. The forest becomes more stable by averaging them.
-->

---

# LightGBM

<div class="algorithm-detail mt-2">
  <div class="algorithm-top">
    <div>
      <img src="/diagram_gradient_boosting.png" />
    </div>
    <div class="text-lg leading-snug">
      <p><b>Runtime idea</b></p>
      <p>Instead of independent trees, LightGBM builds them in a chain where each new tree focuses specifically on fixing the mistakes the previous trees made. The first tree makes rough predictions. The second tree looks at where those were wrong and corrects just the errors. The third corrects the remaining errors, and so on for 500 rounds.</p>
    </div>
  </div>
  <div class="text-lg leading-snug">
    <p>Each tree only applies <b>5% of its correction</b> (the learning rate), forcing the model to take many small careful steps rather than fewer large ones that might overshoot.</p>
    <p class="opacity-70 text-base mt-2">This "learn from your mistakes" strategy is called gradient boosting, and it consistently produces the best results on tabular data like ours.</p>
  </div>
</div>

<!--
This is the model that won. Important features were age, complexity, playtime, and player-count related values.
-->

---

# Demo: Running the Pipeline

<div class="mt-8 border-2 border-dashed border-[#59788E] rounded-lg h-60 flex items-center justify-center text-3xl opacity-80">
  Insert screen recording here
</div>

<div class="mt-8">

```sh
uv run run-notebooks --splits 80_20
```

</div>

<div class="mt-6 text-xl leading-relaxed">
The demo should show the four training notebooks running headlessly, both targets being trained, and the result tables being updated.
</div>

<!--
Keep the recording under 120 seconds. Treat it as proof the pipeline runs, not as the full explanation.
-->

---

# Evaluation: Prediction Error

<div class="text-xl mb-4">
MAE (Mean Absolute Error) is the most intuitive metric — it is the average distance between each prediction and the true value. RMSE (Root Mean Square Error) squares each error before averaging, so large misses are penalised much more heavily than small ones. This makes RMSE the stricter metric and a better choice when outlier errors are costly.
</div>

<div class="grid grid-cols-2 gap-5">
  <img src="/chart_error_quality_score.png" class="w-full rounded" />
  <img src="/chart_error_commercial_score.png" class="w-full rounded" />
</div>

|                      | MAE     | RMSE     |
| -------------------- | ------- | -------- |
| **Quality score**    | 1.3 pts | 2.32 pts |
| **Commercial score** | 6.8 pts | 8.78 pts |

<!--
Use "prediction error", not accuracy. Lower RMSE is better. MAE complements RMSE by showing the typical (non-squared) error.
-->

---

# Evaluation: Error Spread

<div class="text-xl mb-4">
An average doesn't tell you if most predictions are close with a few wild misses, or everything is moderately off. The 50th percentile is the "typical" miss; the 90th percentile shows how bad the hardest 10% of games get.
</div>

<div class="grid grid-cols-2 gap-5">
  <img src="/chart_cum_error_quality_score.png" class="w-full rounded" />
  <img src="/chart_cum_error_commercial_score.png" class="w-full rounded" />
</div>

|                      | 50th percentile (median) | 90th percentile |
| -------------------- | ------------------------ | --------------- |
| **Quality score**    | 0.58 pts                 | 3.40 pts        |
| **Commercial score** | 5.56 pts                 | 14.45 pts       |

<!--
This slide is useful because the median error is easier to understand than RMSE, while the 90th percentile shows the hard cases.
-->

---

# Evaluation: Quality Score

<div class="text-xl mb-4">
R² (Coefficient of Determination) measures how much of the variation in scores the model explains. A perfect model scores 1.0; guessing the average for every game scores 0.
</div>

<img src="/chart_r2_quality_score.png" class="w-full rounded" style="height: 50%; width: auto" />

|                   | R²    | MAE     | RMSE     |
| ----------------- | ----- | ------- | -------- |
| **Quality score** | 0.662 | 1.3 pts | 2.32 pts |

<div class="mt-2 text-xl">
The scatter shows predictions tracking actuals well in the dense mid-range. The residual histogram centres on zero with a slight right skew where the model under-predicts a few high-rated games.
</div>

<!--
The scatter plot shows prediction vs actual; the histogram shows residual distribution. Together they tell the full accuracy story for quality_score.
-->

---

# Evaluation: Commercial Score

<div class="text-xl mb-4">
Similar explanatory power. The remaining scatter is expected: after 500 rounds of corrections, what's left is information our features simply don't contain.
</div>

<img src="/chart_r2_commercial_score.png" class="w-full rounded" style="height: 50%; width: auto" />

|                      | R²    | MAE     | RMSE     |
| -------------------- | ----- | ------- | -------- |
| **Commercial score** | 0.635 | 6.8 pts | 8.78 pts |

<div class="mt-2 text-xl">
No amount of extra boosting rounds can fix what the data doesn't have. Artwork quality, publisher reach, marketing spend, timing, and luck are not in the dataset.
</div>

<!--
Same layout as the quality slide. The pair lets the audience compare both targets side by side across two slides.
-->

---

# Why LightGBM Won

<div class="mt-6">

| Model             | What it taught us                               | Limitation                                        |
| ----------------- | ----------------------------------------------- | ------------------------------------------------- |
| Linear Regression | Feature weights are readable                    | One fixed effect per feature is too rigid         |
| Decision Tree     | Rules are easy to explain                       | One tree groups too many different games together |
| Random Forest     | Averaging many trees improves stability         | Extreme hits get pulled toward the middle         |
| LightGBM          | Sequential correction captures subtler patterns | Still limited by missing real-world factors       |

</div>

<div class="mt-8 text-2xl border-l-4 border-[#6A8F5F] pl-5">
Board game success depends on combinations of design choices, not isolated features.
</div>

<!--
This explains the model comparison in plain language.
-->

---

# Conclusion

<div class="text-3xl leading-tight mt-8">
Yes, board game success is partly predictable from design-time features. With the help of LightGBM we can conclude:
</div>

<div class="mt-10">

| Target             | On average correct within (RSME) | With explained variation (R²) |
| ------------------ | -------------------------------: | ----------------------------: |
| Quality success    |                       2.3 points |                           66% |
| Commercial success |                       8.8 points |                           64% |

</div>

<div class="mt-10 text-2xl leading-relaxed">
The model detects a real signal before a game exists, but it cannot see everything that makes a game succeed.
</div>

<!--
Final speaker line: I cannot perfectly predict success, but I can measure a real signal in the design profile.
-->
