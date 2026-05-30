# How Tree-Based Models Work at Runtime

## Part 1: Decision Tree

### The prediction

A decision tree is a flowchart. A game enters at the top, answers yes/no
questions, and lands on a leaf that holds a predicted score.

```
                    Is GameWeight > 3.0?
                    /                  \
                 Yes                    No
          Is MaxPlayers > 4?        Is Cat:Party = 1?
          /              \          /              \
       Yes                No     Yes                No
    Leaf A             Leaf B   Leaf C            Leaf D
    avg: 78            avg: 72  avg: 55           avg: 52
```

A heavy strategy game for 2 players: GameWeight 3.5 (> 3.0 → left), MaxPlayers
2 (not > 4 → right) → lands on **Leaf B → predicted score: 72**.

A light party game: GameWeight 1.8 (not > 3.0 → right), Cat:Party = 1 (→ left)
→ lands on **Leaf C → predicted score: 55**.

### What a leaf actually holds

Each leaf stores the **average score of all training games that landed there**.
If 200 training games followed the same path to Leaf B, and their actual scores
ranged from 60 to 85, Leaf B's prediction is their average: 72. Every new game
that takes the same path gets that same number.

### How the tree is built (training)

The algorithm starts with all 17,500 training games in one pile and asks: "what
single yes/no question separates high-scoring games from low-scoring ones best?"

It tries **every possible question** exhaustively:

```
Is GameWeight > 1.0?  → measure how well this groups similar scores
Is GameWeight > 1.1?  → measure
Is GameWeight > 1.2?  → measure
...
Is GameWeight > 4.9?  → measure
Is MinPlayers > 1?    → measure
Is MinPlayers > 2?    → measure
...
Does it have Worker Placement?  → measure  (binary: only one threshold)
Does it have Dice Rolling?      → measure
...all 400 features × all their unique values
```

For each candidate question, it splits the 17,500 games into two groups and
measures the total squared error within each group (how spread out are the
scores in each group?). The question that creates the most uniform groups wins.

This is thousands of candidates, but it's just arithmetic on sorted arrays.
Fast.

The winner becomes the root question. Now the algorithm has two groups. It
repeats the same exhaustive search independently on each group. And then on
each sub-group. And so on, up to `MAX_DEPTH = 10` levels deep.

```
Level 1:  1 pile    → best question splits into 2 groups
Level 2:  2 groups  → each finds its own best question → 4 groups  
Level 3:  4 groups  → each finds its own best question → 8 groups
...
Level 10: up to 512 groups → each finds its own best → up to 1,024 leaves
```

The tree stops when:
- It hits depth 10 (no more questions allowed)
- Splitting further doesn't reduce error (all remaining candidates are useless)

In practice your quality tree ends up with 395 leaves, not 1,024, because many
branches stopped early when further splitting didn't help.

### The greedy trade-off

The algorithm is **exhaustive within one level** (it tries every possible
question) but **greedy across levels** (it never reconsiders). Once it commits
to "GameWeight > 3.0" at the top, it never asks "what if I had started with
MinPlayers instead — would the overall tree be better?"

Looking ahead to find the globally optimal combination of all 10 levels of
questions would require evaluating every possible tree, which is computationally
impossible with 400 features. So it settles for: best question now, then best
question for each resulting group, and so on.

This means the same data with the same settings always produces the exact same
tree. It's deterministic.

### Saved model

The `.pkl` file contains one tree: every node's question (feature + threshold),
and every leaf's average score. At prediction time, a game walks down the tree,
answers the questions, and gets the leaf value. No retraining, no recalculation.

## Part 2: Random Forest

### The problem with a single tree

A single tree is fragile. Remove a few games from the training set and the root
question might change completely, which reshuffles the entire tree. One tree's
blind spots are systematic: if the greedy choice at level 1 was suboptimal,
every prediction is affected.

### The fix: build many different trees

A Random Forest trains (typically) 100 trees, but each tree sees a different
version of the data:

**Different rows:** Each tree gets a random sample of ~63% of the 17,500
training games (sampled with replacement, so some games appear twice, others
not at all). This is called "bagging."

**Different features per split:** At each node, instead of evaluating all 400
features, the tree only gets to consider a random subset (e.g. 20). So tree #1
might not even be allowed to ask about GameWeight at level 1, while tree #2 can.

Each tree is still built with the same greedy, exhaustive algorithm as above,
but they see different data and different feature subsets, so they produce
different trees.

### Example: 3 trees

```
Tree #1 (saw 11,000 games, random feature subsets):

                Is ComMaxPlaytime > 90?
                /                     \
             Yes                       No
      Is GameWeight > 2.5?       Is Cat:Family = 1?
      /              \           /              \
   Leaf: 74        Leaf: 68   Leaf: 58        Leaf: 51


Tree #2 (saw a different 11,000 games, different feature subsets):

                Is GameWeight > 3.2?
                /                   \
             Yes                     No
      Is Legacy Game = 1?     Is MinPlayers > 1?
      /              \        /              \
   Leaf: 81       Leaf: 73  Leaf: 56       Leaf: 48


Tree #3 (saw yet another 11,000 games, different feature subsets):

                Is Cat:Strategy = 1?
                /                   \
             Yes                     No
      Is GameWeight > 2.8?     Is MfgPlaytime > 45?
      /              \         /               \
   Leaf: 76        Leaf: 65  Leaf: 54         Leaf: 50
```

Notice: Tree #1 starts with playtime, Tree #2 with complexity, Tree #3 with
category. They ask completely different questions because they saw different
feature subsets at the root.

### Prediction: average all trees

A new game runs through every tree independently and you average the results:

```
Heavy strategy game (GameWeight 3.5, ComMaxPlaytime 120, Cat:Strategy, 2 players):

  Tree #1: ComMaxPlaytime 120 > 90 → Yes → GameWeight 3.5 > 2.5 → Yes → 74
  Tree #2: GameWeight 3.5 > 3.2 → Yes → Legacy = 0 → No → 73
  Tree #3: Cat:Strategy = 1 → Yes → GameWeight 3.5 > 2.8 → Yes → 76

  Final prediction = (74 + 73 + 76) / 3 = 74.3
```

Each tree is noisy on its own, but their errors point in different directions
(because they saw different data). Averaging cancels out the noise. This is the
core insight: many mediocre but *diverse* predictors, averaged together, beat
one carefully built predictor.

### Saved model

The `.pkl` file contains all 100 trees, each with its full structure (questions,
thresholds, leaf values). At prediction time, the game runs through all 100
trees and the results are averaged. The random feature subsets are not saved,
they were only needed during training to create diverse trees.

## Part 3: LightGBM (Gradient Boosting)

### Different philosophy

Random Forest builds many trees **independently** and averages them.
LightGBM builds trees **sequentially**, where each tree corrects the mistakes
of all previous trees.

### Step by step

**Tree #1** is a normal (but shallow, ~depth 4) decision tree trained on actual
scores:

```
Tree #1:
                Is GameWeight > 3.0?
                /                  \
             Yes                    No
          Leaf: 72               Leaf: 53
```

(Simplified to depth 1 for clarity. Real trees are deeper.)

Run all 17,500 training games through it and record the errors:

| Game | Predicted | Actual | Residual (error) |
| --- | --- | --- | --- |
| Gloomhaven | 72 | 85 | +13 |
| Pandemic | 72 | 75 | +3 |
| Catan | 53 | 70 | +17 |
| Uno | 53 | 40 | -13 |
| Dobble | 53 | 52 | -1 |

Tree #1 lumps all complex games together at 72 and all simple games at 53.
Many individual games are significantly off.

**Tree #2** is trained on the **residuals** as the target. It doesn't see the
original scores at all. It only sees: "Gloomhaven: +13, Pandemic: +3, Catan:
+17, Uno: -13, Dobble: -1" and tries to predict those errors.

```
Tree #2 (target = residuals from tree #1):

                Is Cat:Strategy = 1?
                /                  \
             Yes                    No
          Leaf: +8               Leaf: -7
```

Tree #2 discovered that tree #1 tends to underpredict strategy games (positive
residuals) and overpredict non-strategy games (negative residuals).

**Combined prediction** after 2 trees:

```
Gloomhaven:  tree1(72) + tree2(+8) = 80    (actual: 85, still off by 5)
Pandemic:    tree1(72) + tree2(+8) = 80    (actual: 75, off by -5)
Uno:         tree1(53) + tree2(-7) = 46    (actual: 40, off by -6)
Dobble:      tree1(53) + tree2(-7) = 46    (actual: 52, off by -6)
```

Better, but not great. So we compute new residuals:

| Game | Combined prediction | Actual | New residual |
| --- | --- | --- | --- |
| Gloomhaven | 80 | 85 | +5 |
| Pandemic | 80 | 75 | -5 |
| Uno | 46 | 40 | -6 |
| Dobble | 46 | 52 | +6 |

**Tree #3** trains on these new residuals. Maybe it finds that games with high
player interaction tend to be underpredicted:

```
Tree #3 (target = residuals from tree1 + tree2):

                Is MaxPlayers > 5?
                /                 \
             Yes                   No
          Leaf: +4              Leaf: -3
```

Each tree chips away at the remaining error. After 300 trees:

```
final_score = tree1(game) + tree2(game) + tree3(game) + ... + tree300(game)
```

### Why the individual trees are shallow

Each tree only needs to capture one small pattern in the remaining error. A
depth-4 tree with 16 leaves is enough for that. Making correction trees deep
would overfit: tree #2 would memorize the specific errors of tree #1 on the
training data rather than learning general patterns.

This is the opposite of Random Forest, where each tree tries to predict the
full score and needs to be deep enough to do a reasonable job.

### Learning rate

In practice, each tree's contribution is multiplied by a small number (say 0.1)
before adding:

```
final_score = tree1(game) + 0.1 × tree2(game) + 0.1 × tree3(game) + ...
```

This slows down the correction process on purpose. Without it, tree #2 might
overcorrect, then tree #3 overcorrects the overcorrection, and the predictions
oscillate wildly. A small learning rate means each tree makes a gentle nudge,
and you need more trees, but the result is much more stable.

### Saved model

The saved model contains all 300 trees in order, plus the learning rate. At
prediction time, the game runs through every tree and the outputs are summed
(not averaged like Random Forest). The order matters because tree #2's output
is a correction magnitude (+8), not a full score.

## Summary: what's in each saved model

| Model | Saved | Prediction |
| --- | --- | --- |
| Linear Regression | 401 numbers (400 weights + intercept) | Multiply features by weights, sum |
| Decision Tree | 1 tree (questions + leaf averages) | Walk down, get leaf value |
| Random Forest | 100 independent trees | Walk all 100, average results |
| LightGBM | ~300 sequential trees + learning rate | Walk all 300, sum results |
