# How Linear Regression Works at Runtime

## The prediction formula

Every prediction is one formula:

```
score = intercept + (w1 × feature1) + (w2 × feature2) + ... + (w400 × feature400)
```

That's it. 400 multiplications, added together. No branching, no trees, no
decisions. The entire model is just 401 numbers: 400 weights and 1 intercept.

## Training: how those 401 numbers are found

### Start with 1 feature

Forget 400 features for a moment. Say we only have GameWeight, and we want to
predict quality_score. We can plot this:

```
quality_score
    90 |
    80 |          ·  ·
    70 |       · · ·· ·
    60 |    · ·· · ·
    50 |  · · ·
    40 | ·
       +------------------
       1    2    3    4    5
                GameWeight
```

Each dot is one game. Linear regression draws the straight line that gets as
close as possible to all the dots:

```
quality_score
    90 |              /
    80 |          · /· 
    70 |       · ·/·· ·
    60 |    · ··/ ·
    50 |  · · /
    40 | ·  /
       +------------------
       1    2    3    4    5
                GameWeight
```

That line has two numbers: a **slope** (how much quality_score goes up per unit
of GameWeight) and an **intercept** (the score when GameWeight is 0). Say the
best line has slope = 8 and intercept = 40. Then:

```
predicted_score = 40 + (8 × GameWeight)
```

A game with GameWeight 3.0 gets: 40 + (8 × 3.0) = **64**

### How "best" is defined

"Best" means: the line where the total squared distance from every dot to the
line is as small as possible.

```
quality_score
    80 |          ·  ·        The vertical gaps between
    70 |       · |·· ·        each dot and the line are
    60 |    · ··| ·           the errors. Square each one,
    50 |  · · |               add them up. The best line
    40 | ·  |                 makes that sum smallest.
       +------------------
```

Why squared? Because errors above and below the line should both count (squaring
makes negatives positive), and because big misses should hurt more than small
ones (squaring a 10-point error gives 100, while squaring a 2-point error gives
only 4).

### Finding the best line with 1 feature

With 1 feature, we need to find 2 numbers: slope and intercept. We have 17,500
games, each giving us one equation:

```
Game 1:    actual_score_1 ≈ intercept + slope × GameWeight_1
Game 2:    actual_score_2 ≈ intercept + slope × GameWeight_2
...
Game 17500: actual_score_17500 ≈ intercept + slope × GameWeight_17500
```

17,500 equations, 2 unknowns. No pair of numbers will satisfy all equations
exactly (the dots don't form a perfect line), but there's exactly one pair that
minimizes the total squared error. Calculus gives a direct formula for it, no
searching required.

### Scaling up to 400 features

With 2 features (say GameWeight and MinPlayers), the line becomes a flat plane
in 3D space. Instead of fitting a line through dots on a 2D graph, you're
tilting a sheet of paper in 3D to get as close to the dots as possible. Now you
need 3 numbers: intercept, slope for GameWeight, slope for MinPlayers.

With 400 features, it's the same idea in 401 dimensions. We can't visualize it,
but the math is identical. We need 401 numbers (400 slopes + 1 intercept), we
have 17,500 equations (one per game), and there's one unique solution that
minimizes total squared error.

```
Game 1:    score_1 ≈ intercept + w1×GameWeight_1 + w2×MinPlayers_1 + ... + w400×Bingo_1
Game 2:    score_2 ≈ intercept + w1×GameWeight_2 + w2×MinPlayers_2 + ... + w400×Bingo_2
...
Game 17500: ...
```

17,500 equations, 401 unknowns. Linear algebra solves this in one shot. No
iteration, no guessing, no trees. The computer essentially inverts a big matrix
(a 400×400 grid of numbers that captures how all features relate to each other)
and out pop the 401 weights. This is why training takes 0.3 seconds.

### Concrete example

Say the trained model found these weights (real values from the notebook):

```
intercept          = 52.0
TableauBuilding    = +3.8
Cat:Strategy       = +3.1
Legacy Game        = +3.0
GameWeight (scaled)= +2.5
...388 more weights...
Bingo              = +0.2
```

For a heavy strategy game with tableau building, no legacy, GameWeight 3.5:

```
score = 52.0
      + (3.8 × 1)       TableauBuilding = yes
      + (3.1 × 1)       Cat:Strategy = yes
      + (3.0 × 0)       Legacy Game = no
      + (2.5 × 0.75)    GameWeight 3.5 (after scaling, roughly 0.75)
      + ...              388 more terms, most binary features are 0
      + (0.2 × 0)       Bingo = no
      = 52.0 + 3.8 + 3.1 + 0 + 1.875 + ... 
      ≈ 72
```

Most of the 400 terms are `weight × 0` (the game doesn't have that mechanic or
theme), so they contribute nothing. The prediction is really driven by the
handful of features this specific game has.

## Why scaling matters

Before training, the continuous features are scaled to mean=0 and standard
deviation=1. Without this:

- ComMaxPlaytime ranges 0-600
- MinPlayers ranges 1-10

If both matter equally, ComMaxPlaytime's weight would need to be 60x smaller
than MinPlayers' weight just to compensate for the scale difference. The model
would still find the right answer eventually, but Ridge regularization (which
penalizes large weights) would unfairly penalize MinPlayers' bigger weight. 
Scaling puts everything on the same footing first.

Binary features (0/1) don't need scaling because they're already on a comparable
scale.

## Ridge: why not just find the perfect weights?

With 400 features and only 17,500 games, some features are correlated.
ComMinPlaytime and ComMaxPlaytime move together. The model might learn:

```
ComMinPlaytime = +50
ComMaxPlaytime = -48
```

This cancels out to roughly +2, which is the real effect of playtime. But those
extreme weights are fragile: a new game where min and max playtime don't move
together gets a wildly wrong prediction.

Ridge adds a penalty: "find good weights, but also keep them small." It trades
a tiny bit of accuracy on the training games for much more stable predictions on
new games. The penalty strength is a dial: too low and weights blow up, too high
and the model becomes too conservative. RidgeCV automatically tests several
values and picks the best one.

## The fundamental limitation

The formula is always:

```
score = intercept + (w1 × feature1) + (w2 × feature2) + ...
```

Each feature adds a fixed number of points, always. TableauBuilding is always
+3.8 whether the game is a 4.5-complexity euro or a 1.5-complexity kids' game.
In reality, tableau building probably matters more in complex games. But the
formula has no way to express "TableauBuilding adds more when GameWeight is
high." Each weight is independent.

That's why the Gloomhaven commercial prediction overshoots to 109.4 on a 0-100
scale. The model can't learn "these features combine differently for top-tier
games." It just keeps adding positive weights and goes past 100.

Trees don't have this problem because they can nest questions: "IF GameWeight >
3 AND has TableauBuilding THEN predict 82." The prediction depends on the
combination of features, not each one independently.
