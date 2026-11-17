# Prospective 50-match evaluation

This log is maintained in GitHub on the `development` branch. Do not use a separate user spreadsheet.

## Rules
1. Include the next 50 eligible fixtures in chronological order across the six supported competitions (Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and Allsvenskan). Do not select matches because the model shows a large disagreement or a favourable-looking probability.
2. Freeze each forecast before kickoff. Save the timestamp, model version/commit, Elo H/D/A, Dixon–Coles H/D/A, and bookmaker 1X2 odds/probabilities available at that time.
3. Use the same match set for every model comparison. Record closing odds separately from the earlier snapshot when available; never substitute post-kickoff prices.
4. After full time, record the score and outcome. Calculate Brier score and log loss for each probability vector. Accuracy is secondary; do not use it alone.
5. Keep missing odds, lineups, injuries and statistics explicitly blank/NA. Never treat missing data as zero or infer a lineup.
6. Do not change model parameters during this 50-match cohort. Any model change starts a separately versioned cohort.
7. Report results only after 50 eligible matches are complete, while showing interim counts and metrics transparently. This small sample is exploratory and cannot establish a durable edge.

## Model disagreement investigation
The app's current forecast is Dynamic Elo. Its Elo distribution is generated from the pre-match rating difference plus a fixed 65-point home advantage and a heuristic draw mapping:
- expected home score from Elo: logistic function of the effective rating difference;
- draw probability: 0.265 * exp(-abs(Elo difference) / 650), clipped to 12%-30%;
- home/away win probabilities split the remaining probability by the Elo expected score.

Dixon–Coles is a separate goals model fitted to historical home/away goals with time decay, regularisation, a fitted home advantage and low-score correlation (rho). It converts expected goals into a score matrix and sums scorelines into 1X2 probabilities. The app then applies a small rest-days xG adjustment (currently capped at +/-0.10 xG, 0.012 per rest-day difference) before recomputing the Dixon–Coles distribution.

These are structurally different models, so a disagreement is not automatically evidence of an error. For each large disagreement, inspect:
- the effective Elo difference and the Elo-derived draw probability;
- Dixon–Coles expected goals before and after rest adjustment;
- fitted rho and the resulting H/D/A probabilities;
- recent form/table context only as descriptive evidence, not as a replacement for calibrated probabilities.

The 21 November 2026 Mjällby–Malmö screenshot showed Elo 39.5/25.6/34.9 and bookmaker no-vig 30.4/26.1/43.5; a previous diagnostic screenshot showed Dixon–Coles 31.9/29.1/38.9. This is a historical screenshot observation, not yet a frozen prospective evaluation row. The exact expected goals and rho for that example still need to be captured from the expanded diagnostics before attributing the gap.

## Metrics
For each forecast vector p and realised one-hot outcome y:
- multiclass Brier = sum((p-y)^2), lower is better;
- log loss = -ln(probability assigned to the realised outcome), lower is better;
- accuracy = whether the highest-probability outcome occurred.

Compare Elo, Dixon–Coles and no-vig bookmaker probabilities on identical matches. Report mean scores, sample size, and uncertainty intervals. Never claim improvement from a handful of matches or from accuracy alone.
