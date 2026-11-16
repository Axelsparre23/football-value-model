# Football Match Calculator

A quick Streamlit football match research tool for the Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and Allsvenskan. It estimates home-win/draw/away-win probabilities and explains the main inputs.

## Using the calculator

Choose a league, home team and away team, then press **Calculate match prediction**. Lineups, player availability, recent form and historical evaluation are optional expandable sections. A warning appears when the latest available match data is more than 21 days old.

## Current model

- **Dixon–Coles:** time-decayed team attack/defence strengths, home advantage and low-score correlation.
- **Dynamic Elo:** team ratings update chronologically after every completed result. A home-advantage adjustment is included in the pre-match rating difference.
- **Current prediction default:** Elo supplies the displayed home/draw/away probabilities because it outperformed Dixon–Coles and the tested blend in the available Allsvenskan holdout. Dixon–Coles remains an independent expected-goals and probability cross-check. This result has not yet been established for every league.
- **Player impact:** manually entered starting-XI ratings and unavailable-player ratings adjust expected goals using position weights. The app shows the method and caps adjustments. This is a transparent heuristic, not a validated player-value model.
- **Team profile:** recent form, rolling-365-day table position, points, goal difference, goals for/against, and available shots / shots-on-target indicators. It describes observed performance and does not claim to infer a coach's exact tactical system.
- **Match context:** rest days are calculated from each team's latest prior result. A small capped xG adjustment is applied; this needs out-of-sample validation.
- **Evaluation:** chronological training/validation/test comparison, with blend-weight selection and temperature calibration on validation data only. The final test block is held back from tuning.

## Player input format

Enter one player per line:

```
Player name | rating from 1 to 10 | position
```

Examples: `Alex Smith | 7.6 | ST`, `Jordan Jones | 7.2 | CB`. Positions recognised include GK, CB, LB, RB, DF, DM, CM, AM, LM, RM, LW, RW, ST and FW. Ratings are optional, but a line without a rating does not affect the calculated player impact. Enter unavailable players separately using the same format.

Lineup and absence information is manual. No automatic injury, suspension, confirmed-lineup or player-rating feed is connected. Do not treat an expected lineup as confirmed.

## Automatic team-news integration

- Optional provider: [API-Football / API-Sports](https://www.api-football.com/). Its free plan advertises 100 requests/day and endpoints for fixtures, lineups, player statistics and injuries. Free-plan season coverage is restricted and must be checked for each competition/season.
- Configure `API_FOOTBALL_KEY` as a Codespaces environment variable or in `.streamlit/secrets.toml`. Never commit the key to GitHub.
- The app searches for the selected home/away fixture around the selected date, then retrieves confirmed lineups, season-average player ratings where available and fixture-level absences. Missing data stays missing; expected lineups are never fabricated.
- SofaScore is not scraped: it has no official public API and its terms restrict automated extraction and reuse.

## Data sources

- [Football-Data.co.uk](https://www.football-data.co.uk/): historical results for the five major European leagues from 2010/11 onward; detailed statistics vary by league and season.
- [open-matchday](https://github.com/oladeeayo/open-matchday): current-season results and selected match statistics where upstream providers supply them.
- [Swedish Football Dataset](https://github.com/Mongosaurusrex/swedish-football-dataset): Allsvenskan historical results, 2011–2025 (CC BY 4.0; attribution: Nathan Dygant).
- [TopMarx/allsvenskan](https://github.com/TopMarx/allsvenskan): current-season Allsvenskan fixtures and completed scores from the public Fantasy Allsvenskan API mirror; repository is MIT-licensed, while underlying league data remains third-party data.

## Important limitations

- The optional API-Football connection may not return current-season data for every competition on the free tier. No provider guarantees complete player ratings, injuries, suspensions or confirmed XIs across all six competitions. xG, possession, PPDA and detailed tactical events remain inconsistently available.
- The table profile uses a rolling 365-day window as a season-to-date approximation. This is useful for context but is not an official competition table at every point in the season.
- Elo's standalone draw mapping, player impact coefficients and rest-day adjustment remain heuristic and are not yet validated as live prediction inputs.
- The research evaluator now uses whole-date chronological partitions (60% training, 20% validation, 20% final test), selects the Dixon–Coles/Elo blend weight on validation data, fits temperature calibration on validation data, and reports raw and calibrated Brier, log-loss and ranked probability scores on the untouched final block.
- This is still one chronological split, not repeated multi-season validation. The model must be tested across several seasons and compared with a no-vig market benchmark before any predictive edge can be claimed.
- Historical data coverage and statistics availability differ by league. Missing data is displayed as unavailable, not treated as zero.

## 2025 season walk-forward backtest

Run the dedicated season test with:

```bash
python backtest_2025.py
```

- European competitions: 2025/26 season (1 August 2025–30 June 2026).
- Allsvenskan: 2025 calendar season.
- The app's displayed Elo probabilities are evaluated using pre-match Elo snapshots. No result from the fixture's date is used before that date's predictions.
- Dixon–Coles is refitted using only results before each match date, with the same half-life and regularisation settings as the app.
- Where historical odds are present, the no-vig closing market is included as a benchmark. Model-vs-market scores use the exact same odds-covered fixtures.
- Outputs are written to `output/backtest_2025_predictions.csv` and `output/backtest_2025_summary.csv`.

The test evaluates the core 1X2 models. Historical lineups, injuries and manually entered player ratings are not reconstructed. Do not interpret a positive result on one season as proof of a persistent predictive or financial edge.

## Run in GitHub Codespaces

```bash
git pull origin development
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Data is cached locally in `data/cache/`. Respect source licenses and terms. This is a statistical research tool only; it has no bookmaker connection or bet-placement functionality.

## Multi-season draw-calibration experiment

Run the repeated walk-forward evaluation with:

```bash
python multi_season_draw_backtest.py
```

- European leagues: 2021/22 through 2025/26; Allsvenskan: calendar seasons 2021–2025.
- For every league-season, both calibration methods are fitted only on matches before that evaluation season.
- Compares baseline Elo, the existing quadratic draw calibration, a more flexible piecewise-linear draw calibration, and the no-vig closing market where odds exist.
- Reports Brier score, log loss, ranked probability score and accuracy, including results by league-season and the number of league-seasons where calibration beats baseline Elo.
- Outputs: `output/multi_season_draw_predictions.csv` and `output/multi_season_draw_summary.csv`.

This is a research experiment only. It does not change the live app model. A calibration should only be considered for production if it improves proper scoring rules consistently across multiple held-out seasons and remains competitive against the market.


## League-specific Elo parameter optimisation experiment

Run the walk-forward K-factor and home-advantage experiment with:

```bash
python league_elo_optimisation.py
```

- For each league and target season, selects the K-factor and home advantage using only the two immediately preceding seasons.
- Evaluates the selected parameters on the next season, with strictly pre-match Elo snapshots.
- Compares optimised Elo with the fixed K=22 / home advantage=65 baseline.
- Compares both Elo versions with the no-vig market on exactly the same odds-covered fixtures.
- Reports Brier score, log loss, ranked probability score and accuracy, including league-season results.
- Tests K values 12, 16, 20, 22, 26, 30 and 34, and home advantages 45, 55, 65, 75 and 85 Elo points.
- Outputs: `output/league_elo_optimisation_predictions.csv` and `output/league_elo_optimisation_summary.csv`.

This is an experiment only. It does not change the production calculator or its Elo settings. Parameters are selected using historical validation data before each target season; the target season is not used for tuning. A change should only be considered after consistent out-of-sample improvements across leagues and seasons.


## Poisson attack/defence experiment

Run the separate attack/defence model evaluation with:

```bash
python attack_defence_backtest.py
```

- Fits regularised team attack and defence strengths from the two seasons immediately before each target season.
- Converts expected home and away goals into 1X2 probabilities using independent Poisson goal distributions.
- Tests on the following season without fitting on target-season results.
- Compares with the existing pre-match Elo model and no-vig market on exactly the same odds-covered fixtures.
- Reports Brier score, log loss, RPS, accuracy and results by league-season.
- Outputs: `output/attack_defence_predictions.csv` and `output/attack_defence_summary.csv`.

This is a standalone research experiment. It does not change the live calculator. The model uses goals, not xG, because historical xG coverage is incomplete across the six competitions.


## Hybrid Elo + recent form experiment

Run:

```bash
python hybrid_form_backtest.py
```

This walk-forward experiment combines pre-match Elo with exponentially weighted team form: goals for/against, points per match, shots and shots on target. Match statistics are used only when available; missing values are not fabricated. A regularised multinomial model is trained on the two seasons before each target season and evaluated on the untouched target season.

It compares against baseline Elo and no-vig closing market probabilities on identical odds-covered fixtures. It writes:
- `output/hybrid_form_predictions.csv`
- `output/hybrid_form_summary.csv`

This is a research experiment and does not alter the live calculator.


## Seasonal Elo regression experiment

Run:

```bash
python seasonal_elo_backtest.py
```

Tests whether partially regressing team Elo ratings toward the league mean at the start of each new season improves predictions. Candidate carryover factors are compared on 2021–2024 development seasons; the best factor by Brier score is then evaluated on the untouched 2025 holdout (2025/26 for the European leagues and 2025 for Allsvenskan). The final holdout is also compared with baseline Elo and the no-vig market on the same odds-covered fixtures.

This is a research experiment only. It does not change the live calculator. Outputs:
- `output/seasonal_elo_predictions.csv`
- `output/seasonal_elo_summary.csv`

## Market-anchored residual model experiment

Run:

```bash
python market_residual_backtest.py
```

This experiment tests whether pre-match Elo and recent team-performance signals add predictive information beyond no-vig closing market probabilities. The market probabilities are the model's fixed starting point; a regularised multinomial model learns only a correction from Elo difference, recent goals, points, shots and shots on target. Ridge strength is selected on the second prior season after fitting on the first, then the model is refit on both prior seasons before testing. The regularisation penalty is applied on the loss scale rather than divided by sample size.

For each target season, the correction is trained only on the two immediately preceding seasons. Predictions are compared with the no-vig market and baseline Elo on exactly the same odds-covered fixtures. Evaluation covers 2021/22–2025/26 in the five major European leagues where historical odds are available. Allsvenskan is skipped when sufficient historical odds are unavailable.

Outputs:
- `output/market_residual_predictions.csv`
- `output/market_residual_summary.csv`

This is a research-only experiment. It does not change the live calculator. A lower Brier score and log loss than the market across multiple held-out seasons would be evidence of improved probability forecasts, but not proof of a persistent betting or financial edge.


## Market calibration research

Run the chronological test of a restrained market-only calibration layer:

```bash
python market_calibration_backtest.py
```

The experiment compares raw no-vig closing 1X2 probabilities with a calibrated version on identical, unseen fixtures. It fits on the first prior season, selects regularisation on the next prior season, refits on both prior seasons, and tests on the target season. It uses no team features. Results are written to `output/market_calibration_predictions.csv` and `output/market_calibration_summary.csv`.

Calibration is a research experiment only. It is not used by the live calculator unless it demonstrates repeatable out-of-sample improvement.


## Bookmaker price-source and margin-removal experiment

Run:

```bash
python market_price_method_backtest.py
```

Compares proportional, power, additive and Shin margin removal across average and Bet365 opening/closing prices. It reports source coverage, aggregate results, a common-fixture comparison across all four price sources, and average-closing results by league. This is a market benchmark diagnostic only; it does not change the live calculator.


## Opening-to-closing market movement experiment

First run the price-source experiment to generate the required fixture-level file, then run:

```bash
python market_movement_backtest.py
```

This experiment uses additive margin removal and compares average opening odds, average closing odds, and a fixed 50/50 opening-closing blend on identical fixtures. It measures the size and direction of probability changes, tests whether larger movements are associated with larger improvements in closing-market scores, and reports results by league and season.

Movement deciles are descriptive, not fitted trading signals. The blend is fixed in advance to avoid selecting a weight on the same evaluation sample. Outputs are saved under `output/market_movement_*.csv`.

This is a probability-forecast research diagnostic only. It does not establish that market movement can be predicted in advance or that any betting strategy would be profitable.


## Directional market movement experiment

Run after `market_movement_backtest.py`:

```bash
python market_direction_backtest.py
```

For home, draw and away probabilities separately, this groups fixtures by signed opening-to-closing probability movement (largest decreases through largest increases). It compares mean opening probability, closing probability and realised outcome frequency, plus binary Brier scores. Results are also broken down by league and season.

The groups are descriptive and use the full historical sample; they are not a fitted or deployable prediction rule. Outputs: `output/market_direction_*.csv`.


## Market movement significance test

Run after `market_movement_backtest.py`:

```bash
python market_movement_significance.py
```

This performs paired fixture-level bootstrap resampling (2,000 replicates) to estimate 95% confidence intervals for opening-to-closing Brier and log-loss improvements, overall and by outcome, league and season. Positive values mean closing odds performed better. This quantifies uncertainty; it does not establish betting profitability.
