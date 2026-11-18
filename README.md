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

## Prospective study evaluation dashboard

Open the **Prospective Evaluation** page from the Streamlit sidebar. It reads `research/prospective_50_match_log.csv` in read-only mode and reports Elo, Dixon–Coles and market Brier score, log loss, ranked probability score, accuracy, calibration and paired bootstrap uncertainty intervals once results are recorded.

The dashboard does not change forecasts or tune the live model. It only scores completed fixtures with valid pre-match probabilities. The strict market comparison requires a named single bookmaker, source and timestamp; synthetic best-price aggregates and unidentified operators are excluded. Closing prices are eligible only when `closing_bookmaker_name`, `closing_odds_source` and `closing_odds_timestamp_utc` are recorded. The original recorded prices remain in the log for audit.

## Isolated experimental model

`research/experimental_model.py` contains a separate market-anchored multinomial residual model. It starts from no-vig market probabilities and learns a regularised correction from pre-match Elo/form features. The existing `market_residual_backtest.py` uses this module in a walk-forward experiment: it selects regularisation on the prior season and evaluates the next untouched season. This candidate is not imported by `app.py` and cannot change the live predictions or the frozen prospective cohort.

Run the experiment with:

```bash
python market_residual_backtest.py
```

It requires historical odds coverage and writes separate files under `output/`. Treat it as an experiment until repeated out-of-sample comparisons, paired uncertainty intervals and data-coverage checks support it.

## Player input format

Enter one player per line:

```
Player name | rating from 1 to 10 | position
```

Examples: `Alex Smith | 7.6 | ST`, `Jordan Jones | 7.2 | CB`. Positions recognised include GK, CB, LB, RB, DF, DM, CM, AM, LM, RM, LW, RW, ST and FW. Ratings are optional, but a line without a rating does not affect the calculated player impact. Enter unavailable players separately using the same format.

Live confirmed lineups and match events can be loaded from Highlightly when configured. Expected lineups are not fabricated; when confirmed XIs have not been released, the app reports them as unavailable.

## Automatic team-news integration

- Optional provider: [Highlightly Football API](https://highlightly.net/football-api/). Its free tier advertises 100 requests/day and provides match data, confirmed lineups and live events, subject to competition coverage and plan restrictions.
- Configure `HIGHLIGHTLY_API_KEY` as a Streamlit secret or environment variable. Never commit the key to GitHub.
- The match screen searches for the selected fixture and retrieves confirmed lineups and live events. The free tier does not include bookmaker odds, and this integration does not claim to provide a verified injuries/suspensions feed. Manual bookmaker odds remain available in the benchmark section.
- SofaScore is not scraped: it has no official public API and its terms restrict automated extraction and reuse.

## Data sources

- [Football-Data.co.uk](https://www.football-data.co.uk/): historical results for the five major European leagues from 2010/11 onward; detailed statistics vary by league and season.
- [open-matchday](https://github.com/oladeeayo/open-matchday): current-season results and selected match statistics where upstream providers supply them.
- [Swedish Football Dataset](https://github.com/Mongosaurusrex/swedish-football-dataset): Allsvenskan historical results, 2011–2025 (CC BY 4.0; attribution: Nathan Dygant).
- [TopMarx/allsvenskan](https://github.com/TopMarx/allsvenskan): current-season Allsvenskan fixtures and completed scores from the public Fantasy Allsvenskan API mirror; repository is MIT-licensed, while underlying league data remains third-party data.

## Important limitations

- Highlightly's free tier may restrict competition data and does not include bookmaker odds. The current integration does not provide verified injury/suspension data. Confirmed lineups are available only within the provider's published pre-match window. Coverage must be tested for each selected competition.
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


## Final chronological market-versus-Elo test

Run:

```bash
python chronological_market_test.py
```

This is the final planned research test for this phase. It uses matched fixtures with average opening and closing odds and chronological Elo snapshots. Protocol: train on 2021-2023, select regularisation on 2024, refit on 2021-2024, then evaluate once on the untouched 2025 season. It compares Elo, opening market, closing market, Elo+opening, Elo+closing, and Elo+both market snapshots. Paired confidence intervals use 2,000 league-week cluster bootstrap samples and compare candidates against both Elo and closing market.

All comparisons use the same fixtures with both opening and closing prices; competitions/seasons without adequate odds coverage are naturally excluded. This is a prediction-quality research test, not a profitability test. It does not change the live calculator.

Outputs:
- `output/chronological_market_all_fixtures.csv`
- `output/chronological_market_2025_predictions.csv`
- `output/chronological_market_2025_summary.csv`
- `output/chronological_market_2025_bootstrap.csv`
- `output/chronological_market_selected_models.csv`


## Match-analysis improvements

The calculator now shows last-10-match attacking and defensive shot indicators where the source provides them: shots and shots on target created, plus shots and shots on target conceded. These are descriptive matchup indicators, not claims about a coach's exact tactical system and not additional Elo multipliers.

An optional bookmaker benchmark accepts manually entered decimal home/draw/away odds and converts them to proportional no-vig probabilities. It is displayed separately from the model forecast and is never blended into the prediction.

The main 1X2 forecast remains dynamic Elo. The lineup/absence/rest-adjusted Dixon–Coles probabilities are shown as a separate scenario only. Player impact coefficients and rest adjustments have not been validated as 1X2 improvements, so they are not allowed to silently alter the main forecast.


## Optional live match intelligence module

The `match_intelligence.py` module uses Highlightly for fixture lookup, confirmed lineups and live match events. It searches by league, date and both team names, and reports per-source status and retrieval timestamps. Set `HIGHLIGHTLY_API_KEY` as a Streamlit secret or environment variable. Never commit API credentials. Without a key, the module returns a clear `not_configured` state and does not invent data.

The free tier does not include bookmaker odds, and verified injuries/suspensions are not part of this integration. The app labels these gaps explicitly. Competition coverage on the free tier must be verified in the live account.

**Model status:** live outputs are contextual only and do not change Elo or Dixon–Coles probabilities. Do not claim an accuracy improvement until new features are evaluated using chronological out-of-sample tests.

Run its unit tests with:

```bash
python -m unittest tests.test_match_intelligence
```


## Live match information: Highlightly

The optional live match-information panel uses Highlightly. Configure the secret
`HIGHLIGHTLY_API_KEY` in the Streamlit deployment (or as an environment
variable locally). The free tier currently documents 100 requests per day and
supports match data, confirmed lineups, live events and match statistics,
subject to competition coverage and plan limits.

The free tier does **not** include bookmaker odds. The current integration also
does not claim to provide verified injuries or suspensions. The app labels
these sources as plan-restricted or unsupported instead of interpreting missing
data as evidence that no players are unavailable. Manual bookmaker odds remain
available in the market benchmark section.

Highlightly data is contextual only and does not alter Elo or Dixon-Coles
probabilities. No accuracy improvement is claimed until features are evaluated
using chronological out-of-sample tests.
