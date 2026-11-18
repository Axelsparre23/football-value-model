# Model and data audit — November 2026

## Scope

This is a read-only audit and experiment plan. It does not change the live prediction model or the 40 frozen prospective forecasts.

## Findings from the current code

1. **The current market benchmark is stronger than standalone Elo in the available 2025 test.** The recorded 2025 comparison found lower Brier, log loss and RPS for the no-vig market than Elo in each of the five major European leagues. This is evidence that the model needs to demonstrate incremental information beyond the market, not merely improve its own accuracy.
2. **The live probability model does not use bookmaker prices.** Odds are an optional comparison in the app; the displayed forecast is based on Elo. The existing `market_residual_backtest.py` is a separate offline experiment that tests whether pre-match Elo/form features add information on top of market probabilities. It must remain experimental until repeated out-of-sample results support it.
3. **Observed historical xG is currently absent from the main data pipeline.** `data_manager.download_data` fills `HomeXG` and `AwayXG` with missing values. The app also fits Dixon–Coles with `xg_weight=0.0`. Therefore the live model is not currently learning from historical shot-level xG; its expected goals are inferred from goal history.
4. **Player and availability inputs are not production-grade.** Player ratings and absence impacts are manual heuristics. Highlightly supplies fixtures, confirmed lineups and events in the current integration, but the project has not verified an injuries/suspensions feed from its configured free tier. Missing availability must remain unknown, never be interpreted as a full-strength squad.
5. **Several context adjustments are heuristic.** The fixed Elo draw mapping, fixed K/home advantage, rest-day xG adjustment and manually rated player-impact coefficients need validation on untouched seasons before activation.
6. **Elo has no explicit season-start regression-to-mean step.** `elo_history` accepts a `carryover` parameter but does not use it. This is a candidate for a controlled experiment, not an immediate production change.
7. **The historical evaluation framework is still limited.** The current chronological ensemble evaluator uses one train/validation/test split. The existing experiment scripts explore draw calibration, league-specific Elo parameters, market calibration, form and market residuals, but their results must be run and retained before drawing conclusions.
8. **The prospective 40-match cohort is small and user-selected.** It is useful as a live process check, not a representative league-wide estimate. The market comparison must also exclude synthetic best-price triples and unidentified operators, and require a named source plus timestamp for each price snapshot.

## Prioritised experimental questions

1. Does a market-anchored residual model using Elo and pre-match team form beat the no-vig market on identical fixtures across multiple untouched seasons?
2. Does a modest Elo season-start regression improve calibration and log loss across leagues?
3. Does league-specific K/home advantage improve out-of-sample scores consistently, rather than only in the tuning season?
4. Does a learned draw calibration improve RPS and log loss without harming home/away calibration?
5. Does adding verified shot-level xG improve results over goal-only Dixon–Coles on the exact same fixtures?
6. Do rest-day and player-availability features add value after controlling for the market, with missingness and source timestamps handled explicitly?

## Data-source decision

| Source | Best use | Main limitation | Decision |
|---|---|---|---|
| Highlightly (existing free integration) | Fixture lookup, confirmed lineups, live events | No verified injury feed or odds in the configured free tier; competition coverage must be checked | Keep for lineups/events; do not infer injuries |
| football-data.org | Fixtures, results, tables; paid lineup/stat add-ons | Free tier is limited; no demonstrated xG or injury feed | Not sufficient as the main enrichment provider |
| Sportmonks | Injuries/suspensions, lineups, player/team statistics, xG and broad competition coverage | Commercial; Growth starts at €99/month and Pro at €249/month on the current pricing page | Best all-in-one paid candidate, but do not subscribe yet |
| StatsBomb Open Data | Historical event/shot research and selected competition lineups | Selected competitions/seasons only; not a live feed; attribution required | Use as a small offline research source where coverage overlaps |
| openplay dataset | Historical event features and shot-level xG across major competitions | Large static snapshot, not live; no Allsvenskan coverage identified; known gaps/issues are documented | Potential historical xG experiment; defer bulk ingestion until a specific test needs it |

## Implementation policy

- No paid data subscription without a separate decision.
- Continue using Highlightly for confirmed lineups/events where returned.
- Keep injury status explicitly unavailable unless a source and timestamp support it.
- Do not add xG, lineup or injury features to the live model until they have matched historical coverage and pass a leakage-safe multi-season comparison.
- Run experiments on exact shared fixture sets and report sample sizes, missing-data coverage, calibration and paired uncertainty intervals.
- Keep each candidate model and its output separate from `app.py` and the prospective forecast log.

## Source references

- Highlightly API documentation: https://highlightly.net/football-api/documentation/
- football-data.org pricing and API documentation: https://www.football-data.org/pricing and https://www.football-data.org/documentation/api
- Sportmonks football API and pricing: https://www.sportmonks.com/football-api/ and https://www.sportmonks.com/football-api/plans-pricing/
- StatsBomb Open Data: https://github.com/hudl/open-data
- openplay dataset: https://github.com/nlbair/openplay
