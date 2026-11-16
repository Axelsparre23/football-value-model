# Football Match Calculator

A quick Streamlit football match research tool for the Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and Allsvenskan. It estimates home-win/draw/away-win probabilities and explains the main inputs.

## Current model

- **Dixon–Coles:** time-decayed team attack/defence strengths, home advantage and low-score correlation.
- **Dynamic Elo:** team ratings update chronologically after every completed result. A home-advantage adjustment is included in the pre-match rating difference.
- **Ensemble:** Dixon–Coles probabilities and Elo baseline probabilities are blended. The Elo share is configurable in the app; the default is 25%. This is a research setting, not an optimised weight.
- **Player impact:** manually entered starting-XI ratings and unavailable-player ratings adjust expected goals using position weights. The app shows the method and caps adjustments. This is a transparent heuristic, not a validated player-value model.
- **Team profile:** recent form, rolling-365-day table position, points, goal difference, goals for/against, and available shots / shots-on-target indicators. It describes observed performance and does not claim to infer a coach's exact tactical system.
- **Match context:** rest days are calculated from each team's latest prior result. A small capped xG adjustment is applied; this needs out-of-sample validation.
- **Evaluation:** chronological 80/20 holdout comparison of Dixon–Coles and the Elo ensemble using Brier score, log loss and accuracy. The evaluation is per selected league and does not use future results to predict earlier holdout fixtures.

## Player input format

Enter one player per line:

```
Player name | rating from 1 to 10 | position
```

Examples: `Alex Smith | 7.6 | ST`, `Jordan Jones | 7.2 | CB`. Positions recognised include GK, CB, LB, RB, DF, DM, CM, AM, LM, RM, LW, RW, ST and FW. Ratings are optional, but a line without a rating does not affect the calculated player impact. Enter unavailable players separately using the same format.

Lineup and absence information is manual. No automatic injury, suspension, confirmed-lineup or player-rating feed is connected. Do not treat an expected lineup as confirmed.

## Data sources

- [Football-Data.co.uk](https://www.football-data.co.uk/): historical results for the five major European leagues from 2010/11 onward; detailed statistics vary by league and season.
- [open-matchday](https://github.com/oladeeayo/open-matchday): current-season results and selected match statistics where upstream providers supply them.
- [Swedish Football Dataset](https://github.com/Mongosaurusrex/swedish-football-dataset): Allsvenskan historical results, 2011–2025 (CC BY 4.0; attribution: Nathan Dygant).

## Important limitations

- There is no consistent free source connected for player-level ratings, injuries, suspensions, confirmed XIs, xG, possession, PPDA or detailed tactical events across all six competitions.
- The table profile uses a rolling 365-day window as a season-to-date approximation. This is useful for context but is not an official competition table at every point in the season.
- Elo probabilities, the ensemble weight, player impact coefficients and rest-day adjustment have not yet been optimised through nested chronological validation.
- A single 80/20 holdout is an initial check, not proof of calibration, future accuracy or profitability. The model must be tested across multiple seasons and compared with a no-vig market benchmark before any edge can be claimed.
- Historical data coverage and statistics availability differ by league. Missing data is displayed as unavailable, not treated as zero.

## Run in GitHub Codespaces

```bash
git pull origin development
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Data is cached locally in `data/cache/`. Respect source licenses and terms. This is a statistical research tool only; it has no bookmaker connection or bet-placement functionality.
