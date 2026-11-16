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

## Run in GitHub Codespaces

```bash
git pull origin development
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Data is cached locally in `data/cache/`. Respect source licenses and terms. This is a statistical research tool only; it has no bookmaker connection or bet-placement functionality.
