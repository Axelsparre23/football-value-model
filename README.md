# Football Match Calculator

A quick Streamlit match predictor for the Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and Allsvenskan. It produces estimated home-win/draw/away-win probabilities and an explanation of the main model inputs.

## Data sources

- [Football-Data.co.uk](https://www.football-data.co.uk/): historical results and odds for the five major European leagues. The loader uses historical match results and retains available match-stat columns.
- [open-matchday](https://github.com/oladeeayo/open-matchday): daily published current-season CSVs for the five major leagues and Allsvenskan. It combines public feeds including ESPN, fixturedownload, OpenLigaDB, Football-Data.co.uk and Soccerway. It contains results and fixtures broadly, but detailed shots/corners/fouls/cards are only present where the upstream provider supplies them.
- [Understat](https://understat.com/): a potential xG and shot-level source for the five major leagues. It has no official public developer API, so this app does not scrape it automatically.
- [API-Football coverage](https://www.api-football.com/coverage): lists Allsvenskan and the five major leagues, but detailed fixture/lineup/player-stat availability varies by league and season. It is a possible optional integration if a free API key and permitted endpoint access are confirmed; it is not silently treated as a free, complete injury feed.

## What the calculator uses

- Dixon–Coles goal model with time-decayed historical goals, attack/defence strengths and home advantage.
- Recent form from each team's last five completed matches (points, goals for and goals against).
- Available match statistics in the recent-match panel: shots, shots on target, corners and fouls. Missing values are shown as unavailable, not zero.
- Optional manual player-absence inputs and tactical assumptions. You enter the names/positions and explicitly choose the expected-goal effect; the app does not invent injury news or hidden player weights.
- A most-likely score derived from the adjusted goal rates.

## Important coverage caveats

Allsvenskan is a calendar-year league. The current open-matchday feed lists Allsvenskan results/fixtures, but its league index does not show the same detailed match-stat coverage as the top five. The model therefore relies on available Allsvenskan results and clearly reports unavailable stats. xG, confirmed lineups, injuries and detailed tactical data are not yet automatically integrated consistently across all six leagues.

The manual injury/tactical sliders are assumptions supplied by the user, not verified facts or empirically calibrated player values. The current model's percentages should be treated as estimates, not established accuracy. It has not yet been validated against a held-out season or bookmaker benchmark for the new six-league data pipeline.

## Run in GitHub Codespaces

```bash
git pull origin development
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Data is cached locally in `data/cache/`. The app refreshes the current-season feed on its cache refresh cycle. Respect the source repositories' terms. The open-matchday dataset is published for informational, non-commercial use and requires attribution; its terms and upstream provider terms apply.

This is a statistical research tool only. No bookmaker connections or automated bet placement.
