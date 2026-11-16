# Football Value Model

Research project for football match probability modelling, historical walk-forward evaluation and simulated/paper-only selections. No bookmaker connections or automated bet placement.

## Included

- Original Dixon–Coles v3 model preserved as `football_value_model_v3.py`.
- Automatic download and local caching of historical results and odds from Football-Data.co.uk.
- Five leagues: Premier League, La Liga, Bundesliga, Serie A and Ligue 1.
- Chronological walk-forward backtest, no-vig market benchmark, Brier score and log loss.
- Command-line runner and Streamlit web interface.

## Run in GitHub Codespaces or locally

```bash
python -m pip install -r requirements.txt
python run.py
```

Web interface:

```bash
streamlit run app.py
```

The first run downloads available CSVs into `data/cache/`; subsequent runs reuse the cache and refresh the current season. Unpublished season files may not yet exist.

## Current limitations

The automatic feed currently supplies historical results and odds, not historical xG. The underlying v3 script can optionally blend an xG CSV, but automatic xG ingestion is not enabled yet. Injuries, confirmed lineups, player-level impact, live events and tactical changes are also not integrated. Those require suitable data access, time-stamped pre-match snapshots and leakage-safe validation.

Data source: [Football-Data.co.uk](https://www.football-data.co.uk/). Respect provider terms and avoid excessive requests.

This is a statistical research and paper-evaluation tool. Historical simulated returns do not establish future profitability.
