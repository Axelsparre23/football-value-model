"""Leakage-safe 2025-season backtest of the app's displayed Elo forecast.

European leagues: 2025/26 season (2025-08-01 through 2026-06-30).
Allsvenskan: 2025 calendar season.

For every target match date:
- Elo uses the pre-date ratings from the chronological Elo history.
- Dixon-Coles is fitted only on matches before that date, matching the app's
  current model settings (half-life 300 days, 240 for Allsvenskan).
- All fixtures on the same date are forecast before any result from that date
  updates Elo or enters Dixon-Coles training.
- The no-vig market is a benchmark where historical odds are available.

Run: python backtest_2025.py
Outputs: output/backtest_2025_predictions.csv and
         output/backtest_2025_summary.csv
"""
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities
from football_value_model_v3 import DixonColesModel, odds_of, devig

OUTPUT = Path("output")
EUROPE_START = pd.Timestamp("2025-08-01")
EUROPE_END = pd.Timestamp("2026-06-30")
SWE_START = pd.Timestamp("2025-01-01")
SWE_END = pd.Timestamp("2025-12-31")


def outcome_index(home_goals, away_goals):
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def score_metrics(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    one_hot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(one_hot, axis=1)[:, :2]
    return {
        "Matches": int(len(y)),
        "Brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "Log loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def target_matches(data, league):
    if league == "SWE":
        start, end = SWE_START, SWE_END
    else:
        start, end = EUROPE_START, EUROPE_END
    return data[(data.Date >= start) & (data.Date <= end)].copy()


def run(include_dc=False):
    print("Downloading/loading historical results and odds...")
    data = download_data(
        Path("data/cache"), first_start_year=2010,
        leagues=tuple(LEAGUES), refresh_current=True
    )
    download_failures = data.attrs.get("download_failures", [])
    if download_failures:
        print("\nWARNING: Some historical data sources could not be refreshed:")
        for failure in download_failures:
            print(f"  - {failure}")
        print("Check target-season coverage before interpreting the results.\n")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    print("Building chronological pre-match Elo snapshots separately for each league...")
    prediction_rows = []
    for league, league_name in LEAGUES.items():
        league_data = data[data.League == league].copy()
        # Match the app: Elo ratings are isolated within the selected league.
        elo_rows, _ = elo_history(league_data)
        elo_lookup = {}
        for row in elo_rows.to_dict("records"):
            key = (pd.Timestamp(row["Date"]).normalize(),
                   str(row["HomeTeam"]), str(row["AwayTeam"]))
            elo_lookup[key] = row

        targets = target_matches(league_data, league)
        if targets.empty:
            print(f"{league_name}: no target-season results found; skipped.")
            continue

        half_life = 240 if league == "SWE" else 300
        print(f"{league_name}: scoring {len(targets)} fixtures across "
              f"{targets.Date.dt.normalize().nunique()} match dates...")
        for match_date, day in targets.groupby(targets.Date.dt.normalize(), sort=True):
            # Strictly pre-match training set: no same-day or future result leakage.
            train = league_data[league_data.Date < match_date].copy()
            if len(train) < 100 or train.HomeTeam.nunique() < 4 or train.AwayTeam.nunique() < 4:
                warnings.warn(f"{league_name} {match_date.date()}: insufficient prior data; date skipped.")
                continue

            # Dixon-Coles fitting is expensive; the default evaluates the app's Elo model only.
            dc_model = None
            if include_dc:
                dc_model = DixonColesModel(
                    half_life=half_life, reg=0.04, xg_weight=0.0
                ).fit(train, pd.Timestamp(match_date))

            for _, match in day.iterrows():
                key = (pd.Timestamp(match.Date).normalize(),
                       str(match.HomeTeam), str(match.AwayTeam))
                elo_snapshot = elo_lookup.get(key)
                if elo_snapshot is None:
                    continue
                try:
                    elo_p = elo_probabilities(float(elo_snapshot["EloDiff"]))
                    if include_dc:
                        dc_p, (xg_home, xg_away) = dc_model.probabilities(
                            str(match.HomeTeam), str(match.AwayTeam)
                        )
                except (KeyError, ValueError, OverflowError):
                    continue

                actual = outcome_index(float(match.FTHG), float(match.FTAG))
                record = {
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": league,
                    "LeagueName": league_name,
                    "HomeTeam": match.HomeTeam,
                    "AwayTeam": match.AwayTeam,
                    "HomeGoals": int(match.FTHG),
                    "AwayGoals": int(match.FTAG),
                    "Actual": ["Home", "Draw", "Away"][actual],
                    "EloDiff": float(elo_snapshot["EloDiff"]),
                    "EloHome": float(elo_snapshot["EloHome"]),
                    "EloAway": float(elo_snapshot["EloAway"]),
                    "EloPHome": float(elo_p[0]),
                    "EloPDraw": float(elo_p[1]),
                    "EloPAway": float(elo_p[2]),
                    "TrainingMatchesBeforeKickoff": int(len(train)),
                }
                if include_dc:
                    record.update({
                        "DCPHome": float(dc_p[0]),
                        "DCPDraw": float(dc_p[1]),
                        "DCPAway": float(dc_p[2]),
                        "DCXGHome": float(xg_home),
                        "DCXGAway": float(xg_away),
                    })

                odds_pair = odds_of(match)
                if odds_pair is not None:
                    market_odds, _best_odds = odds_pair
                    try:
                        market_p = devig(market_odds)
                        record.update({
                            "MarketPHome": float(market_p[0]),
                            "MarketPDraw": float(market_p[1]),
                            "MarketPAway": float(market_p[2]),
                        })
                    except (ValueError, FloatingPointError):
                        pass
                prediction_rows.append(record)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No fixtures could be scored. Check historical data coverage.")
    for column in ("MarketPHome", "MarketPDraw", "MarketPAway"):
        if column not in predictions.columns:
            predictions[column] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "backtest_2025_predictions.csv", index=False)

    summary_rows = []
    for league, group in predictions.groupby(["League", "LeagueName"], sort=True):
        y = group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
        model_specs = [
            ("App Elo (displayed prediction)", ["EloPHome", "EloPDraw", "EloPAway"]),
        ]
        if include_dc:
            model_specs.append(
                ("Dixon-Coles (cross-check)", ["DCPHome", "DCPDraw", "DCPAway"])
            )
        for model_name, cols in model_specs:
            metrics = score_metrics(group[cols].to_numpy(), y)
            summary_rows.append({"League": league[0], "LeagueName": league[1],
                                 "Model": model_name, "BenchmarkMatches": len(group), **metrics})

        market_cols = ["MarketPHome", "MarketPDraw", "MarketPAway"]
        market_rows = group.dropna(subset=market_cols)
        if not market_rows.empty:
            market_y = market_rows.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
            market_specs = [
                ("App Elo (same odds subset)", ["EloPHome", "EloPDraw", "EloPAway"]),
                ("No-vig closing market", market_cols),
            ]
            if include_dc:
                market_specs.insert(
                    1, ("Dixon-Coles (same odds subset)", ["DCPHome", "DCPDraw", "DCPAway"])
                )
            for model_name, cols in market_specs:
                metrics = score_metrics(market_rows[cols].to_numpy(), market_y)
                summary_rows.append({"League": league[0], "LeagueName": league[1],
                                     "Model": model_name, "BenchmarkMatches": len(market_rows), **metrics})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "backtest_2025_summary.csv", index=False)

    print("\n=== 2025 SEASON WALK-FORWARD BACKTEST ===")
    print("Lower Brier, log loss and RPS are better; higher accuracy is better.")
    print("Market comparisons use only fixtures with historical odds.")
    print(summary.to_string(index=False, formatters={
        "Brier": "{:.4f}".format, "Log loss": "{:.4f}".format,
        "RPS": "{:.4f}".format, "Accuracy": "{:.1%}".format,
    }))
    print(f"\nScored fixtures: {len(predictions):,}")
    print(f"Fixtures with odds: {predictions[['MarketPHome','MarketPDraw','MarketPAway']].notna().all(axis=1).sum():,}")
    print(f"Prediction detail: {OUTPUT / 'backtest_2025_predictions.csv'}")
    print(f"Summary: {OUTPUT / 'backtest_2025_summary.csv'}")
    print("\nIMPORTANT: Elo probabilities are the app's current displayed 1X2 method.")
    if not include_dc:
        print("Dixon-Coles was skipped for speed. Re-run with --include-dc for the slower cross-check.")
    print("Historical lineups, injuries and manual player ratings are not reconstructed,")
    print("so this test evaluates the core model without those optional inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the football model.")
    parser.add_argument(
        "--include-dc", action="store_true",
        help="Also fit Dixon-Coles separately before every match date (much slower)."
    )
    args = parser.parse_args()
    run(include_dc=args.include_dc)
