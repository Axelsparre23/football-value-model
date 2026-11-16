"""Season-transition Elo experiment.

Tests whether regressing team ratings toward the league mean between seasons
improves out-of-sample 1X2 probabilities. The final season is held out:
carryover is selected only on the earlier development seasons.

Run: python seasonal_elo_backtest.py
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
CARRYOVERS = (0.35, 0.50, 0.65, 0.80, 0.90, 1.00)
INITIAL = 1500.0
K = 22.0
HOME_ADVANTAGE = 65.0
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}


def season_key(date, league):
    date = pd.Timestamp(date)
    if league == "SWE":
        return date.year
    return date.year if date.month >= 8 else date.year - 1


def seasonal_elo(data, carryover):
    """Calculate strictly pre-match Elo, shrinking ratings at season boundaries."""
    ratings = {}
    rows = []
    ordered = data.sort_values(["Date", "League"]).copy()
    ordered["_Season"] = [season_key(d, league) for d, league in zip(ordered.Date, ordered.League)]
    previous_season = None

    for (league, season), season_data in ordered.groupby(["League", "_Season"], sort=True):
        if previous_season is not None:
            # Each league is processed independently by caller; this guards boundaries.
            pass
        if rows:
            # Regression toward the league's neutral prior at each new season.
            ratings = {team: INITIAL + carryover * (rating - INITIAL)
                       for team, rating in ratings.items()}

        for _, day in season_data.groupby(season_data.Date.dt.normalize(), sort=True):
            updates = []
            for _, match in day.iterrows():
                home, away = str(match.HomeTeam), str(match.AwayTeam)
                rh, ra = ratings.get(home, INITIAL), ratings.get(away, INITIAL)
                diff = rh + HOME_ADVANTAGE - ra
                rows.append({"Date": match.Date, "HomeTeam": home, "AwayTeam": away,
                             "EloDiff": diff, "Season": int(season)})
                hg, ag = int(match.FTHG), int(match.FTAG)
                actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
                expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
                margin = abs(hg - ag)
                multiplier = (math.log1p(margin) *
                              (2.2 / (2.2 + 0.001 * abs(rh - ra))) if margin else 1.0)
                change = K * multiplier * (actual - expected)
                updates.append((home, away, change))
            # All matches on the same date use the same pre-match ratings.
            for home, away, change in updates:
                ratings[home] = ratings.get(home, INITIAL) + change
                ratings[away] = ratings.get(away, INITIAL) - change
    return pd.DataFrame(rows)


def metrics(probs, actual):
    p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(actual, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": len(y),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def add_market(row, match):
    try:
        odds_pair = odds_of(pd.Series(match.to_dict()))
        if odds_pair is None:
            return
        probs = devig(odds_pair[0])
        row.update({"MarketHome": float(probs[0]), "MarketDraw": float(probs[1]),
                    "MarketAway": float(probs[2])})
    except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
        pass


def main():
    print("Loading historical results...", flush=True)
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data[(data.FTHG >= 0) & (data.FTAG >= 0)].copy()
    data = data.sort_values(["Date", "League"]).reset_index(drop=True)

    all_predictions = []
    for code, league_name in LEAGUES.items():
        league = data[data.League == code].copy()
        if league.empty:
            continue
        print(f"Evaluating {league_name}...", flush=True)
        baseline, _ = elo_history(league)
        baseline = league[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].merge(
            baseline[["Date", "HomeTeam", "AwayTeam", "EloDiff"]],
            on=["Date", "HomeTeam", "AwayTeam"], validate="one_to_one")
        baseline["Season"] = [season_key(d, code) for d in baseline.Date]

        seasonal_by_carry = {}
        for carry in CARRYOVERS:
            seasonal_by_carry[carry] = seasonal_elo(league, carry)

        for year in range(2021, 2026):
            if code == "SWE":
                mask = baseline.Season == year
                season_label = str(year)
            else:
                mask = baseline.Season == year
                season_label = f"{year}/{str(year + 1)[-2:]}"
            target = baseline.loc[mask].copy()
            if len(target) < 50:
                continue
            base_probs = np.vstack([elo_probabilities(v) for v in target.EloDiff])
            for carry, elo_frame in seasonal_by_carry.items():
                scored = target[["Date", "HomeTeam", "AwayTeam"]].merge(
                    elo_frame[["Date", "HomeTeam", "AwayTeam", "EloDiff"]],
                    on=["Date", "HomeTeam", "AwayTeam"], suffixes=("_base", "_seasonal"),
                    validate="one_to_one")
                seasonal_probs = np.vstack([elo_probabilities(v) for v in scored.EloDiff])
                for i, (_, match) in enumerate(target.iterrows()):
                    actual = 0 if match.FTHG > match.FTAG else (1 if match.FTHG == match.FTAG else 2)
                    row = {
                        "League": code, "LeagueName": league_name, "Season": season_label,
                        "Date": pd.Timestamp(match.Date).date().isoformat(),
                        "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                        "Actual": ["Home", "Draw", "Away"][actual],
                        "Carryover": carry,
                        "BaseHome": float(base_probs[i, 0]), "BaseDraw": float(base_probs[i, 1]),
                        "BaseAway": float(base_probs[i, 2]),
                        "SeasonalHome": float(seasonal_probs[i, 0]),
                        "SeasonalDraw": float(seasonal_probs[i, 1]),
                        "SeasonalAway": float(seasonal_probs[i, 2]),
                    }
                    if carry == CARRYOVERS[0]:
                        add_market(row, match)
                    all_predictions.append(row)

    pred = pd.DataFrame(all_predictions)
    if pred.empty:
        raise RuntimeError("No fixtures were scored.")
    for col in ("MarketHome", "MarketDraw", "MarketAway"):
        if col not in pred:
            pred[col] = np.nan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "seasonal_elo_predictions.csv"
    summary_path = OUTPUT / "seasonal_elo_summary.csv"
    pred.to_csv(pred_path, index=False)

    actual = pred.drop_duplicates(["League", "Season", "Date", "HomeTeam", "AwayTeam"]).Actual.map(OUTCOME_MAP).to_numpy()
    unique = pred.drop_duplicates(["League", "Season", "Date", "HomeTeam", "AwayTeam"])
    dev_seasons = unique.Season.str.contains("/") & (unique.Season.str[:4].astype(int) <= 2024)
    # Allsvenskan uses calendar seasons; 2021-2024 are development, 2025 is final holdout.
    dev_mask = np.where(unique.League.eq("SWE"), unique.Season.astype(int) <= 2024,
                        unique.Season.str[:4].astype(int) <= 2024)
    test_mask = ~dev_mask
    summaries = []
    print("\n=== DEVELOPMENT PERIOD (2021-2024) ===")
    dev_rows = unique.loc[dev_mask]
    for carry in CARRYOVERS:
        group = pred[pred.Carryover == carry]
        group = group.merge(dev_rows[["League", "Season", "Date", "HomeTeam", "AwayTeam"]],
                            on=["League", "Season", "Date", "HomeTeam", "AwayTeam"])
        y = group.Actual.map(OUTCOME_MAP).to_numpy()
        p = group[["SeasonalHome", "SeasonalDraw", "SeasonalAway"]].to_numpy()
        m = metrics(p, y)
        summaries.append({"Scope": "Development 2021-2024", "Carryover": carry, **m})
        print(f"Carryover={carry:.2f} N={m['N']:5d} Brier={m['Brier']:.4f} "
              f"LogLoss={m['LogLoss']:.4f} RPS={m['RPS']:.4f} Acc={m['Accuracy']:.4f}")

    dev_summary = [s for s in summaries if s["Scope"].startswith("Development")]
    selected = min(dev_summary, key=lambda x: x["Brier"])["Carryover"]
    print(f"\nSelected carryover using development only: {selected:.2f}")
    print("\n=== FINAL HELD-OUT SEASON (2025) ===")
    final_rows = unique.loc[test_mask]
    selected_rows = pred[(pred.Carryover == selected)].merge(
        final_rows[["League", "Season", "Date", "HomeTeam", "AwayTeam"]],
        on=["League", "Season", "Date", "HomeTeam", "AwayTeam"])
    y = selected_rows.Actual.map(OUTCOME_MAP).to_numpy()
    for name, cols in {
        "Seasonal Elo": ["SeasonalHome", "SeasonalDraw", "SeasonalAway"],
        "Baseline Elo": ["BaseHome", "BaseDraw", "BaseAway"],
    }.items():
        m = metrics(selected_rows[cols].to_numpy(), y)
        summaries.append({"Scope": "Final holdout 2025", "Model": name, "Carryover": selected, **m})
        print(f"{name:16s} N={m['N']:4d} Brier={m['Brier']:.4f} "
              f"LogLoss={m['LogLoss']:.4f} RPS={m['RPS']:.4f} Acc={m['Accuracy']:.4f}")

    market_mask = selected_rows[["MarketHome", "MarketDraw", "MarketAway"]].notna().all(axis=1)
    market_rows = selected_rows.loc[market_mask]
    if len(market_rows):
        for name, cols in {
            "Seasonal Elo": ["SeasonalHome", "SeasonalDraw", "SeasonalAway"],
            "Baseline Elo": ["BaseHome", "BaseDraw", "BaseAway"],
            "No-vig market": ["MarketHome", "MarketDraw", "MarketAway"],
        }.items():
            m = metrics(market_rows[cols].to_numpy(),
                        market_rows.Actual.map(OUTCOME_MAP).to_numpy())
            summaries.append({"Scope": "Final holdout market-covered", "Model": name,
                              "Carryover": selected, **m})
            print(f"{name:16s} market sample N={m['N']:4d} Brier={m['Brier']:.4f} "
                  f"LogLoss={m['LogLoss']:.4f} RPS={m['RPS']:.4f} Acc={m['Accuracy']:.4f}")

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print("Research only: live calculator unchanged.")


if __name__ == "__main__":
    main()
