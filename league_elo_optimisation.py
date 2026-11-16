"""Walk-forward optimisation of league-specific Elo K-factor and home advantage.

Each target season is predicted using parameters selected only on the two
immediately preceding seasons. Elo snapshots are always pre-match. The
experiment does not alter the production calculator.

Run: python league_elo_optimisation.py
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
EUROPE_YEARS = range(2021, 2026)
SWEDEN_YEARS = range(2021, 2026)
K_VALUES = [12.0, 16.0, 20.0, 22.0, 26.0, 30.0, 34.0]
HOME_ADVANTAGES = [45.0, 55.0, 65.0, 75.0, 85.0]
BASELINE = (22.0, 65.0)
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}
MARKET_COLS = ["MarketHome", "MarketDraw", "MarketAway"]


def key(date, home, away):
    return (pd.Timestamp(date).normalize(), str(home), str(away))


def snapshots(data, k, home_advantage):
    rows, _ = elo_history(data, k=k, home_advantage=home_advantage)
    return {
        key(row.Date, row.HomeTeam, row.AwayTeam): float(row.EloDiff)
        for row in rows.itertuples(index=False)
    }


def outcomes(frame):
    return np.where(frame.FTHG > frame.FTAG, 0,
                    np.where(frame.FTHG == frame.FTAG, 1, 2)).astype(int)


def probabilities(diffs):
    return np.vstack([elo_probabilities(float(diff)) for diff in diffs])


def metrics(probs, actual):
    p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(actual, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": int(len(y)),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def tune_parameters(validation, candidate_lookups):
    """Choose K/home advantage by validation Brier using precomputed snapshots."""
    validation_keys = [
        key(row.Date, row.HomeTeam, row.AwayTeam)
        for row in validation.itertuples(index=False)
    ]
    actual = outcomes(validation)
    best = None
    for (k_value, home_advantage), lookup in candidate_lookups.items():
        diffs = [lookup.get(item) for item in validation_keys]
        valid = np.array([value is not None for value in diffs])
        if valid.sum() < 100:
            continue
        p = probabilities(np.asarray(diffs, dtype=float)[valid])
        score = metrics(p, actual[valid])["Brier"]
        candidate = (score, k_value, home_advantage)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not tune Elo parameters: too few validation fixtures.")
    return best[1], best[2], best[0]


def main():
    print("Loading historical results...")
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    prediction_rows = []
    for league_code, league_name in LEAGUES.items():
        league_data = data[data.League == league_code].copy()
        years = SWEDEN_YEARS if league_code == "SWE" else EUROPE_YEARS
        candidates = list(dict.fromkeys([BASELINE] + list(product(K_VALUES, HOME_ADVANTAGES))))
        candidate_lookups = {}
        print(f"Precomputing {len(candidates)} Elo configurations for {league_name}...", flush=True)
        for index, (candidate_k, candidate_ha) in enumerate(candidates, start=1):
            candidate_lookups[(candidate_k, candidate_ha)] = snapshots(
                league_data, candidate_k, candidate_ha
            )
            if index % 5 == 0 or index == len(candidates):
                print(f"  {index}/{len(candidates)} configurations ready", flush=True)
        for year in years:
            if league_code == "SWE":
                start = pd.Timestamp(f"{year}-01-01")
                end = pd.Timestamp(f"{year}-12-31")
                validation_start = start - pd.DateOffset(years=2)
                season = str(year)
            else:
                start = pd.Timestamp(f"{year}-08-01")
                end = pd.Timestamp(f"{year + 1}-06-30")
                validation_start = start - pd.DateOffset(years=2)
                season = f"{year}/{str(year + 1)[-2:]}"

            history = league_data[league_data.Date < start].copy()
            validation = history[history.Date >= validation_start].copy()
            target = league_data[(league_data.Date >= start) &
                                 (league_data.Date <= end)].copy()
            if len(validation) < 100 or len(target) < 50:
                print(f"{league_name:16s} {season}: skipped "
                      f"(validation={len(validation)}, target={len(target)})")
                continue

            best_k, best_ha, validation_brier = tune_parameters(validation, candidate_lookups)

            # Reuse full chronological pre-match snapshots. Future results cannot
            # affect an earlier snapshot because elo_history updates only after
            # each date block has been scored.
            tuned_lookup = candidate_lookups[(best_k, best_ha)]
            baseline_lookup = candidate_lookups[BASELINE]

            scored = 0
            for match in target.itertuples(index=False):
                match_key = key(match.Date, match.HomeTeam, match.AwayTeam)
                tuned_diff = tuned_lookup.get(match_key)
                baseline_diff = baseline_lookup.get(match_key)
                if tuned_diff is None or baseline_diff is None:
                    continue
                tuned_p = probabilities([tuned_diff])[0]
                baseline_p = probabilities([baseline_diff])[0]
                actual = (0 if match.FTHG > match.FTAG else
                          1 if match.FTHG == match.FTAG else 2)
                row = {
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": league_code, "LeagueName": league_name,
                    "Season": season, "HomeTeam": match.HomeTeam,
                    "AwayTeam": match.AwayTeam,
                    "Actual": ["Home", "Draw", "Away"][actual],
                    "SelectedK": best_k, "SelectedHomeAdvantage": best_ha,
                    "ValidationBrier": validation_brier,
                    "BaselineHome": baseline_p[0], "BaselineDraw": baseline_p[1],
                    "BaselineAway": baseline_p[2],
                    "TunedHome": tuned_p[0], "TunedDraw": tuned_p[1],
                    "TunedAway": tuned_p[2],
                }
                try:
                    odds_pair = odds_of(pd.Series(match._asdict()))
                    if odds_pair is not None:
                        market, _ = odds_pair
                        market_p = devig(market)
                        row.update(dict(zip(MARKET_COLS, map(float, market_p))))
                except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
                    pass
                prediction_rows.append(row)
                scored += 1

            print(f"{league_name:16s} {season}: validation={len(validation):4d}, "
                  f"test={scored:3d}, K={best_k:4.0f}, HA={best_ha:4.0f}, "
                  f"validation Brier={validation_brier:.4f}")

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No target fixtures were scored.")
    for col in MARKET_COLS:
        if col not in predictions:
            predictions[col] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT / "league_elo_optimisation_predictions.csv"
    summary_path = OUTPUT / "league_elo_optimisation_summary.csv"
    predictions.to_csv(prediction_path, index=False)

    actual = predictions.Actual.map(OUTCOME_MAP).to_numpy()
    model_specs = {
        "Baseline Elo": ["BaselineHome", "BaselineDraw", "BaselineAway"],
        "Optimised Elo": ["TunedHome", "TunedDraw", "TunedAway"],
    }
    print("\n=== AGGREGATE OUT-OF-SAMPLE RESULTS ===")
    summary_rows = []
    for label, cols in model_specs.items():
        result = metrics(predictions[cols].to_numpy(), actual)
        summary_rows.append({"Scope": "All matches", "Model": label, **result})
        print(f"{label:16s} N={result['N']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}")

    # Fair benchmark: all three models are scored on precisely the odds-covered
    # fixtures, not on different match populations.
    market_mask = predictions[MARKET_COLS].notna().all(axis=1)
    if market_mask.any():
        subset = predictions.loc[market_mask]
        subset_actual = subset.Actual.map(OUTCOME_MAP).to_numpy()
        print(f"\n=== SAME-SAMPLE MARKET COMPARISON (N={len(subset)}) ===")
        for label, cols in model_specs.items():
            result = metrics(subset[cols].to_numpy(), subset_actual)
            summary_rows.append({"Scope": "Odds-covered matches", "Model": label, **result})
            print(f"{label:16s} Brier={result['Brier']:.4f} "
                  f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
                  f"Accuracy={result['Accuracy']:.4f}")
        market_result = metrics(subset[MARKET_COLS].to_numpy(), subset_actual)
        summary_rows.append({"Scope": "Odds-covered matches", "Model": "No-vig market",
                             **market_result})
        print(f"{'No-vig market':16s} Brier={market_result['Brier']:.4f} "
              f"LogLoss={market_result['LogLoss']:.4f} RPS={market_result['RPS']:.4f} "
              f"Accuracy={market_result['Accuracy']:.4f}")

    print("\n=== RESULTS BY LEAGUE AND SEASON ===")
    for (league, season), group in predictions.groupby(["LeagueName", "Season"], sort=True):
        y = group.Actual.map(OUTCOME_MAP).to_numpy()
        base = metrics(group[model_specs["Baseline Elo"]].to_numpy(), y)
        tuned = metrics(group[model_specs["Optimised Elo"]].to_numpy(), y)
        print(f"{league:16s} {season:7s} N={len(group):3d} "
              f"Baseline Brier={base['Brier']:.4f} | Tuned Brier={tuned['Brier']:.4f} "
              f"| K={group.SelectedK.iloc[0]:.0f} "
              f"| HA={group.SelectedHomeAdvantage.iloc[0]:.0f}")

    improved = 0
    tested = 0
    for _, group in predictions.groupby(["League", "Season"]):
        y = group.Actual.map(OUTCOME_MAP).to_numpy()
        base = metrics(group[model_specs["Baseline Elo"]].to_numpy(), y)["Brier"]
        tuned = metrics(group[model_specs["Optimised Elo"]].to_numpy(), y)["Brier"]
        improved += int(tuned < base)
        tested += 1
    print(f"\nOptimised Elo beats baseline Brier in {improved}/{tested} league-seasons.")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {prediction_path}")
    print(f"Saved {summary_path}")
    print("Research only: selected parameters are not automatically applied to the app.")


if __name__ == "__main__":
    main()
