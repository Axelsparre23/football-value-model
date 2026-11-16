"""Walk-forward hybrid Elo + recent-form model experiment.

Uses only pre-match information. Adds exponentially weighted goals, points,
shots and shots-on-target form to Elo, then fits a regularised multinomial
logistic model on the two seasons before each test season.

Run: python hybrid_form_backtest.py
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
YEARS = range(2021, 2026)
FEATURES = [
    "EloDiff",
    "GoalForDiff", "GoalAgainstDiff", "PointsPerMatchDiff",
    "ShotsForDiff", "ShotsAgainstDiff",
    "ShotsOnTargetForDiff", "ShotsOnTargetAgainstDiff",
    "HomeStatsCoverage", "AwayStatsCoverage",
]
MARKET_COLS = ["MarketHome", "MarketDraw", "MarketAway"]
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}
ALPHA = 0.20
RIDGE = 0.08


def fixture_key(date, home, away):
    return (pd.Timestamp(date).normalize(), str(home), str(away))


def metric_value(row, col):
    value = row.get(col, np.nan)
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def build_pre_match_features(league_data):
    """Create features from team state before each calendar date is updated."""
    elo_rows, _ = elo_history(league_data)
    elo_lookup = {
        fixture_key(r.Date, r.HomeTeam, r.AwayTeam): float(r.EloDiff)
        for r in elo_rows.itertuples(index=False)
    }
    # Values are EWMAs. Counts track how much actual match history supports them.
    defaults = {
        "gf": 1.45, "ga": 1.45, "ppm": 1.30,
        "shots_for": 12.0, "shots_against": 12.0,
        "sot_for": 4.0, "sot_against": 4.0,
    }
    states = {}
    records = []

    def state(team):
        if team not in states:
            states[team] = {**defaults, "n": 0, "stats_n": 0}
        return states[team]

    ordered = league_data.sort_values(["Date", "HomeTeam", "AwayTeam"])
    for date, day in ordered.groupby(ordered.Date.dt.normalize(), sort=True):
        updates = []
        for _, match in day.iterrows():
            home, away = str(match.HomeTeam), str(match.AwayTeam)
            hs, aws = state(home), state(away)
            key = fixture_key(date, home, away)
            elo_diff = elo_lookup.get(key, 0.0)
            # Positive GoalAgainstDiff means the home side has recently conceded more.
            features = {
                "Date": pd.Timestamp(match.Date),
                "HomeTeam": home, "AwayTeam": away,
                "EloDiff": elo_diff,
                "GoalForDiff": hs["gf"] - aws["gf"],
                "GoalAgainstDiff": hs["ga"] - aws["ga"],
                "PointsPerMatchDiff": hs["ppm"] - aws["ppm"],
                "ShotsForDiff": hs["shots_for"] - aws["shots_for"],
                "ShotsAgainstDiff": hs["shots_against"] - aws["shots_against"],
                "ShotsOnTargetForDiff": hs["sot_for"] - aws["sot_for"],
                "ShotsOnTargetAgainstDiff": hs["sot_against"] - aws["sot_against"],
                "HomeStatsCoverage": hs["stats_n"] / max(1, hs["n"]),
                "AwayStatsCoverage": aws["stats_n"] / max(1, aws["n"]),
            }
            records.append(features)

            hg, ag = float(match.FTHG), float(match.FTAG)
            hp = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
            ap = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
            h_stats = {
                "shots_for": metric_value(match, "HS"),
                "shots_against": metric_value(match, "AS"),
                "sot_for": metric_value(match, "HST"),
                "sot_against": metric_value(match, "AST"),
            }
            a_stats = {
                "shots_for": metric_value(match, "AS"),
                "shots_against": metric_value(match, "HS"),
                "sot_for": metric_value(match, "AST"),
                "sot_against": metric_value(match, "HST"),
            }
            updates.append((home, away, hg, ag, hp, ap, h_stats, a_stats))

        # Apply outcomes only after every match on this date has been featured.
        for home, away, hg, ag, hp, ap, h_stats, a_stats in updates:
            hs, aws = state(home), state(away)
            for key, value in (("gf", hg), ("ga", ag), ("ppm", hp)):
                hs[key] = (1 - ALPHA) * hs[key] + ALPHA * value
            for key, value in (("gf", ag), ("ga", hg), ("ppm", ap)):
                aws[key] = (1 - ALPHA) * aws[key] + ALPHA * value
            h_valid = a_valid = False
            for key, value in h_stats.items():
                if np.isfinite(value):
                    hs[key] = (1 - ALPHA) * hs[key] + ALPHA * value
                    h_valid = True
            for key, value in a_stats.items():
                if np.isfinite(value):
                    aws[key] = (1 - ALPHA) * aws[key] + ALPHA * value
                    a_valid = True
            hs["n"] += 1
            aws["n"] += 1
            hs["stats_n"] += int(h_valid)
            aws["stats_n"] += int(a_valid)

    return pd.DataFrame(records)


def fit_multinomial(train_x, train_y):
    """Standardised, L2-regularised multinomial logistic regression."""
    x = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=int)
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x = np.where(np.isfinite(x), x, means)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    z = np.clip((x - means) / scales, -8.0, 8.0)
    design = np.column_stack([np.ones(len(z)), z])
    n_features = design.shape[1]

    def objective(flat):
        weights = flat.reshape(3, n_features)
        logits = design @ weights.T
        log_norm = np.logaddexp.reduce(logits, axis=1)
        loss = np.mean(log_norm - logits[np.arange(len(y)), y])
        penalty = RIDGE * np.sum(weights[:, 1:] ** 2) / (2 * len(y))
        probs = softmax(logits, axis=1)
        onehot = np.eye(3)[y]
        grad = ((probs - onehot).T @ design) / len(y)
        grad[:, 1:] += (RIDGE / len(y)) * weights[:, 1:]
        return float(loss + penalty), grad.ravel()

    result = minimize(objective, np.zeros(3 * n_features), jac=True, method="L-BFGS-B",
                      options={"maxiter": 800, "ftol": 1e-10})
    if not np.isfinite(result.fun):
        raise RuntimeError("Multinomial model optimisation failed.")
    return {"weights": result.x.reshape(3, n_features), "means": means, "scales": scales}


def predict(model, x):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, model["means"])
    z = np.clip((x - model["means"]) / model["scales"], -8.0, 8.0)
    design = np.column_stack([np.ones(len(z)), z])
    return softmax(design @ model["weights"].T, axis=1)


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


def main():
    print("Loading historical results...", flush=True)
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League"]).reset_index(drop=True)

    all_rows = []
    for code, league_name in LEAGUES.items():
        league = data[data.League == code].copy()
        if league.empty:
            continue
        print(f"Building pre-match form features for {league_name}...", flush=True)
        features = build_pre_match_features(league)
        merged = league.merge(features, on=["Date", "HomeTeam", "AwayTeam"],
                              how="inner", validate="one_to_one")
        years = YEARS
        for year in years:
            if code == "SWE":
                start, end, season = (pd.Timestamp(f"{year}-01-01"),
                                      pd.Timestamp(f"{year}-12-31 23:59:59"), str(year))
            else:
                start, end, season = (pd.Timestamp(f"{year}-08-01"),
                                      pd.Timestamp(f"{year+1}-06-30 23:59:59"),
                                      f"{year}/{str(year+1)[-2:]}")
            train_start = start - pd.DateOffset(years=2)
            train = merged[(merged.Date >= train_start) & (merged.Date < start)].copy()
            test = merged[(merged.Date >= start) & (merged.Date <= end)].copy()
            if len(train) < 250 or len(test) < 50:
                print(f"  {season}: skipped (train={len(train)}, test={len(test)})", flush=True)
                continue
            x_train = train[FEATURES].to_numpy(dtype=float)
            y_train = np.where(train.FTHG > train.FTAG, 0,
                               np.where(train.FTHG == train.FTAG, 1, 2))
            model = fit_multinomial(x_train, y_train)
            hybrid_probs = predict(model, test[FEATURES].to_numpy(dtype=float))
            for i, (_, match) in enumerate(test.iterrows()):
                actual = 0 if match.FTHG > match.FTAG else (1 if match.FTHG == match.FTAG else 2)
                elo_probs = elo_probabilities(float(match.EloDiff))
                row = {
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": code, "LeagueName": league_name, "Season": season,
                    "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                    "Actual": ["Home", "Draw", "Away"][actual],
                    "HybridHome": float(hybrid_probs[i, 0]),
                    "HybridDraw": float(hybrid_probs[i, 1]),
                    "HybridAway": float(hybrid_probs[i, 2]),
                    "EloHome": float(elo_probs[0]),
                    "EloDraw": float(elo_probs[1]),
                    "EloAway": float(elo_probs[2]),
                }
                try:
                    odds_pair = odds_of(pd.Series(match.to_dict()))
                    if odds_pair is not None:
                        market_probs = devig(odds_pair[0])
                        row.update(dict(zip(MARKET_COLS, map(float, market_probs))))
                except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
                    pass
                all_rows.append(row)
            print(f"  {season}: train={len(train):4d}, test={len(test):3d}, fit=ok", flush=True)

    predictions = pd.DataFrame(all_rows)
    if predictions.empty:
        raise RuntimeError("No fixtures were scored.")
    for col in MARKET_COLS:
        if col not in predictions:
            predictions[col] = np.nan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "hybrid_form_predictions.csv"
    summary_path = OUTPUT / "hybrid_form_summary.csv"
    predictions.to_csv(pred_path, index=False)

    specs = {
        "Hybrid Elo + form": ["HybridHome", "HybridDraw", "HybridAway"],
        "Baseline Elo": ["EloHome", "EloDraw", "EloAway"],
    }
    actual = predictions.Actual.map(OUTCOME_MAP).to_numpy()
    summaries = []
    print("\n=== AGGREGATE OUT-OF-SAMPLE RESULTS ===")
    for name, cols in specs.items():
        result = metrics(predictions[cols].to_numpy(), actual)
        summaries.append({"Scope": "All matches", "Model": name, **result})
        print(f"{name:20s} N={result['N']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}")

    market_mask = predictions[MARKET_COLS].notna().all(axis=1)
    if market_mask.any():
        subset = predictions.loc[market_mask]
        y = subset.Actual.map(OUTCOME_MAP).to_numpy()
        print(f"\n=== SAME-SAMPLE MARKET COMPARISON (N={len(subset)}) ===")
        for name, cols in specs.items():
            result = metrics(subset[cols].to_numpy(), y)
            summaries.append({"Scope": "Odds-covered matches", "Model": name, **result})
            print(f"{name:20s} Brier={result['Brier']:.4f} "
                  f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
                  f"Accuracy={result['Accuracy']:.4f}")
        result = metrics(subset[MARKET_COLS].to_numpy(), y)
        summaries.append({"Scope": "Odds-covered matches", "Model": "No-vig market", **result})
        print(f"{'No-vig market':20s} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}")

    print("\n=== RESULTS BY LEAGUE AND SEASON ===")
    for (league_name, season), group in predictions.groupby(["LeagueName", "Season"], sort=True):
        y = group.Actual.map(OUTCOME_MAP).to_numpy()
        hybrid = metrics(group[specs["Hybrid Elo + form"]].to_numpy(), y)
        elo = metrics(group[specs["Baseline Elo"]].to_numpy(), y)
        print(f"{league_name:16s} {season:7s} N={len(group):3d} "
              f"Hybrid Brier={hybrid['Brier']:.4f} | Elo Brier={elo['Brier']:.4f}")
        summaries.append({"Scope": f"{league_name} {season}", "Model": "Hybrid Elo + form", **hybrid})
        summaries.append({"Scope": f"{league_name} {season}", "Model": "Baseline Elo", **elo})

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print("Research only: the live calculator has not been changed.")


if __name__ == "__main__":
    main()
