"""Strict chronological test of market information beyond Elo.

Protocol:
  Train: 2021-2023 seasons
  Validate/select ridge: 2024 season
  Final refit: 2021-2024 seasons
  One untouched test: 2025 season

Evaluates identical fixtures with average opening and closing odds:
  - Elo only
  - Opening market only
  - Closing market only
  - Elo + opening market (regularised multinomial stack)
  - Elo + closing market (regularised multinomial stack)
  - Elo + opening + closing market (tests incremental closing information)

Market probabilities use additive no-vig conversion. Elo snapshots are
chronological and pre-match. Confidence intervals resample league-week
clusters, not individual fixtures.

Run: python chronological_market_test.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities

OUTPUT = Path("output")
YEARS = range(2021, 2026)
RIDGE_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_BOOT = 2000
SEED = 20261116
OUTCOME_NAMES = ["Home", "Draw", "Away"]


def season_bounds(code, year):
    if code == "SWE":
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")
    return pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year+1}-06-30 23:59:59")


def additive_no_vig(row, cols):
    values = pd.to_numeric(row[list(cols)], errors="coerce").to_numpy(dtype=float)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        return None
    if np.any(values <= 1.01) or np.any(values > 100):
        return None
    implied = 1.0 / values
    overround = float(implied.sum())
    probs = np.maximum(implied - (overround - 1.0) / 3.0, 1e-9)
    probs /= probs.sum()
    return probs


def outcome_index(home_goals, away_goals):
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def centred_log_probs(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    logs = np.log(p)
    return logs - logs.mean(axis=1, keepdims=True)


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": int(len(y)),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def fit_stack(x, y, ridge):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    z = np.clip((x - means) / scales, -8.0, 8.0)
    design = np.column_stack([np.ones(len(z)), z])
    n_features = design.shape[1]

    def objective(flat):
        weights = flat.reshape(3, n_features)
        logits = design @ weights.T
        log_norm = np.logaddexp.reduce(logits, axis=1)
        loss = np.mean(log_norm - logits[np.arange(len(y)), y])
        penalty = ridge * np.sum(weights[:, 1:] ** 2) / 2.0
        probs = softmax(logits, axis=1)
        grad = ((probs - np.eye(3)[y]).T @ design) / len(y)
        grad[:, 1:] += ridge * weights[:, 1:]
        return float(loss + penalty), grad.ravel()

    result = minimize(
        objective, np.zeros(3 * n_features), jac=True, method="L-BFGS-B",
        options={"maxiter": 1200, "ftol": 1e-10},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Stack fit failed: {result.message}")
    return {"weights": result.x.reshape(3, n_features), "means": means, "scales": scales}


def predict_stack(model, x):
    x = np.asarray(x, dtype=float)
    z = np.clip((x - model["means"]) / model["scales"], -8.0, 8.0)
    design = np.column_stack([np.ones(len(z)), z])
    return softmax(design @ model["weights"].T, axis=1)


def choose_ridge(x_train, y_train, x_val, y_val):
    best = None
    for ridge in RIDGE_CANDIDATES:
        model = fit_stack(x_train, y_train, ridge)
        p = predict_stack(model, x_val)
        loss = metrics(p, y_val)["LogLoss"]
        if best is None or loss < best[0]:
            best = (loss, ridge)
    return float(best[1]), float(best[0])


def paired_cluster_bootstrap(predictions, y, cluster, rng):
    """Paired 95% CIs for model-vs-reference score differences."""
    names = list(predictions)
    references = ["Closing market", "Elo"]
    rows = []
    unique_clusters = pd.unique(cluster)
    cluster_indices = [np.flatnonzero(cluster == key) for key in unique_clusters]
    n_clusters = len(cluster_indices)

    # Per-fixture loss arrays, so resampling preserves pairing.
    onehot = np.eye(3)[y]
    losses = {}
    for name, p in predictions.items():
        p = np.clip(np.asarray(p), 1e-9, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        losses[name] = {
            "brier": np.sum((p - onehot) ** 2, axis=1),
            "logloss": -np.log(p[np.arange(len(y)), y]),
        }

    for reference in references:
        for name in names:
            if name == reference:
                continue
            for metric_name in ("brier", "logloss"):
                # Positive gain means the candidate is better than the reference.
                delta = losses[reference][metric_name] - losses[name][metric_name]
                observed = float(delta.mean())
                boot = np.empty(N_BOOT)
                for start in range(0, N_BOOT, 100):
                    count = min(100, N_BOOT - start)
                    chosen = rng.integers(0, n_clusters, size=(count, n_clusters))
                    sums = np.zeros(count)
                    counts = np.zeros(count)
                    for j in range(n_clusters):
                        idx = cluster_indices[chosen[:, j]]
                        for b in range(count):
                            sums[b] += delta[idx[b]].sum()
                            counts[b] += len(idx[b])
                    boot[start:start + count] = sums / counts
                rows.append({
                    "Candidate": name, "Reference": reference, "Metric": metric_name,
                    "Gain": observed, "CI95Low": float(np.quantile(boot, 0.025)),
                    "CI95High": float(np.quantile(boot, 0.975)),
                    "Clusters": n_clusters, "Fixtures": len(y),
                })
    return rows


def main():
    print("Loading historical results and odds...", flush=True)
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()

    rows = []
    for code, league_name in LEAGUES.items():
        league = data[data.League == code].sort_values("Date").copy()
        if league.empty:
            continue
        print(f"Building chronological Elo: {league_name}", flush=True)
        elo_rows, _ = elo_history(league)
        elo_map = {
            (pd.Timestamp(r.Date).normalize(), str(r.HomeTeam), str(r.AwayTeam)): float(r.EloDiff)
            for r in elo_rows.itertuples(index=False)
        }
        for year in YEARS:
            start, end = season_bounds(code, year)
            season = league[(league.Date >= start) & (league.Date <= end)]
            for _, match in season.iterrows():
                key = (pd.Timestamp(match.Date).normalize(), str(match.HomeTeam), str(match.AwayTeam))
                diff = elo_map.get(key)
                if diff is None:
                    continue
                opening = additive_no_vig(match, ("AvgH", "AvgD", "AvgA"))
                closing = additive_no_vig(match, ("AvgCH", "AvgCD", "AvgCA"))
                # All comparisons use the exact same fixture set with both price snapshots.
                if opening is None or closing is None:
                    continue
                elo_p = elo_probabilities(diff)
                rows.append({
                    "Date": pd.Timestamp(match.Date).normalize(),
                    "League": code, "LeagueName": league_name, "Season": year,
                    "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                    "Actual": outcome_index(float(match.FTHG), float(match.FTAG)),
                    "EloDiff": float(diff),
                    "EloHome": float(elo_p[0]), "EloDraw": float(elo_p[1]), "EloAway": float(elo_p[2]),
                    "OpenHome": float(opening[0]), "OpenDraw": float(opening[1]), "OpenAway": float(opening[2]),
                    "CloseHome": float(closing[0]), "CloseDraw": float(closing[1]), "CloseAway": float(closing[2]),
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No fixtures with Elo, average opening odds and average closing odds.")
    frame["Season"] = frame.Season.astype(int)
    train = frame[frame.Season.between(2021, 2023)].copy()
    validation = frame[frame.Season == 2024].copy()
    test = frame[frame.Season == 2025].copy()
    if min(len(train), len(validation), len(test)) < 100:
        raise RuntimeError(
            f"Insufficient matched fixtures: train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

    def matrices(part):
        elo = part[["EloHome", "EloDraw", "EloAway"]].to_numpy(float)
        opening = part[["OpenHome", "OpenDraw", "OpenAway"]].to_numpy(float)
        closing = part[["CloseHome", "CloseDraw", "CloseAway"]].to_numpy(float)
        return {
            "Elo": centred_log_probs(elo),
            "Opening": centred_log_probs(opening),
            "Closing": centred_log_probs(closing),
        }

    train_x, val_x, test_x = matrices(train), matrices(validation), matrices(test)
    y_train, y_val = train.Actual.to_numpy(int), validation.Actual.to_numpy(int)
    y_test = test.Actual.to_numpy(int)

    combinations = {
        "Elo + opening": ["Elo", "Opening"],
        "Elo + closing": ["Elo", "Closing"],
        "Elo + opening + closing": ["Elo", "Opening", "Closing"],
    }
    selected = {}
    print(f"Matched fixtures: train 2021-23 N={len(train)}, validation 2024 N={len(validation)}, test 2025 N={len(test)}")
    print("\n=== VALIDATION: RIDGE SELECTION (2024 ONLY) ===")
    for name, sources in combinations.items():
        xtr = np.column_stack([train_x[s] for s in sources])
        xva = np.column_stack([val_x[s] for s in sources])
        ridge, val_loss = choose_ridge(xtr, y_train, xva, y_val)
        selected[name] = ridge
        print(f"{name:27s} ridge={ridge:g} validation logloss={val_loss:.5f}")

    # Refit using all pre-test seasons after selecting hyperparameters on 2024.
    pretest = frame[frame.Season.between(2021, 2024)].copy()
    pre_x = matrices(pretest)
    y_pre = pretest.Actual.to_numpy(int)
    test_predictions = {
        "Elo": test[["EloHome", "EloDraw", "EloAway"]].to_numpy(float),
        "Opening market": test[["OpenHome", "OpenDraw", "OpenAway"]].to_numpy(float),
        "Closing market": test[["CloseHome", "CloseDraw", "CloseAway"]].to_numpy(float),
    }
    fitted_rows = []
    print("\n=== FINAL TEST: 2025 ONLY ===")
    for name, sources in combinations.items():
        x_pre = np.column_stack([pre_x[s] for s in sources])
        x_test = np.column_stack([test_x[s] for s in sources])
        model = fit_stack(x_pre, y_pre, selected[name])
        p = predict_stack(model, x_test)
        test_predictions[name] = p
        fitted_rows.append({"Model": name, "SelectedRidge": selected[name]})

    summary_rows = []
    for name, p in test_predictions.items():
        result = metrics(p, y_test)
        summary_rows.append({"Model": name, **result})
        print(
            f"{name:27s} N={result['N']:4d} Brier={result['Brier']:.5f} "
            f"LogLoss={result['LogLoss']:.5f} RPS={result['RPS']:.5f} "
            f"Acc={result['Accuracy']:.4f}"
        )

    # League-week clusters keep matches in the same competition and calendar week together.
    dates = pd.to_datetime(test.Date)
    iso = dates.dt.isocalendar()
    clusters = (test.League.astype(str) + "_" + iso.year.astype(str) + "_" +
                iso.week.astype(str)).to_numpy()
    rng = np.random.default_rng(SEED)
    ci_rows = paired_cluster_bootstrap(test_predictions, y_test, clusters, rng)
    ci_frame = pd.DataFrame(ci_rows)
    print("\n=== PAIRED LEAGUE-WEEK BOOTSTRAP VS CLOSING MARKET ===")
    for r in ci_frame.itertuples(index=False):
        print(
            f"vs {r.Reference:14s} | {r.Candidate:27s} {r.Metric:7s} "
            f"gain={r.Gain:+.6f} 95% CI [{r.CI95Low:+.6f}, {r.CI95High:+.6f}]"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "chronological_market_all_fixtures.csv", index=False)
    test_out = test[["Date", "League", "LeagueName", "Season", "HomeTeam", "AwayTeam", "Actual"]].copy()
    for name, p in test_predictions.items():
        slug = name.lower().replace(" ", "_").replace("+", "plus")
        for j, outcome in enumerate(OUTCOME_NAMES):
            test_out[f"{slug}_{outcome}"] = p[:, j]
    test_out.to_csv(OUTPUT / "chronological_market_2025_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "chronological_market_2025_summary.csv", index=False)
    ci_frame.to_csv(OUTPUT / "chronological_market_2025_bootstrap.csv", index=False)
    pd.DataFrame(fitted_rows).to_csv(OUTPUT / "chronological_market_selected_models.csv", index=False)
    print("\nSaved chronological market test outputs under output/.")
    print("No production app changes were made.", flush=True)


if __name__ == "__main__":
    main()
