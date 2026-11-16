"""Market-anchored residual model: test whether pre-match team signals improve on no-vig closing odds.

For each target season, fit only on the two preceding seasons. The market
probabilities are a fixed log-probability offset; a regularised multinomial
model learns whether Elo and pre-match form signals add information.

European leagues: 2021/22 through 2025/26.
Allsvenskan: evaluated against Elo only where historical odds are unavailable.

Run: python market_residual_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from data_manager import download_data, LEAGUES
from advanced_model import elo_probabilities
from football_value_model_v3 import odds_of, devig
from hybrid_form_backtest import build_pre_match_features

OUTPUT = Path("output")
YEARS = range(2021, 2026)
FEATURES = [
    "EloDiff",
    "GoalForDiff",
    "GoalAgainstDiff",
    "PointsPerMatchDiff",
    "ShotsForDiff",
    "ShotsAgainstDiff",
    "ShotsOnTargetForDiff",
    "ShotsOnTargetAgainstDiff",
    "HomeStatsCoverage",
    "AwayStatsCoverage",
]
MARKET_COLS = ["MarketHome", "MarketDraw", "MarketAway"]
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}
RIDGE_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]


def metrics(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
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


def fit_residual_model(x, market_p, y, ridge):
    """Fit logits = log(market probability) + learned correction."""
    x = np.asarray(x, dtype=float)
    market_p = np.clip(np.asarray(market_p, dtype=float), 1e-7, 1.0)
    market_p /= market_p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)

    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x = np.where(np.isfinite(x), x, means)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    z = np.clip((x - means) / scales, -6.0, 6.0)

    # Center the offset so numerical magnitudes remain stable.
    offset = np.log(market_p)
    offset -= offset.mean(axis=1, keepdims=True)
    design = np.column_stack([np.ones(len(z)), z])
    n_features = design.shape[1]

    def objective(flat):
        weights = flat.reshape(3, n_features)
        logits = offset + design @ weights.T
        log_norm = np.logaddexp.reduce(logits, axis=1)
        loss = np.mean(log_norm - logits[np.arange(len(y)), y])
        penalty = ridge * np.sum(weights[:, 1:] ** 2) / 2.0
        probs = softmax(logits, axis=1)
        grad = ((probs - np.eye(3)[y]).T @ design) / len(y)
        grad[:, 1:] += ridge * weights[:, 1:]
        return float(loss + penalty), grad.ravel()

    result = minimize(
        objective, np.zeros(3 * n_features), jac=True, method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Residual model fit failed: {result.message}")
    return {"weights": result.x.reshape(3, n_features), "means": means,
            "scales": scales}


def choose_ridge(train, validation):
    """Choose regularisation on the immediately following season, never the test season."""
    x_fit = train[FEATURES].to_numpy(dtype=float)
    y_fit = np.where(train.FTHG > train.FTAG, 0,
                     np.where(train.FTHG == train.FTAG, 1, 2))
    p_fit = train[MARKET_COLS].to_numpy(dtype=float)
    x_val = validation[FEATURES].to_numpy(dtype=float)
    y_val = np.where(validation.FTHG > validation.FTAG, 0,
                     np.where(validation.FTHG == validation.FTAG, 1, 2))
    p_val = validation[MARKET_COLS].to_numpy(dtype=float)
    scores = []
    for ridge in RIDGE_CANDIDATES:
        candidate = fit_residual_model(x_fit, p_fit, y_fit, ridge)
        probabilities = predict_residual_model(candidate, x_val, p_val)
        loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(y_val)), y_val], 1e-9, 1.0)))
        scores.append((float(loss), float(ridge)))
    scores.sort()
    return scores[0][1], scores[0][0]


def predict_residual_model(model, x, market_p):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, model["means"])
    z = np.clip((x - model["means"]) / model["scales"], -6.0, 6.0)
    design = np.column_stack([np.ones(len(z)), z])
    p = np.clip(np.asarray(market_p, dtype=float), 1e-7, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    offset = np.log(p)
    offset -= offset.mean(axis=1, keepdims=True)
    return softmax(offset + design @ model["weights"].T, axis=1)


def season_bounds(code, year):
    if code == "SWE":
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")
    return pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year+1}-06-30 23:59:59")


def add_market_columns(frame):
    rows = []
    for _, row in frame.iterrows():
        market = None
        try:
            pair = odds_of(row)
            if pair is not None:
                market = devig(pair[0])
        except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
            pass
        rows.append(market if market is not None else [np.nan, np.nan, np.nan])
    out = frame.copy()
    out[MARKET_COLS] = np.asarray(rows, dtype=float)
    return out


def main():
    print("Loading historical results and closing odds...", flush=True)
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    all_predictions = []
    for code, league_name in LEAGUES.items():
        league = data[data.League == code].copy()
        if league.empty:
            continue
        print(f"Building pre-match signals: {league_name}...", flush=True)
        features = build_pre_match_features(league)
        merged = league.merge(features, on=["Date", "HomeTeam", "AwayTeam"],
                              how="inner", validate="one_to_one")
        merged = add_market_columns(merged)

        for year in YEARS:
            start, end = season_bounds(code, year)
            train_start, _ = season_bounds(code, year - 2)
            train = merged[(merged.Date >= train_start) & (merged.Date < start)].copy()
            test = merged[(merged.Date >= start) & (merged.Date <= end)].copy()
            if len(train) < 250 or len(test) < 40:
                print(f"  {year}: skipped (train={len(train)}, test={len(test)})", flush=True)
                continue

            # Fit on the first prior season, select regularisation on the second,
            # then refit on both prior seasons before evaluating the untouched test season.
            validation_start, _ = season_bounds(code, year - 1)
            fit_market = train[train.Date < validation_start].dropna(subset=MARKET_COLS)
            validation_market = train[train.Date >= validation_start].dropna(subset=MARKET_COLS)
            train_market = train.dropna(subset=MARKET_COLS)
            test_market = test.dropna(subset=MARKET_COLS)
            if (len(fit_market) < 200 or len(validation_market) < 200
                    or len(train_market) < 400 or len(test_market) < 30):
                print(f"  {year}: market model skipped (fit={len(fit_market)}, "
                      f"validation={len(validation_market)}, train={len(train_market)}, "
                      f"test={len(test_market)})", flush=True)
                continue

            selected_ridge, validation_logloss = choose_ridge(fit_market, validation_market)
            x_train = train_market[FEATURES].to_numpy(dtype=float)
            y_train = np.where(train_market.FTHG > train_market.FTAG, 0,
                               np.where(train_market.FTHG == train_market.FTAG, 1, 2))
            market_train = train_market[MARKET_COLS].to_numpy(dtype=float)
            model = fit_residual_model(x_train, market_train, y_train, selected_ridge)

            x_test = test_market[FEATURES].to_numpy(dtype=float)
            market_test = test_market[MARKET_COLS].to_numpy(dtype=float)
            residual_p = predict_residual_model(model, x_test, market_test)

            for i, (_, match) in enumerate(test_market.iterrows()):
                actual = 0 if match.FTHG > match.FTAG else (1 if match.FTHG == match.FTAG else 2)
                elo_p = elo_probabilities(float(match.EloDiff))
                all_predictions.append({
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": code, "LeagueName": league_name, "Season": str(year),
                    "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                    "Actual": ["Home", "Draw", "Away"][actual],
                    "MarketHome": float(market_test[i, 0]),
                    "MarketDraw": float(market_test[i, 1]),
                    "MarketAway": float(market_test[i, 2]),
                    "ResidualHome": float(residual_p[i, 0]),
                    "ResidualDraw": float(residual_p[i, 1]),
                    "ResidualAway": float(residual_p[i, 2]),
                    "EloHome": float(elo_p[0]), "EloDraw": float(elo_p[1]),
                    "EloAway": float(elo_p[2]),
                    "TrainMatches": int(len(train_market)),
                    "SelectedRidge": float(selected_ridge),
                    "ValidationLogLoss": float(validation_logloss),
                })
            print(f"  {year}: train={len(train_market)} odds-covered, "
                  f"test={len(test_market)} odds-covered; ridge={selected_ridge:g}; "
                  f"validation logloss={validation_logloss:.4f}", flush=True)

    predictions = pd.DataFrame(all_predictions)
    if predictions.empty:
        raise RuntimeError("No market-covered fixtures could be evaluated.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "market_residual_predictions.csv"
    summary_path = OUTPUT / "market_residual_summary.csv"
    predictions.to_csv(pred_path, index=False)

    y = predictions.Actual.map(OUTCOME_MAP).to_numpy()
    market_cols = ["MarketHome", "MarketDraw", "MarketAway"]
    residual_cols = ["ResidualHome", "ResidualDraw", "ResidualAway"]
    elo_cols = ["EloHome", "EloDraw", "EloAway"]
    print("\n=== AGGREGATE WALK-FORWARD RESULTS: IDENTICAL FIXTURES ===", flush=True)
    summaries = []
    for name, cols in [
        ("No-vig closing market", market_cols),
        ("Market + learned Elo/form residual", residual_cols),
        ("Baseline Elo", elo_cols),
    ]:
        result = metrics(predictions[cols].to_numpy(), y)
        summaries.append({"Scope": "All odds-covered matches", "Model": name, **result})
        print(f"{name:39s} N={result['N']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}", flush=True)

    print("\n=== RESULTS BY LEAGUE AND SEASON ===", flush=True)
    for (league_name, season), group in predictions.groupby(["LeagueName", "Season"], sort=True):
        actual = group.Actual.map(OUTCOME_MAP).to_numpy()
        row = {"Scope": f"{league_name} {season}"}
        for name, cols in [
            ("Market", market_cols), ("Residual", residual_cols), ("Elo", elo_cols)
        ]:
            result = metrics(group[cols].to_numpy(), actual)
            row.update({f"{name}_{k}": v for k, v in result.items()})
        summaries.append(row)
        print(f"{league_name:16s} {season}: "
              f"market={row['Market_Brier']:.4f} "
              f"residual={row['Residual_Brier']:.4f} "
              f"Elo={row['Elo_Brier']:.4f} N={len(group)}", flush=True)

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print("Research only: the live calculator has not been changed.", flush=True)


if __name__ == "__main__":
    main()
