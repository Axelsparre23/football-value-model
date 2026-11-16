"""Walk-forward calibration test for no-vig closing 1X2 market probabilities.

Fits a restrained three-class calibration layer (shared temperature + class biases).
For each target season: fit on the first prior season, select regularisation on the
second prior season, refit on both prior seasons, then evaluate the untouched target.
No team features are used. Compare raw market and calibrated market on identical fixtures.

Run: python market_calibration_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from data_manager import download_data, LEAGUES
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
YEARS = range(2021, 2026)
MARKET_COLS = ["MarketHome", "MarketDraw", "MarketAway"]
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}
RIDGE_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def season_bounds(code, year):
    if code == "SWE":
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")
    return pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year+1}-06-30 23:59:59")


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": int(len(y)),
        "Brier": float(np.mean(np.sum((p-onehot)**2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p-cumulative_y)**2, axis=1)/2)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def fit_calibrator(market_p, y, ridge):
    p = np.clip(np.asarray(market_p, dtype=float), 1e-7, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    logits = np.log(p)
    # A shared temperature plus three class offsets; offsets are centred
    # to avoid an unidentified common shift.
    def objective(theta):
        log_temp = theta[0]
        bias = theta[1:] - np.mean(theta[1:])
        z = logits * np.exp(-log_temp) + bias
        log_norm = np.logaddexp.reduce(z, axis=1)
        loss = np.mean(log_norm - z[np.arange(len(y)), y])
        penalty = 0.5 * ridge * (log_temp**2 + np.sum(bias**2))
        probs = softmax(z, axis=1)
        residual = probs - np.eye(3)[y]
        grad_logits = (residual.T @ logits).sum() / len(y)
        grad_log_temp = -np.exp(-log_temp) * grad_logits + ridge * log_temp
        grad_bias = residual.mean(axis=0) + ridge * bias
        grad_raw_bias = grad_bias - grad_bias.mean()
        return float(loss + penalty), np.r_[grad_log_temp, grad_raw_bias]
    result = minimize(objective, np.zeros(4), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-11})
    if not np.isfinite(result.fun):
        raise RuntimeError("Calibration optimisation returned a non-finite result.")
    theta = result.x
    return {"log_temp": float(theta[0]),
            "bias": (theta[1:] - theta[1:].mean()).astype(float)}


def apply_calibrator(model, market_p):
    p = np.clip(np.asarray(market_p, dtype=float), 1e-7, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    z = np.log(p) * np.exp(-model["log_temp"]) + model["bias"]
    return softmax(z, axis=1)


def choose_ridge(fit, validation):
    x_fit = fit[MARKET_COLS].to_numpy(dtype=float)
    y_fit = fit.Actual.map(OUTCOME_MAP).to_numpy()
    x_val = validation[MARKET_COLS].to_numpy(dtype=float)
    y_val = validation.Actual.map(OUTCOME_MAP).to_numpy()
    scores = []
    for ridge in RIDGE_CANDIDATES:
        model = fit_calibrator(x_fit, y_fit, ridge)
        p = apply_calibrator(model, x_val)
        loss = metrics(p, y_val)["LogLoss"]
        scores.append((loss, ridge))
    scores.sort()
    return float(scores[0][1]), float(scores[0][0])


def add_market(frame):
    rows = []
    for _, row in frame.iterrows():
        try:
            pair = odds_of(row)
            probs = devig(pair[0]) if pair is not None else None
        except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
            probs = None
        rows.append(probs if probs is not None else [np.nan, np.nan, np.nan])
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
    rows = []
    for code, league_name in LEAGUES.items():
        league = add_market(data[data.League == code].copy())
        if league.empty:
            continue
        for year in YEARS:
            start, end = season_bounds(code, year)
            train_start, _ = season_bounds(code, year-2)
            train = league[(league.Date >= train_start) & (league.Date < start)].dropna(subset=MARKET_COLS).copy()
            test = league[(league.Date >= start) & (league.Date <= end)].dropna(subset=MARKET_COLS).copy()
            validation_start, _ = season_bounds(code, year-1)
            fit = train[train.Date < validation_start].copy()
            validation = train[train.Date >= validation_start].copy()
            if min(len(fit), len(validation), len(test)) < 100:
                print(f"{league_name} {year}: skipped fit={len(fit)} validation={len(validation)} test={len(test)}", flush=True)
                continue
            ridge, val_loss = choose_ridge(fit, validation)
            refit = fit_calibrator(train[MARKET_COLS].to_numpy(dtype=float),
                                   train.Actual.map(OUTCOME_MAP).to_numpy() if "Actual" in train else
                                   np.where(train.FTHG > train.FTAG, 0, np.where(train.FTHG == train.FTAG, 1, 2)),
                                   ridge)
            market_test = test[MARKET_COLS].to_numpy(dtype=float)
            calibrated = apply_calibrator(refit, market_test)
            for i, (_, match) in enumerate(test.iterrows()):
                actual = "Home" if match.FTHG > match.FTAG else ("Draw" if match.FTHG == match.FTAG else "Away")
                rows.append({
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": code, "LeagueName": league_name, "Season": str(year),
                    "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                    "Actual": actual,
                    "MarketHome": float(market_test[i,0]), "MarketDraw": float(market_test[i,1]), "MarketAway": float(market_test[i,2]),
                    "CalibratedHome": float(calibrated[i,0]), "CalibratedDraw": float(calibrated[i,1]), "CalibratedAway": float(calibrated[i,2]),
                    "SelectedRidge": ridge, "ValidationLogLoss": val_loss,
                })
            print(f"{league_name} {year}: fit={len(fit)} validation={len(validation)} test={len(test)} ridge={ridge:g} val_logloss={val_loss:.4f}", flush=True)
    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("No odds-covered matches were evaluated.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUTPUT / "market_calibration_predictions.csv", index=False)
    y = pred.Actual.map(OUTCOME_MAP).to_numpy()
    market_cols = ["MarketHome", "MarketDraw", "MarketAway"]
    calibrated_cols = ["CalibratedHome", "CalibratedDraw", "CalibratedAway"]
    print("\n=== AGGREGATE WALK-FORWARD RESULTS: IDENTICAL FIXTURES ===", flush=True)
    summaries = []
    for name, cols in [("Raw no-vig market", market_cols), ("Calibrated market", calibrated_cols)]:
        score = metrics(pred[cols].to_numpy(), y)
        summaries.append({"Scope":"All odds-covered matches","Model":name, **score})
        print(f"{name:24s} N={score['N']:5d} Brier={score['Brier']:.4f} LogLoss={score['LogLoss']:.4f} RPS={score['RPS']:.4f} Accuracy={score['Accuracy']:.4f}", flush=True)
    print("\n=== BY LEAGUE AND SEASON ===", flush=True)
    for (league, season), group in pred.groupby(["LeagueName","Season"], sort=True):
        actual = group.Actual.map(OUTCOME_MAP).to_numpy()
        m = metrics(group[market_cols].to_numpy(), actual)
        c = metrics(group[calibrated_cols].to_numpy(), actual)
        summaries.append({"Scope":f"{league} {season}","Model":"Raw market",**m})
        summaries.append({"Scope":f"{league} {season}","Model":"Calibrated market",**c})
        print(f"{league:16s} {season}: raw={m['Brier']:.4f} calibrated={c['Brier']:.4f} | LL {m['LogLoss']:.4f}->{c['LogLoss']:.4f}", flush=True)
    pd.DataFrame(summaries).to_csv(OUTPUT / "market_calibration_summary.csv", index=False)
    print("\nSaved output/market_calibration_predictions.csv")
    print("Saved output/market_calibration_summary.csv")


if __name__ == "__main__":
    main()
