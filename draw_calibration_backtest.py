"""Out-of-sample test of league-specific draw calibration for Elo.

Fits a small logistic draw-probability model using only matches before each
competition's 2025 evaluation period. The Elo home/away split is preserved;
only the draw probability is recalibrated. This is an experiment, not the
production app model.

Run: python draw_calibration_backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from data_manager import download_data, LEAGUES
from advanced_model import elo_history, elo_probabilities
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
EUROPE_START = pd.Timestamp("2025-08-01")
EUROPE_END = pd.Timestamp("2026-06-30")
SWE_START = pd.Timestamp("2025-01-01")
SWE_END = pd.Timestamp("2025-12-31")
OUTCOMES = np.array(["Home", "Draw", "Away"])
BANDS = [0, 25, 50, 100, 150, 250, 1000]
BAND_LABELS = ["0-25", "25-50", "50-100", "100-150", "150-250", "250+"]


def outcome_index(home_goals, away_goals):
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def draw_features(elo_diff):
    # Scale for stable optimisation. Quadratic term allows a non-monotonic
    # relationship if the historical evidence supports one.
    x = np.clip(np.abs(np.asarray(elo_diff, dtype=float)) / 200.0, 0.0, 5.0)
    return np.column_stack([np.ones(len(x)), x, x * x])


def fit_draw_model(train):
    x = draw_features(train["EloDiff"].to_numpy())
    y = (train["Actual"].to_numpy() == "Draw").astype(float)
    if len(y) < 300 or y.min() == y.max():
        raise ValueError("Insufficient historical matches to fit draw calibration.")

    def objective(beta):
        z = np.clip(x @ beta, -25, 25)
        # Mean logistic loss plus light L2 regularisation on the two slopes.
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        penalty = 0.02 * np.sum(beta[1:] ** 2)
        return float(loss + penalty)

    initial_rate = np.clip(y.mean(), 0.05, 0.50)
    initial = np.array([np.log(initial_rate / (1.0 - initial_rate)), 0.0, 0.0])
    result = minimize(objective, initial, method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError("Draw calibration optimisation failed.")
    return result.x


def calibrated_probabilities(elo_diff, beta):
    base = elo_probabilities(float(elo_diff))
    draw = float(np.clip(expit(draw_features([elo_diff]) @ beta)[0], 0.08, 0.40))
    decisive = 1.0 - draw
    base_decisive = base[0] + base[2]
    if base_decisive <= 0:
        home_share = 0.5
    else:
        home_share = base[0] / base_decisive
    return np.array([decisive * home_share, draw, decisive * (1.0 - home_share)])


def metrics(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "Matches": int(len(y)),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "Log loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def main():
    print("Loading historical results...")
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    print("Building leakage-safe pre-match Elo snapshots...")
    elo_rows, _ = elo_history(data)
    elo_lookup = {
        (pd.Timestamp(r["Date"]).normalize(), str(r["HomeTeam"]), str(r["AwayTeam"])): r
        for r in elo_rows.to_dict("records")
    }

    prediction_rows = []
    for league, league_name in LEAGUES.items():
        start, end = (SWE_START, SWE_END) if league == "SWE" else (EUROPE_START, EUROPE_END)
        league_data = data[data.League == league].copy()
        train = league_data[league_data.Date < start].copy()
        targets = league_data[(league_data.Date >= start) & (league_data.Date <= end)].copy()
        if targets.empty:
            print(f"{league_name}: no target fixtures found; skipped.")
            continue
        if len(train) < 300:
            print(f"{league_name}: only {len(train)} prior matches; skipped.")
            continue

        train_rows = []
        for _, m in train.iterrows():
            key = (pd.Timestamp(m.Date).normalize(), str(m.HomeTeam), str(m.AwayTeam))
            snap = elo_lookup.get(key)
            if snap is None:
                continue
            train_rows.append({
                "EloDiff": float(snap["EloDiff"]),
                "Actual": OUTCOMES[outcome_index(float(m.FTHG), float(m.FTAG))],
            })
        train_frame = pd.DataFrame(train_rows)
        beta = fit_draw_model(train_frame)
        print(f"{league_name}: fitted on {len(train_frame)} pre-period matches; "
              f"draw coefficients={np.round(beta, 4).tolist()}")

        for _, m in targets.iterrows():
            key = (pd.Timestamp(m.Date).normalize(), str(m.HomeTeam), str(m.AwayTeam))
            snap = elo_lookup.get(key)
            if snap is None:
                continue
            diff = float(snap["EloDiff"])
            base = elo_probabilities(diff)
            calibrated = calibrated_probabilities(diff, beta)
            row = {
                "Date": pd.Timestamp(m.Date).date().isoformat(),
                "League": league,
                "LeagueName": league_name,
                "HomeTeam": m.HomeTeam,
                "AwayTeam": m.AwayTeam,
                "Actual": OUTCOMES[outcome_index(float(m.FTHG), float(m.FTAG))],
                "EloDiff": diff,
                "EloPHome": base[0], "EloPDraw": base[1], "EloPAway": base[2],
                "CalPHome": calibrated[0], "CalPDraw": calibrated[1], "CalPAway": calibrated[2],
            }
            odds_pair = odds_of(m)
            if odds_pair is not None:
                try:
                    market, _ = odds_pair
                    mp = devig(market)
                    row.update({"MarketPHome": float(mp[0]), "MarketPDraw": float(mp[1]),
                                "MarketPAway": float(mp[2])})
                except (ValueError, FloatingPointError):
                    pass
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No fixtures could be scored.")
    market_cols = ["MarketPHome", "MarketPDraw", "MarketPAway"]
    for col in market_cols:
        if col not in predictions:
            predictions[col] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "draw_calibration_predictions.csv", index=False)

    print("\n=== OUT-OF-SAMPLE RESULTS BY LEAGUE ===")
    summary_rows = []
    for (code, name), group in predictions.groupby(["League", "LeagueName"], sort=True):
        y = group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
        for label, cols in [
            ("Baseline Elo", ["EloPHome", "EloPDraw", "EloPAway"]),
            ("Calibrated Elo", ["CalPHome", "CalPDraw", "CalPAway"]),
        ]:
            result = metrics(group[cols].to_numpy(), y)
            summary_rows.append({"League": code, "LeagueName": name, "Model": label, **result})
            print(f"{name:16s} {label:16s} " +
                  " ".join(f"{k}={v:.4f}" for k, v in result.items() if k != "Matches") +
                  f" N={result['Matches']}")
        market = group.dropna(subset=market_cols)
        if not market.empty:
            my = market.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
            for label, cols in [
                ("Baseline Elo (odds subset)", ["EloPHome", "EloPDraw", "EloPAway"]),
                ("Calibrated Elo (odds subset)", ["CalPHome", "CalPDraw", "CalPAway"]),
                ("No-vig market", market_cols),
            ]:
                result = metrics(market[cols].to_numpy(), my)
                summary_rows.append({"League": code, "LeagueName": name, "Model": label, **result})
                print(f"{name:16s} {label:28s} Brier={result['Brier']:.4f} "
                      f"LogLoss={result['Log loss']:.4f} RPS={result['RPS']:.4f} "
                      f"Acc={result['Accuracy']:.3f} N={result['Matches']}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "draw_calibration_summary.csv", index=False)

    print("\n=== ELO-DIFFERENCE DIAGNOSTIC ===")
    predictions["EloBand"] = pd.cut(predictions.EloDiff.abs(), BANDS,
                                    labels=BAND_LABELS, include_lowest=True)
    for band, group in predictions.groupby("EloBand", observed=True):
        y = group.Actual.to_numpy()
        base = group[["EloPHome", "EloPDraw", "EloPAway"]].to_numpy(float)
        cal = group[["CalPHome", "CalPDraw", "CalPAway"]].to_numpy(float)
        print(f"{str(band):8s} N={len(group):4d} "
              f"actual_draw={np.mean(y == 'Draw'):.3f} "
              f"base_draw={base[:, 1].mean():.3f} "
              f"cal_draw={cal[:, 1].mean():.3f} "
              f"base_acc={np.mean(OUTCOMES[base.argmax(axis=1)] == y):.3f} "
              f"cal_acc={np.mean(OUTCOMES[cal.argmax(axis=1)] == y):.3f}")

    print("\nSaved output/draw_calibration_predictions.csv")
    print("Saved output/draw_calibration_summary.csv")
    print("Do not replace the app model unless calibrated Elo improves proper scoring rules "
          "on the out-of-sample period.")


if __name__ == "__main__":
    main()
