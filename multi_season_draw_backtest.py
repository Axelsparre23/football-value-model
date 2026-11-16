"""Multi-season walk-forward test of flexible draw-probability calibration.

For each league and evaluation season, both calibrators are fitted only on
matches completed before that season. Elo ratings are chronological pre-match
snapshots. No target-season results are used to fit the draw model.

European leagues: 2021/22 through 2025/26.
Allsvenskan: calendar seasons 2021 through 2025.

Run: python multi_season_draw_backtest.py
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
EUROPE_YEARS = range(2021, 2026)
SWEDEN_YEARS = range(2021, 2026)
KNOTS = np.array([0, 25, 50, 75, 100, 150, 200, 300, 450, 700], dtype=float)
OUTCOMES = np.array(["Home", "Draw", "Away"])
MARKET_COLS = ["MarketPHome", "MarketPDraw", "MarketPAway"]


def outcome_index(home_goals, away_goals):
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def quadratic_features(elo_diff):
    x = np.clip(np.abs(np.asarray(elo_diff, dtype=float)) / 200.0, 0.0, 5.0)
    return np.column_stack([np.ones(len(x)), x, x * x])


def flexible_features(elo_diff):
    """Piecewise-linear basis over absolute Elo difference; can be non-monotonic."""
    x = np.clip(np.abs(np.asarray(elo_diff, dtype=float)), KNOTS[0], KNOTS[-1])
    basis = np.zeros((len(x), len(KNOTS)), dtype=float)
    right = np.searchsorted(KNOTS, x, side="right")
    right = np.clip(right, 1, len(KNOTS) - 1)
    left = right - 1
    span = KNOTS[right] - KNOTS[left]
    fraction = np.divide(x - KNOTS[left], span, out=np.zeros_like(x), where=span != 0)
    basis[np.arange(len(x)), left] = 1.0 - fraction
    basis[np.arange(len(x)), right] = fraction
    # First basis column is absorbed into the intercept to avoid collinearity.
    return np.column_stack([np.ones(len(x)), basis[:, 1:]])


def fit_logistic(features, labels, penalty):
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(y) < 300 or y.min() == y.max():
        raise ValueError("Insufficient historical matches to fit draw calibration.")

    def objective(beta):
        z = np.clip(x @ beta, -30.0, 30.0)
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        regularisation = penalty * np.sum(beta[1:] ** 2)
        return float(loss + regularisation)

    def gradient(beta):
        z = np.clip(x @ beta, -30.0, 30.0)
        residual = expit(z) - y
        grad = (x.T @ residual) / len(y)
        grad[1:] += 2.0 * penalty * beta[1:]
        return grad

    rate = np.clip(y.mean(), 0.05, 0.50)
    initial = np.zeros(x.shape[1], dtype=float)
    initial[0] = np.log(rate / (1.0 - rate))
    result = minimize(objective, initial, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Draw calibration fit failed: {result.message}")
    return result.x


def make_probabilities(elo_diff, beta, feature_function):
    base = elo_probabilities(float(elo_diff))
    draw = float(np.clip(expit(feature_function([elo_diff]) @ beta)[0], 0.08, 0.40))
    decisive = 1.0 - draw
    base_decisive = base[0] + base[2]
    home_share = base[0] / base_decisive if base_decisive > 0 else 0.5
    return np.array([decisive * home_share, draw, decisive * (1.0 - home_share)])


def score_metrics(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "Matches": int(len(y)),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def match_key(match):
    if isinstance(match, dict):
        date, home, away = match["Date"], match["HomeTeam"], match["AwayTeam"]
    else:
        date, home, away = match.Date, match.HomeTeam, match.AwayTeam
    return (pd.Timestamp(date).normalize(), str(home), str(away))


def main():
    print("Loading historical results...")
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    print("Building leakage-safe pre-match Elo snapshots...")
    elo_rows, _ = elo_history(data)
    elo_lookup = {match_key(row): row for row in elo_rows.to_dict("records")}

    prediction_rows = []
    for league_code, league_name in LEAGUES.items():
        league_data = data[data.League == league_code].copy()
        years = SWEDEN_YEARS if league_code == "SWE" else EUROPE_YEARS
        for year in years:
            if league_code == "SWE":
                start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
                season = str(year)
            else:
                start, end = pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year + 1}-06-30")
                season = f"{year}/{str(year + 1)[-2:]}"

            train = league_data[league_data.Date < start]
            targets = league_data[(league_data.Date >= start) & (league_data.Date <= end)]
            if len(targets) < 50 or len(train) < 300:
                print(f"{league_name} {season}: skipped (train={len(train)}, target={len(targets)})")
                continue

            train_diffs, train_labels = [], []
            for match in train.itertuples(index=False):
                snap = elo_lookup.get(match_key(match))
                if snap is None:
                    continue
                train_diffs.append(float(snap["EloDiff"]))
                train_labels.append(int(float(match.FTHG) == float(match.FTAG)))
            train_diffs = np.asarray(train_diffs, dtype=float)
            train_labels = np.asarray(train_labels, dtype=float)
            if len(train_labels) < 300:
                print(f"{league_name} {season}: skipped (usable train={len(train_labels)})")
                continue

            beta_quad = fit_logistic(quadratic_features(train_diffs), train_labels, penalty=0.02)
            beta_flex = fit_logistic(flexible_features(train_diffs), train_labels, penalty=0.08)
            print(f"{league_name:16s} {season}: train={len(train_labels):5d}, "
                  f"test={len(targets):3d}, draw-rate={train_labels.mean():.3f}")

            for match in targets.itertuples(index=False):
                snap = elo_lookup.get(match_key(match))
                if snap is None:
                    continue
                diff = float(snap["EloDiff"])
                base = elo_probabilities(diff)
                quad = make_probabilities(diff, beta_quad, quadratic_features)
                flex = make_probabilities(diff, beta_flex, flexible_features)
                actual = outcome_index(float(match.FTHG), float(match.FTAG))
                row = {
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": league_code, "LeagueName": league_name, "Season": season,
                    "HomeTeam": match.HomeTeam, "AwayTeam": match.AwayTeam,
                    "Actual": OUTCOMES[actual], "EloDiff": diff,
                    "BaseHome": base[0], "BaseDraw": base[1], "BaseAway": base[2],
                    "QuadHome": quad[0], "QuadDraw": quad[1], "QuadAway": quad[2],
                    "FlexHome": flex[0], "FlexDraw": flex[1], "FlexAway": flex[2],
                    "TrainingMatches": len(train_labels),
                }
                odds_pair = odds_of(match._asdict())
                if odds_pair is not None:
                    try:
                        market, _ = odds_pair
                        mp = devig(market)
                        row.update(dict(zip(MARKET_COLS, map(float, mp))))
                    except (ValueError, FloatingPointError, TypeError):
                        pass
                prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No evaluation fixtures were scored.")
    for col in MARKET_COLS:
        if col not in predictions:
            predictions[col] = np.nan

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "multi_season_draw_predictions.csv"
    summary_path = OUTPUT / "multi_season_draw_summary.csv"
    predictions.to_csv(pred_path, index=False)

    specs = {
        "Baseline Elo": ["BaseHome", "BaseDraw", "BaseAway"],
        "Quadratic calibration": ["QuadHome", "QuadDraw", "QuadAway"],
        "Flexible calibration": ["FlexHome", "FlexDraw", "FlexAway"],
    }
    summary_rows = []

    def report(label, group, cols):
        y = group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
        result = score_metrics(group[cols].to_numpy(), y)
        summary_rows.append({"Scope": label, "Model": label if label in specs else "No-vig market",
                             **result})
        return result

    print("\n=== AGGREGATE OUT-OF-SAMPLE RESULTS ===")
    for model, cols in specs.items():
        result = report(model, predictions, cols)
        print(f"{model:24s} N={result['Matches']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.3f}")

    odds = predictions.dropna(subset=MARKET_COLS)
    if not odds.empty:
        y = odds.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
        print(f"{'No-vig market':24s} N={len(odds):5d} " +
              " ".join(f"{k}={v:.4f}" for k, v in score_metrics(odds[MARKET_COLS], y).items()
                       if k != "Matches"))

    print("\n=== RESULTS BY LEAGUE AND SEASON ===")
    for (code, name, season), group in predictions.groupby(
            ["League", "LeagueName", "Season"], sort=True):
        y = group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
        parts = []
        for model, cols in specs.items():
            m = score_metrics(group[cols].to_numpy(), y)
            summary_rows.append({"Scope": f"{name} {season}", "Model": model, **m})
            parts.append(f"{model.split()[0]} Brier={m['Brier']:.4f}")
        odds_group = group.dropna(subset=MARKET_COLS)
        if len(odds_group):
            oy = odds_group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
            om = score_metrics(odds_group[MARKET_COLS].to_numpy(), oy)
            summary_rows.append({"Scope": f"{name} {season}", "Model": "No-vig market", **om})
            parts.append(f"Market Brier={om['Brier']:.4f}")
        print(f"{name:16s} {season:7s} N={len(group):3d} " + " | ".join(parts))

    # Compare calibrated models to baseline on identical league-season samples.
    print("\n=== CONSISTENCY CHECK ===")
    for model, cols in [("Quadratic", specs["Quadratic calibration"]),
                        ("Flexible", specs["Flexible calibration"])]:
        improved = 0
        tested = 0
        for _, group in predictions.groupby(["League", "Season"]):
            y = group.Actual.map({"Home": 0, "Draw": 1, "Away": 2}).to_numpy()
            base_brier = score_metrics(group[specs["Baseline Elo"]].to_numpy(), y)["Brier"]
            model_brier = score_metrics(group[cols].to_numpy(), y)["Brier"]
            improved += int(model_brier < base_brier)
            tested += 1
        print(f"{model:10s}: lower Brier than baseline in {improved}/{tested} league-seasons")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print("Interpret only aggregate and repeated league-season improvements; do not select a "
          "calibrator from a single season or league.")


if __name__ == "__main__":
    main()
