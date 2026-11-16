"""Walk-forward Poisson attack/defence model experiment.

Fits team attack and defence strengths using only the two seasons before each
target season, then evaluates 1X2 probabilities on the untouched target season.
Compares against pre-match Elo and no-vig closing odds on identical fixtures.

Run: python attack_defence_backtest.py
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from data_manager import download_data, LEAGUES
from advanced_model import elo_history
from football_value_model_v3 import odds_of, devig

OUTPUT = Path("output")
EUROPE_YEARS = range(2021, 2026)
SWEDEN_YEARS = range(2021, 2026)
MARKET_COLS = ["MarketHome", "MarketDraw", "MarketAway"]
OUTCOME_MAP = {"Home": 0, "Draw": 1, "Away": 2}
MAX_GOALS = 10
RIDGE = 0.08


def fixture_key(date, home, away):
    return (pd.Timestamp(date).normalize(), str(home), str(away))


def actual_outcomes(frame):
    return np.where(frame.FTHG > frame.FTAG, 0,
                    np.where(frame.FTHG == frame.FTAG, 1, 2)).astype(int)


def fit_attack_defence(train):
    """Regularised independent-Poisson team attack/defence fit."""
    teams = sorted(set(train.HomeTeam.astype(str)) | set(train.AwayTeam.astype(str)))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    if n < 4 or len(train) < max(100, n * 3):
        raise ValueError("Not enough training fixtures for attack/defence fit.")

    hi = train.HomeTeam.astype(str).map(index).to_numpy(dtype=int)
    ai = train.AwayTeam.astype(str).map(index).to_numpy(dtype=int)
    hg = train.FTHG.to_numpy(dtype=float)
    ag = train.FTAG.to_numpy(dtype=float)
    league_home = float(np.clip(hg.mean(), 0.2, 4.0))
    league_away = float(np.clip(ag.mean(), 0.2, 4.0))
    initial = np.zeros(2 * n + 2, dtype=float)
    initial[2*n] = math.log(league_home)
    initial[2*n+1] = math.log(league_away)

    def unpack(theta):
        attack = theta[:n] - theta[:n].mean()
        defence = theta[n:2*n] - theta[n:2*n].mean()
        return attack, defence, theta[2*n], theta[2*n+1]

    def objective(theta):
        attack, defence, home_intercept, away_intercept = unpack(theta)
        log_h = np.clip(home_intercept + attack[hi] - defence[ai], -3.0, 2.0)
        log_a = np.clip(away_intercept + attack[ai] - defence[hi], -3.0, 2.0)
        loss = np.sum(np.exp(log_h) - hg * log_h + gammaln(hg + 1.0))
        loss += np.sum(np.exp(log_a) - ag * log_a + gammaln(ag + 1.0))
        loss += RIDGE * (np.sum(attack**2) + np.sum(defence**2))
        return float(loss)

    result = minimize(objective, initial, method="L-BFGS-B",
                      options={"maxiter": 1200, "ftol": 1e-10})
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Attack/defence fit failed: {result.message}")
    attack, defence, home_intercept, away_intercept = unpack(result.x)
    return {
        "teams": index, "attack": attack, "defence": defence,
        "home_intercept": float(home_intercept),
        "away_intercept": float(away_intercept),
        "converged": bool(result.success),
    }


def score_probabilities(model, home, away):
    teams = model["teams"]
    if home not in teams or away not in teams:
        return None
    hi, ai = teams[home], teams[away]
    lam_h = math.exp(float(np.clip(
        model["home_intercept"] + model["attack"][hi] - model["defence"][ai], -3, 2
    )))
    lam_a = math.exp(float(np.clip(
        model["away_intercept"] + model["attack"][ai] - model["defence"][hi], -3, 2
    )))
    # Include the complete 10+ goal tail in the final bucket.
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals, lam_h)
    pa = poisson.pmf(goals, lam_a)
    ph[-1] = poisson.sf(MAX_GOALS - 1, lam_h)
    pa[-1] = poisson.sf(MAX_GOALS - 1, lam_a)
    matrix = np.outer(ph, pa)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    probs = np.array([home_win, draw, away_win], dtype=float)
    return probs / probs.sum(), lam_h, lam_a


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
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data = data.dropna(subset=["FTHG", "FTAG"])
    data = data.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    rows = []
    for code, league_name in LEAGUES.items():
        league = data[data.League == code].copy()
        if league.empty:
            continue
        print(f"Building Elo benchmark for {league_name}...", flush=True)
        elo_rows, _ = elo_history(league)
        elo_lookup = {
            fixture_key(r.Date, r.HomeTeam, r.AwayTeam): float(r.EloDiff)
            for r in elo_rows.itertuples(index=False)
        }
        years = SWEDEN_YEARS if code == "SWE" else EUROPE_YEARS
        for year in years:
            if code == "SWE":
                start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")
                season = str(year)
            else:
                start, end = pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year+1}-06-30 23:59:59")
                season = f"{year}/{str(year+1)[-2:]}"
            train_start = start - pd.DateOffset(years=2)
            train = league[(league.Date >= train_start) & (league.Date < start)].copy()
            target = league[(league.Date >= start) & (league.Date <= end)].copy()
            if len(train) < 250 or len(target) < 50:
                print(f"  {season}: skipped (train={len(train)}, test={len(target)})", flush=True)
                continue
            model = fit_attack_defence(train)
            scored = 0
            for match in target.itertuples(index=False):
                key = fixture_key(match.Date, match.HomeTeam, match.AwayTeam)
                result = score_probabilities(model, str(match.HomeTeam), str(match.AwayTeam))
                elo_diff = elo_lookup.get(key)
                if result is None or elo_diff is None:
                    continue
                ad_probs, xg_home, xg_away = result
                # Existing production Elo probability mapping, unchanged.
                from advanced_model import elo_probabilities
                elo_probs = elo_probabilities(elo_diff)
                actual = 0 if match.FTHG > match.FTAG else (1 if match.FTHG == match.FTAG else 2)
                row = {
                    "Date": pd.Timestamp(match.Date).date().isoformat(),
                    "League": code, "LeagueName": league_name, "Season": season,
                    "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                    "Actual": ["Home", "Draw", "Away"][actual],
                    "AttackDefenceHome": ad_probs[0], "AttackDefenceDraw": ad_probs[1],
                    "AttackDefenceAway": ad_probs[2],
                    "EloHome": elo_probs[0], "EloDraw": elo_probs[1], "EloAway": elo_probs[2],
                    "ExpectedGoalsHome": xg_home, "ExpectedGoalsAway": xg_away,
                    "TrainingMatches": len(train), "FitConverged": model["converged"],
                }
                try:
                    odds_pair = odds_of(pd.Series(match._asdict()))
                    if odds_pair is not None:
                        market, _ = odds_pair
                        market_probs = devig(market)
                        for col, value in zip(MARKET_COLS, market_probs):
                            row[col] = float(value)
                except (ValueError, FloatingPointError, TypeError, KeyError, AttributeError):
                    pass
                rows.append(row)
                scored += 1
            print(f"  {season}: train={len(train):4d}, test={scored:3d}, "
                  f"fit={'ok' if model['converged'] else 'warning'}", flush=True)

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No fixtures were scored.")
    for col in MARKET_COLS:
        if col not in predictions:
            predictions[col] = np.nan
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "attack_defence_predictions.csv"
    summary_path = OUTPUT / "attack_defence_summary.csv"
    predictions.to_csv(pred_path, index=False)

    specs = {
        "Attack/defence": ["AttackDefenceHome", "AttackDefenceDraw", "AttackDefenceAway"],
        "Baseline Elo": ["EloHome", "EloDraw", "EloAway"],
    }
    actual = predictions.Actual.map(OUTCOME_MAP).to_numpy()
    print("\n=== AGGREGATE OUT-OF-SAMPLE RESULTS ===")
    summary = []
    for name, cols in specs.items():
        result = metrics(predictions[cols].to_numpy(), actual)
        summary.append({"Scope": "All matches", "Model": name, **result})
        print(f"{name:16s} N={result['N']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}")

    market_mask = predictions[MARKET_COLS].notna().all(axis=1)
    if market_mask.any():
        subset = predictions.loc[market_mask]
        y = subset.Actual.map(OUTCOME_MAP).to_numpy()
        print(f"\n=== SAME-SAMPLE MARKET COMPARISON (N={len(subset)}) ===")
        for name, cols in specs.items():
            result = metrics(subset[cols].to_numpy(), y)
            summary.append({"Scope": "Odds-covered matches", "Model": name, **result})
            print(f"{name:16s} Brier={result['Brier']:.4f} "
                  f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
                  f"Accuracy={result['Accuracy']:.4f}")
        result = metrics(subset[MARKET_COLS].to_numpy(), y)
        summary.append({"Scope": "Odds-covered matches", "Model": "No-vig market", **result})
        print(f"{'No-vig market':16s} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}")

    print("\n=== RESULTS BY LEAGUE AND SEASON ===")
    for (name, season), group in predictions.groupby(["LeagueName", "Season"], sort=True):
        y = group.Actual.map(OUTCOME_MAP).to_numpy()
        ad = metrics(group[specs["Attack/defence"]].to_numpy(), y)
        elo = metrics(group[specs["Baseline Elo"]].to_numpy(), y)
        print(f"{name:16s} {season:7s} N={len(group):3d} "
              f"Attack/defence Brier={ad['Brier']:.4f} | Elo Brier={elo['Brier']:.4f}")
        summary.append({"Scope": f"{name} {season}", "Model": "Attack/defence", **ad})
        summary.append({"Scope": f"{name} {season}", "Model": "Baseline Elo", **elo})

    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print("Research only: this experiment does not change the live calculator.")


if __name__ == "__main__":
    main()
