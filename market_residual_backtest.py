"""Market-anchored residual model: test whether pre-match team signals improve on no-vig closing odds.

For each target season, fit only on the two preceding seasons. The market
probabilities are a fixed log-probability offset; a regularised multinomial
model learns whether Elo and pre-match form signals add information.

European leagues: 2021/22 through 2025/26.
Allsvenskan: excluded from the market-residual fit where consistent historical 1X2 odds are unavailable.

Run: python market_residual_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from data_manager import download_data, LEAGUES
from advanced_model import elo_probabilities
from football_value_model_v3 import odds_of, devig
from hybrid_form_backtest import build_pre_match_features
from research.experimental_model import fit_residual_model, predict_residual_model

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
ABLATION_GROUPS = {
    "Elo only": ["EloDiff"],
    "Form only": ["GoalForDiff", "GoalAgainstDiff", "PointsPerMatchDiff"],
    "Shots only": ["ShotsForDiff", "ShotsAgainstDiff", "ShotsOnTargetForDiff", "ShotsOnTargetAgainstDiff"],
    "Elo + form": ["EloDiff", "GoalForDiff", "GoalAgainstDiff", "PointsPerMatchDiff"],
    "All signals": FEATURES,
}
RIDGE_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
NO_CORRECTION = -1.0  # exact market-only fallback selected on validation data


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


def _per_match_losses(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "Brier": np.sum((p - onehot) ** 2, axis=1),
        "LogLoss": -np.log(p[np.arange(len(y)), y]),
        "RPS": np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0,
    }


def paired_bootstrap_vs_market(model_probabilities, market_probabilities, outcomes,
                               n_boot=2000, seed=42):
    """Paired match-level bootstrap; negative deltas favour the tested model."""
    model_losses = _per_match_losses(model_probabilities, outcomes)
    market_losses = _per_match_losses(market_probabilities, outcomes)
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    result = {}
    for metric_name in ("Brier", "LogLoss", "RPS"):
        differences = model_losses[metric_name] - market_losses[metric_name]
        sampled = rng.integers(0, n, size=(n_boot, n))
        means = differences[sampled].mean(axis=1)
        result[f"{metric_name}_delta_vs_market"] = float(differences.mean())
        result[f"{metric_name}_ci95_low"] = float(np.quantile(means, 0.025))
        result[f"{metric_name}_ci95_high"] = float(np.quantile(means, 0.975))
    return result


def calibration_summary(probabilities, outcomes, model_name, scope, bins=10):
    """Classwise equal-width reliability bins and a simple weighted ECE."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    rows = []
    for class_index, class_name in enumerate(("Home", "Draw", "Away")):
        confidence = p[:, class_index]
        observed = (y == class_index).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for bin_index in range(bins):
            mask = ((confidence >= edges[bin_index]) &
                    ((confidence < edges[bin_index + 1]) if bin_index < bins - 1
                     else (confidence <= edges[bin_index + 1])))
            count = int(mask.sum())
            if count == 0:
                continue
            mean_probability = float(confidence[mask].mean())
            observed_frequency = float(observed[mask].mean())
            ece += count / len(y) * abs(mean_probability - observed_frequency)
            rows.append({
                "Scope": scope, "Model": model_name, "Outcome": class_name,
                "BinLow": float(edges[bin_index]), "BinHigh": float(edges[bin_index + 1]),
                "N": count, "MeanPredicted": mean_probability,
                "ObservedFrequency": observed_frequency,
            })
        for row in rows:
            if row["Scope"] == scope and row["Model"] == model_name and row["Outcome"] == class_name:
                row["ClassECE"] = float(ece)
    return rows


def choose_ridge(train, validation, feature_cols):
    """Choose regularisation on the immediately following season, never the test season."""
    x_fit = train[feature_cols].to_numpy(dtype=float)
    y_fit = np.where(train.FTHG > train.FTAG, 0,
                     np.where(train.FTHG == train.FTAG, 1, 2))
    p_fit = train[MARKET_COLS].to_numpy(dtype=float)
    x_val = validation[feature_cols].to_numpy(dtype=float)
    y_val = np.where(validation.FTHG > validation.FTAG, 0,
                     np.where(validation.FTHG == validation.FTAG, 1, 2))
    p_val = validation[MARKET_COLS].to_numpy(dtype=float)
    # Include the unchanged market as a candidate. Never force a learned
    # correction when it loses to the market on the validation season.
    market_loss = -np.mean(np.log(np.clip(p_val[np.arange(len(y_val)), y_val], 1e-9, 1.0)))
    scores = [(float(market_loss), NO_CORRECTION)]
    for ridge in RIDGE_CANDIDATES:
        candidate = fit_residual_model(x_fit, p_fit, y_fit, ridge)
        probabilities = predict_residual_model(candidate, x_val, p_val)
        loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(y_val)), y_val], 1e-9, 1.0)))
        scores.append((float(loss), float(ridge)))
    scores.sort(key=lambda item: (item[0], item[1]))
    return scores[0][1], scores[0][0]


def fit_selected_candidate(features, market_probabilities, outcomes, ridge):
    if ridge == NO_CORRECTION:
        return None
    return fit_residual_model(features, market_probabilities, outcomes, ridge)


def predict_selected_candidate(model, features, market_probabilities):
    if model is None:
        return np.asarray(market_probabilities, dtype=float).copy()
    return predict_residual_model(model, features, market_probabilities)


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
    failures = data.attrs.get("download_failures", [])
    if failures:
        print(f"WARNING: {len(failures)} data-source download/read issues:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
    else:
        print("Data-source download check: no reported failures.", flush=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    all_predictions = []
    all_ablation_predictions = []
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

            selected_ridge, validation_logloss = choose_ridge(fit_market, validation_market, FEATURES)
            x_train = train_market[FEATURES].to_numpy(dtype=float)
            y_train = np.where(train_market.FTHG > train_market.FTAG, 0,
                               np.where(train_market.FTHG == train_market.FTAG, 1, 2))
            market_train = train_market[MARKET_COLS].to_numpy(dtype=float)
            model = fit_selected_candidate(x_train, market_train, y_train, selected_ridge)

            x_test = test_market[FEATURES].to_numpy(dtype=float)
            market_test = test_market[MARKET_COLS].to_numpy(dtype=float)
            y_test = np.where(test_market.FTHG > test_market.FTAG, 0,
                              np.where(test_market.FTHG == test_market.FTAG, 1, 2))
            residual_p = predict_selected_candidate(model, x_test, market_test)

            # Feature ablation: each candidate selects ridge on the prior validation
            # season, then is refit on both prior seasons before touching the test season.
            for group_name, feature_cols in ABLATION_GROUPS.items():
                candidate_ridge, candidate_validation_loss = choose_ridge(
                    fit_market, validation_market, feature_cols
                )
                candidate_model = fit_selected_candidate(
                    train_market[feature_cols].to_numpy(dtype=float),
                    market_train, y_train, candidate_ridge
                )
                candidate_p = predict_selected_candidate(
                    candidate_model, test_market[feature_cols].to_numpy(dtype=float), market_test
                )
                for j, (_, candidate_match) in enumerate(test_market.iterrows()):
                    all_ablation_predictions.append({
                        "Date": pd.Timestamp(candidate_match.Date).date().isoformat(),
                        "League": code, "LeagueName": league_name, "Season": str(year),
                        "HomeTeam": str(candidate_match.HomeTeam),
                        "AwayTeam": str(candidate_match.AwayTeam),
                        "Actual": ["Home", "Draw", "Away"][int(y_test[j])],
                        "FeatureGroup": group_name,
                        "MarketHome": float(market_test[j, 0]),
                        "MarketDraw": float(market_test[j, 1]),
                        "MarketAway": float(market_test[j, 2]),
                        "PredHome": float(candidate_p[j, 0]),
                        "PredDraw": float(candidate_p[j, 1]),
                        "PredAway": float(candidate_p[j, 2]),
                        "SelectedRidge": float(candidate_ridge),
                        "ValidationLogLoss": float(candidate_validation_loss),
                    })

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
                    "TrainFixturesTotal": int(len(train)),
                    "TestFixturesTotal": int(len(test)),
                    "TestOddsCoverage": float(len(test_market) / max(1, len(test))),
                    "SelectedRidge": float(selected_ridge),
                    "ValidationLogLoss": float(validation_logloss),
                })
            print(f"  {year}: train odds coverage={len(train_market)}/{len(train)}, "
                  f"test odds coverage={len(test_market)}/{len(test)} "
                  f"({len(test_market) / max(1, len(test)):.1%}); "
                  f"selection={'market-only' if selected_ridge == NO_CORRECTION else 'ridge=' + format(selected_ridge, 'g')}; "
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
    uncertainty_rows = []
    calibration_rows = []
    for name, cols in [
        ("No-vig closing market", market_cols),
        ("Market + learned Elo/form residual", residual_cols),
        ("Baseline Elo", elo_cols),
    ]:
        result = metrics(predictions[cols].to_numpy(), y)
        summaries.append({"Scope": "All odds-covered matches", "Model": name, **result})
        if name != "No-vig closing market":
            comparison = paired_bootstrap_vs_market(
                predictions[cols].to_numpy(), predictions[market_cols].to_numpy(), y
            )
            uncertainty_rows.append({"Scope": "All odds-covered matches", "Model": name, **comparison})
            print(f"  vs market: Brier delta={comparison['Brier_delta_vs_market']:+.4f} "
                  f"(95% CI {comparison['Brier_ci95_low']:+.4f} to {comparison['Brier_ci95_high']:+.4f})",
                  flush=True)
        calibration_rows.extend(calibration_summary(
            predictions[cols].to_numpy(), y, name, "All odds-covered matches"
        ))
        print(f"{name:39s} N={result['N']:5d} Brier={result['Brier']:.4f} "
              f"LogLoss={result['LogLoss']:.4f} RPS={result['RPS']:.4f} "
              f"Accuracy={result['Accuracy']:.4f}", flush=True)

    print("\n=== RESULTS BY LEAGUE AND SEASON ===", flush=True)
    for (league_name, season), group in predictions.groupby(["LeagueName", "Season"], sort=True):
        actual = group.Actual.map(OUTCOME_MAP).to_numpy()
        row = {
            "Scope": f"{league_name} {season}",
            "TestFixturesTotal": int(group["TestFixturesTotal"].iloc[0]),
            "OddsCoveredTestFixtures": int(len(group)),
            "TestOddsCoverage": float(group["TestOddsCoverage"].iloc[0]),
        }
        for name, cols in [
            ("Market", market_cols), ("Residual", residual_cols), ("Elo", elo_cols)
        ]:
            result = metrics(group[cols].to_numpy(), actual)
            row.update({f"{name}_{k}": v for k, v in result.items()})
        summaries.append(row)
        for name, cols in [("Residual", residual_cols), ("Elo", elo_cols)]:
            comparison = paired_bootstrap_vs_market(
                group[cols].to_numpy(), group[market_cols].to_numpy(), actual,
                n_boot=2000, seed=42 + int(season) + len(league_name)
            )
            uncertainty_rows.append({"Scope": f"{league_name} {season}", "Model": name, **comparison})
        for name, cols in [("Market", market_cols), ("Residual", residual_cols), ("Elo", elo_cols)]:
            calibration_rows.extend(calibration_summary(
                group[cols].to_numpy(), actual, name, f"{league_name} {season}"
            ))
        print(f"{league_name:16s} {season}: "
              f"market={row['Market_Brier']:.4f} "
              f"residual={row['Residual_Brier']:.4f} "
              f"Elo={row['Elo_Brier']:.4f} N={len(group)}", flush=True)

    uncertainty_path = OUTPUT / "market_residual_uncertainty.csv"
    calibration_path = OUTPUT / "market_residual_calibration.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(uncertainty_rows).to_csv(uncertainty_path, index=False)
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    print(f"\nSaved {pred_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {uncertainty_path}")
    print(f"Saved {calibration_path}")
    # Save a paired, identical-fixture feature-ablation report against the market.
    ablation = pd.DataFrame(all_ablation_predictions)
    if not ablation.empty:
        ablation_pred_path = OUTPUT / "market_residual_ablation_predictions.csv"
        ablation_summary_path = OUTPUT / "market_residual_ablation_summary.csv"
        ablation_uncertainty_path = OUTPUT / "market_residual_ablation_uncertainty.csv"
        ablation.to_csv(ablation_pred_path, index=False)
        ablation_summaries = []
        ablation_uncertainty = []
        for group_name, group in [("All seasons", ablation)] + [
            (f"{league} {season}", part)
            for (league, season), part in ablation.groupby(["LeagueName", "Season"], sort=True)
        ]:
            actual = group.Actual.map(OUTCOME_MAP).to_numpy()
            market_p = group[["MarketHome", "MarketDraw", "MarketAway"]].to_numpy()
            for feature_name, candidate in group.groupby("FeatureGroup", sort=False):
                candidate_p = candidate[["PredHome", "PredDraw", "PredAway"]].to_numpy()
                # Grouping preserves row order; verify fixture alignment before paired scoring.
                keys = ["Date", "League", "HomeTeam", "AwayTeam", "Actual"]
                market_rows = group[group.FeatureGroup == feature_name]
                if not market_rows[keys].reset_index(drop=True).equals(candidate[keys].reset_index(drop=True)):
                    raise RuntimeError("Ablation fixture alignment check failed.")
                market_baseline = candidate[["MarketHome", "MarketDraw", "MarketAway"]].to_numpy()
                candidate_actual = candidate.Actual.map(OUTCOME_MAP).to_numpy()
                score = metrics(candidate_p, candidate_actual)
                ablation_summaries.append({
                    "Scope": group_name, "FeatureGroup": feature_name, **score,
                    "SelectedRidgeMean": float(candidate.SelectedRidge.mean()),
                })
                comparison = paired_bootstrap_vs_market(
                    candidate_p, market_baseline, candidate_actual,
                    n_boot=2000, seed=42 + len(group_name) + len(feature_name)
                )
                ablation_uncertainty.append({
                    "Scope": group_name, "FeatureGroup": feature_name, **comparison
                })
        pd.DataFrame(ablation_summaries).to_csv(ablation_summary_path, index=False)
        pd.DataFrame(ablation_uncertainty).to_csv(ablation_uncertainty_path, index=False)
        print("\\n=== FEATURE ABLATION: BRIER / LOGLOSS / DELTA VS MARKET ===", flush=True)
        overall = pd.DataFrame(ablation_summaries)
        for _, row in overall[overall.Scope == "All seasons"].iterrows():
            delta = next(
                item["Brier_delta_vs_market"] for item in ablation_uncertainty
                if item["Scope"] == "All seasons" and item["FeatureGroup"] == row.FeatureGroup
            )
            print(f"{row.FeatureGroup:16s} N={row.N:5d} Brier={row.Brier:.4f} "
                  f"LogLoss={row.LogLoss:.4f} Brier delta={delta:+.4f}", flush=True)
        print(f"Saved {ablation_pred_path}")
        print(f"Saved {ablation_summary_path}")
        print(f"Saved {ablation_uncertainty_path}")

    print("Research only: the live calculator has not been changed.", flush=True)


if __name__ == "__main__":
    main()
