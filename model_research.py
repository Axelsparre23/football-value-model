"""Leakage-aware chronological evaluation for Dixon-Coles + Elo."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from advanced_model import elo_history, elo_probabilities, blended_probabilities


def _normalised_probs(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    return p / p.sum()


def _temperature(p, temperature):
    return softmax(np.log(_normalised_probs(p)) / float(temperature))


def _fit_temperature(probabilities, outcomes):
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)

    def objective(log_t):
        t = math.exp(float(log_t))
        calibrated = np.vstack([_temperature(row, t) for row in probs])
        return float(-np.mean(np.log(np.clip(calibrated[np.arange(len(y)), y], 1e-9, 1.0))))

    result = minimize_scalar(
        objective, bounds=(math.log(0.65), math.log(1.8)), method="bounded"
    )
    return float(math.exp(result.x)) if result.success else 1.0


def _metrics(probabilities, outcomes):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    one_hot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(one_hot, axis=1)[:, :2]
    return {
        "Brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "Log loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "Matches": int(len(y)),
    }


def _score_block(block, model, elo_lookup, elo_weight=None, temperature=1.0):
    dc_rows, elo_rows, actuals = [], [], []
    for _, match in block.iterrows():
        key = (pd.Timestamp(match.Date).normalize(), str(match.HomeTeam), str(match.AwayTeam))
        elo = elo_lookup.get(key)
        if elo is None:
            continue
        try:
            dc, _ = model.probabilities(match.HomeTeam, match.AwayTeam)
        except (KeyError, ValueError, OverflowError):
            continue
        elo_p = elo_probabilities(float(elo["EloDiff"]))
        actual = 0 if match.FTHG > match.FTAG else (1 if match.FTHG == match.FTAG else 2)
        dc_rows.append(_normalised_probs(dc))
        elo_rows.append(_normalised_probs(elo_p))
        actuals.append(actual)

    if not actuals:
        return None
    dc_arr = np.asarray(dc_rows)
    elo_arr = np.asarray(elo_rows)
    y = np.asarray(actuals, dtype=int)
    result = {
        "dixon_coles": dc_arr,
        "elo": elo_arr,
        "outcomes": y,
    }
    if elo_weight is not None:
        blend = np.vstack([
            blended_probabilities(d, e, elo_weight)
            for d, e in zip(dc_arr, elo_arr)
        ])
        result["ensemble"] = np.vstack([_temperature(row, temperature) for row in blend])
    return result


def chronological_ensemble_evaluation(data, dc_model_class):
    """Tune blend weight/calibration on a middle time block; report once on final block.

    All partitions use whole match dates. Elo snapshots are generated chronologically,
    with every fixture on the same date using the same pre-date ratings.
    """
    df = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    dates = np.array(sorted(df.Date.dt.normalize().unique()))
    if len(df) < 500 or len(dates) < 60:
        raise ValueError("Need at least 500 matches across 60 match dates for a useful three-way evaluation.")

    date_blocks = np.array_split(dates, 5)
    train_dates = np.concatenate(date_blocks[:3])
    validation_dates = date_blocks[3]
    test_dates = date_blocks[4]
    train_end = pd.Timestamp(train_dates[-1])
    validation_end = pd.Timestamp(validation_dates[-1])
    train = df[df.Date.dt.normalize().isin(train_dates)].copy()
    train_validation = df[df.Date.dt.normalize().isin(np.concatenate([train_dates, validation_dates]))].copy()
    validation = df[df.Date.dt.normalize().isin(validation_dates)].copy()
    test = df[df.Date.dt.normalize().isin(test_dates)].copy()

    if len(train) < 300 or len(validation) < 30 or len(test) < 30:
        raise ValueError("The chronological partitions are too small to evaluate reliably.")

    # Build pre-match Elo snapshots once. This function deliberately groups by date,
    # so no result from a matchday can influence another fixture on that same date.
    elo_rows, _ = elo_history(df)
    elo_lookup = {}
    for row in elo_rows.to_dict("records"):
        key = (pd.Timestamp(row["Date"]).normalize(), str(row["HomeTeam"]), str(row["AwayTeam"]))
        elo_lookup[key] = row

    half_life = 240 if str(df.iloc[0].get("League", "")) == "SWE" else 300
    validation_model = dc_model_class(half_life=half_life, reg=0.04, xg_weight=0.0)
    validation_model.fit(train, train_end + pd.Timedelta(days=1))
    validation_scores = _score_block(validation, validation_model, elo_lookup)
    if validation_scores is None:
        raise ValueError("No validation fixtures could be scored; check team names and historical coverage.")

    # Select the ensemble weight on validation only, never on the final test block.
    weights = np.round(np.arange(0.0, 0.501, 0.025), 3)
    best_weight, best_loss = 0.0, float("inf")
    for weight in weights:
        blend = np.vstack([
            blended_probabilities(d, e, float(weight))
            for d, e in zip(validation_scores["dixon_coles"], validation_scores["elo"])
        ])
        y = validation_scores["outcomes"]
        loss = float(-np.mean(np.log(np.clip(blend[np.arange(len(y)), y], 1e-9, 1.0))))
        if loss < best_loss:
            best_weight, best_loss = float(weight), loss

    validation_blend = np.vstack([
        blended_probabilities(d, e, best_weight)
        for d, e in zip(validation_scores["dixon_coles"], validation_scores["elo"])
    ])
    temperature = _fit_temperature(validation_blend, validation_scores["outcomes"])
    dc_temperature = _fit_temperature(validation_scores["dixon_coles"], validation_scores["outcomes"])
    elo_temperature = _fit_temperature(validation_scores["elo"], validation_scores["outcomes"])

    # Refit the goal model with all information available before the final test period.
    test_model = dc_model_class(half_life=half_life, reg=0.04, xg_weight=0.0)
    test_model.fit(train_validation, validation_end + pd.Timedelta(days=1))
    test_scores = _score_block(test, test_model, elo_lookup, best_weight, temperature)
    if test_scores is None:
        raise ValueError("No final test fixtures could be scored; check team names and historical coverage.")

    dc_metrics = _metrics(test_scores["dixon_coles"], test_scores["outcomes"])
    elo_metrics = _metrics(test_scores["elo"], test_scores["outcomes"])
    raw_blend = np.vstack([
        blended_probabilities(d, e, best_weight)
        for d, e in zip(test_scores["dixon_coles"], test_scores["elo"])
    ])
    raw_metrics = _metrics(raw_blend, test_scores["outcomes"])
    calibrated_metrics = _metrics(test_scores["ensemble"], test_scores["outcomes"])
    calibrated_dc = np.vstack([_temperature(row, dc_temperature) for row in test_scores["dixon_coles"]])
    calibrated_elo = np.vstack([_temperature(row, elo_temperature) for row in test_scores["elo"]])
    calibrated_dc_metrics = _metrics(calibrated_dc, test_scores["outcomes"])
    calibrated_elo_metrics = _metrics(calibrated_elo, test_scores["outcomes"])

    return {
        "train_matches": int(len(train)),
        "validation_matches": int(len(validation_scores["outcomes"])),
        "test_matches": int(len(test_scores["outcomes"])),
        "cutoff": str(pd.Timestamp(test_dates[0]).date()),
        "validation_end": str(validation_end.date()),
        "elo_weight": best_weight,
        "temperature": temperature,
        "dc_temperature": dc_temperature,
        "elo_temperature": elo_temperature,
        "validation_log_loss": best_loss,
        "dixon_coles": dc_metrics,
        "elo": elo_metrics,
        "elo_ensemble": raw_metrics,
        "calibrated_dixon_coles": calibrated_dc_metrics,
        "calibrated_elo": calibrated_elo_metrics,
        "calibrated_ensemble": calibrated_metrics,
    }
