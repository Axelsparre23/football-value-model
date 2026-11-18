"""Leakage-safe evaluation helpers for the prospective forecast log.

This module only scores recorded pre-match probabilities against later outcomes.
It never modifies the source CSV or refits the live prediction model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUTCOMES = {"H": 0, "HOME": 0, "HOME_WIN": 0, "D": 1, "DRAW": 1,
            "A": 2, "AWAY": 2, "AWAY_WIN": 2}
MODEL_COLUMNS = {
    "Elo": ("elo_home_pct", "elo_draw_pct", "elo_away_pct"),
    "Dixon-Coles": ("dc_home_pct", "dc_draw_pct", "dc_away_pct"),
}
MARKET_OPEN = ("market_home_novig_pct", "market_draw_novig_pct", "market_away_novig_pct")
MARKET_CLOSE = ("closing_market_home_novig_pct", "closing_market_draw_novig_pct",
                "closing_market_away_novig_pct")


def _known_single_bookmaker(value):
    """Reject synthetic best-price aggregates and unidentified operators."""
    source = str(value or "").strip().casefold()
    if not source or source in {"nan", "none", "unknown"}:
        return False
    rejected = ("aggregat", "best price", "best odds", "source/operator not recorded",
                "operator not recorded", "not recorded")
    return not any(token in source for token in rejected)


def _has_provenance(source, timestamp):
    source_text = str(source or "").strip().casefold()
    time_text = str(timestamp or "").strip().casefold()
    return (
        bool(source_text) and source_text not in {"nan", "none", "unknown"}
        and bool(time_text) and time_text not in {"nan", "none", "unknown"}
    )


def load_log(path: str | Path) -> pd.DataFrame:
    """Load the immutable forecast log; caller controls when/if outcomes are appended."""
    frame = pd.read_csv(path)
    required = {"match_id", "kickoff_local", "home_team", "away_team"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast log is missing required columns: {sorted(missing)}")
    return frame


def _actual(row):
    value = str(row.get("actual_outcome", "")).strip().upper()
    if value in OUTCOMES:
        return OUTCOMES[value]
    hg, ag = row.get("final_home_goals"), row.get("final_away_goals")
    if pd.notna(hg) and pd.notna(ag):
        return 0 if float(hg) > float(ag) else (1 if float(hg) == float(ag) else 2)
    return None


def _probabilities(row, columns):
    values = [pd.to_numeric(row.get(c), errors="coerce") for c in columns]
    if not all(pd.notna(v) and np.isfinite(float(v)) and 0 <= float(v) <= 100 for v in values):
        return None
    p = np.asarray(values, dtype=float) / 100.0
    total = p.sum()
    if total <= 0 or abs(total - 1.0) > 0.025:
        return None
    return p / total


def scored_rows(log: pd.DataFrame) -> pd.DataFrame:
    """Return completed fixtures with validated pre-match model and market probabilities."""
    rows = []
    for _, row in log.iterrows():
        actual = _actual(row)
        if actual is None:
            continue
        item = row.to_dict()
        item["actual_index"] = int(actual)
        for name, columns in MODEL_COLUMNS.items():
            item[f"{name}_p"] = _probabilities(row, columns)
        close = _probabilities(row, MARKET_CLOSE)
        opened = _probabilities(row, MARKET_OPEN)
        close_operator = row.get("closing_bookmaker_name", "")
        open_operator = row.get("bookmaker_name", "")
        close_valid = (
            close is not None
            and _known_single_bookmaker(close_operator)
            and _has_provenance(row.get("closing_odds_source", ""),
                                row.get("closing_odds_timestamp_utc", ""))
        )
        open_valid = (
            opened is not None
            and _known_single_bookmaker(open_operator)
            and _has_provenance(row.get("odds_source", ""),
                                row.get("odds_timestamp_utc", ""))
        )
        item["Recorded_market_p"] = close if close is not None else opened
        if close_valid:
            item["Market_p"] = close
            item["market_snapshot"] = "closing"
            item["market_source_status"] = "verified single operator name recorded"
        elif open_valid:
            item["Market_p"] = opened
            item["market_snapshot"] = "pre-match"
            item["market_source_status"] = "single operator name recorded; not closing"
        else:
            item["Market_p"] = None
            item["market_snapshot"] = "unverified / aggregated / unavailable"
            item["market_source_status"] = "excluded from strict market comparison"
        if any(item[f"{name}_p"] is not None for name in MODEL_COLUMNS) or item["Market_p"] is not None or item["Recorded_market_p"] is not None:
            rows.append(item)
    return pd.DataFrame(rows)


def score_probabilities(probabilities, outcomes) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(y) or len(y) == 0:
        raise ValueError("Expected a non-empty N×3 probability matrix and N outcomes.")
    p = np.clip(p, 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "matches": int(len(y)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "rps": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2)),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
    }


def metric_table(scored: pd.DataFrame) -> pd.DataFrame:
    result = []
    for name in ("Elo", "Dixon-Coles", "Market"):
        subset = scored[scored[f"{name}_p"].notna()] if f"{name}_p" in scored else scored.iloc[0:0]
        if subset.empty:
            continue
        metrics = score_probabilities(np.vstack(subset[f"{name}_p"]), subset.actual_index)
        result.append({"Model": name, **metrics})
    return pd.DataFrame(result)


def paired_brier_bootstrap(scored: pd.DataFrame, first="Dixon-Coles", second="Market",
                           iterations=5000, seed=20261118) -> dict | None:
    """Bootstrap paired per-match Brier differences; positive means second is better."""
    common = scored[scored[f"{first}_p"].notna() & scored[f"{second}_p"].notna()]
    if len(common) < 2:
        return None
    y = common.actual_index.to_numpy(dtype=int)
    onehot = np.eye(3)[y]
    p1 = np.vstack(common[f"{first}_p"])
    p2 = np.vstack(common[f"{second}_p"])
    d1 = np.sum((p1 - onehot) ** 2, axis=1)
    d2 = np.sum((p2 - onehot) ** 2, axis=1)
    differences = d1 - d2
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(iterations, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"matches": int(len(common)), "mean_difference": float(differences.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "interpretation": f"{second} lower Brier" if differences.mean() > 0 else f"{first} lower Brier"}


def calibration_table(scored: pd.DataFrame, model="Elo", bins=10) -> pd.DataFrame:
    subset = scored[scored[f"{model}_p"].notna()]
    if subset.empty:
        return pd.DataFrame(columns=["Outcome", "Bin", "Mean predicted", "Observed frequency", "Count"])
    p = np.vstack(subset[f"{model}_p"])
    y = subset.actual_index.to_numpy(dtype=int)
    labels = ("Home win", "Draw", "Away win")
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for outcome, label in enumerate(labels):
        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            mask = (p[:, outcome] >= lower) & (
                (p[:, outcome] < upper) if index < bins - 1 else (p[:, outcome] <= upper)
            )
            if mask.any():
                rows.append({"Outcome": label, "Bin": f"{lower:.1f}–{upper:.1f}",
                             "Mean predicted": float(p[mask, outcome].mean()),
                             "Observed frequency": float((y[mask] == outcome).mean()),
                             "Count": int(mask.sum())})
    return pd.DataFrame(rows)
