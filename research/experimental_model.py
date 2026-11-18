"""Experimental market-anchored multinomial residual model.

This model is deliberately separate from app.py and the prospective forecast
cohort. It starts from no-vig market probabilities and learns a regularised
correction from pre-match features. It is only a research candidate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def _validate_inputs(features, market_probabilities, outcomes=None):
    x = np.asarray(features, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    if x.ndim != 2 or market.ndim != 2 or market.shape != (len(x), 3):
        raise ValueError("Expected an N×F feature matrix and an N×3 market probability matrix.")
    if len(x) == 0 or not np.all(np.isfinite(market)):
        raise ValueError("Training/prediction inputs must contain rows and finite market probabilities.")
    if np.any(market <= 0):
        raise ValueError("Market probabilities must be strictly positive.")
    market = market / market.sum(axis=1, keepdims=True)
    y = None
    if outcomes is not None:
        y = np.asarray(outcomes, dtype=int)
        if y.shape != (len(x),) or np.any((y < 0) | (y > 2)):
            raise ValueError("Outcomes must be an N-length array coded 0=home, 1=draw, 2=away.")
    return x, market, y


def fit_residual_model(features, market_probabilities, outcomes, ridge=0.1):
    """Fit logits = log(no-vig market probabilities) + learned feature correction."""
    x, market, y = _validate_inputs(features, market_probabilities, outcomes)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite non-negative number.")

    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    means = np.divide(np.where(finite, x, 0.0).sum(axis=0), counts,
                      out=np.zeros(x.shape[1], dtype=float), where=counts > 0)
    x = np.where(finite, x, means)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    z = np.clip((x - means) / scales, -6.0, 6.0)

    offset = np.log(market)
    offset -= offset.mean(axis=1, keepdims=True)
    design = np.column_stack([np.ones(len(z)), z])
    n_features = design.shape[1]
    onehot = np.eye(3)[y]

    def objective(flat):
        weights = flat.reshape(3, n_features)
        logits = offset + design @ weights.T
        log_norm = np.logaddexp.reduce(logits, axis=1)
        loss = np.mean(log_norm - logits[np.arange(len(y)), y])
        penalty = ridge * np.sum(weights[:, 1:] ** 2) / 2.0
        probabilities = softmax(logits, axis=1)
        gradient = ((probabilities - onehot).T @ design) / len(y)
        gradient[:, 1:] += ridge * weights[:, 1:]
        return float(loss + penalty), gradient.ravel()

    result = minimize(
        objective, np.zeros(3 * n_features), jac=True, method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Experimental residual-model fit failed: {result.message}")
    return {
        "weights": result.x.reshape(3, n_features),
        "means": means,
        "scales": scales,
        "ridge": float(ridge),
        "training_matches": int(len(y)),
        "feature_count": int(x.shape[1]),
    }


def predict_residual_model(model, features, market_probabilities):
    """Predict from the market prior plus a learned correction; never refits."""
    x, market, _ = _validate_inputs(features, market_probabilities)
    if x.shape[1] != model["feature_count"]:
        raise ValueError("Prediction feature count differs from the fitted model.")
    finite = np.isfinite(x)
    x = np.where(finite, x, model["means"])
    z = np.clip((x - model["means"]) / model["scales"], -6.0, 6.0)
    design = np.column_stack([np.ones(len(z)), z])
    offset = np.log(market)
    offset -= offset.mean(axis=1, keepdims=True)
    return softmax(offset + design @ model["weights"].T, axis=1)
