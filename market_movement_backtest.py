"""Measure opening-to-closing market movement and its predictive information.

Uses the additive no-vig conversion selected by the price-method experiment.
Compares opening, closing and a fixed 50/50 blend on identical fixtures.
Movement bins are descriptive only; no thresholds or blend weights are fitted.

Run: python market_movement_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("output/market_price_method_predictions.csv")
OUTPUT = Path("output")
METHOD = "additive"
OPEN_SOURCE = "Average opening"
CLOSE_SOURCE = "Average closing"


def score(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": len(y),
        "Brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run python market_price_method_backtest.py first."
        )
    raw = pd.read_csv(INPUT)
    raw = raw[raw.Method == METHOD].copy()
    key = ["Date", "League", "HomeTeam", "AwayTeam", "Season", "Actual"]
    fields = ["Home", "Draw", "Away", "Overround"]

    opening = raw[raw.Source == OPEN_SOURCE][key + fields].rename(
        columns={c: f"{c}_open" for c in fields}
    )
    closing = raw[raw.Source == CLOSE_SOURCE][key + fields].rename(
        columns={c: f"{c}_close" for c in fields}
    )
    data = opening.merge(closing, on=key, how="inner", validate="one_to_one")
    if data.empty:
        raise RuntimeError("No fixtures have both average opening and closing prices.")

    open_p = data[["Home_open", "Draw_open", "Away_open"]].to_numpy(float)
    close_p = data[["Home_close", "Draw_close", "Away_close"]].to_numpy(float)
    y = data.Actual.to_numpy(int)
    blend_p = 0.5 * open_p + 0.5 * close_p
    movement = close_p - open_p
    data["Movement_L1"] = np.abs(movement).sum(axis=1)
    data["Movement_max"] = np.abs(movement).max(axis=1)
    data["Home_move_pp"] = movement[:, 0] * 100
    data["Draw_move_pp"] = movement[:, 1] * 100
    data["Away_move_pp"] = movement[:, 2] * 100
    data["Closing_overround_change_pp"] = (
        data.Overround_close - data.Overround_open
    ) * 100

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / "market_movement_fixture_data.csv", index=False)

    print(f"Matched opening/closing fixtures: N={len(data)}")
    print("\n=== OVERALL: IDENTICAL FIXTURES ===")
    summary = []
    for name, p in [
        ("Average opening", open_p),
        ("50/50 opening-closing blend", blend_p),
        ("Average closing", close_p),
    ]:
        result = score(p, y)
        summary.append({"Model": name, **result})
        print(
            f"{name:30s} N={result['N']:5d} "
            f"Brier={result['Brier']:.4f} LogLoss={result['LogLoss']:.4f} "
            f"RPS={result['RPS']:.4f} Acc={result['Accuracy']:.4f}"
        )

    print("\n=== MOVEMENT SIZE ===")
    print(
        f"Mean total absolute probability movement (L1): "
        f"{data.Movement_L1.mean() * 100:.2f} percentage points"
    )
    print(
        f"Median total absolute movement: "
        f"{data.Movement_L1.median() * 100:.2f} percentage points"
    )
    print(
        f"Mean maximum single-outcome movement: "
        f"{data.Movement_max.mean() * 100:.2f} percentage points"
    )
    print(
        f"Mean overround change: "
        f"{data.Closing_overround_change_pp.mean():+.3f} percentage points"
    )

    # Rank-based bins avoid arbitrary thresholds and keep bin sizes similar.
    data["Movement_decile"] = pd.qcut(
        data.Movement_L1.rank(method="first"), 10, labels=False
    ) + 1
    print("\n=== DOES BIGGER MOVEMENT RELATE TO CLOSING-MARKET IMPROVEMENT? ===")
    movement_rows = []
    for decile, group in data.groupby("Movement_decile", sort=True):
        idx = group.index.to_numpy()
        loc = data.index.get_indexer(idx)
        op = score(open_p[loc], group.Actual.to_numpy())
        cl = score(close_p[loc], group.Actual.to_numpy())
        row = {
            "MovementDecile": int(decile),
            "N": len(group),
            "MeanMovementL1_pp": group.Movement_L1.mean() * 100,
            "OpeningBrier": op["Brier"],
            "ClosingBrier": cl["Brier"],
            "BrierImprovement": op["Brier"] - cl["Brier"],
            "OpeningLogLoss": op["LogLoss"],
            "ClosingLogLoss": cl["LogLoss"],
            "LogLossImprovement": op["LogLoss"] - cl["LogLoss"],
        }
        movement_rows.append(row)
        print(
            f"D{int(decile):02d} N={len(group):4d} "
            f"mean movement={row['MeanMovementL1_pp']:.2f}pp "
            f"Brier open/close={op['Brier']:.4f}/{cl['Brier']:.4f} "
            f"gain={row['BrierImprovement']:+.4f} "
            f"LogLoss gain={row['LogLossImprovement']:+.4f}"
        )

    print("\n=== BY LEAGUE: OPENING VS CLOSING ===")
    league_rows = []
    for league, group in data.groupby("League", sort=True):
        loc = data.index.get_indexer(group.index)
        op = score(open_p[loc], group.Actual.to_numpy())
        cl = score(close_p[loc], group.Actual.to_numpy())
        league_rows.append({
            "League": league, "N": len(group),
            "OpeningBrier": op["Brier"], "ClosingBrier": cl["Brier"],
            "BrierImprovement": op["Brier"] - cl["Brier"],
            "OpeningLogLoss": op["LogLoss"], "ClosingLogLoss": cl["LogLoss"],
            "LogLossImprovement": op["LogLoss"] - cl["LogLoss"],
            "MeanMovementL1_pp": group.Movement_L1.mean() * 100,
        })
        print(
            f"{league:5s} N={len(group):4d} "
            f"Brier gain={op['Brier']-cl['Brier']:+.4f} "
            f"LogLoss gain={op['LogLoss']-cl['LogLoss']:+.4f} "
            f"mean movement={group.Movement_L1.mean()*100:.2f}pp"
        )

    print("\n=== BY SEASON: OPENING VS CLOSING ===")
    season_rows = []
    for season, group in data.groupby("Season", sort=True):
        loc = data.index.get_indexer(group.index)
        op = score(open_p[loc], group.Actual.to_numpy())
        cl = score(close_p[loc], group.Actual.to_numpy())
        season_rows.append({
            "Season": season, "N": len(group),
            "OpeningBrier": op["Brier"], "ClosingBrier": cl["Brier"],
            "BrierImprovement": op["Brier"] - cl["Brier"],
            "OpeningLogLoss": op["LogLoss"], "ClosingLogLoss": cl["LogLoss"],
            "LogLossImprovement": op["LogLoss"] - cl["LogLoss"],
        })
        print(
            f"{season:6s} N={len(group):4d} "
            f"Brier gain={op['Brier']-cl['Brier']:+.4f} "
            f"LogLoss gain={op['LogLoss']-cl['LogLoss']:+.4f}"
        )

    pd.DataFrame(summary).to_csv(OUTPUT / "market_movement_summary.csv", index=False)
    pd.DataFrame(movement_rows).to_csv(OUTPUT / "market_movement_deciles.csv", index=False)
    pd.DataFrame(league_rows).to_csv(OUTPUT / "market_movement_leagues.csv", index=False)
    pd.DataFrame(season_rows).to_csv(OUTPUT / "market_movement_seasons.csv", index=False)
    print("\nSaved market movement outputs under output/.")


if __name__ == "__main__":
    main()
