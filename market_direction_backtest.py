"""Test whether the direction of market probability movement is informative.

Requires output/market_movement_fixture_data.csv from market_movement_backtest.py.
For each 1X2 outcome, groups matches by signed opening-to-closing probability
change, then compares opening/closing probability with the realised frequency.
Also reports binary Brier score by movement direction and league.

Run: python market_direction_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("output/market_movement_fixture_data.csv")
OUTPUT = Path("output")
OUTCOMES = {
    "Home": ("Home_move_pp", "Home_open", "Home_close", 0),
    "Draw": ("Draw_move_pp", "Draw_open", "Draw_close", 1),
    "Away": ("Away_move_pp", "Away_open", "Away_close", 2),
}


def binary_brier(prob, actual):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((prob - actual) ** 2))


def evaluate_group(group, move_col, open_col, close_col, outcome_code):
    actual = (group.Actual.to_numpy(int) == outcome_code).astype(int)
    p_open = group[open_col].to_numpy(float)
    p_close = group[close_col].to_numpy(float)
    return {
        "N": len(group),
        "MeanMove_pp": group[move_col].mean(),
        "MeanOpenProb": p_open.mean(),
        "MeanCloseProb": p_close.mean(),
        "ActualRate": actual.mean(),
        "OpenBrier": binary_brier(p_open, actual),
        "CloseBrier": binary_brier(p_close, actual),
        "BrierImprovement": binary_brier(p_open, actual) - binary_brier(p_close, actual),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run python market_movement_backtest.py first."
        )
    data = pd.read_csv(INPUT)
    required = {"Actual", "League", "Season"}
    for move_col, open_col, close_col, _ in OUTCOMES.values():
        required.update({move_col, open_col, close_col})
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")

    print(f"Fixtures: N={len(data)}")
    summary_rows = []
    decile_rows = []
    league_rows = []
    season_rows = []

    for outcome, (move_col, open_col, close_col, code) in OUTCOMES.items():
        print(f"\n=== {outcome.upper()} PROBABILITY MOVEMENT ===")
        actual = (data.Actual.to_numpy(int) == code).astype(int)
        overall_open = binary_brier(data[open_col], actual)
        overall_close = binary_brier(data[close_col], actual)
        print(
            f"Overall binary Brier: opening={overall_open:.5f} "
            f"closing={overall_close:.5f} "
            f"improvement={overall_open-overall_close:+.5f}"
        )

        # Five signed movement groups: strongest decreases through strongest increases.
        # Quantile groups are outcome-specific and have approximately equal sample sizes.
        ranked = data[move_col].rank(method="first")
        data[f"{outcome}_movement_group"] = pd.qcut(
            ranked, 5, labels=[
                "Largest decrease", "Moderate decrease", "Near zero",
                "Moderate increase", "Largest increase"
            ]
        )
        print("Movement group            N   Mean change   Open prob   Close prob   Actual rate   Brier gain")
        for group_name, group in data.groupby(f"{outcome}_movement_group", observed=False, sort=False):
            result = evaluate_group(group, move_col, open_col, close_col, code)
            row = {"Outcome": outcome, "MovementGroup": str(group_name), **result}
            decile_rows.append(row)
            print(
                f"{str(group_name):24s} {result['N']:4d} "
                f"{result['MeanMove_pp']:+8.2f}pp "
                f"{result['MeanOpenProb']*100:8.2f}% "
                f"{result['MeanCloseProb']*100:9.2f}% "
                f"{result['ActualRate']*100:10.2f}% "
                f"{result['BrierImprovement']:+.5f}"
            )

        # Check whether observed outcome frequency increases with the closing probability.
        corr = float(np.corrcoef(data[close_col], actual)[0, 1])
        mean_move = float(data[move_col].mean())
        summary_rows.append({
            "Outcome": outcome, "N": len(data),
            "OpeningBinaryBrier": overall_open,
            "ClosingBinaryBrier": overall_close,
            "BrierImprovement": overall_open - overall_close,
            "CorrelationClosingProbabilityActual": corr,
            "MeanMovement_pp": mean_move,
        })

        for league, group in data.groupby("League", sort=True):
            result = evaluate_group(group, move_col, open_col, close_col, code)
            league_rows.append({"Outcome": outcome, "League": league, **result})
        for season, group in data.groupby("Season", sort=True):
            result = evaluate_group(group, move_col, open_col, close_col, code)
            season_rows.append({"Outcome": outcome, "Season": season, **result})

    print("\n=== BY LEAGUE: BINARY BRIER IMPROVEMENT FROM OPEN TO CLOSE ===")
    league_df = pd.DataFrame(league_rows)
    for outcome in OUTCOMES:
        part = league_df[league_df.Outcome == outcome]
        print(outcome + ": " + " | ".join(
            f"{r.League} {r.BrierImprovement:+.5f}" for r in part.itertuples()
        ))

    print("\n=== BY SEASON: BINARY BRIER IMPROVEMENT FROM OPEN TO CLOSE ===")
    season_df = pd.DataFrame(season_rows)
    for outcome in OUTCOMES:
        part = season_df[season_df.Outcome == outcome]
        print(outcome + ": " + " | ".join(
            f"{r.Season} {r.BrierImprovement:+.5f}" for r in part.itertuples()
        ))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "market_direction_summary.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(OUTPUT / "market_direction_groups.csv", index=False)
    league_df.to_csv(OUTPUT / "market_direction_leagues.csv", index=False)
    season_df.to_csv(OUTPUT / "market_direction_seasons.csv", index=False)
    print("\nSaved market_direction_*.csv under output/.")


if __name__ == "__main__":
    main()
