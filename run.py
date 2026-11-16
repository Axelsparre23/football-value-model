"""One-command historical model run. Simulated/paper evaluation only."""
import argparse
from pathlib import Path
from data_manager import download_data
from football_value_model_v3 import run_backtest, report

def main():
    parser = argparse.ArgumentParser(description="Football Value Model — historical paper backtest")
    parser.add_argument("--first-season", type=int, default=2016)
    parser.add_argument("--min-train", type=int, default=300)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--refit-matchdays", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    print("Downloading/caching historical results and odds...")
    data = download_data(first_start_year=args.first_season)
    print(f"Loaded {len(data):,} matches across {data.League.nunique()} leagues.")
    predictions = run_backtest(data, min_train=args.min_train,
        refit_matchdays=args.refit_matchdays, min_edge=args.min_edge)
    report(predictions, args.output)

if __name__ == "__main__":
    main()
