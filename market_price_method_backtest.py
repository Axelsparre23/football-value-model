"""Compare bookmaker price sources and overround-removal methods.

Uses identical fixtures within each comparison. No fitted team model and no
future information: this is a market-probability construction experiment.

Run: python market_price_method_backtest.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from data_manager import download_data, LEAGUES

OUTPUT = Path("output")
YEARS = range(2021, 2026)
SOURCES = {
    "Average closing": ("AvgCH", "AvgCD", "AvgCA"),
    "Bet365 closing": ("B365CH", "B365CD", "B365CA"),
    "Average opening": ("AvgH", "AvgD", "AvgA"),
    "Bet365 opening": ("B365H", "B365D", "B365A"),
}


def valid_odds(row, cols):
    if not all(c in row.index for c in cols):
        return None
    vals = pd.to_numeric(row[list(cols)], errors="coerce").to_numpy(dtype=float)
    if len(vals) != 3 or not np.all(np.isfinite(vals)) or np.any(vals <= 1.01) or np.any(vals > 100):
        return None
    return vals


def remove_margin(odds, method):
    q = 1.0 / np.asarray(odds, dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q <= 0):
        return None
    overround = float(q.sum())
    if method == "proportional":
        p = q / overround
    elif method == "additive":
        p = np.maximum(q - (overround - 1.0) / 3.0, 1e-9)
        p /= p.sum()
    elif method == "power":
        def total_error(power):
            return float(np.sum(q ** power) - 1.0)
        try:
            power = brentq(total_error, 1e-4, 20.0)
            p = q ** power
            p /= p.sum()
        except (ValueError, RuntimeError, FloatingPointError):
            return None
    elif method == "shin":
        # Standard Shin three-outcome transformation.
        def probs(z):
            z = min(max(float(z), 0.0), 1.0 - 1e-10)
            return (np.sqrt(z*z + 4.0*(1.0-z)*q*q/overround) - z) / (2.0*(1.0-z))
        try:
            z = brentq(lambda z: float(probs(z).sum() - 1.0), 0.0, 1.0-1e-9)
            p = probs(z)
            p /= p.sum()
        except (ValueError, ZeroDivisionError, FloatingPointError):
            p = q / overround
    else:
        raise ValueError(method)
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        return None
    return p / p.sum()


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    onehot = np.eye(3)[y]
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_y = np.cumsum(onehot, axis=1)[:, :2]
    return {
        "N": int(len(y)),
        "Brier": float(np.mean(np.sum((p-onehot)**2, axis=1))),
        "LogLoss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "RPS": float(np.mean(np.sum((cumulative_p-cumulative_y)**2, axis=1)/2)),
        "Accuracy": float(np.mean(p.argmax(axis=1) == y)),
    }


def season_bounds(code, year):
    if code == "SWE":
        return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59:59")
    return pd.Timestamp(f"{year}-08-01"), pd.Timestamp(f"{year+1}-06-30 23:59:59")


def main():
    print("Loading historical results and bookmaker prices...", flush=True)
    data = download_data(Path("data/cache"), first_start_year=2010,
                         leagues=tuple(LEAGUES), refresh_current=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    data["Actual"] = np.where(data.FTHG > data.FTAG, 0,
                              np.where(data.FTHG == data.FTAG, 1, 2))
    rows = []
    methods = ["proportional", "power", "additive", "shin"]

    for code, league_name in LEAGUES.items():
        league = data[data.League == code]
        for year in YEARS:
            start, end = season_bounds(code, year)
            season = league[(league.Date >= start) & (league.Date <= end)]
            if season.empty:
                continue
            for _, match in season.iterrows():
                for source, cols in SOURCES.items():
                    odds = valid_odds(match, cols)
                    if odds is None:
                        continue
                    for method in methods:
                        p = remove_margin(odds, method)
                        if p is None:
                            continue
                        rows.append({
                            "Date": pd.Timestamp(match.Date).date().isoformat(),
                            "League": code, "LeagueName": league_name, "Season": str(year),
                            "HomeTeam": str(match.HomeTeam), "AwayTeam": str(match.AwayTeam),
                            "Actual": int(match.Actual), "Source": source, "Method": method,
                            "Home": float(p[0]), "Draw": float(p[1]), "Away": float(p[2]),
                            "Overround": float(np.sum(1.0/odds)-1.0),
                        })

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("No bookmaker odds were found in the selected seasons.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUTPUT / "market_price_method_predictions.csv", index=False)

    print("\n=== SOURCE COVERAGE ===", flush=True)
    for source, group in pred[pred.Method == "shin"].groupby("Source"):
        print(f"{source:20s} N={group[['Date','League','HomeTeam','AwayTeam']].drop_duplicates().shape[0]}", flush=True)

    print("\n=== AGGREGATE RESULTS BY PRICE SOURCE AND METHOD ===", flush=True)
    summary = []
    for (source, method), group in pred.groupby(["Source", "Method"], sort=False):
        score = metrics(group[["Home","Draw","Away"]].to_numpy(), group.Actual.to_numpy())
        summary.append({"Source":source,"Method":method,**score})
        print(f"{source:20s} {method:13s} N={score['N']:5d} Brier={score['Brier']:.4f} LogLoss={score['LogLoss']:.4f} RPS={score['RPS']:.4f} Acc={score['Accuracy']:.4f}", flush=True)

    print("\n=== FAIR METHOD COMPARISON: FIXTURES WITH ALL FOUR PRICE SOURCES ===", flush=True)
    key = ["Date","League","HomeTeam","AwayTeam"]
    source_presence = pred[pred.Method == "shin"].groupby(key).Source.nunique()
    common_keys = source_presence[source_presence == len(SOURCES)].reset_index()[key]
    common = pred.merge(common_keys, on=key, how="inner")
    for source in SOURCES:
        for method in methods:
            group = common[(common.Source == source) & (common.Method == method)]
            if group.empty:
                continue
            score = metrics(group[["Home","Draw","Away"]].to_numpy(), group.Actual.to_numpy())
            print(f"{source:20s} {method:13s} N={score['N']:5d} Brier={score['Brier']:.4f} LogLoss={score['LogLoss']:.4f} RPS={score['RPS']:.4f}", flush=True)

    print("\n=== BY LEAGUE: AVERAGE CLOSING ODDS ===", flush=True)
    close = pred[pred.Source == "Average closing"]
    for league, group in close.groupby("LeagueName", sort=True):
        for method in methods:
            part = group[group.Method == method]
            if part.empty:
                continue
            score = metrics(part[["Home","Draw","Away"]].to_numpy(), part.Actual.to_numpy())
            print(f"{league:16s} {method:13s} N={score['N']:5d} Brier={score['Brier']:.4f} LogLoss={score['LogLoss']:.4f}", flush=True)

    pd.DataFrame(summary).to_csv(OUTPUT / "market_price_method_summary.csv", index=False)
    print("\nSaved output/market_price_method_predictions.csv")
    print("Saved output/market_price_method_summary.csv")


if __name__ == "__main__":
    main()
