"""Multi-source football data loader.

Historical top-five results: Football-Data.co.uk.
Current-season results/fixtures: open-matchday GitHub dataset.
The open-matchday provider documents uneven match-stat coverage; missing values stay NaN.
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import requests

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "SWE": "Allsvenskan",
}
OPEN_MATCHDAY = {
    "E0": ("ENG_1_premier-league", "2627"),
    "SP1": ("ESP_1_la-liga", "2627"),
    "D1": ("GER_1_bundesliga", "2627"),
    "I1": ("ITA_1_serie-a", "2627"),
    "F1": ("FRA_1_ligue-1", "2627"),
    "SWE": ("SWE_1_allsvenskan", "2026"),
}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
OPEN_URL = "https://raw.githubusercontent.com/oladeeayo/open-matchday/master/data/{folder}/results_{season}.csv"
USER_AGENT = "FootballValueModel/2.0 (personal research project)"

def season_codes(first_start_year: int = 2016, current_start_year: int | None = None) -> list[str]:
    now = pd.Timestamp.utcnow()
    current_start_year = current_start_year or (now.year if now.month >= 7 else now.year - 1)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first_start_year, current_start_year + 1)]

def _normalise_open_matchday(raw: pd.DataFrame, league_code: str) -> pd.DataFrame:
    mapping = {
        "date": "Date", "home": "HomeTeam", "away": "AwayTeam",
        "ft_home": "FTHG", "ft_away": "FTAG",
        "home_shots": "HS", "away_shots": "AS",
        "home_sot": "HST", "away_sot": "AST",
        "home_corners": "HC", "away_corners": "AC",
        "home_fouls": "HF", "away_fouls": "AF",
        "home_yellow": "HY", "away_yellow": "AY",
        "home_red": "HR", "away_red": "AR",
    }
    out = raw.rename(columns=mapping).copy()
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not set(required).issubset(out.columns):
        raise ValueError("Open-matchday file is missing required result columns.")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = float("nan")
    out["League"] = league_code
    return out.dropna(subset=required).copy()

def download_data(cache_dir: Path = Path("data/cache"), first_start_year: int = 2016,
                  leagues: tuple[str, ...] = tuple(LEAGUES), refresh_current: bool = True) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    frames, failures = [], []

    # Historical results and odds for the five European leagues.
    top_five = [code for code in leagues if code in ("E0", "SP1", "D1", "I1", "F1")]
    if top_five:
        seasons = season_codes(first_start_year)
        current = seasons[-1]
        for season in seasons:
            for league in top_five:
                path = cache_dir / f"{season}_{league}.csv"
                if not path.exists() or (refresh_current and season == current):
                    try:
                        response = session.get(BASE_URL.format(season=season, league=league), timeout=20)
                        if response.status_code == 404:
                            failures.append(f"Football-Data {season}/{league}: not published")
                            continue
                        response.raise_for_status()
                        if response.content.strip():
                            path.write_bytes(response.content)
                    except requests.RequestException as exc:
                        failures.append(f"Football-Data {season}/{league}: {exc}")
                        if not path.exists():
                            continue
                    time.sleep(0.12)
                if path.exists():
                    try:
                        raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
                        if {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(raw.columns):
                            raw["League"] = league
                            frames.append(raw)
                    except Exception as exc:
                        failures.append(f"{path.name}: {exc}")

    # Current-season results/fixtures for all six leagues, including Allsvenskan.
    for league in leagues:
        folder_season = OPEN_MATCHDAY.get(league)
        if not folder_season:
            continue
        folder, season = folder_season
        path = cache_dir / f"open_matchday_{league}_{season}.csv"
        if not path.exists() or refresh_current:
            try:
                response = session.get(OPEN_URL.format(folder=folder, season=season), timeout=25)
                response.raise_for_status()
                path.write_bytes(response.content)
            except requests.RequestException as exc:
                failures.append(f"open-matchday {league}: {exc}")
        if path.exists():
            try:
                raw = pd.read_csv(path)
                frames.append(_normalise_open_matchday(raw, league))
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")

    if not frames:
        raise RuntimeError("No match data loaded. Check internet access and provider availability.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data[(data.FTHG >= 0) & (data.FTAG >= 0)]
    data = data.sort_values(["Date", "League"]).drop_duplicates(
        ["Date", "League", "HomeTeam", "AwayTeam"], keep="last"
    ).reset_index(drop=True)
    for col in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
        if col not in data:
            data[col] = float("nan")
    data["HomeXG"] = float("nan")
    data["AwayXG"] = float("nan")
    data.attrs["download_failures"] = failures
    return data
