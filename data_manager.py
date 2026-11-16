"""Automatic historical data downloader for Football-Data.co.uk."""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import requests

LEAGUES = {"E0":"Premier League", "SP1":"La Liga", "D1":"Bundesliga", "I1":"Serie A", "F1":"Ligue 1"}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
USER_AGENT = "FootballValueModel/1.0 (historical research; GitHub project)"

def season_codes(first_start_year: int = 2016, current_start_year: int | None = None) -> list[str]:
    now = pd.Timestamp.utcnow()
    current_start_year = current_start_year or (now.year if now.month >= 7 else now.year - 1)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first_start_year, current_start_year + 1)]

def download_data(cache_dir: Path = Path("data/cache"), first_start_year: int = 2016,
                  leagues: tuple[str, ...] = tuple(LEAGUES), refresh_current: bool = True) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seasons = season_codes(first_start_year)
    current = seasons[-1]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    frames, failures = [], []
    for season in seasons:
        for league in leagues:
            path = cache_dir / f"{season}_{league}.csv"
            if not path.exists() or (refresh_current and season == current):
                try:
                    response = session.get(BASE_URL.format(season=season, league=league), timeout=25)
                    if response.status_code == 404:
                        failures.append(f"{season}/{league}: not published")
                        continue
                    response.raise_for_status()
                    if response.content.strip():
                        path.write_bytes(response.content)
                    else:
                        continue
                except requests.RequestException as exc:
                    failures.append(f"{season}/{league}: {exc}")
                    if not path.exists():
                        continue
                time.sleep(0.15)
            try:
                raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
                if not {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(raw.columns):
                    failures.append(f"{path.name}: missing required columns")
                    continue
                raw["League"] = league
                frames.append(raw)
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")
    if not frames:
        raise RuntimeError("No historical data loaded. Check internet access and provider availability.")
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    for col in ("FTHG","FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
    data = data[(data.FTHG >= 0) & (data.FTAG >= 0)]
    data = data.sort_values(["Date","League"]).drop_duplicates(
        ["Date","League","HomeTeam","AwayTeam"], keep="last").reset_index(drop=True)
    data["HomeXG"] = float("nan")
    data["AwayXG"] = float("nan")
    data.attrs["download_failures"] = failures
    data.attrs["seasons"] = seasons
    return data
