"""Optional API-Football match intelligence.

This module retrieves real fixture-level information when API_FOOTBALL_KEY is
configured. Missing or unsupported provider data remains unavailable; it is
never replaced with estimates or fabricated values.

The functions are deliberately separate from the Elo forecast. Do not feed
these fields into production probabilities until they pass chronological,
out-of-sample validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import requests

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = {
    "E0": 39,   # Premier League
    "SP1": 140, # La Liga
    "D1": 78,   # Bundesliga
    "I1": 135,  # Serie A
    "F1": 61,   # Ligue 1
    "SWE": 113, # Allsvenskan
}
REQUEST_TIMEOUT = 15


class APIFootballError(RuntimeError):
    """Raised when the provider request fails or returns an API error."""


class APIFootballClient:
    def __init__(self, api_key: str | None = None, timeout: int = REQUEST_TIMEOUT):
        self.api_key = (api_key or os.getenv("API_FOOTBALL_KEY", "")).strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": self.api_key,
            "User-Agent": "FootballValueModel/2.1",
        })

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            raise APIFootballError("API_FOOTBALL_KEY is not configured.")
        response = self.session.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors")
        if errors:
            raise APIFootballError(f"API-Football error: {errors}")
        return payload

    def find_fixture(
        self, league_code: str, season: int, match_date: str,
        home_team: str, away_team: str,
    ) -> dict[str, Any] | None:
        """Find the selected fixture on the provider's local date."""
        if league_code not in LEAGUE_IDS:
            raise ValueError(f"Unsupported league code: {league_code}")
        payload = self._get("fixtures", {
            "league": LEAGUE_IDS[league_code],
            "season": int(season),
            "date": match_date,
        })
        home = _normalise_name(home_team)
        away = _normalise_name(away_team)
        for item in payload.get("response", []):
            teams = item.get("teams", {})
            if (
                _normalise_name(teams.get("home", {}).get("name", "")) == home
                and _normalise_name(teams.get("away", {}).get("name", "")) == away
            ):
                return item
        return None

    def fixture_data(self, fixture_id: int) -> dict[str, Any]:
        """Return lineups, absences, odds and team statistics with source status."""
        if not self.configured:
            return unavailable_result("API_FOOTBALL_KEY is not configured.")

        fetched_at = datetime.now(timezone.utc).isoformat()
        endpoints = {
            "lineups": ("fixtures/lineups", {"fixture": int(fixture_id)}),
            "absences": ("injuries", {"fixture": int(fixture_id)}),
            "odds": ("odds", {"fixture": int(fixture_id)}),
        }
        data: dict[str, Any] = {
            "provider": "API-Football",
            "fixture_id": int(fixture_id),
            "fetched_at": fetched_at,
            "lineups": [],
            "absences": [],
            "odds": [],
            "team_statistics": [],
            "sources": {},
        }
        for key, (endpoint, params) in endpoints.items():
            try:
                payload = self._get(endpoint, params)
                data[key] = payload.get("response", [])
                data["sources"][key] = _source_status(
                    "available" if data[key] else "no_data",
                    fetched_at, payload.get("results", 0),
                )
            except (requests.RequestException, ValueError, APIFootballError) as exc:
                data["sources"][key] = _source_status("error", fetched_at, 0, str(exc))

        # Team statistics require league, season and team identifiers. The
        # fixture endpoint provides team IDs and competition metadata.
        try:
            fixture_payload = self._get("fixtures", {"id": int(fixture_id)})
            fixtures = fixture_payload.get("response", [])
            if fixtures:
                fixture = fixtures[0]
                league = fixture.get("league", {})
                season = league.get("season")
                league_id = league.get("id")
                for side in ("home", "away"):
                    team_id = fixture.get("teams", {}).get(side, {}).get("id")
                    if not all((team_id, season, league_id)):
                        continue
                    payload = self._get("teams/statistics", {
                        "league": league_id, "season": season, "team": team_id,
                    })
                    data["team_statistics"].append({
                        "side": side,
                        "team_id": team_id,
                        "statistics": payload.get("response"),
                    })
                data["sources"]["team_statistics"] = _source_status(
                    "available" if data["team_statistics"] else "no_data",
                    fetched_at, len(data["team_statistics"]),
                )
            else:
                data["sources"]["team_statistics"] = _source_status(
                    "no_data", fetched_at, 0, "Fixture metadata unavailable."
                )
        except (requests.RequestException, ValueError, APIFootballError) as exc:
            data["sources"]["team_statistics"] = _source_status(
                "error", fetched_at, 0, str(exc)
            )

        data["overall_status"] = (
            "available" if any(s["status"] == "available" for s in data["sources"].values())
            else "partial" if any(s["status"] == "error" for s in data["sources"].values())
            else "no_data"
        )
        return data


def _normalise_name(value: str) -> str:
    return " ".join(str(value).casefold().replace(".", "").split())


def _source_status(
    status: str, fetched_at: str, result_count: int, message: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "fetched_at": fetched_at,
        "result_count": int(result_count),
    }
    if message:
        result["message"] = message
    return result


def unavailable_result(reason: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    keys = ("lineups", "absences", "odds", "team_statistics")
    return {
        "provider": "API-Football",
        "overall_status": "not_configured",
        "fetched_at": now,
        "lineups": [],
        "absences": [],
        "odds": [],
        "team_statistics": [],
        "sources": {
            key: _source_status("not_configured", now, 0, reason)
            for key in keys
        },
    }
