"""Highlightly match intelligence integration.

The provider supplies fixtures, confirmed lineups and live match events.
The free tier does not include bookmaker odds; this module marks odds and
injury/absence data unavailable rather than implying that no records exist.
Match context is kept separate from model probabilities until validated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import os
import re
import unicodedata
from typing import Any

import requests

BASE_URL = "https://soccer.highlightly.net"
LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "SWE": "Allsvenskan",
}
REQUEST_TIMEOUT = 15


class HighlightlyError(RuntimeError):
    """Raised when Highlightly returns an error."""


class HighlightlyClient:
    def __init__(self, api_key: str | None = None, timeout: int = REQUEST_TIMEOUT):
        self.api_key = (api_key or os.getenv("HIGHLIGHTLY_API_KEY", "")).strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-key": self.api_key,
            "User-Agent": "FootballValueModel/2.2",
        })

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        if not self.configured:
            raise HighlightlyError("HIGHLIGHTLY_API_KEY is not configured.")
        response = self.session.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            params=params or {},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise HighlightlyError(str(payload["error"]))
        return payload

    @staticmethod
    def _items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data", [])
            return data if isinstance(data, list) else []
        return []

    def find_fixture(
        self, league_code: str, season: int, match_date: str,
        home_team: str, away_team: str,
    ) -> dict[str, Any] | None:
        if league_code not in LEAGUE_NAMES:
            raise ValueError(f"Unsupported league code: {league_code}")
        payload = self._get("matches", {
            "date": match_date,
            "leagueName": LEAGUE_NAMES[league_code],
            "timezone": "Europe/Stockholm",
            "limit": 100,
        })
        home = _normalise_name(home_team)
        away = _normalise_name(away_team)
        candidates = []
        for item in self._items(payload):
            h = item.get("homeTeam") or {}
            a = item.get("awayTeam") or {}
            home_score = _team_name_score(home, h.get("name", ""))
            away_score = _team_name_score(away, a.get("name", ""))
            if home_score >= 0.78 and away_score >= 0.78:
                candidates.append((home_score + away_score, item))
        if not candidates:
            return None
        candidates.sort(key=lambda row: row[0], reverse=True)
        return candidates[0][1]

    def fixture_data(self, fixture_id: int) -> dict[str, Any]:
        if not self.configured:
            return unavailable_result("HIGHLIGHTLY_API_KEY is not configured.")

        fetched_at = datetime.now(timezone.utc).isoformat()
        data: dict[str, Any] = {
            "provider": "Highlightly",
            "fixture_id": int(fixture_id),
            "fetched_at": fetched_at,
            "lineups": [],
            "absences": [],
            "odds": [],
            "events": [],
            "team_statistics": [],
            "sources": {},
        }

        try:
            payload = self._get(f"lineups/{int(fixture_id)}")
            if isinstance(payload, dict):
                payload = payload.get("data", payload)
                for side in ("homeTeam", "awayTeam"):
                    team = payload.get(side) or {}
                    if not team:
                        continue
                    players = [
                        player
                        for row in (team.get("initialLineup") or [])
                        for player in (row if isinstance(row, list) else [])
                    ]
                    data["lineups"].append({
                        "team": {"name": team.get("name", side)},
                        "formation": team.get("formation"),
                        "startXI": [{"player": {"name": p.get("name", "Unknown")}}
                                    for p in players],
                        "substitutes": team.get("substitutes") or [],
                    })
            data["sources"]["lineups"] = _source_status(
                "available" if data["lineups"] else "no_data", fetched_at,
                len(data["lineups"]),
            )
        except (requests.RequestException, ValueError, HighlightlyError) as exc:
            data["sources"]["lineups"] = _source_status("error", fetched_at, 0, str(exc))

        try:
            payload = self._get(f"events/{int(fixture_id)}")
            data["events"] = self._items(payload)
            data["sources"]["events"] = _source_status(
                "available" if data["events"] else "no_data", fetched_at,
                len(data["events"]),
            )
        except (requests.RequestException, ValueError, HighlightlyError) as exc:
            data["sources"]["events"] = _source_status("error", fetched_at, 0, str(exc))

        data["sources"]["absences"] = _source_status(
            "not_supported", fetched_at, 0,
            "A verified injuries/suspensions feed is not documented for this integration.",
        )
        data["sources"]["odds"] = _source_status(
            "plan_restricted", fetched_at, 0,
            "Bookmaker odds are not included in Highlightly's free plan.",
        )
        data["sources"]["team_statistics"] = _source_status(
            "not_requested", fetched_at, 0,
            "Use the existing historical match data for team context.",
        )
        data["overall_status"] = (
            "available" if any(s["status"] == "available" for s in data["sources"].values())
            else "partial" if any(s["status"] == "error" for s in data["sources"].values())
            else "no_data"
        )
        return data


def _normalise_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.replace(".", "")
    tokens = re.sub(r"[^a-zA-Z0-9 ]", " ", value).casefold().split()
    ignored = {"fc", "if", "aif", "afc", "cf", "fk", "club"}
    return " ".join(token for token in tokens if token not in ignored)


def _team_name_score(selected: str, provider_name: str) -> float:
    candidate = _normalise_name(provider_name)
    if not selected or not candidate:
        return 0.0
    if selected == candidate:
        return 1.0
    if selected in candidate or candidate in selected:
        return 0.92
    return SequenceMatcher(None, selected, candidate).ratio()


def _source_status(status: str, fetched_at: str, result_count: int,
                   message: str | None = None) -> dict[str, Any]:
    result = {"status": status, "fetched_at": fetched_at,
              "result_count": int(result_count)}
    if message:
        result["message"] = message
    return result


def unavailable_result(reason: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    keys = ("lineups", "events", "absences", "odds", "team_statistics")
    return {
        "provider": "Highlightly",
        "overall_status": "not_configured",
        "fetched_at": now,
        "lineups": [], "events": [], "absences": [], "odds": [],
        "team_statistics": [],
        "sources": {key: _source_status("not_configured", now, 0, reason)
                    for key in keys},
    }


def extract_1x2_odds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility helper; odds are not supplied by the free Highlightly plan."""
    return []


def compare_odds_snapshots(
    previous: list[dict[str, Any]], current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare odds snapshots if a future paid provider adapter supplies them."""
    old = {row["bookmaker"]: row for row in previous or []}
    changes = []
    for row in current or []:
        prior = old.get(row["bookmaker"])
        if prior:
            changes.append({
                "bookmaker": row["bookmaker"],
                "home": round(row["home"] - prior["home"], 3),
                "draw": round(row["draw"] - prior["draw"], 3),
                "away": round(row["away"] - prior["away"], 3),
            })
    return changes
