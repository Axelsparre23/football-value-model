import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from match_intelligence import (
    DAILY_REQUEST_BUDGET, HighlightlyClient, HighlightlyError, LEAGUE_NAMES,
    _USAGE, _USAGE_LOCK, _normalise_name, _record_api_request,
    _record_api_response, get_api_usage, unavailable_result,
)


class MatchIntelligenceTests(unittest.TestCase):
    def setUp(self):
        with _USAGE_LOCK:
            self.previous_usage = dict(_USAGE)
            _USAGE.update(
                date=datetime.now(timezone.utc).date().isoformat(),
                requests=0, rate_limited=False, provider_remaining=None,
            )

    def tearDown(self):
        with _USAGE_LOCK:
            _USAGE.clear()
            _USAGE.update(self.previous_usage)

    def test_api_usage_counter_tracks_requests_and_remaining_estimate(self):
        _record_api_request()
        _record_api_request()
        usage = get_api_usage()
        self.assertEqual(usage["requests_today"], 2)
        self.assertEqual(usage["remaining_estimate"], DAILY_REQUEST_BUDGET - 2)
        self.assertFalse(usage["rate_limited"])

    def test_provider_remaining_header_can_lower_estimate(self):
        response = Mock(status_code=200, headers={"x-ratelimit-remaining": "7"})
        _record_api_response(response)
        usage = get_api_usage()
        self.assertEqual(usage["remaining_estimate"], 7)

    def test_http_429_marks_quota_as_exhausted(self):
        response = Mock(status_code=429, headers={})
        _record_api_response(response)
        usage = get_api_usage()
        self.assertTrue(usage["rate_limited"])
        self.assertEqual(usage["remaining_estimate"], 0)

    def test_all_six_leagues_are_mapped(self):
        self.assertEqual(set(LEAGUE_NAMES), {"E0", "SP1", "D1", "I1", "F1", "SWE"})
        self.assertEqual(LEAGUE_NAMES["E0"], "Premier League")
        self.assertEqual(LEAGUE_NAMES["SWE"], "Allsvenskan")

    def test_team_name_normalisation(self):
        self.assertEqual(_normalise_name("  A.C. Milan "), "ac milan")
        self.assertEqual(_normalise_name("Real   Madrid"), "real madrid")

    def test_missing_key_returns_explicit_unavailable_state(self):
        client = HighlightlyClient(api_key="")
        self.assertFalse(client.configured)
        with self.assertRaises(HighlightlyError):
            client._get("matches", {"date": "2026-11-17"})
        result = unavailable_result("HIGHLIGHTLY_API_KEY is not configured.")
        self.assertEqual(result["overall_status"], "not_configured")
        self.assertEqual(result["sources"]["lineups"]["status"], "not_configured")

    def test_fixture_lookup_matches_home_and_away(self):
        client = HighlightlyClient(api_key="test-key")
        client._get = Mock(return_value={"data": [{
            "id": 123,
            "homeTeam": {"name": "Real Madrid"},
            "awayTeam": {"name": "Barcelona"},
        }]})
        found = client.find_fixture("SP1", 2026, "2026-11-17",
                                    " Real   Madrid ", "Barcelona")
        self.assertEqual(found["id"], 123)
        client._get.assert_called_once()

    def test_lineups_are_normalised_and_free_plan_limits_are_explicit(self):
        client = HighlightlyClient(api_key="test-key")
        client._get = Mock(side_effect=[
            {"homeTeam": {
                "name": "Real Madrid", "formation": "4-3-3",
                "initialLineup": [[{"name": "Player One"}]],
                "substitutes": [],
            }, "awayTeam": {
                "name": "Barcelona", "formation": "4-4-2",
                "initialLineup": [[{"name": "Player Two"}]],
                "substitutes": [],
            }},
            [{"type": "Goal", "time": "12", "player": "Player One"}],
        ])
        result = client.fixture_data(123)
        self.assertEqual(len(result["lineups"]), 2)
        self.assertEqual(result["lineups"][0]["startXI"][0]["player"]["name"], "Player One")
        self.assertEqual(result["events"][0]["type"], "Goal")
        self.assertEqual(result["sources"]["odds"]["status"], "plan_restricted")
        self.assertEqual(result["sources"]["absences"]["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
