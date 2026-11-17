import unittest
from unittest.mock import Mock, patch

from match_intelligence import (
    APIFootballClient,
    APIFootballError,
    LEAGUE_IDS,
    _normalise_name,
    unavailable_result,
)


class MatchIntelligenceTests(unittest.TestCase):
    def test_all_six_leagues_have_provider_ids(self):
        self.assertEqual(set(LEAGUE_IDS), {"E0", "SP1", "D1", "I1", "F1", "SWE"})
        self.assertEqual(len(set(LEAGUE_IDS.values())), 6)

    def test_team_name_normalisation(self):
        self.assertEqual(_normalise_name("  A.C. Milan "), "ac milan")
        self.assertEqual(_normalise_name("Real   Madrid"), "real madrid")

    def test_missing_key_returns_explicit_unavailable_state(self):
        client = APIFootballClient(api_key="")
        self.assertFalse(client.configured)
        with self.assertRaises(APIFootballError):
            client._get("fixtures", {"date": "2026-11-17"})

        result = unavailable_result("API_FOOTBALL_KEY is not configured.")
        self.assertEqual(result["overall_status"], "not_configured")
        self.assertEqual(result["lineups"], [])
        self.assertEqual(result["sources"]["odds"]["status"], "not_configured")

    def test_fixture_lookup_matches_home_and_away_order(self):
        client = APIFootballClient(api_key="test-key")
        client._get = Mock(return_value={
            "response": [{
                "fixture": {"id": 123},
                "teams": {
                    "home": {"name": "Real Madrid"},
                    "away": {"name": "Barcelona"},
                },
            }]
        })
        found = client.find_fixture(
            "SP1", 2026, "2026-11-17", " Real   Madrid ", "Barcelona"
        )
        self.assertEqual(found["fixture"]["id"], 123)
        client._get.assert_called_once()

    def test_unknown_league_is_rejected(self):
        client = APIFootballClient(api_key="test-key")
        with self.assertRaises(ValueError):
            client.find_fixture("UNKNOWN", 2026, "2026-11-17", "A", "B")


if __name__ == "__main__":
    unittest.main()
