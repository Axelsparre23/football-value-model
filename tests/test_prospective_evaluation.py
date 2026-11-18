import unittest
import numpy as np
import pandas as pd

from research.prospective_evaluation import scored_rows, score_probabilities, metric_table, paired_brier_bootstrap


class ProspectiveEvaluationTests(unittest.TestCase):
    def make_log(self):
        return pd.DataFrame([
            {"match_id": 1, "kickoff_local": "2026-11-20", "home_team": "A", "away_team": "B",
             "elo_home_pct": 50, "elo_draw_pct": 25, "elo_away_pct": 25,
             "dc_home_pct": 45, "dc_draw_pct": 30, "dc_away_pct": 25,
             "market_home_novig_pct": 48, "market_draw_novig_pct": 27, "market_away_novig_pct": 25, "bookmaker_name": "Bet365",
             "final_home_goals": 2, "final_away_goals": 0, "actual_outcome": ""},
            {"match_id": 2, "kickoff_local": "2026-11-21", "home_team": "C", "away_team": "D",
             "elo_home_pct": 30, "elo_draw_pct": 30, "elo_away_pct": 40,
             "dc_home_pct": 32, "dc_draw_pct": 28, "dc_away_pct": 40,
             "market_home_novig_pct": 29, "market_draw_novig_pct": 30, "market_away_novig_pct": 41,
             "final_home_goals": np.nan, "final_away_goals": np.nan, "actual_outcome": ""},
        ])

    def test_only_completed_fixtures_are_scored(self):
        scored = scored_rows(self.make_log())
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0].actual_index, 0)

    def test_scores_are_correct_and_probabilities_normalised(self):
        result = score_probabilities([[0.5, 0.25, 0.25]], [0])
        self.assertAlmostEqual(result["brier"], 0.375)
        self.assertAlmostEqual(result["log_loss"], -np.log(0.5))
        self.assertEqual(result["accuracy"], 1.0)

    def test_market_uses_closing_when_present_else_opening(self):
        log = self.make_log()
        log.loc[0, ["closing_market_home_novig_pct", "closing_market_draw_novig_pct",
                    "closing_market_away_novig_pct"]] = [40, 30, 30]
        log.loc[0, "closing_bookmaker_name"] = "Bet365"
        scored = scored_rows(log)
        self.assertEqual(scored.iloc[0].market_snapshot, "closing")
        self.assertTrue(np.allclose(scored.iloc[0].Market_p, [0.4, 0.3, 0.3]))

    def test_invalid_probability_triplet_is_excluded(self):
        log = self.make_log()
        log.loc[0, ["elo_home_pct", "elo_draw_pct", "elo_away_pct"]] = [90, 20, 10]
        scored = scored_rows(log)
        self.assertIsNone(scored.iloc[0]["Elo_p"])

    def test_aggregated_best_prices_are_not_used_as_strict_market_benchmark(self):
        log = self.make_log()
        log.loc[0, "bookmaker_name"] = "Aggregated best prices across 20 bookmakers"
        scored = scored_rows(log)
        self.assertIsNone(scored.iloc[0]["Market_p"])
        self.assertIsNotNone(scored.iloc[0]["Recorded_market_p"])

    def test_bootstrap_is_reproducible(self):
        log = self.make_log()
        log.loc[1, "final_home_goals"] = 0
        log.loc[1, "final_away_goals"] = 1
        log.loc[1, "bookmaker_name"] = "Bet365"
        scored = scored_rows(log)
        scored = scored.loc[scored.Elo_p.notna() & scored.Market_p.notna()]
        self.assertEqual(len(scored), 2)
        a = paired_brier_bootstrap(scored, "Elo", "Market", iterations=500, seed=7)
        b = paired_brier_bootstrap(scored, "Elo", "Market", iterations=500, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
