import unittest

import numpy as np
import pandas as pd

from advanced_model import backtest_80_20


class FakeDixonColes:
    def __init__(self, **kwargs):
        self.fitted_until = None

    def fit(self, train, as_of):
        self.fitted_until = pd.Timestamp(as_of)
        self.training_dates = pd.to_datetime(train["Date"])
        return self

    def probabilities(self, home, away):
        return np.array([0.40, 0.30, 0.30]), (1.2, 0.9)


class ChronologicalBacktestTests(unittest.TestCase):
    def make_data(self):
        teams = ["A", "B", "C", "D", "E", "F"]
        fixtures = [
            (teams[i], teams[j])
            for i in range(len(teams))
            for j in range(i + 1, len(teams))
        ][:15]
        rows = []
        for day in range(10):
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
            for index, (home, away) in enumerate(fixtures):
                rows.append({
                    "Date": date, "League": "TEST", "HomeTeam": home, "AwayTeam": away,
                    "FTHG": index % 3, "FTAG": (index + day) % 2,
                })
        return pd.DataFrame(rows)

    def test_holdout_boundary_never_splits_a_match_date(self):
        result = backtest_80_20(self.make_data(), FakeDixonColes, None)
        self.assertEqual(result["train_matches"], 120)
        self.assertEqual(result["test_matches"], 30)
        self.assertEqual(result["cutoff"], "2025-01-09")
        self.assertTrue(all(row["date"] >= result["cutoff"] for row in result["predictions"]))

    def test_all_fixtures_on_same_date_use_same_pre_date_elo_snapshot(self):
        result = backtest_80_20(self.make_data(), FakeDixonColes, None)
        same_fixture = [
            row for row in result["predictions"]
            if row["date"] == result["cutoff"]
            and row["home_team"] == "A" and row["away_team"] == "B"
        ]
        self.assertGreaterEqual(len(same_fixture), 1)
        self.assertEqual(len({row["elo_diff"] for row in same_fixture}), 1)

    def test_requires_multiple_dates(self):
        data = self.make_data()
        data["Date"] = pd.Timestamp("2025-01-01")
        with self.assertRaisesRegex(ValueError, "distinct match dates"):
            backtest_80_20(data, FakeDixonColes, None)


if __name__ == "__main__":
    unittest.main()
