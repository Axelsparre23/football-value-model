import unittest
import numpy as np

from market_residual_backtest import (
    _per_match_losses,
    paired_bootstrap_vs_market,
    calibration_summary,
)


class MarketResidualEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([0, 1, 2, 0, 2, 1])
        self.market = np.array([
            [0.55, 0.25, 0.20],
            [0.40, 0.30, 0.30],
            [0.20, 0.25, 0.55],
            [0.50, 0.28, 0.22],
            [0.25, 0.25, 0.50],
            [0.35, 0.35, 0.30],
        ])
        self.model = np.array([
            [0.60, 0.22, 0.18],
            [0.38, 0.34, 0.28],
            [0.18, 0.25, 0.57],
            [0.52, 0.27, 0.21],
            [0.22, 0.25, 0.53],
            [0.33, 0.38, 0.29],
        ])

    def test_per_match_losses_are_finite_and_aligned(self):
        losses = _per_match_losses(self.market, self.actual)
        self.assertEqual(set(losses), {"Brier", "LogLoss", "RPS"})
        for values in losses.values():
            self.assertEqual(values.shape, self.actual.shape)
            self.assertTrue(np.all(np.isfinite(values)))

    def test_paired_bootstrap_is_reproducible(self):
        first = paired_bootstrap_vs_market(self.model, self.market, self.actual,
                                           n_boot=250, seed=7)
        second = paired_bootstrap_vs_market(self.model, self.market, self.actual,
                                            n_boot=250, seed=7)
        self.assertEqual(first, second)
        for metric in ("Brier", "LogLoss", "RPS"):
            self.assertLessEqual(first[f"{metric}_ci95_low"],
                                 first[f"{metric}_ci95_high"])

    def test_calibration_bins_cover_each_outcome(self):
        rows = calibration_summary(self.model, self.actual, "Residual", "test", bins=5)
        self.assertEqual({row["Outcome"] for row in rows}, {"Home", "Draw", "Away"})
        for outcome in ("Home", "Draw", "Away"):
            self.assertEqual(sum(row["N"] for row in rows if row["Outcome"] == outcome),
                             len(self.actual))
            self.assertTrue(all(np.isfinite(row["ClassECE"]) for row in rows
                                if row["Outcome"] == outcome))


if __name__ == "__main__":
    unittest.main()
