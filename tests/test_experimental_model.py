import unittest
import numpy as np

from research.experimental_model import fit_residual_model, predict_residual_model


class ExperimentalResidualModelTests(unittest.TestCase):
    def make_data(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(180, 4))
        x[::11, 1] = np.nan
        market = np.tile([0.50, 0.27, 0.23], (len(x), 1))
        y = rng.choice(3, size=len(x), p=market[0])
        return x, market, y

    def test_fit_predict_returns_valid_probabilities(self):
        x, market, y = self.make_data()
        model = fit_residual_model(x, market, y, ridge=0.1)
        probabilities = predict_residual_model(model, x, market)
        self.assertEqual(probabilities.shape, (len(x), 3))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertEqual(model["training_matches"], len(x))

    def test_prediction_imputes_missing_features_from_training_means(self):
        x, market, y = self.make_data()
        model = fit_residual_model(x, market, y)
        x_new = x[:5].copy()
        x_new[0, 2] = np.nan
        p = predict_residual_model(model, x_new, market[:5])
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0))

    def test_rejects_invalid_market_probabilities(self):
        x, market, y = self.make_data()
        market[0] = [0.0, 0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            fit_residual_model(x, market, y)

    def test_rejects_feature_schema_mismatch(self):
        x, market, y = self.make_data()
        model = fit_residual_model(x, market, y)
        with self.assertRaisesRegex(ValueError, "feature count"):
            predict_residual_model(model, x[:, :2], market)


if __name__ == "__main__":
    unittest.main()
