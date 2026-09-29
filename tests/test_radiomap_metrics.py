import math
import unittest

import torch

from patch_flow.radiomap_metrics import RadioMapMetricAccumulator


class TestRadioMapMetricAccumulator(unittest.TestCase):
    def test_global_rmse_nmse_and_auxiliary_metrics(self):
        target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        prediction = torch.tensor([[[[2.0, 2.0], [1.0, 4.0]]]])
        building = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
        tx = torch.zeros_like(target)
        tx[:, :, 0, 0] = 1
        target_bands = torch.ones(1, 2, 2, 2)
        predicted_bands = target_bands.clone()
        predicted_bands[:, 1] += 1

        tracker = RadioMapMetricAccumulator(num_bands=2, tx_window=1)
        tracker.update(
            prediction,
            target,
            building=building,
            tx=tx,
            predicted_mean=prediction.mean((-2, -1), keepdim=True),
            predicted_bands=predicted_bands,
            target_bands=target_bands,
            band_scales=torch.tensor([2.0, 3.0]),
        )
        metrics = tracker.compute()

        self.assertAlmostEqual(metrics["radiomap"]["rmse"], math.sqrt(5 / 4))
        self.assertAlmostEqual(metrics["radiomap"]["nmse"], 5 / 30)
        self.assertAlmostEqual(metrics["radiomap"]["nrmse"], math.sqrt(5 / 30))
        self.assertAlmostEqual(metrics["radiomap"]["mae"], 3 / 4)
        self.assertAlmostEqual(metrics["radiomap"]["bias"], -1 / 4)
        self.assertAlmostEqual(metrics["mean"]["rmse"], 1 / 4)
        self.assertEqual(metrics["tx_neighborhood"]["count"], 1)
        self.assertAlmostEqual(metrics["bands"][0]["rmse"], 0.0)
        self.assertAlmostEqual(metrics["bands"][1]["rmse"], 3.0)
        self.assertAlmostEqual(metrics["bands"][1]["nmse"], 1.0)

    def test_micro_nmse_differs_from_mean_sample_nmse(self):
        target = torch.tensor([[[[1.0]]], [[[10.0]]]])
        prediction = torch.tensor([[[[2.0]]], [[[11.0]]]])
        tracker = RadioMapMetricAccumulator(num_bands=1)
        tracker.update(prediction, target)
        metrics = tracker.compute()["radiomap"]

        self.assertAlmostEqual(metrics["nmse"], 2 / 101)
        self.assertAlmostEqual(metrics["sample_nmse_mean"], (1.0 + 0.01) / 2)

    def test_rejects_partial_band_arguments(self):
        tracker = RadioMapMetricAccumulator(num_bands=1)
        image = torch.zeros(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            tracker.update(image, image, predicted_bands=torch.zeros(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
