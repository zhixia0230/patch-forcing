import unittest

import torch

from patch_flow.band_flow import FrequencyBandFlow
from patch_flow.band_integrators import (
    BandContinuousAsyncSampler,
    BandEulerSampler,
    BandMacroSyncAdaptiveSampler,
)
from patch_flow.models.band_unet import FrequencyBandUNet
from patch_flow.radiomap_time_sampling import (
    MacroSynchronizedTimeSampler,
    StructuredBandTimeSampler,
)
from patch_flow.trainer_radiomap import RadiomapBandFlowTrainer


class IdentityProjector(torch.nn.Module):
    num_bands = 16

    @staticmethod
    def project_bands(bands):
        return bands


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    @staticmethod
    def encode_condition(condition):
        return (condition, condition, condition, condition)

    def forward(
        self,
        x,
        band_time,
        mean_state,
        mean_time,
        condition_features=None,
        return_aux=False,
    ):
        del mean_state, condition_features
        self.calls.append((band_time.detach().clone(), mean_time.detach().clone()))
        velocity = torch.zeros_like(x)
        uncertainty = torch.arange(16, device=x.device, dtype=x.dtype)[None].expand(x.shape[0], -1)
        mean_velocity = torch.full(
            (x.shape[0], 1, 1, 1), float(len(self.calls)), device=x.device, dtype=x.dtype
        )
        return (velocity, uncertainty, mean_velocity) if return_aux else velocity


class FakeDecomposer(torch.nn.Module):
    num_bands = 16

    def transform(self, radiomap):
        mean = radiomap.mean(dim=(-2, -1), keepdim=True)
        bands = torch.zeros(
            radiomap.shape[0], 16, *radiomap.shape[-2:], device=radiomap.device, dtype=radiomap.dtype
        )
        bands[:, 0] = (radiomap - mean)[:, 0]
        return bands, mean

    @staticmethod
    def normalize_mean(mean):
        return mean

    @staticmethod
    def denormalize_mean(mean):
        return mean

    @staticmethod
    def project_bands(bands):
        return bands

    def band_limited_noise(self, batch_size, *, device, dtype, generator=None):
        del generator
        return torch.zeros(batch_size, 16, 16, 16, device=device, dtype=dtype)

    @staticmethod
    def inverse_transform(bands, mean):
        return mean + bands.sum(dim=1, keepdim=True)


class TestBandMeanFlow(unittest.TestCase):
    def _small_model(self):
        return FrequencyBandUNet(
            base_channels=8,
            channel_multipliers=(1, 2, 4, 4),
            condition_channels=(4, 8, 8, 8),
            time_embedding_dim=8,
            gradient_checkpointing=False,
        )

    def test_model_outputs_and_four_scale_conditioning(self):
        model = self._small_model()
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        times = torch.rand(2, 16)
        mean_time = times.mean(dim=1).view(2, 1, 1, 1)
        condition = torch.zeros(2, 2, 16, 16)
        condition[:, 1, 5, 7] = 1
        outputs = model(
            x,
            times,
            torch.randn(2, 1, 1, 1),
            mean_time,
            condition=condition,
            return_aux=True,
        )
        velocity, uncertainty, mean_velocity = outputs
        self.assertEqual(velocity.shape, x.shape)
        self.assertEqual(uncertainty.shape, (2, 16))
        self.assertEqual(mean_velocity.shape, (2, 1, 1, 1))
        condition_features = model.encode_condition(condition)
        self.assertEqual([item.shape[-2:] for item in condition_features], [(16, 16), (8, 8), (4, 4), (2, 2)])
        (velocity.square().mean() + uncertainty.square().mean() + mean_velocity.square().mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_mean_training_time_is_average_band_time(self):
        flow = FrequencyBandFlow(num_bands=16)
        mean0 = torch.tensor([[[[-1.0]]]])
        mean1 = torch.tensor([[[[3.0]]]])
        band_time = torch.arange(16, dtype=torch.float32)[None] / 16
        state, velocity, mean_time = flow.get_mean_interpolants(mean1, mean0, band_time)
        self.assertTrue(torch.equal(mean_time, band_time.mean(dim=1).view(1, 1, 1, 1)))
        self.assertTrue(torch.equal(velocity, mean1 - mean0))
        self.assertTrue(torch.equal(state, (1 - mean_time) * mean0 + mean_time * mean1))

    def test_structured_times_respect_r4(self):
        sampler = StructuredBandTimeSampler()
        times, branches = sampler.sample_with_branches((4096, 16))
        odds = times / (1 - times)
        ratio = odds.max(dim=1).values / odds.min(dim=1).values
        self.assertLessEqual(float(ratio.max()), 4.0001)
        self.assertTrue(torch.all((times > 0) & (times < 1)))
        counts = torch.bincount(branches, minlength=5)
        self.assertTrue(torch.all(counts > 0))

    def test_macro_synchronized_training_times_have_bounded_temporary_leads(self):
        sampler = MacroSynchronizedTimeSampler()
        times, branches = sampler.sample_with_branches((4096, 16))
        spread = times.max(dim=1).values - times.min(dim=1).values
        self.assertLessEqual(float(spread.max()), 0.02001)
        self.assertTrue(torch.all((times > 0) & (times < 1)))
        self.assertTrue(torch.all(torch.bincount(branches, minlength=4) > 0))
        synchronized = branches != 2
        self.assertTrue(torch.equal(spread[synchronized], torch.zeros_like(spread[synchronized])))

    def test_euler_integrates_mean_by_actual_average_clock_delta(self):
        model = RecordingModel()
        _, mean, diagnostics = BandEulerSampler()(
            model=model,
            x=torch.zeros(1, 16, 4, 4),
            mean=torch.zeros(1, 1, 1, 1),
            timesteps=torch.tensor([0.25, 0.5, 1.0]),
            decomposer=IdentityProjector(),
            condition=torch.zeros(1, 2, 4, 4),
            progress=False,
        )
        self.assertTrue(torch.equal(mean, torch.tensor([[[[1.25]]]])))
        self.assertEqual(diagnostics["final_mean_time"], 1.0)

    def test_continuous_async_finishes_all_clocks(self):
        model = RecordingModel()
        _, _, diagnostics = BandContinuousAsyncSampler()(
            model=model,
            x=torch.zeros(2, 16, 4, 4),
            mean=torch.zeros(2, 1, 1, 1),
            timesteps=torch.linspace(0, 1, 11),
            decomposer=IdentityProjector(),
            condition=torch.zeros(2, 2, 4, 4),
            progress=False,
        )
        self.assertEqual(diagnostics["nfe"], 10)
        self.assertEqual(diagnostics["final_mean_time"], 1.0)
        self.assertLessEqual(diagnostics["maximum_observed_odds_ratio"], 4.001)

    def test_macro_sync_sampler_bounds_steps_and_resynchronizes(self):
        model = RecordingModel()
        _, _, diagnostics = BandMacroSyncAdaptiveSampler(max_reliable_logvar=4.0)(
            model=model,
            x=torch.zeros(2, 16, 4, 4),
            mean=torch.zeros(2, 1, 1, 1),
            timesteps=torch.linspace(0, 1, 26),
            decomposer=IdentityProjector(),
            condition=torch.zeros(2, 2, 4, 4),
            progress=False,
        )
        self.assertEqual(diagnostics["nfe"], 80)
        self.assertEqual(diagnostics["macro_rounds"], 25)
        self.assertEqual(diagnostics["macro_resynchronizations"], 25)
        self.assertAlmostEqual(diagnostics["maximum_step"], 0.02, places=5)
        self.assertLessEqual(diagnostics["maximum_time_gap"], 0.02001)
        self.assertEqual(diagnostics["final_mean_time"], 1.0)

    def test_trainer_uses_conditions_and_mean_weight(self):
        model = self._small_model()
        trainer = RadiomapBandFlowTrainer(
            model=model,
            decomposer=FakeDecomposer(),
            flow=FrequencyBandFlow(num_bands=16),
            ema_rate=0,
            mean_weight=0.2,
        )
        radiomap = torch.rand(2, 1, 16, 16)
        band_times = torch.rand(2, 16)
        batch = {
            "image": radiomap,
            "building": torch.zeros_like(radiomap),
            "tx": torch.zeros_like(radiomap),
        }
        batch["tx"][:, :, 3, 4] = 1
        losses = trainer.compute_losses(
            batch,
            band_source=torch.zeros(2, 16, 16, 16),
            mean_source=torch.zeros(2, 1, 1, 1),
            band_times=band_times,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.equal(losses["mean_time"], band_times.mean(dim=1).view(2, 1, 1, 1)))
        expected = losses["flow_loss"] + 0.2 * losses["mean_loss"] + 0.01 * losses["uncertainty_loss"]
        self.assertTrue(torch.allclose(losses["loss"], expected))
        losses["loss"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.mean_velocity_head.parameters()))


if __name__ == "__main__":
    unittest.main()
