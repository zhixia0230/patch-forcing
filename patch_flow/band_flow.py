"""Flow-matching interpolation for frequency bands and the radiomap mean."""

from __future__ import annotations

from collections.abc import Callable

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from jutils import instantiate_from_config


class FrequencyBandFlow(nn.Module):
    """Linear flow matching with one time per non-DC frequency band.

    The radiomap mean is a separate scalar flow state.  Its time is defined as
    ``mean(band_times)`` and therefore has no independently sampled clock.
    """

    def __init__(
        self,
        num_bands: int = 16,
        timestep_sampler: dict | DictConfig | Callable | None = None,
    ) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError(f"num_bands must be positive, got {num_bands}.")
        self.num_bands = int(num_bands)
        if timestep_sampler is None:
            self.t_sampler = torch.rand
        elif isinstance(timestep_sampler, (dict, DictConfig)):
            self.t_sampler = instantiate_from_config(timestep_sampler)
        elif callable(timestep_sampler):
            self.t_sampler = timestep_sampler
        else:
            raise TypeError(
                "timestep_sampler must be None, a config, or a callable; "
                f"got {type(timestep_sampler).__name__}."
            )

    def get_interpolants(
        self,
        x1: Tensor,
        x0: Tensor,
        t: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return band state, target velocity, and band times.

        ``x0`` is required because the source must be fixed-band projected
        noise.  Silently substituting full-spectrum ``randn_like`` here would
        violate the frequency-band state space.
        """

        if x1.ndim != 4 or x1.shape[1] != self.num_bands:
            raise ValueError(f"Expected x1 [B, {self.num_bands}, H, W], got {tuple(x1.shape)}.")
        if x0.shape != x1.shape:
            raise ValueError(f"x0 and x1 must have equal shapes, got {tuple(x0.shape)} and {tuple(x1.shape)}.")
        if t is None:
            t = self.t_sampler((x1.shape[0], self.num_bands), device=x1.device, dtype=x1.dtype)
        if t.shape != (x1.shape[0], self.num_bands):
            raise ValueError(f"Expected t {(x1.shape[0], self.num_bands)}, got {tuple(t.shape)}.")
        if not torch.isfinite(t).all() or not torch.all((t >= 0) & (t <= 1)):
            raise ValueError("Band flow times must be finite and lie in [0, 1].")

        t_image = t[:, :, None, None]
        state = (1.0 - t_image) * x0 + t_image * x1
        target_velocity = x1 - x0
        return state, target_velocity, t

    def get_mean_interpolants(
        self,
        mean1: Tensor,
        mean0: Tensor,
        band_times: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return scalar-mean state, target velocity, and ``t_mu``.

        Inputs are normalized mean values shaped ``[B, 1, 1, 1]``.  There is
        exactly one mean state per radiomap, not one mean per frequency band.
        """

        if mean1.ndim != 4 or mean1.shape[1:] != (1, 1, 1):
            raise ValueError(f"Expected mean1 [B, 1, 1, 1], got {tuple(mean1.shape)}.")
        if mean0.shape != mean1.shape:
            raise ValueError(
                f"mean0 and mean1 must have equal shapes, got {tuple(mean0.shape)} and {tuple(mean1.shape)}."
            )
        expected_times = (mean1.shape[0], self.num_bands)
        if band_times.shape != expected_times:
            raise ValueError(
                f"Expected band_times {expected_times}, got {tuple(band_times.shape)}."
            )
        if not torch.isfinite(band_times).all() or not torch.all((band_times >= 0) & (band_times <= 1)):
            raise ValueError("Band flow times must be finite and lie in [0, 1].")

        mean_time = band_times.mean(dim=1).view(-1, 1, 1, 1)
        state = (1.0 - mean_time) * mean0 + mean_time * mean1
        target_velocity = mean1 - mean0
        return state, target_velocity, mean_time
