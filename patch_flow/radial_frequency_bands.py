"""Fixed 16-band RadioMap FFT representation and spectrally shaped noise."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


class FixedRadialBandDecomposer(nn.Module):
    """Load the training-only FFT masks/scales and keep all states band limited.

    ``spectral_mix_rho`` interpolates between uniform power inside each band
    (rho=0) and the fixed training reference spectrum (rho=1).  It is not a
    correlation coefficient between band channels.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        spectral_mix_rho: float = 0.0,
        mean_center: float = 0.27602527060913873,
        mean_scale: float = 0.07821975390838161,
    ) -> None:
        super().__init__()
        if not 0.0 <= spectral_mix_rho <= 1.0:
            raise ValueError("spectral_mix_rho must lie in [0, 1].")
        artifact_path = Path(artifact_path).expanduser()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frequency-band artifact not found: {artifact_path}")
        payload = np.load(artifact_path)
        required = {"masks", "reference_spectrum", "raw_rms"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Band artifact is missing keys: {sorted(missing)}")

        masks = torch.from_numpy(payload["masks"].astype(np.bool_))
        reference = torch.from_numpy(payload["reference_spectrum"].astype(np.float64))
        scales = torch.from_numpy(payload["raw_rms"].astype(np.float64))
        if masks.ndim != 3 or reference.shape != masks.shape[1:] or scales.shape != (masks.shape[0],):
            raise ValueError("Incompatible masks, reference spectrum, or raw RMS shapes.")
        if not torch.all(scales > 0) or not torch.isfinite(scales).all():
            raise ValueError("Every fixed band scale must be finite and positive.")

        coverage = masks.to(torch.int16).sum(dim=0)
        shifted_dc = (masks.shape[-2] // 2, masks.shape[-1] // 2)
        expected = torch.ones_like(coverage)
        expected[shifted_dc] = 0
        if not torch.equal(coverage, expected):
            raise ValueError("Masks must cover each non-DC frequency exactly once.")

        # Enforce exact conjugate symmetry before taking a square root.  The
        # supplied statistic is already symmetric up to floating round-off.
        unshifted = torch.fft.ifftshift(reference, dim=(-2, -1))
        height, width = reference.shape
        negative_y = (-torch.arange(height)) % height
        negative_x = (-torch.arange(width)) % width
        conjugate = unshifted.index_select(0, negative_y).index_select(1, negative_x)
        reference = torch.fft.fftshift(0.5 * (unshifted + conjugate), dim=(-2, -1))
        reference = reference.clamp_min(0)

        counts = masks.sum(dim=(-2, -1), dtype=torch.int64)
        uniform = masks.to(torch.float64) / counts[:, None, None]
        band_reference = masks.to(torch.float64) * reference[None]
        reference_totals = band_reference.sum(dim=(-2, -1), keepdim=True)
        if not torch.all(reference_totals > 0):
            raise ValueError("Every band must have positive reference-spectrum mass.")
        band_reference = band_reference / reference_totals
        power = (1.0 - spectral_mix_rho) * uniform + spectral_mix_rho * band_reference
        power = power / power.sum(dim=(-2, -1), keepdim=True)

        self.num_bands = int(masks.shape[0])
        self.height = int(height)
        self.width = int(width)
        self.spectral_mix_rho = float(spectral_mix_rho)
        self.register_buffer("masks", masks)
        self.register_buffer("raw_rms", scales)
        self.register_buffer("reference_spectrum", reference)
        self.register_buffer("frequency_counts", counts)
        self.register_buffer("source_power", power)
        self.register_buffer("mean_center", torch.tensor(float(mean_center), dtype=torch.float64))
        self.register_buffer("mean_scale", torch.tensor(float(mean_scale), dtype=torch.float64))

    def _validate_map(self, x: Tensor) -> None:
        expected = (1, self.height, self.width)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected radiomap [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("Radiomap input must be floating point.")

    def transform(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return fixed-RMS normalized bands and the scalar spatial mean."""

        self._validate_map(x)
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = fft_input.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(fft_input - mean, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        )
        complex_bands = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * self.masks.to(x.device)[None], dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        )
        scales = self.raw_rms.to(device=x.device, dtype=complex_bands.real.dtype)
        bands = complex_bands.real / scales[None, :, None, None]
        return bands, mean

    def project_bands(self, bands: Tensor) -> Tensor:
        expected = (self.num_bands, self.height, self.width)
        if bands.ndim != 4 or tuple(bands.shape[1:]) != expected:
            raise ValueError(f"Expected bands [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(bands.shape)}.")
        fft_input = bands.float() if bands.dtype in (torch.float16, torch.bfloat16) else bands
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(fft_input, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        )
        projected = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * self.masks.to(bands.device)[None], dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        )
        return projected.real

    def band_limited_noise(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw unit-expected-RMS Gaussian fields with the configured spectrum."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("FFT source noise must use float32 or float64.")
        device = self.masks.device if device is None else torch.device(device)
        white = torch.randn(
            batch_size,
            self.num_bands,
            self.height,
            self.width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(white, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        )
        power = self.source_power.to(device=device, dtype=dtype)
        multiplier = torch.sqrt((self.height * self.width) * power)
        shaped = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * multiplier[None], dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        )
        return shaped.real

    def normalize_mean(self, mean: Tensor) -> Tensor:
        center = self.mean_center.to(device=mean.device, dtype=mean.dtype)
        scale = self.mean_scale.to(device=mean.device, dtype=mean.dtype)
        return (mean - center) / scale

    def denormalize_mean(self, normalized_mean: Tensor) -> Tensor:
        center = self.mean_center.to(device=normalized_mean.device, dtype=normalized_mean.dtype)
        scale = self.mean_scale.to(device=normalized_mean.device, dtype=normalized_mean.dtype)
        return normalized_mean * scale + center

    def inverse_transform(self, bands: Tensor, mean: Tensor) -> Tensor:
        expected = (self.num_bands, self.height, self.width)
        if bands.ndim != 4 or tuple(bands.shape[1:]) != expected:
            raise ValueError(f"Expected bands [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(bands.shape)}.")
        if mean.shape != (bands.shape[0], 1, 1, 1):
            raise ValueError(f"Expected mean [B, 1, 1, 1], got {tuple(mean.shape)}.")
        scales = self.raw_rms.to(device=bands.device, dtype=bands.dtype)
        return mean + (bands * scales[None, :, None, None]).sum(dim=1, keepdim=True)

    @torch.no_grad()
    def validate_numerics(self, x: Tensor) -> dict[str, float]:
        bands, mean = self.transform(x)
        reconstructed = self.inverse_transform(bands, mean)
        projected = self.project_bands(bands)
        return {
            "max_reconstruction_error": float((reconstructed - x).abs().max().item()),
            "max_projection_error": float((projected - bands).abs().max().item()),
            "mean_scale": float(self.mean_scale.item()),
            "sum_band_scale_squared": float(self.raw_rms.square().sum().item()),
        }
