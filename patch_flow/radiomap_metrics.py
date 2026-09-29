"""Streaming metrics for generated RadioMap samples.

The primary RMSE and NMSE values are micro-averages: squared errors and
target energies are accumulated over the complete evaluation set before the
ratio is computed.  Per-sample averages are reported separately so the two
aggregation conventions cannot be confused.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_image_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            "prediction and target must have identical [B, 1, H, W] shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.shape[0] == 0:
        raise ValueError("Metric batches must not be empty.")


def _error_summary(
    squared_error: float,
    absolute_error: float,
    target_energy: float,
    signed_error: float,
    count: int,
    eps: float,
    *,
    include_psnr: bool = False,
) -> dict[str, float | int]:
    if count < 1:
        raise RuntimeError("Cannot compute metrics without observations.")
    mse = squared_error / count
    nmse = squared_error / max(target_energy, eps)
    result: dict[str, float | int] = {
        "count": count,
        "rmse": math.sqrt(mse),
        "mae": absolute_error / count,
        "bias": signed_error / count,
        "nmse": nmse,
        "nrmse": math.sqrt(nmse),
        "nmse_db": 10.0 * math.log10(max(nmse, eps)),
    }
    if include_psnr:
        # RadioMap PNG values are represented on the [0, 1] scale.
        result["psnr_db"] = 10.0 * math.log10(1.0 / max(mse, eps))
    return result


class RadioMapMetricAccumulator:
    """Accumulate full-map, spatial-region, scalar-mean, and band metrics."""

    def __init__(self, num_bands: int = 16, tx_window: int = 11, eps: float = 1e-12) -> None:
        if num_bands < 1:
            raise ValueError("num_bands must be positive.")
        if tx_window < 1 or tx_window % 2 == 0:
            raise ValueError("tx_window must be a positive odd integer.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.num_bands = int(num_bands)
        self.tx_window = int(tx_window)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.num_samples = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.signed_error = 0.0
        self.target_energy = 0.0
        self.pixel_count = 0
        self.sample_rmse_sum = 0.0
        self.sample_nmse_sum = 0.0
        self.sample_pearson_sum = 0.0
        self.below_zero = 0
        self.above_one = 0

        self.mean_squared_error = 0.0
        self.mean_absolute_error = 0.0
        self.mean_signed_error = 0.0
        self.mean_target_energy = 0.0
        self.mean_count = 0

        self.region_sums = {
            "building_boundary": dict(squared=0.0, absolute=0.0, signed=0.0, energy=0.0, count=0),
            "tx_neighborhood": dict(squared=0.0, absolute=0.0, signed=0.0, energy=0.0, count=0),
        }
        self.band_squared_error = torch.zeros(self.num_bands, dtype=torch.float64)
        self.band_absolute_error = torch.zeros(self.num_bands, dtype=torch.float64)
        self.band_signed_error = torch.zeros(self.num_bands, dtype=torch.float64)
        self.band_target_energy = torch.zeros(self.num_bands, dtype=torch.float64)
        self.band_count = torch.zeros(self.num_bands, dtype=torch.int64)

    @staticmethod
    def _accumulate_region(store: dict[str, float | int], error: Tensor, target: Tensor, mask: Tensor) -> None:
        selected_error = error[mask]
        if selected_error.numel() == 0:
            return
        selected_target = target[mask]
        store["squared"] += selected_error.square().sum().item()
        store["absolute"] += selected_error.abs().sum().item()
        store["signed"] += selected_error.sum().item()
        store["energy"] += selected_target.square().sum().item()
        store["count"] += selected_error.numel()

    @torch.no_grad()
    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        building: Tensor | None = None,
        tx: Tensor | None = None,
        predicted_mean: Tensor | None = None,
        predicted_bands: Tensor | None = None,
        target_bands: Tensor | None = None,
        band_scales: Tensor | None = None,
    ) -> None:
        _validate_image_pair(prediction, target)
        prediction = prediction.detach().double()
        target = target.detach().double()
        error = prediction - target
        batch = prediction.shape[0]

        self.num_samples += batch
        self.squared_error += error.square().sum().item()
        self.absolute_error += error.abs().sum().item()
        self.signed_error += error.sum().item()
        self.target_energy += target.square().sum().item()
        self.pixel_count += error.numel()
        self.below_zero += (prediction < 0).sum().item()
        self.above_one += (prediction > 1).sum().item()

        flat_error = error.flatten(1)
        flat_prediction = prediction.flatten(1)
        flat_target = target.flatten(1)
        sample_squared_error = flat_error.square().sum(dim=1)
        sample_target_energy = flat_target.square().sum(dim=1)
        self.sample_rmse_sum += torch.sqrt(sample_squared_error / flat_error.shape[1]).sum().item()
        self.sample_nmse_sum += (sample_squared_error / sample_target_energy.clamp_min(self.eps)).sum().item()
        centered_prediction = flat_prediction - flat_prediction.mean(dim=1, keepdim=True)
        centered_target = flat_target - flat_target.mean(dim=1, keepdim=True)
        denominator = (
            centered_prediction.square().sum(dim=1).sqrt()
            * centered_target.square().sum(dim=1).sqrt()
        )
        correlation = (centered_prediction * centered_target).sum(dim=1) / denominator.clamp_min(self.eps)
        correlation = torch.where(denominator > self.eps, correlation, torch.zeros_like(correlation))
        self.sample_pearson_sum += correlation.sum().item()

        target_mean = target.mean(dim=(-2, -1), keepdim=True)
        if predicted_mean is None:
            predicted_mean = prediction.mean(dim=(-2, -1), keepdim=True)
        if predicted_mean.shape != target_mean.shape:
            raise ValueError(
                f"predicted_mean must have shape {tuple(target_mean.shape)}, got {tuple(predicted_mean.shape)}."
            )
        mean_error = predicted_mean.detach().double() - target_mean
        self.mean_squared_error += mean_error.square().sum().item()
        self.mean_absolute_error += mean_error.abs().sum().item()
        self.mean_signed_error += mean_error.sum().item()
        self.mean_target_energy += target_mean.square().sum().item()
        self.mean_count += mean_error.numel()

        if building is not None:
            if building.shape != target.shape:
                raise ValueError("building must have the same shape as target.")
            building = building.detach().float()
            dilated = F.max_pool2d(building, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool2d(-building, kernel_size=3, stride=1, padding=1)
            self._accumulate_region(
                self.region_sums["building_boundary"], error, target, (dilated - eroded) > 0
            )

        if tx is not None:
            if tx.shape != target.shape:
                raise ValueError("tx must have the same shape as target.")
            radius = self.tx_window // 2
            tx_mask = F.max_pool2d(
                tx.detach().float(), kernel_size=self.tx_window, stride=1, padding=radius
            ) > 0
            self._accumulate_region(self.region_sums["tx_neighborhood"], error, target, tx_mask)

        band_arguments = (predicted_bands, target_bands, band_scales)
        if any(value is not None for value in band_arguments):
            if any(value is None for value in band_arguments):
                raise ValueError(
                    "predicted_bands, target_bands, and band_scales must be provided together."
                )
            expected = (batch, self.num_bands, *target.shape[-2:])
            if predicted_bands.shape != expected or target_bands.shape != expected:
                raise ValueError(f"Band tensors must have shape {expected}.")
            if band_scales.numel() != self.num_bands:
                raise ValueError(f"Expected {self.num_bands} band scales.")
            scales = band_scales.detach().double().reshape(1, self.num_bands, 1, 1)
            raw_prediction = predicted_bands.detach().double() * scales
            raw_target = target_bands.detach().double() * scales
            band_error = raw_prediction - raw_target
            reduce_dimensions = (0, 2, 3)
            self.band_squared_error += band_error.square().sum(reduce_dimensions).cpu()
            self.band_absolute_error += band_error.abs().sum(reduce_dimensions).cpu()
            self.band_signed_error += band_error.sum(reduce_dimensions).cpu()
            self.band_target_energy += raw_target.square().sum(reduce_dimensions).cpu()
            observations = batch * target.shape[-2] * target.shape[-1]
            self.band_count += observations

    def compute(self) -> dict[str, object]:
        radiomap = _error_summary(
            self.squared_error,
            self.absolute_error,
            self.target_energy,
            self.signed_error,
            self.pixel_count,
            self.eps,
            include_psnr=True,
        )
        radiomap.update(
            {
                "sample_rmse_mean": self.sample_rmse_sum / self.num_samples,
                "sample_nmse_mean": self.sample_nmse_sum / self.num_samples,
                "sample_pearson_mean": self.sample_pearson_sum / self.num_samples,
            }
        )
        result: dict[str, object] = {
            "num_samples": self.num_samples,
            "radiomap": radiomap,
            "mean": _error_summary(
                self.mean_squared_error,
                self.mean_absolute_error,
                self.mean_target_energy,
                self.mean_signed_error,
                self.mean_count,
                self.eps,
            ),
            "range": {
                "below_zero_fraction": self.below_zero / self.pixel_count,
                "above_one_fraction": self.above_one / self.pixel_count,
            },
        }
        for name, values in self.region_sums.items():
            if values["count"]:
                result[name] = _error_summary(
                    values["squared"],
                    values["absolute"],
                    values["energy"],
                    values["signed"],
                    values["count"],
                    self.eps,
                )
        if bool((self.band_count > 0).any()):
            result["bands"] = [
                {
                    "band": index,
                    **_error_summary(
                        self.band_squared_error[index].item(),
                        self.band_absolute_error[index].item(),
                        self.band_target_energy[index].item(),
                        self.band_signed_error[index].item(),
                        int(self.band_count[index].item()),
                        self.eps,
                    ),
                }
                for index in range(self.num_bands)
            ]
        return result
