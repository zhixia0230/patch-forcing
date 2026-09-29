"""Trainer for conditional 16-band and scalar-mean RadioMap flow matching."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor, nn

from jutils import instantiate_from_config, load_partial_from_config

from patch_flow.band_flow import FrequencyBandFlow


def _instantiate_module(component, name: str) -> nn.Module:
    if isinstance(component, nn.Module):
        return component
    if isinstance(component, (dict, DictConfig)):
        result = instantiate_from_config(component)
        if isinstance(result, nn.Module):
            return result
    raise TypeError(f"{name} must instantiate to nn.Module.")


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer)


class RadiomapBandFlowTrainer(LightningModule):
    """Train band velocities/uncertainties and one scalar mean velocity."""

    def __init__(
        self,
        model,
        decomposer,
        flow: FrequencyBandFlow | dict | DictConfig,
        input_key: str = "image",
        building_key: str = "building",
        tx_key: str = "tx",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ema_rate: float = 0.999,
        uncertainty_weight: float = 0.01,
        mean_weight: float = 0.2,
        logvar_min: float = -12.0,
        logvar_max: float = 8.0,
        validation_seed: int = 42,
        lr_scheduler_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = _instantiate_module(model, "model")
        self.decomposer = _instantiate_module(decomposer, "decomposer")
        self.flow = (
            flow
            if isinstance(flow, FrequencyBandFlow)
            else _instantiate_module(flow, "flow")
        )
        if not isinstance(self.flow, FrequencyBandFlow):
            raise TypeError("flow must be FrequencyBandFlow.")
        expected_bands = self.flow.num_bands
        if self.model.num_bands != expected_bands or self.decomposer.num_bands != expected_bands:
            raise ValueError("Model, decomposer, and flow must use the same number of bands.")
        self.decomposer.requires_grad_(False)
        self.decomposer.eval()

        self.input_key = str(input_key)
        self.building_key = str(building_key)
        self.tx_key = str(tx_key)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.ema_rate = float(ema_rate)
        self.uncertainty_weight = float(uncertainty_weight)
        self.mean_weight = float(mean_weight)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.validation_seed = int(validation_seed)
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.ema_model = deepcopy(self.model) if self.ema_rate > 0 else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            _update_ema(self.ema_model, self.model, decay=0.0)
        self._validation_outputs: list[dict[str, Tensor]] = []

    def train(self, mode: bool = True):
        super().train(mode)
        self.decomposer.eval()
        if self.ema_model is not None:
            self.ema_model.eval()
        return self

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        result = {"optimizer": optimizer}
        if self.lr_scheduler_cfg is not None:
            result["lr_scheduler"] = load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
        return result

    def _condition(self, batch: Mapping[str, Tensor]) -> Tensor:
        for key in (self.building_key, self.tx_key):
            if key not in batch:
                raise KeyError(f"Batch does not contain required condition {key!r}.")
        building, transmitter = batch[self.building_key], batch[self.tx_key]
        if building.ndim != 4 or transmitter.shape != building.shape or building.shape[1] != 1:
            raise ValueError("Building and TX conditions must both have shape [B, 1, H, W].")
        return torch.cat((building.float(), transmitter.float()), dim=1)

    def compute_losses(
        self,
        batch: Mapping[str, Tensor],
        *,
        band_source: Tensor | None = None,
        mean_source: Tensor | None = None,
        band_times: Tensor | None = None,
        model: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        if self.input_key not in batch:
            raise KeyError(f"Batch does not contain {self.input_key!r}.")
        radiomap = batch[self.input_key].float()
        if radiomap.ndim != 4 or radiomap.shape[1] != 1:
            raise ValueError(f"Expected scalar radiomap [B, 1, H, W], got {tuple(radiomap.shape)}.")
        condition = self._condition(batch)
        target_bands, pixel_mean = self.decomposer.transform(radiomap)
        normalized_mean = self.decomposer.normalize_mean(pixel_mean)
        if band_source is None:
            band_source = self.decomposer.band_limited_noise(
                radiomap.shape[0], device=radiomap.device, dtype=radiomap.dtype
            )
        noisy_bands, target_velocity, band_times = self.flow.get_interpolants(
            x1=target_bands, x0=band_source, t=band_times
        )
        if mean_source is None:
            mean_source = torch.randn_like(normalized_mean)
        noisy_mean, target_mean_velocity, mean_time = self.flow.get_mean_interpolants(
            mean1=normalized_mean, mean0=mean_source, band_times=band_times
        )
        prediction_model = self.model if model is None else model
        predicted_velocity, log_variance, predicted_mean_velocity = prediction_model(
            noisy_bands,
            band_times,
            noisy_mean,
            mean_time,
            condition=condition,
            return_aux=True,
        )
        predicted_velocity = self.decomposer.project_bands(predicted_velocity)
        band_mse = (predicted_velocity - target_velocity).square().mean(dim=(-2, -1))
        flow_loss = band_mse.mean()
        if log_variance.shape != band_mse.shape:
            raise ValueError(f"Expected logvar {tuple(band_mse.shape)}, got {tuple(log_variance.shape)}.")
        log_variance = log_variance.float().clamp(self.logvar_min, self.logvar_max)
        uncertainty_loss = 0.5 * (
            torch.exp(-log_variance) * band_mse.detach() + log_variance
        ).mean()
        if predicted_mean_velocity.shape != target_mean_velocity.shape:
            raise ValueError("The mean head must predict one scalar velocity per sample.")
        mean_loss = (predicted_mean_velocity - target_mean_velocity).square().mean()
        loss = flow_loss + self.mean_weight * mean_loss + self.uncertainty_weight * uncertainty_loss
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "uncertainty_loss": uncertainty_loss,
            "mean_loss": mean_loss,
            "band_mse": band_mse.mean(dim=0),
            "mean_time": mean_time.detach(),
        }

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        losses = self.compute_losses(batch)
        return losses["loss"], {
            "flow_loss": losses["flow_loss"],
            "uncertainty_loss": losses["uncertainty_loss"],
            "mean_loss": losses["mean_loss"],
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        del outputs, batch, batch_idx
        if self.ema_model is not None:
            _update_ema(self.ema_model, self.model, self.ema_rate)

    @torch.no_grad()
    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> None:
        device = batch[self.input_key].device
        fork_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.validation_seed + batch_idx)
            validation_model = self.ema_model if self.ema_model is not None else self.model
            losses = self.compute_losses(batch, model=validation_model)
        self._validation_outputs.append(
            {name: value.detach().cpu() for name, value in losses.items()}
        )

    @torch.no_grad()
    def on_validation_epoch_end(self) -> dict[str, Tensor] | None:
        if not self._validation_outputs:
            return None
        metrics: dict[str, Tensor] = {}
        for name in ("loss", "flow_loss", "uncertainty_loss", "mean_loss"):
            value = torch.stack([output[name] for output in self._validation_outputs]).mean()
            self.log(f"val/{name}", value)
            metrics[f"val/{name}"] = value
        core = metrics["val/flow_loss"] + self.mean_weight * metrics["val/mean_loss"]
        self.log("val/core_loss", core)
        metrics["val/core_loss"] = core
        band_mse = torch.stack([output["band_mse"] for output in self._validation_outputs]).mean(0)
        for index, value in enumerate(band_mse):
            self.log(f"val/band_{index:02d}_mse", value)
        self._validation_outputs.clear()
        return metrics

    @torch.no_grad()
    def sample(self, batch: Mapping[str, Tensor], sampler, num_steps: int = 50, progress: bool = False):
        reference = batch[self.input_key]
        condition = self._condition(batch).to(reference.device)
        model = self.ema_model if self.ema_model is not None else self.model
        source = self.decomposer.band_limited_noise(
            reference.shape[0], device=reference.device, dtype=reference.dtype
        )
        mean_source = torch.randn(
            reference.shape[0], 1, 1, 1, device=reference.device, dtype=reference.dtype
        )
        timesteps = torch.linspace(0, 1, num_steps + 1, device=reference.device, dtype=reference.dtype)
        bands, normalized_mean, diagnostics = sampler(
            model=model,
            x=source,
            mean=mean_source,
            timesteps=timesteps,
            decomposer=self.decomposer,
            condition=condition,
            progress=progress,
        )
        pixel_mean = self.decomposer.denormalize_mean(normalized_mean)
        radiomap = self.decomposer.inverse_transform(bands, pixel_mean)
        return radiomap, bands, pixel_mean, diagnostics
