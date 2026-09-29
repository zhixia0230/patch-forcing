"""Conditional U-Net for asynchronous RadioMap frequency-band flow matching."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class BandTimeEmbedding(nn.Module):
    def __init__(self, num_bands: int, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even.")
        self.num_bands = int(num_bands)
        half = embedding_dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.band_embedding = nn.Parameter(torch.randn(num_bands, embedding_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(num_bands * embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, times: Tensor) -> Tensor:
        if times.ndim != 2 or times.shape[1] != self.num_bands:
            raise ValueError(f"Expected band times [B, {self.num_bands}], got {tuple(times.shape)}.")
        angles = times[..., None] * self.frequencies.to(times.dtype) * (2 * math.pi)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        embedding = embedding + self.band_embedding[None].to(embedding.dtype)
        return self.mlp(embedding.flatten(1))


class MeanStateEmbedding(nn.Module):
    """Encode the current noisy scalar mean and its average band clock."""

    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even.")
        half = embedding_dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + 1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, state: Tensor, time: Tensor) -> Tensor:
        batch = state.shape[0]
        expected = (batch, 1, 1, 1)
        if state.shape != expected or time.shape != expected:
            raise ValueError(f"Expected mean state/time {expected}, got {tuple(state.shape)} and {tuple(time.shape)}.")
        flat_time = time.reshape(batch, 1)
        angles = flat_time[..., None] * self.frequencies.to(flat_time.dtype) * (2 * math.pi)
        time_embedding = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return self.mlp(torch.cat((time_embedding, state.reshape(batch, 1).to(time_embedding.dtype)), dim=1))


class ConditionResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SpatialConditionPyramid(nn.Module):
    """Four-scale building/TX encoder with raw inputs repeated at each scale."""

    def __init__(self, channels: Sequence[int] = (32, 64, 128, 128)) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("The condition pyramid must have four resolutions.")
        self.channels = tuple(int(value) for value in channels)
        self.stem = nn.Conv2d(2, self.channels[0], 3, padding=1)
        self.blocks = nn.ModuleList([ConditionResBlock(value) for value in self.channels])
        self.down = nn.ModuleList(
            [
                nn.Conv2d(self.channels[index], self.channels[index + 1], 3, stride=2, padding=1)
                for index in range(3)
            ]
        )
        self.merge = nn.ModuleList(
            [nn.Conv2d(self.channels[index] + 2, self.channels[index], 3, padding=1) for index in range(1, 4)]
        )

    @staticmethod
    def _raw_at_size(condition: Tensor, size: tuple[int, int]) -> Tensor:
        building = F.adaptive_avg_pool2d(condition[:, :1], size)
        transmitter = F.adaptive_max_pool2d(condition[:, 1:2], size)
        return torch.cat((building, transmitter), dim=1)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if condition.ndim != 4 or condition.shape[1] != 2:
            raise ValueError(f"Expected building/TX condition [B, 2, H, W], got {tuple(condition.shape)}.")
        features: list[Tensor] = []
        h = self.blocks[0](self.stem(condition))
        features.append(h)
        for level in range(1, 4):
            h = self.down[level - 1](h)
            raw = self._raw_at_size(condition, h.shape[-2:])
            h = self.blocks[level](self.merge[level - 1](torch.cat((h, raw), dim=1)))
            features.append(h)
        return tuple(features)  # type: ignore[return-value]


class ModulatedResBlock(nn.Module):
    """Residual block independently modulated by band times and mean state."""

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(embedding_dim, 2 * out_channels)
        self.mean_projection = nn.Linear(embedding_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor, time_embedding: Tensor, mean_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        time_scale, time_shift = self.time_projection(F.silu(time_embedding)).chunk(2, dim=1)
        mean_scale, mean_shift = self.mean_projection(F.silu(mean_embedding)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + time_scale[:, :, None, None] + mean_scale[:, :, None, None])
        h = h + time_shift[:, :, None, None] + mean_shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


def _zero_projection(in_channels: int, out_channels: int) -> nn.Conv2d:
    projection = nn.Conv2d(in_channels, out_channels, 1)
    nn.init.zeros_(projection.weight)
    nn.init.zeros_(projection.bias)
    return projection


class FrequencyBandUNet(nn.Module):
    """Four-resolution U-Net with 14 spatial-condition injection points."""

    def __init__(
        self,
        num_bands: int = 16,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        condition_channels: Sequence[int] = (32, 64, 128, 128),
        time_embedding_dim: int = 32,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive.")
        if len(channel_multipliers) != 4 or len(condition_channels) != 4:
            raise ValueError("The RadioMap model requires four resolution channel entries.")
        self.num_bands = int(num_bands)
        self.predict_uncertainty = True
        self.predict_mean_uncertainty = False
        self.gradient_checkpointing = bool(gradient_checkpointing)
        channels = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        condition_channels = [int(value) for value in condition_channels]
        embedding_dim = base_channels * 4

        self.time_embedding = BandTimeEmbedding(num_bands, time_embedding_dim, embedding_dim)
        self.mean_embedding = MeanStateEmbedding(time_embedding_dim, embedding_dim)
        self.condition_encoder = SpatialConditionPyramid(condition_channels)
        self.input = nn.Conv2d(num_bands, channels[0], 3, padding=1)

        self.encoder = nn.ModuleList()
        for level in range(3):
            self.encoder.append(
                nn.ModuleDict(
                    {
                        "res1": ModulatedResBlock(channels[level], channels[level], embedding_dim, dropout),
                        "res2": ModulatedResBlock(channels[level], channels[level], embedding_dim, dropout),
                        "cond1": _zero_projection(condition_channels[level], channels[level]),
                        "cond2": _zero_projection(condition_channels[level], channels[level]),
                        "down": Downsample(channels[level], channels[level + 1]),
                    }
                )
            )

        self.middle1 = ModulatedResBlock(channels[3], channels[3], embedding_dim, dropout)
        self.middle2 = ModulatedResBlock(channels[3], channels[3], embedding_dim, dropout)
        self.middle_cond1 = _zero_projection(condition_channels[3], channels[3])
        self.middle_cond2 = _zero_projection(condition_channels[3], channels[3])

        self.decoder = nn.ModuleList()
        for level in reversed(range(3)):
            self.decoder.append(
                nn.ModuleDict(
                    {
                        "up": Upsample(channels[level + 1], channels[level]),
                        "res1": ModulatedResBlock(2 * channels[level], channels[level], embedding_dim, dropout),
                        "res2": ModulatedResBlock(channels[level], channels[level], embedding_dim, dropout),
                        "cond1": _zero_projection(condition_channels[level], channels[level]),
                        "cond2": _zero_projection(condition_channels[level], channels[level]),
                    }
                )
            )

        self.output_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.velocity_head = nn.Conv2d(channels[0], num_bands, 3, padding=1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.SiLU(),
            nn.Linear(channels[0], num_bands),
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)
        self.mean_velocity_head = nn.Sequential(
            nn.Linear(channels[3] + embedding_dim, channels[3]),
            nn.SiLU(),
            nn.Linear(channels[3], 1),
        )

    def encode_condition(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.condition_encoder(condition)

    def _run(self, module: nn.Module, *inputs: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(
        self,
        x: Tensor,
        band_time: Tensor,
        mean_state: Tensor,
        mean_time: Tensor,
        condition: Tensor | None = None,
        condition_features: Sequence[Tensor] | None = None,
        return_uncertainty: bool = False,
        return_aux: bool = False,
    ):
        if x.ndim != 4 or x.shape[1] != self.num_bands:
            raise ValueError(f"Expected x [B, {self.num_bands}, H, W], got {tuple(x.shape)}.")
        batch = x.shape[0]
        if band_time.shape != (batch, self.num_bands):
            raise ValueError(f"Expected band_time {(batch, self.num_bands)}, got {tuple(band_time.shape)}.")
        if mean_state.shape != (batch, 1, 1, 1) or mean_time.shape != (batch, 1, 1, 1):
            raise ValueError("Mean state and mean time must have shape [B, 1, 1, 1].")
        expected_mean_time = band_time.mean(dim=1).view(batch, 1, 1, 1)
        if not torch.allclose(mean_time, expected_mean_time, atol=2e-6, rtol=0):
            raise ValueError("mean_time must equal mean(band_time) for every sample.")
        if condition_features is None:
            if condition is None:
                raise ValueError("Either condition or condition_features must be provided.")
            condition_features = self.encode_condition(condition)
        if len(condition_features) != 4:
            raise ValueError("condition_features must contain four spatial scales.")

        time_embedding = self.time_embedding(band_time)
        mean_embedding = self.mean_embedding(mean_state, mean_time)
        h = self.input(x)
        skips: list[Tensor] = []
        for level, stage in enumerate(self.encoder):
            h = self._run(stage["res1"], h, time_embedding, mean_embedding)
            h = h + stage["cond1"](condition_features[level])
            h = self._run(stage["res2"], h, time_embedding, mean_embedding)
            h = h + stage["cond2"](condition_features[level])
            skips.append(h)
            h = stage["down"](h)

        h = self._run(self.middle1, h, time_embedding, mean_embedding)
        h = h + self.middle_cond1(condition_features[3])
        h = self._run(self.middle2, h, time_embedding, mean_embedding)
        h = h + self.middle_cond2(condition_features[3])
        bottleneck = h

        for stage, level in zip(self.decoder, reversed(range(3)), strict=True):
            h = stage["up"](h, skips[level].shape[-2:])
            h = torch.cat((h, skips[level]), dim=1)
            h = self._run(stage["res1"], h, time_embedding, mean_embedding)
            h = h + stage["cond1"](condition_features[level])
            h = self._run(stage["res2"], h, time_embedding, mean_embedding)
            h = h + stage["cond2"](condition_features[level])

        h = F.silu(self.output_norm(h))
        velocity = self.velocity_head(h)
        log_variance = self.uncertainty_head(h.mean(dim=(-2, -1)))
        mean_features = torch.cat((bottleneck.mean(dim=(-2, -1)), F.silu(mean_embedding)), dim=1)
        mean_velocity = self.mean_velocity_head(mean_features).view(batch, 1, 1, 1)
        if return_aux:
            return velocity, log_variance, mean_velocity
        if return_uncertainty:
            return velocity, log_variance
        return velocity
