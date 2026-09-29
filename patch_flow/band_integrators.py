"""Inference integrators for asynchronous frequency-band and mean flows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor
from tqdm import tqdm


class BandProjector(Protocol):
    num_bands: int

    def project_bands(self, bands: Tensor) -> Tensor: ...


def _validate_mean_source(mean: Tensor, batch: int) -> None:
    if mean.shape != (batch, 1, 1, 1):
        raise ValueError(f"Expected normalized mean source [B, 1, 1, 1], got {tuple(mean.shape)}.")


def _mean_time(band_time: Tensor) -> Tensor:
    return band_time.mean(dim=1).view(-1, 1, 1, 1)


def _advance_mean(
    mean_state: Tensor,
    mean_time: Tensor,
    mean_velocity: Tensor,
    next_band_time: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Euler-update the mean by its actual clock increment.

    A model evaluation does not imply a mean update.  If ``mean(band_time)``
    stays unchanged, ``delta_mean_time`` is zero and the mean state is held.
    """

    next_mean_time = _mean_time(next_band_time)
    delta_mean_time = next_mean_time - mean_time
    if bool((delta_mean_time < -2e-6).any()):
        raise RuntimeError("Mean time cannot move backwards.")
    delta_mean_time = delta_mean_time.clamp_min(0)
    mean_state = mean_state + delta_mean_time * mean_velocity
    return mean_state, next_mean_time, delta_mean_time


def _validate_model_outputs(
    velocity: Tensor,
    log_variance: Tensor,
    mean_velocity: Tensor,
    state: Tensor,
) -> None:
    if velocity.shape != state.shape:
        raise ValueError(f"Expected band velocity {tuple(state.shape)}, got {tuple(velocity.shape)}.")
    if log_variance.shape != state.shape[:2]:
        raise ValueError(
            f"Expected one uncertainty per band shaped {tuple(state.shape[:2])}, got {tuple(log_variance.shape)}."
        )
    if mean_velocity.shape != (state.shape[0], 1, 1, 1):
        raise ValueError(
            "Expected one scalar mean velocity per sample shaped "
            f"{(state.shape[0], 1, 1, 1)}, got {tuple(mean_velocity.shape)}."
        )


class BandEulerSampler:
    """Synchronous Euler integration for 16 bands and one scalar mean."""

    def __repr__(self) -> str:
        return "BandEuler"

    @torch.no_grad()
    def __call__(
        self,
        model,
        x: Tensor,
        mean: Tensor,
        timesteps: Sequence[float] | Tensor,
        decomposer: BandProjector,
        condition: Tensor,
        progress: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, int | float]]:
        if x.ndim != 4 or x.shape[1] != decomposer.num_bands:
            raise ValueError(f"Expected source [B, {decomposer.num_bands}, H, W], got {tuple(x.shape)}.")
        batch, num_bands = x.shape[:2]
        _validate_mean_source(mean, batch)
        timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
        if timesteps.ndim != 1 or timesteps.numel() < 2:
            raise ValueError("timesteps must be one-dimensional with at least two values.")
        if not torch.all(timesteps[1:] >= timesteps[:-1]):
            raise ValueError("timesteps must be monotonically non-decreasing.")
        if timesteps[0] < 0 or timesteps[0] >= 1 or timesteps[-1] != 1:
            raise ValueError("timesteps must start in [0, 1) and end at 1.")

        state = decomposer.project_bands(x)
        mean_state = mean
        start_time = timesteps[0]
        band_time = torch.full((batch, num_bands), start_time, device=x.device, dtype=x.dtype)
        mean_time = _mean_time(band_time)
        condition_features = model.encode_condition(condition)
        evaluations = 0
        mean_advances = 0
        iterator = zip(timesteps[:-1], timesteps[1:])
        for current, following in tqdm(iterator, total=timesteps.numel() - 1, disable=not progress):
            velocity, log_variance, mean_velocity = model(
                state,
                band_time,
                mean_state,
                mean_time,
                condition_features=condition_features,
                return_aux=True,
            )
            _validate_model_outputs(velocity, log_variance, mean_velocity, state)
            velocity = decomposer.project_bands(velocity)
            next_band_time = torch.full_like(band_time, following)
            step = next_band_time[:, :, None, None] - band_time[:, :, None, None]
            state = decomposer.project_bands(state + step * velocity)
            mean_state, mean_time, delta = _advance_mean(
                mean_state, mean_time, mean_velocity, next_band_time
            )
            mean_advances += int((delta > 0).sum().item())
            band_time = next_band_time
            evaluations += 1

        return state, mean_state, {
            "nfe": evaluations,
            "mean_time_advances": mean_advances,
            "final_mean_time": float(mean_time.min().item()),
        }


class BandDualLoopSampler:
    """Dual-loop integration using uncertainty only to rank the 16 bands."""

    def __init__(self, p: float = 0.4, n_inner: int = 4) -> None:
        if not 0 < p < 1:
            raise ValueError(f"p must lie in (0, 1), got {p}.")
        if n_inner < 1:
            raise ValueError(f"n_inner must be positive, got {n_inner}.")
        self.p = float(p)
        self.n_inner = int(n_inner)

    def __repr__(self) -> str:
        return f"BandDualLoop-p{self.p * 100:.0f}-inner{self.n_inner}"

    def compute_certain_mask(self, log_variance: Tensor) -> Tensor:
        if log_variance.ndim != 2:
            raise ValueError(f"Expected band log-variance [B, K], got {tuple(log_variance.shape)}.")
        threshold = torch.quantile(log_variance.double(), self.p, dim=1, keepdim=True)
        return log_variance < threshold.to(log_variance.dtype)

    @torch.no_grad()
    def __call__(
        self,
        model,
        x: Tensor,
        mean: Tensor,
        timesteps: Sequence[float] | Tensor,
        decomposer: BandProjector,
        condition: Tensor,
        progress: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, int | float]]:
        if x.ndim != 4 or x.shape[1] != decomposer.num_bands:
            raise ValueError(f"Expected source [B, {decomposer.num_bands}, H, W], got {tuple(x.shape)}.")
        batch, num_bands = x.shape[:2]
        _validate_mean_source(mean, batch)
        timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
        if timesteps.ndim != 1 or timesteps.numel() < 2:
            raise ValueError("timesteps must be one-dimensional with at least two values.")
        if not torch.all(timesteps[1:] >= timesteps[:-1]):
            raise ValueError("timesteps must be monotonically non-decreasing.")
        if timesteps[0] < 0 or timesteps[0] >= 1 or timesteps[-1] != 1:
            raise ValueError("timesteps must start in [0, 1) and end at 1.")

        state = decomposer.project_bands(x)
        mean_state = mean
        start_time = timesteps[0]
        band_time = torch.full((batch, num_bands), start_time, device=x.device, dtype=x.dtype)
        mean_time = _mean_time(band_time)
        condition_features = model.encode_condition(condition)
        evaluations = 0
        difficult_updates = 0
        mean_advances = 0
        zero_delta_mean_calls = 0
        iterator = zip(timesteps[:-1], timesteps[1:])

        for current, following in tqdm(iterator, total=timesteps.numel() - 1, disable=not progress):
            expected_current = torch.full_like(band_time, current)
            if not torch.allclose(band_time, expected_current, atol=2e-6, rtol=0):
                raise RuntimeError("Dual-loop band clocks did not meet at the macro-step boundary.")
            if not torch.allclose(mean_time, _mean_time(band_time), atol=2e-6, rtol=0):
                raise RuntimeError("Mean clock is inconsistent with the band clocks.")

            velocity, log_variance, mean_velocity = model(
                state,
                band_time,
                mean_state,
                mean_time,
                condition_features=condition_features,
                return_aux=True,
            )
            _validate_model_outputs(velocity, log_variance, mean_velocity, state)
            evaluations += 1
            velocity = decomposer.project_bands(velocity)
            certain = self.compute_certain_mask(log_variance)
            difficult = ~certain
            full_step = following - current
            inner_step = full_step / self.n_inner
            certain_image = certain[:, :, None, None]
            difficult_image = difficult[:, :, None, None]

            state = state + full_step * velocity * certain_image + inner_step * velocity * difficult_image
            next_band_time = band_time + full_step * certain + inner_step * difficult
            state = decomposer.project_bands(state)
            mean_state, mean_time, delta = _advance_mean(
                mean_state, mean_time, mean_velocity, next_band_time
            )
            mean_advances += int((delta > 0).sum().item())
            zero_delta_mean_calls += int((delta == 0).sum().item())
            band_time = next_band_time
            difficult_updates += int(difficult.sum().item())

            for _ in range(self.n_inner - 1):
                velocity, log_variance, mean_velocity = model(
                    state,
                    band_time,
                    mean_state,
                    mean_time,
                    condition_features=condition_features,
                    return_aux=True,
                )
                _validate_model_outputs(velocity, log_variance, mean_velocity, state)
                evaluations += 1
                velocity = decomposer.project_bands(velocity)
                state = decomposer.project_bands(state + inner_step * velocity * difficult_image)
                next_band_time = band_time + inner_step * difficult
                mean_state, mean_time, delta = _advance_mean(
                    mean_state, mean_time, mean_velocity, next_band_time
                )
                mean_advances += int((delta > 0).sum().item())
                zero_delta_mean_calls += int((delta == 0).sum().item())
                band_time = next_band_time
                difficult_updates += int(difficult.sum().item())

            expected_following = torch.full_like(band_time, following)
            if not torch.allclose(band_time, expected_following, atol=2e-6, rtol=0):
                raise RuntimeError("Dual-loop inner steps did not synchronize every frequency band.")
            band_time = expected_following
            mean_time = _mean_time(band_time)

        final_velocity, final_log_variance, final_mean_velocity = model(
            state,
            band_time,
            mean_state,
            mean_time,
            condition_features=condition_features,
            return_aux=True,
        )
        _validate_model_outputs(
            final_velocity, final_log_variance, final_mean_velocity, state
        )
        evaluations += 1
        return state, mean_state, {
            "nfe": evaluations,
            "difficult_band_updates": difficult_updates,
            "final_uncertainty_values": final_log_variance.numel(),
            "mean_time_advances": mean_advances,
            "zero_delta_mean_calls": zero_delta_mean_calls,
            "final_mean_time": float(mean_time.min().item()),
        }


class BandContinuousAsyncSampler:
    """Persistent asynchronous Euler integration with a bounded odds lead.

    Lower predicted log-variance bands receive larger proposed steps.  The
    proposal is then clipped by the R=2/4/2 odds schedule and the temporary
    t=0.95 leader ceiling.  Every unfinished difficult band retains enough
    base progress to reach t=1 within the fixed model-evaluation budget.
    """

    def __init__(
        self,
        start_ratio: float = 2.0,
        middle_ratio: float = 4.0,
        tail_ratio: float = 2.0,
        start_threshold: float = 0.2,
        tail_threshold: float = 0.8,
        leader_ceiling: float = 0.95,
        easy_multiplier: float = 2.0,
        middle_multiplier: float = 1.5,
    ) -> None:
        if not (1 <= start_ratio <= middle_ratio and 1 <= tail_ratio <= middle_ratio):
            raise ValueError("Invalid odds-ratio schedule.")
        if middle_ratio > 4:
            raise ValueError("The first-round middle odds ratio must not exceed 4.")
        if not 0 < start_threshold < tail_threshold < leader_ceiling < 1:
            raise ValueError("Invalid asynchronous phase thresholds.")
        if not 1 <= middle_multiplier <= easy_multiplier <= 2:
            raise ValueError("First-round single-step multipliers must lie in [1, 2].")
        self.start_ratio = float(start_ratio)
        self.middle_ratio = float(middle_ratio)
        self.tail_ratio = float(tail_ratio)
        self.start_threshold = float(start_threshold)
        self.tail_threshold = float(tail_threshold)
        self.leader_ceiling = float(leader_ceiling)
        self.easy_multiplier = float(easy_multiplier)
        self.middle_multiplier = float(middle_multiplier)

    def __repr__(self) -> str:
        return f"BandContinuousAsync-R{self.middle_ratio:g}"

    @staticmethod
    def _step_multipliers(log_variance: Tensor, easy: float, middle: float) -> Tensor:
        batch, bands = log_variance.shape
        order = log_variance.argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(bands, device=order.device)[None].expand(batch, -1))
        easy_count = max(1, bands // 4)
        middle_count = max(1, bands // 2)
        multipliers = torch.ones_like(log_variance)
        multipliers = torch.where(rank < easy_count, torch.full_like(multipliers, easy), multipliers)
        middle_mask = (rank >= easy_count) & (rank < easy_count + middle_count)
        return torch.where(middle_mask, torch.full_like(multipliers, middle), multipliers)

    @staticmethod
    def _clip_odds_span(candidate: Tensor, ratio: Tensor) -> Tensor:
        eps = torch.finfo(candidate.dtype).eps
        safe = candidate.clamp(eps, 1 - eps)
        odds = safe / (1 - safe)
        minimum = odds.min(dim=1, keepdim=True).values
        maximum_allowed = ratio[:, None] * minimum
        time_allowed = maximum_allowed / (1 + maximum_allowed)
        return torch.minimum(candidate, time_allowed)

    @torch.no_grad()
    def __call__(
        self,
        model,
        x: Tensor,
        mean: Tensor,
        timesteps: Sequence[float] | Tensor,
        decomposer: BandProjector,
        condition: Tensor,
        progress: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, int | float]]:
        if x.ndim != 4 or x.shape[1] != decomposer.num_bands:
            raise ValueError(f"Expected source [B, {decomposer.num_bands}, H, W], got {tuple(x.shape)}.")
        batch, num_bands = x.shape[:2]
        _validate_mean_source(mean, batch)
        timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
        if timesteps.ndim != 1 or timesteps.numel() < 2 or timesteps[0] != 0 or timesteps[-1] != 1:
            raise ValueError("timesteps must be one-dimensional, start at 0, and end at 1.")
        num_steps = timesteps.numel() - 1
        state = decomposer.project_bands(x)
        mean_state = mean
        band_time = torch.zeros(batch, num_bands, device=x.device, dtype=x.dtype)
        mean_time = _mean_time(band_time)
        condition_features = model.encode_condition(condition)
        mean_advances = 0
        paused_updates = 0
        maximum_observed_ratio = 1.0

        iterator = range(num_steps)
        for index in tqdm(iterator, total=num_steps, disable=not progress):
            velocity, log_variance, mean_velocity = model(
                state,
                band_time,
                mean_state,
                mean_time,
                condition_features=condition_features,
                return_aux=True,
            )
            _validate_model_outputs(velocity, log_variance, mean_velocity, state)
            velocity = decomposer.project_bands(velocity)
            remaining_calls = num_steps - index
            if remaining_calls == 1:
                next_band_time = torch.ones_like(band_time)
            else:
                base = (1.0 - band_time) / remaining_calls
                multipliers = self._step_multipliers(
                    log_variance.float(), self.easy_multiplier, self.middle_multiplier
                ).to(band_time.dtype)
                candidate = (band_time + base * multipliers).clamp_max(1.0)
                lagging = band_time.min(dim=1).values < self.tail_threshold
                ceiling = torch.where(
                    lagging[:, None],
                    torch.full_like(candidate, self.leader_ceiling),
                    torch.ones_like(candidate),
                )
                candidate = torch.minimum(candidate, ceiling)
                start = band_time.max(dim=1).values < self.start_threshold
                tail = band_time.min(dim=1).values >= self.tail_threshold
                ratio = torch.full((batch,), self.middle_ratio, device=x.device, dtype=x.dtype)
                ratio = torch.where(start, torch.full_like(ratio, self.start_ratio), ratio)
                ratio = torch.where(tail, torch.full_like(ratio, self.tail_ratio), ratio)
                next_band_time = self._clip_odds_span(candidate, ratio)
                next_band_time = torch.maximum(next_band_time, band_time)

            delta = next_band_time - band_time
            paused_updates += int((delta == 0).sum().item())
            state = decomposer.project_bands(state + delta[:, :, None, None] * velocity)
            mean_state, mean_time, mean_delta = _advance_mean(
                mean_state, mean_time, mean_velocity, next_band_time
            )
            mean_advances += int((mean_delta > 0).sum().item())
            band_time = next_band_time
            if index + 1 < num_steps:
                safe = band_time.clamp(1e-6, 1 - 1e-6)
                odds = safe / (1 - safe)
                observed = odds.max(dim=1).values / odds.min(dim=1).values
                maximum_observed_ratio = max(maximum_observed_ratio, float(observed.max().item()))

        return state, mean_state, {
            "nfe": int(num_steps),
            "mean_time_advances": mean_advances,
            "paused_band_updates": paused_updates,
            "maximum_observed_odds_ratio": maximum_observed_ratio,
            "final_mean_time": float(mean_time.min().item()),
        }


class BandMacroSyncAdaptiveSampler:
    """Bounded adaptive substepping with synchronization after every round.

    Low predicted uncertainty controls how many numerical substeps a band
    receives; it never removes the hard ``max_step`` limit.  In the adaptive
    middle stage, reliably low-uncertainty bands can finish a macro round in
    fewer updates and then pause while the remaining bands catch up.  All band
    clocks reach the macro endpoint through real integration before the next
    round begins.
    """

    def __init__(
        self,
        max_macro_step: float = 0.04,
        max_step: float = 0.02,
        difficult_substeps: int = 4,
        reliable_quantile: float = 0.25,
        max_reliable_logvar: float = -2.0,
        confidence_streak: int = 2,
        start_threshold: float = 0.2,
        tail_threshold: float = 0.8,
    ) -> None:
        if not 0 < max_step <= max_macro_step:
            raise ValueError("Expected 0 < max_step <= max_macro_step.")
        if difficult_substeps < 2:
            raise ValueError("difficult_substeps must be at least two.")
        if not 0 < reliable_quantile < 1:
            raise ValueError("reliable_quantile must lie in (0, 1).")
        if confidence_streak < 1:
            raise ValueError("confidence_streak must be positive.")
        if not 0 < start_threshold < tail_threshold < 1:
            raise ValueError("Expected 0 < start_threshold < tail_threshold < 1.")
        self.max_macro_step = float(max_macro_step)
        self.max_step = float(max_step)
        self.difficult_substeps = int(difficult_substeps)
        self.reliable_quantile = float(reliable_quantile)
        self.max_reliable_logvar = float(max_reliable_logvar)
        self.confidence_streak = int(confidence_streak)
        self.start_threshold = float(start_threshold)
        self.tail_threshold = float(tail_threshold)

    def __repr__(self) -> str:
        return "BandMacroSyncAdaptive"

    def _low_uncertainty(self, log_variance: Tensor) -> Tensor:
        band_count = log_variance.shape[1]
        reliable_count = max(1, int(math.ceil(self.reliable_quantile * band_count)))
        order = log_variance.argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(band_count, device=order.device)[None].expand_as(order),
        )
        return (rank < reliable_count) & (log_variance <= self.max_reliable_logvar)

    @torch.no_grad()
    def __call__(
        self,
        model,
        x: Tensor,
        mean: Tensor,
        timesteps: Sequence[float] | Tensor,
        decomposer: BandProjector,
        condition: Tensor,
        progress: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, int | float]]:
        if x.ndim != 4 or x.shape[1] != decomposer.num_bands:
            raise ValueError(f"Expected source [B, {decomposer.num_bands}, H, W], got {tuple(x.shape)}.")
        batch, num_bands = x.shape[:2]
        _validate_mean_source(mean, batch)
        timesteps = torch.as_tensor(timesteps, device=x.device, dtype=x.dtype)
        if timesteps.ndim != 1 or timesteps.numel() < 2 or timesteps[0] != 0 or timesteps[-1] != 1:
            raise ValueError("timesteps must be one-dimensional, start at 0, and end at 1.")
        if not torch.all(timesteps[1:] > timesteps[:-1]):
            raise ValueError("Macro timesteps must be strictly increasing.")
        if bool(((timesteps[1:] - timesteps[:-1]) > self.max_macro_step + 2e-6).any()):
            raise ValueError("A macro interval exceeds max_macro_step.")

        state = decomposer.project_bands(x)
        mean_state = mean
        band_time = torch.zeros(batch, num_bands, device=x.device, dtype=x.dtype)
        mean_time = _mean_time(band_time)
        condition_features = model.encode_condition(condition)
        streak = torch.zeros(batch, num_bands, device=x.device, dtype=torch.int64)
        evaluations = 0
        adaptive_rounds = 0
        reliable_decisions = 0
        paused_updates = 0
        mean_advances = 0
        maximum_step = 0.0
        maximum_time_gap = 0.0

        macro_iterator = zip(timesteps[:-1], timesteps[1:])
        for macro_start, macro_end in tqdm(
            macro_iterator,
            total=timesteps.numel() - 1,
            disable=not progress,
        ):
            expected_start = torch.full_like(band_time, macro_start)
            if not torch.allclose(band_time, expected_start, atol=2e-6, rtol=0):
                raise RuntimeError("Band clocks were not synchronized at a macro-round boundary.")
            interval = float((macro_end - macro_start).item())
            synchronized_phase = bool(
                macro_end <= self.start_threshold + 2e-6
                or macro_start >= self.tail_threshold - 2e-6
            )
            if synchronized_phase:
                # Ignore float32 boundary noise around exact multiples such as
                # 0.04 / 0.02; it must not create an accidental third substep.
                inner_calls = max(
                    1,
                    int(math.ceil((interval - 2e-6) / self.max_step)),
                )
            else:
                inner_calls = self.difficult_substeps
                adaptive_rounds += 1

            for inner_index in range(inner_calls):
                velocity, log_variance, mean_velocity = model(
                    state,
                    band_time,
                    mean_state,
                    mean_time,
                    condition_features=condition_features,
                    return_aux=True,
                )
                _validate_model_outputs(velocity, log_variance, mean_velocity, state)
                evaluations += 1
                velocity = decomposer.project_bands(velocity)
                low_uncertainty = self._low_uncertainty(log_variance.float())
                streak = torch.where(low_uncertainty, streak + 1, torch.zeros_like(streak))
                reliable = streak >= self.confidence_streak

                remaining_calls = inner_calls - inner_index
                remaining = torch.full_like(band_time, macro_end) - band_time
                base_step = remaining / remaining_calls
                if synchronized_phase:
                    proposed = base_step
                else:
                    multiplier = torch.where(
                        reliable,
                        torch.full_like(base_step, 2.0),
                        torch.ones_like(base_step),
                    )
                    proposed = base_step * multiplier
                    reliable_decisions += int(reliable.sum().item())
                delta = torch.minimum(proposed, remaining).clamp_min(0)
                delta = delta.clamp_max(self.max_step)
                if remaining_calls == 1 and bool((remaining > self.max_step + 2e-6).any()):
                    raise RuntimeError("The final substep would exceed max_step.")
                if remaining_calls == 1:
                    delta = remaining

                paused_updates += int((delta == 0).sum().item())
                maximum_step = max(maximum_step, float(delta.max().item()))
                next_band_time = band_time + delta
                state = decomposer.project_bands(
                    state + delta[:, :, None, None] * velocity
                )
                mean_state, mean_time, mean_delta = _advance_mean(
                    mean_state,
                    mean_time,
                    mean_velocity,
                    next_band_time,
                )
                mean_advances += int((mean_delta > 0).sum().item())
                band_time = next_band_time
                maximum_time_gap = max(
                    maximum_time_gap,
                    float((band_time.max(dim=1).values - band_time.min(dim=1).values).max().item()),
                )

            expected_end = torch.full_like(band_time, macro_end)
            if not torch.allclose(band_time, expected_end, atol=2e-6, rtol=0):
                raise RuntimeError("Every band must integrate to the macro endpoint before resynchronizing.")
            band_time = expected_end
            mean_time = _mean_time(band_time)

        return state, mean_state, {
            "nfe": evaluations,
            "macro_rounds": int(timesteps.numel() - 1),
            "adaptive_rounds": adaptive_rounds,
            "macro_resynchronizations": int(timesteps.numel() - 1),
            "reliable_band_decisions": reliable_decisions,
            "paused_band_updates": paused_updates,
            "mean_time_advances": mean_advances,
            "maximum_step": maximum_step,
            "maximum_time_gap": maximum_time_gap,
            "final_mean_time": float(mean_time.min().item()),
        }
