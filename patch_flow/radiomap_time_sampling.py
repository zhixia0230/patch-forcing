"""Structured time-vector sampling matched to continuous asynchronous inference."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class StructuredBandTimeSampler(nn.Module):
    """Sample the 20/20/25/15/20 RadioMap training-time mixture.

    Every non-endpoint row obeys an odds span of at most ``max_ratio``.  The
    pause/catch-up branch samples a state visited while lagging bands close an
    existing lead, rather than equating a pause action with tied clock values.
    """

    branch_names = ("start", "synchronous", "leading", "catchup", "tail")

    def __init__(
        self,
        num_bands: int = 16,
        probabilities: tuple[float, ...] = (0.20, 0.20, 0.25, 0.15, 0.20),
        max_ratio: float = 4.0,
        low_frequency_probability: float = 0.5,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_bands < 2:
            raise ValueError("Structured sampling requires at least two bands.")
        if len(probabilities) != 5 or any(value < 0 for value in probabilities):
            raise ValueError("probabilities must contain five non-negative values.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("probabilities must sum to one.")
        if max_ratio < 1 or max_ratio > 4:
            raise ValueError("The first-round max_ratio must lie in [1, 4].")
        if not 0 <= low_frequency_probability <= 1:
            raise ValueError("low_frequency_probability must lie in [0, 1].")
        if not 0 < epsilon < 0.01:
            raise ValueError("epsilon must lie in (0, 0.01).")
        self.num_bands = int(num_bands)
        self.max_ratio = float(max_ratio)
        self.low_frequency_probability = float(low_frequency_probability)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "cumulative_probabilities",
            torch.tensor(probabilities, dtype=torch.float64).cumsum(0),
            persistent=False,
        )

    @staticmethod
    def _times_from_cap_and_log_ratio(cap: Tensor, log_ratio: Tensor) -> Tensor:
        ratio = log_ratio.exp()
        return cap / (cap + (1.0 - cap) * ratio)

    def _controlled_log_ratios(
        self,
        ratio: Tensor,
        *,
        grouped: bool,
    ) -> Tensor:
        """Create a continuous row with exact ratio endpoints."""

        device, dtype = ratio.device, ratio.dtype
        maximum = ratio.log()
        if not grouped:
            values = torch.rand(self.num_bands, device=device, dtype=dtype) * maximum
            values[0] = 0
            values[-1] = maximum
            return values

        num_leaders = int(torch.randint(1, self.num_bands // 2 + 1, (), device=device).item())
        if bool(torch.rand((), device=device) < self.low_frequency_probability):
            leader_indices = torch.arange(num_leaders, device=device)
        else:
            leader_indices = torch.randperm(self.num_bands, device=device)[:num_leaders]
        leader_mask = torch.zeros(self.num_bands, device=device, dtype=torch.bool)
        leader_mask[leader_indices] = True
        values = torch.empty(self.num_bands, device=device, dtype=dtype)
        values[leader_mask] = torch.rand(num_leaders, device=device, dtype=dtype) * (0.25 * maximum)
        num_lagging = self.num_bands - num_leaders
        values[~leader_mask] = (0.5 + 0.5 * torch.rand(num_lagging, device=device, dtype=dtype)) * maximum
        values[leader_indices[0]] = 0
        lagging_indices = torch.nonzero(~leader_mask, as_tuple=False).flatten()
        values[lagging_indices[0]] = maximum
        return values

    @torch.no_grad()
    def sample_with_branches(
        self,
        shape,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if len(shape) != 2 or int(shape[1]) != self.num_bands:
            raise ValueError(f"Expected time shape [B, {self.num_bands}], got {shape}.")
        batch_size = int(shape[0])
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        device = torch.device(device)
        draws = torch.rand(batch_size, device=device, dtype=torch.float64)
        boundaries = self.cumulative_probabilities.to(device)
        branches = torch.bucketize(draws, boundaries[:-1]).to(torch.int64)
        output = torch.empty(batch_size, self.num_bands, device=device, dtype=dtype)

        for row in range(batch_size):
            branch = int(branches[row].item())
            one = torch.ones((), device=device, dtype=dtype)
            if branch == 0:  # all bands remain in the noisy start region
                cap = self.epsilon + torch.rand((), device=device, dtype=dtype) * (0.2 - self.epsilon)
                ratio = one + torch.rand((), device=device, dtype=dtype)
                log_ratios = self._controlled_log_ratios(ratio, grouped=False)
            elif branch == 1:  # exact synchronization or a narrow neighborhood
                cap = self.epsilon + torch.rand((), device=device, dtype=dtype) * (1 - 2 * self.epsilon)
                if bool(torch.rand((), device=device) < 0.5):
                    output[row].fill_(cap)
                    continue
                ratio = one + torch.rand((), device=device, dtype=dtype)
                log_ratios = self._controlled_log_ratios(ratio, grouped=False)
            elif branch == 2:  # persistent, continuously valued leading state
                cap = 0.2 + torch.rand((), device=device, dtype=dtype) * 0.75
                ratio = torch.exp(torch.rand((), device=device, dtype=dtype) * math.log(self.max_ratio))
                log_ratios = self._controlled_log_ratios(ratio, grouped=True)
            elif branch == 3:  # a point visited during pause and catch-up
                cap = 0.2 + torch.rand((), device=device, dtype=dtype) * 0.75
                initial_ratio = 2.0 * torch.exp(
                    torch.rand((), device=device, dtype=dtype) * math.log(self.max_ratio / 2.0)
                )
                catch_progress = torch.rand((), device=device, dtype=dtype)
                ratio = torch.exp((1.0 - catch_progress) * initial_ratio.log())
                log_ratios = self._controlled_log_ratios(ratio, grouped=True)
            else:  # all clocks are in the common refinement tail
                cap = 0.9 + torch.rand((), device=device, dtype=dtype) * (0.1 - self.epsilon)
                ratio = one + torch.rand((), device=device, dtype=dtype)
                log_ratios = self._controlled_log_ratios(ratio, grouped=False)
            output[row] = self._times_from_cap_and_log_ratio(cap, log_ratios)
        return output, branches

    def forward(self, shape, device="cpu", dtype=torch.float32) -> Tensor:
        times, _ = self.sample_with_branches(shape, device=device, dtype=dtype)
        return times


class MacroSynchronizedTimeSampler(nn.Module):
    """Sample clocks visited by bounded, round-synchronized substepping.

    The asynchronous branch represents a single macro round of width
    ``macro_step``.  Reliable bands may progress twice as far as lagging bands
    inside the round, but the lead is temporary and never exceeds half a macro
    step.  Start, macro-boundary, and tail samples are exactly synchronized.
    """

    branch_names = ("start_sync", "middle_sync", "temporary_lead", "tail_sync")

    def __init__(
        self,
        num_bands: int = 16,
        probabilities: tuple[float, ...] = (0.20, 0.20, 0.40, 0.20),
        macro_step: float = 0.04,
        start_threshold: float = 0.2,
        tail_threshold: float = 0.8,
        max_leaders: int = 4,
        low_frequency_probability: float = 0.5,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_bands < 2:
            raise ValueError("Macro-synchronized sampling requires at least two bands.")
        if len(probabilities) != 4 or any(value < 0 for value in probabilities):
            raise ValueError("probabilities must contain four non-negative values.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("probabilities must sum to one.")
        if not 0 < macro_step < tail_threshold - start_threshold:
            raise ValueError("macro_step must fit inside the adaptive middle interval.")
        if not 0 < start_threshold < tail_threshold < 1:
            raise ValueError("Expected 0 < start_threshold < tail_threshold < 1.")
        if not 1 <= max_leaders < num_bands:
            raise ValueError("max_leaders must lie in [1, num_bands).")
        if not 0 <= low_frequency_probability <= 1:
            raise ValueError("low_frequency_probability must lie in [0, 1].")
        if not 0 < epsilon < min(0.01, macro_step):
            raise ValueError("epsilon must be positive and smaller than macro_step.")
        self.num_bands = int(num_bands)
        self.macro_step = float(macro_step)
        self.start_threshold = float(start_threshold)
        self.tail_threshold = float(tail_threshold)
        self.max_leaders = int(max_leaders)
        self.low_frequency_probability = float(low_frequency_probability)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "cumulative_probabilities",
            torch.tensor(probabilities, dtype=torch.float64).cumsum(0),
            persistent=False,
        )

    def _leader_indices(self, device: torch.device) -> Tensor:
        count = int(torch.randint(1, self.max_leaders + 1, (), device=device).item())
        if bool(torch.rand((), device=device) < self.low_frequency_probability):
            return torch.arange(count, device=device)
        return torch.randperm(self.num_bands, device=device)[:count]

    @torch.no_grad()
    def sample_with_branches(
        self,
        shape,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if len(shape) != 2 or int(shape[1]) != self.num_bands:
            raise ValueError(f"Expected time shape [B, {self.num_bands}], got {shape}.")
        batch_size = int(shape[0])
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        device = torch.device(device)
        draws = torch.rand(batch_size, device=device, dtype=torch.float64)
        branches = torch.bucketize(
            draws, self.cumulative_probabilities.to(device)[:-1]
        ).to(torch.int64)
        output = torch.empty(batch_size, self.num_bands, device=device, dtype=dtype)

        for row in range(batch_size):
            branch = int(branches[row].item())
            if branch == 0:
                time = self.epsilon + torch.rand((), device=device, dtype=dtype) * (
                    self.start_threshold - self.epsilon
                )
                output[row].fill_(time)
            elif branch == 1:
                time = self.start_threshold + torch.rand((), device=device, dtype=dtype) * (
                    self.tail_threshold - self.start_threshold
                )
                output[row].fill_(time)
            elif branch == 2:
                macro_start = self.start_threshold + torch.rand((), device=device, dtype=dtype) * (
                    self.tail_threshold - self.start_threshold - self.macro_step
                )
                progress = self.epsilon + torch.rand((), device=device, dtype=dtype) * (
                    self.macro_step - self.epsilon
                )
                lagging_time = macro_start + progress
                leading_time = macro_start + torch.minimum(
                    2.0 * progress,
                    torch.tensor(self.macro_step, device=device, dtype=dtype),
                )
                output[row].fill_(lagging_time)
                output[row, self._leader_indices(device)] = leading_time
            else:
                time = self.tail_threshold + torch.rand((), device=device, dtype=dtype) * (
                    1.0 - self.epsilon - self.tail_threshold
                )
                output[row].fill_(time)
        return output, branches

    def forward(self, shape, device="cpu", dtype=torch.float32) -> Tensor:
        times, _ = self.sample_with_branches(shape, device=device, dtype=dtype)
        return times
