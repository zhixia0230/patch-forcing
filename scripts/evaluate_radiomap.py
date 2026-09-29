"""Evaluate a RadioMap checkpoint with complete flow-based generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from jutils import instantiate_from_config  # noqa: E402
from patch_flow.band_integrators import (  # noqa: E402
    BandContinuousAsyncSampler,
    BandEulerSampler,
    BandMacroSyncAdaptiveSampler,
)
from patch_flow.radiomap_metrics import RadioMapMetricAccumulator  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RadioMaps and report RMSE, NMSE, MAE, PSNR, and spatial/band metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved training config. Defaults to <checkpoint>/../../config.yaml.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--sampler",
        choices=("euler", "async-r4", "macro-sync-adaptive"),
        default="euler",
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate an evenly spaced subset; omit to evaluate the complete split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tx-window", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--clip-output", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _evenly_spaced_subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be positive when provided.")
    indices = torch.linspace(0, len(dataset) - 1, steps=max_samples).round().to(torch.int64).tolist()
    return Subset(dataset, indices)


def _sampler(name: str):
    if name == "euler":
        return BandEulerSampler()
    if name == "async-r4":
        return BandContinuousAsyncSampler()
    if name == "macro-sync-adaptive":
        return BandMacroSyncAdaptiveSampler()
    raise ValueError(f"Unknown sampler: {name}")


def main() -> None:
    args = _parse_args()
    if args.num_steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("num_steps and batch_size must be positive; num_workers must be non-negative.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else checkpoint_path.parent.parent / "config.yaml"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 evaluation currently requires a CUDA device; use --precision fp32 on CPU.")

    torch.manual_seed(args.seed)
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    data = instantiate_from_config(cfg.data)
    data.setup(None)
    dataset = _evenly_spaced_subset(data.datasets[args.split], args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    module = instantiate_from_config(cfg.trainer)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module = module.to(device).eval()
    sampler = _sampler(args.sampler)
    tracker = RadioMapMetricAccumulator(
        num_bands=module.decomposer.num_bands,
        tx_window=args.tx_window,
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )
    diagnostics: list[dict[str, int | float]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            target = batch[module.input_key].float()
            target_bands, _ = module.decomposer.transform(target)
            fork_devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + batch_index)
                with autocast:
                    prediction, predicted_bands, predicted_mean, batch_diagnostics = module.sample(
                        batch,
                        sampler,
                        num_steps=args.num_steps,
                        progress=False,
                    )
            if args.clip_output:
                prediction = prediction.clamp(0, 1)
                # The generated bands no longer reconstruct a clipped image, so
                # recompute them before reporting band metrics.
                predicted_bands, predicted_mean = module.decomposer.transform(prediction.float())
            tracker.update(
                prediction.float(),
                target,
                building=batch[module.building_key],
                tx=batch[module.tx_key],
                predicted_mean=predicted_mean.float(),
                predicted_bands=predicted_bands.float(),
                target_bands=target_bands,
                band_scales=module.decomposer.raw_rms,
            )
            diagnostics.append(batch_diagnostics)
            print(f"Evaluated {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "evaluation": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint.get("global_step", -1)),
            "config": str(config_path),
            "split": args.split,
            "sampler": str(sampler),
            "num_steps": args.num_steps,
            "seed": args.seed,
            "precision": args.precision,
            "clip_output": args.clip_output,
            "elapsed_seconds": elapsed,
            "seconds_per_sample": elapsed / len(dataset),
        },
        "metrics": tracker.compute(),
        "sampler_diagnostics": {
            "nfe_per_batch": sorted({int(item["nfe"]) for item in diagnostics}),
            "maximum_observed_odds_ratio": max(
                (float(item.get("maximum_observed_odds_ratio", 1.0)) for item in diagnostics),
                default=1.0,
            ),
            "final_mean_time_min": min(
                (float(item["final_mean_time"]) for item in diagnostics), default=float("nan")
            ),
        },
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
