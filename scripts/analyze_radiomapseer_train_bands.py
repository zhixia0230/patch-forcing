"""Analyze fixed non-DC RadioMapSeer bands using training data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import gammaln
from scipy.stats import chi2
import torch
from torch.utils.data import DataLoader, Dataset


PIXEL_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
SAMPLE_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
SNR_TIMES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/rw/Downloads/RadioMapSeer"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("/home/rw/radiomap1/artifacts/prepared_dpm_core_seed42/splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/radiomapseer_train_bands_seed42"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pixel-samples-per-batch", type=int, default=256)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class DPMTrainingDataset(Dataset):
    def __init__(self, dpm_root: Path, scene_ids: list[int], size: int = 256) -> None:
        self.dpm_root = dpm_root
        self.size = int(size)
        self.samples = [
            (int(scene_id), tx_id, dpm_root / f"{scene_id}_{tx_id}.png")
            for scene_id in scene_ids
            for tx_id in range(80)
        ]
        missing = [str(path) for _, _, path in self.samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} DPM files; first entries: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        scene_id, tx_id, path = self.samples[index]
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (self.size, self.size):
                raise ValueError(f"Expected L/{self.size}x{self.size}, got {image.mode}/{image.size}: {path}")
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array), index, scene_id, tx_id


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fit_sequential_bands(shell_energy: np.ndarray, num_bands: int) -> list[tuple[int, int, float, float]]:
    bands: list[tuple[int, int, float, float]] = []
    start = 0
    remaining_bands = num_bands
    while remaining_bands > 1:
        target = float(shell_energy[start:].sum() / remaining_bands)
        last_legal = len(shell_energy) - remaining_bands
        cumulative = np.cumsum(shell_energy[start : last_legal + 1])
        end = start + int(np.argmin(np.abs(cumulative - target)))
        actual = float(shell_energy[start : end + 1].sum())
        bands.append((start, end, actual, target))
        start = end + 1
        remaining_bands -= 1
    actual = float(shell_energy[start:].sum())
    bands.append((start, len(shell_energy) - 1, actual, actual))
    return bands


def quantile_dict(values: np.ndarray, quantiles: tuple[float, ...], prefix: str) -> dict[str, float]:
    result = np.quantile(values, quantiles)
    return {
        f"{prefix}_p{int(round(q * 1000)):03d}": float(value)
        for q, value in zip(quantiles, result, strict=True)
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_one(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float64).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.float64)[None, None] / 255.0


def decompose(x: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    centered = x - mean
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
    band_spectra = spectrum * masks[None]
    complex_bands = torch.fft.ifft2(
        torch.fft.ifftshift(band_spectra, dim=(-2, -1)), norm="ortho"
    )
    return complex_bands, mean


def plot_masks(masks: np.ndarray, rows: list[dict], output: Path) -> None:
    num_bands = len(rows)
    columns = min(4, num_bands)
    plot_rows = math.ceil(num_bands / columns)
    figure, axes = plt.subplots(
        plot_rows, columns, figsize=(3 * columns, 3 * plot_rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat[:num_bands]):
        axis.imshow(masks[index], cmap="gray", interpolation="nearest")
        row = rows[index]
        axis.set_title(
            f"band {index:02d}: q={row['q_min']}-{row['q_max']}\nN={row['frequency_count']:,}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[num_bands:]:
        axis.axis("off")
    figure.suptitle(f"{num_bands} fixed non-DC radial FFT masks (fftshift layout)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_energy(rows: list[dict], output: Path) -> None:
    bands = np.arange(len(rows))
    reference = np.array([row["reference_ac_fraction"] for row in rows]) * 100
    pooled = np.array([row["pooled_ac_fraction"] for row in rows]) * 100
    rms = np.array([row["raw_rms"] for row in rows])
    counts = np.array([row["frequency_count"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    width = 0.38
    axes[0].bar(bands - width / 2, reference, width, label="per-map normalized AC")
    axes[0].bar(bands + width / 2, pooled, width, label="pooled raw AC")
    nominal_fraction = 100.0 / len(rows)
    axes[0].axhline(
        nominal_fraction,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"nominal {nominal_fraction:.2f}%",
    )
    axes[0].set_ylabel("AC energy (%)")
    axes[0].legend(fontsize=8)
    axes[1].bar(bands, rms)
    axes[1].set_ylabel("raw band RMS")
    axes[2].bar(bands, counts)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("frequency points (log scale)")
    for axis in axes:
        axis.set_xlabel("band")
        axis.set_xticks(bands)
        axis.tick_params(axis="x", labelrotation=90)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Training-set band energy, scale, and frequency support")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_rms_distributions(per_sample_rms: np.ndarray, raw_rms: np.ndarray, output: Path) -> None:
    normalized = per_sample_rms / raw_rms[None]
    bands = np.arange(raw_rms.size)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, values, title, reference in (
        (axes[0], per_sample_rms, "Raw per-sample band RMS", raw_rms),
        (
            axes[1],
            normalized,
            "Per-sample RMS after fixed train-RMS normalization",
            np.ones(raw_rms.size),
        ),
    ):
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        axis.fill_between(bands, q05, q95, alpha=0.2, label="p05-p95")
        axis.fill_between(bands, q25, q75, alpha=0.35, label="p25-p75")
        axis.plot(bands, q50, marker="o", label="median")
        axis.plot(bands, reference, linestyle="--", marker=".", label="pooled RMS")
        axis.set_title(title)
        axis.set_xlabel("band")
        axis.set_xticks(bands)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_pixel_distributions(pixel_samples: np.ndarray, raw_rms: np.ndarray, output: Path) -> None:
    bands = np.arange(raw_rms.size)
    normalized = pixel_samples / raw_rms[:, None]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, values, title in (
        (axes[0], pixel_samples, "Sampled raw spatial-band values"),
        (axes[1], normalized, "Sampled values after fixed train-RMS normalization"),
    ):
        q01, q05, q25, q50, q75, q95, q99 = np.quantile(
            values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], axis=1
        )
        axis.fill_between(bands, q01, q99, alpha=0.15, label="p01-p99")
        axis.fill_between(bands, q05, q95, alpha=0.25, label="p05-p95")
        axis.fill_between(bands, q25, q75, alpha=0.4, label="p25-p75")
        axis.plot(bands, q50, color="black", marker=".", label="median")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("band")
        axis.set_xticks(bands)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_noise(rows: list[dict], output: Path) -> None:
    bands = np.arange(len(rows))
    factor = np.array([row["noise_equalization_factor"] for row in rows])
    p05 = np.array([row["unit_noise_rms_p05"] for row in rows])
    p50 = np.array([row["unit_noise_rms_p50"] for row in rows])
    p95 = np.array([row["unit_noise_rms_p95"] for row in rows])
    raw_scale = np.array([row["recommended_raw_noise_rms"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(bands, factor)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sqrt{HW/N_k}$ (log scale)")
    axes[1].fill_between(bands, p05, p95, alpha=0.25, label="p05-p95")
    axes[1].plot(bands, p50, marker="o", label="median")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, label=r"$E[RMS^2]=1$")
    axes[1].set_ylabel("equalized unit-noise RMS")
    axes[1].legend(fontsize=8)
    axes[2].bar(bands, raw_scale)
    axes[2].set_ylabel("recommended raw-space noise RMS")
    for axis in axes:
        axis.set_xlabel("band")
        axis.set_xticks(bands)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Band-limited noise calibration")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_snr(snr_rows: list[dict], output: Path) -> None:
    num_bands = max(int(row["band"]) for row in snr_rows) + 1
    matrix = np.empty((num_bands, len(SNR_TIMES)), dtype=np.float64)
    for row in snr_rows:
        matrix[int(row["band"]), SNR_TIMES.index(float(row["time"]))] = row["snr_db_p50"]
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-20, vmax=20)
    axis.set_xticks(np.arange(len(SNR_TIMES)), labels=[str(value) for value in SNR_TIMES])
    axis.set_yticks(
        np.arange(num_bands), labels=[f"{index:02d}" for index in range(num_bands)]
    )
    axis.set_xlabel("flow time t")
    axis.set_ylabel("band")
    axis.set_title("Median realized SNR after fixed RMS normalization")
    figure.colorbar(image, ax=axis, label="SNR (dB)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_example(
    image: np.ndarray,
    centered: np.ndarray,
    bands: np.ndarray,
    reconstructed: np.ndarray,
    pixel_rows: list[dict],
    title: str,
    output: Path,
) -> None:
    num_bands = bands.shape[0]
    error = np.abs(reconstructed - image)
    panels = [(image, "GT [0,1]", "viridis", 0.0, 1.0)]
    center_limit = max(abs(float(centered.min())), abs(float(centered.max())), 1e-12)
    panels.append((centered, "GT - spatial mean", "coolwarm", -center_limit, center_limit))
    for index in range(num_bands):
        limit = max(
            abs(pixel_rows[index]["pixel_sample_p010"]),
            abs(pixel_rows[index]["pixel_sample_p990"]),
            1e-8,
        )
        panels.append((bands[index], f"band {index:02d}", "coolwarm", -limit, limit))
    panels.append((reconstructed, "reconstruction", "viridis", 0.0, 1.0))
    panels.append((error, "absolute error", "magma", 0.0, max(float(error.max()), 1e-15)))

    columns = 4 if len(panels) <= 12 else 5
    plot_rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        plot_rows, columns, figsize=(3.2 * columns, 3 * plot_rows), squeeze=False
    )
    for axis, (values, panel_title, cmap, vmin, vmax) in zip(axes.flat, panels):
        image_artist = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")
        figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.02)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.pixel_samples_per_batch < 1
        or args.num_bands < 1
    ):
        raise ValueError("Invalid batch/worker/sample configuration.")
    num_bands = int(args.num_bands)

    split_bytes = args.split_manifest.read_bytes()
    splits = json.loads(split_bytes)
    old_train = [int(value) for value in splits["train"]]
    old_validation = [int(value) for value in splits["val"]]
    old_test = [int(value) for value in splits["test"]]
    train_scene_ids = old_train + old_validation
    if len(old_train) != 500 or len(old_validation) != 100 or len(old_test) != 100:
        raise ValueError("Authoritative manifest must contain the expected 500/100/100 split.")
    if len(train_scene_ids) != 600 or len(set(train_scene_ids)) != 600:
        raise ValueError("Derived training split must contain 600 unique scenes.")
    if set(train_scene_ids) & set(old_test):
        raise ValueError("Training scenes overlap the authoritative test scenes.")

    dpm_root = args.dataset_root / "gain" / "DPM"
    dataset = DPMTrainingDataset(dpm_root, train_scene_ids)
    if len(dataset) != 48_000:
        raise ValueError(f"Expected 48,000 training samples, got {len(dataset)}.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    pin_memory = device.type == "cuda"
    height = width = 256
    pixels = height * width

    print("Pass 1/2: fitting training-only non-DC radial boundaries", flush=True)
    reference_sum = torch.zeros(height, width, dtype=torch.float64, device=device)
    raw_power_sum = torch.zeros_like(reference_sum)
    mean_sum = torch.zeros((), dtype=torch.float64, device=device)
    mean_square_sum = torch.zeros_like(mean_sum)
    image_square_sum = torch.zeros_like(mean_sum)
    valid_samples = 0
    zero_ac_samples = 0
    processed = 0
    for raw, _, _, _ in make_loader(dataset, args.batch_size, args.num_workers, pin_memory):
        x = raw.to(device=device, dtype=torch.float64, non_blocking=True).div_(255.0).unsqueeze(1)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        centered = x - mean
        spectrum = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
        power = spectrum.abs().square().squeeze(1)
        power[:, height // 2, width // 2] = 0
        ac_energy = power.sum(dim=(-2, -1))
        valid = ac_energy > 0
        if valid.any():
            reference_sum += (power[valid] / ac_energy[valid, None, None]).sum(dim=0)
            raw_power_sum += power[valid].sum(dim=0)
            valid_samples += int(valid.sum().item())
        zero_ac_samples += int((~valid).sum().item())
        flat_mean = mean.flatten()
        mean_sum += flat_mean.sum()
        mean_square_sum += flat_mean.square().sum()
        image_square_sum += x.square().sum()
        processed += x.shape[0]
        if processed % 8_000 == 0:
            print(f"  pass 1: {processed}/{len(dataset)}", flush=True)

    if valid_samples == 0:
        raise RuntimeError("All training radiomaps have zero non-DC energy.")
    reference_spectrum = (reference_sum / valid_samples).cpu().numpy()
    raw_power_sum_np = raw_power_sum.cpu().numpy()
    reference_spectrum[height // 2, width // 2] = 0
    reference_spectrum /= reference_spectrum.sum()

    frequency = np.rint(np.fft.fftshift(np.fft.fftfreq(height)) * height).astype(np.int64)
    q = frequency[:, None] ** 2 + frequency[None, :] ** 2
    shell_q, shell_inverse = np.unique(q, return_inverse=True)
    shell_energy = np.bincount(
        shell_inverse.reshape(-1), weights=reference_spectrum.reshape(-1)
    )
    if shell_q[0] != 0:
        raise RuntimeError("The first radial shell is not DC.")
    active_q = shell_q[1:]
    active_energy = shell_energy[1:]
    active_energy /= active_energy.sum()
    fitted = fit_sequential_bands(active_energy, num_bands=num_bands)
    shell_index = shell_inverse.reshape(height, width)
    masks = np.stack(
        [(shell_index >= start + 1) & (shell_index <= end + 1) for start, end, _, _ in fitted]
    )
    coverage = masks.sum(axis=0)
    if not np.array_equal(coverage, (q > 0).astype(coverage.dtype)):
        raise RuntimeError("Masks do not cover every non-DC frequency exactly once.")
    if np.any(masks.sum(axis=(1, 2)) == 0):
        raise RuntimeError("At least one fitted mask is empty.")
    unshifted = np.fft.ifftshift(masks, axes=(-2, -1))
    negative = (-np.arange(height)) % height
    if not np.array_equal(unshifted, unshifted[:, negative][:, :, negative]):
        raise RuntimeError("A fitted mask is not conjugate symmetric.")

    raw_band_energy = np.array([raw_power_sum_np[mask].sum() for mask in masks])
    raw_rms = np.sqrt(raw_band_energy / (len(dataset) * pixels))
    raw_ac_total = float(raw_band_energy.sum())
    image_ms = float(image_square_sum.item() / (len(dataset) * pixels))
    mean_center = float(mean_sum.item() / len(dataset))
    mean_ms = float(mean_square_sum.item() / len(dataset))
    mean_scale = math.sqrt(max(0.0, mean_ms - mean_center**2))
    ac_ms = raw_ac_total / (len(dataset) * pixels)
    parseval_error = abs(image_ms - mean_ms - ac_ms)

    print("Pass 2/2: exact per-sample RMS and spatial-band distributions", flush=True)
    num_samples = len(dataset)
    num_batches = math.ceil(num_samples / args.batch_size)
    sample_band_rms = np.empty((num_samples, num_bands), dtype=np.float32)
    sample_energy_fraction = np.empty((num_samples, num_bands), dtype=np.float32)
    sample_mean = np.empty(num_samples, dtype=np.float32)
    sample_ac_rms = np.empty(num_samples, dtype=np.float32)
    sample_total_rms = np.empty(num_samples, dtype=np.float32)
    sample_scene = np.empty(num_samples, dtype=np.int32)
    sample_tx = np.empty(num_samples, dtype=np.int16)
    pixel_samples = np.empty(
        (num_bands, num_batches * args.pixel_samples_per_batch), dtype=np.float32
    )
    pixel_min = np.full(num_bands, np.inf, dtype=np.float64)
    pixel_max = np.full(num_bands, -np.inf, dtype=np.float64)
    pixel_sum = np.zeros(num_bands, dtype=np.float64)
    pixel_abs_sum = np.zeros(num_bands, dtype=np.float64)
    pixel_positive = np.zeros(num_bands, dtype=np.int64)
    pixel_negative = np.zeros(num_bands, dtype=np.int64)
    pixel_near_zero = np.zeros(num_bands, dtype=np.int64)
    total_band_values = 0
    max_imaginary = 0.0
    max_reconstruction_error = 0.0
    reconstruction_square_error = 0.0
    reconstruction_values = 0
    uint8_failures = 0
    masks_t = torch.from_numpy(masks).to(device=device)
    masks_double = masks_t.to(torch.float64)
    sample_generator = torch.Generator(device=device).manual_seed(20260927)
    processed = 0
    for batch_index, (raw, indices, scenes, tx_ids) in enumerate(
        make_loader(dataset, args.batch_size, args.num_workers, pin_memory)
    ):
        x = raw.to(device=device, dtype=torch.float64, non_blocking=True).div_(255.0).unsqueeze(1)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        centered = x - mean
        spectrum = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
        power = spectrum.abs().square().squeeze(1)
        power[:, height // 2, width // 2] = 0
        ac_energy = power.sum(dim=(-2, -1))
        band_energy = torch.einsum("bhw,khw->bk", power, masks_double)
        band_rms = torch.sqrt(band_energy / pixels)
        array_indices = indices.numpy()
        sample_band_rms[array_indices] = band_rms.cpu().numpy().astype(np.float32)
        sample_energy_fraction[array_indices] = (
            band_energy / ac_energy[:, None]
        ).cpu().numpy().astype(np.float32)
        sample_mean[array_indices] = mean.flatten().cpu().numpy().astype(np.float32)
        sample_ac_rms[array_indices] = torch.sqrt(ac_energy / pixels).cpu().numpy().astype(np.float32)
        sample_total_rms[array_indices] = torch.sqrt(
            x.square().sum(dim=(-2, -1)).flatten() / pixels
        ).cpu().numpy().astype(np.float32)
        sample_scene[array_indices] = scenes.numpy().astype(np.int32)
        sample_tx[array_indices] = tx_ids.numpy().astype(np.int16)

        band_spectra = spectrum * masks_t[None]
        complex_bands = torch.fft.ifft2(
            torch.fft.ifftshift(band_spectra, dim=(-2, -1)), norm="ortho"
        )
        max_imaginary = max(max_imaginary, float(complex_bands.imag.abs().max().item()))
        bands = complex_bands.real
        batch_count = bands.shape[0]
        batch_values = batch_count * pixels
        pixel_min = np.minimum(pixel_min, bands.amin(dim=(0, 2, 3)).cpu().numpy())
        pixel_max = np.maximum(pixel_max, bands.amax(dim=(0, 2, 3)).cpu().numpy())
        pixel_sum += bands.sum(dim=(0, 2, 3)).cpu().numpy()
        pixel_abs_sum += bands.abs().sum(dim=(0, 2, 3)).cpu().numpy()
        pixel_positive += (bands > 0).sum(dim=(0, 2, 3)).cpu().numpy()
        pixel_negative += (bands < 0).sum(dim=(0, 2, 3)).cpu().numpy()
        pixel_near_zero += (bands.abs() <= 1e-8).sum(dim=(0, 2, 3)).cpu().numpy()
        total_band_values += batch_values

        flat_count = batch_count * pixels
        positions = torch.randint(
            flat_count,
            (args.pixel_samples_per_batch,),
            device=device,
            generator=sample_generator,
        )
        sample_start = batch_index * args.pixel_samples_per_batch
        sample_stop = sample_start + args.pixel_samples_per_batch
        for band_index in range(num_bands):
            pixel_samples[band_index, sample_start:sample_stop] = (
                bands[:, band_index].reshape(-1)[positions].cpu().numpy().astype(np.float32)
            )

        reconstruction = bands.sum(dim=1, keepdim=True) + mean
        error = reconstruction - x
        max_reconstruction_error = max(max_reconstruction_error, float(error.abs().max().item()))
        reconstruction_square_error += float(error.square().sum().item())
        reconstruction_values += error.numel()
        recovered = torch.round(reconstruction * 255.0).to(torch.int16)
        uint8_failures += int((recovered != raw.to(device=device, dtype=torch.int16).unsqueeze(1)).sum().item())

        processed += batch_count
        if processed % 4_000 == 0:
            print(f"  pass 2: {processed}/{len(dataset)}", flush=True)

    reconstruction_rmse = math.sqrt(reconstruction_square_error / reconstruction_values)
    if uint8_failures:
        raise RuntimeError(f"Float64 reconstruction failed for {uint8_failures} uint8 pixels.")

    pixel_rows: list[dict] = []
    snr_rows: list[dict] = []
    rng = np.random.default_rng(42)
    for band_index, (start, end, reference_fraction, target) in enumerate(fitted):
        count = int(masks[band_index].sum())
        rms_values = sample_band_rms[:, band_index].astype(np.float64)
        normalized_rms = rms_values / raw_rms[band_index]
        energy_values = sample_energy_fraction[:, band_index].astype(np.float64)
        samples = pixel_samples[band_index].astype(np.float64)
        noise_rms_samples = np.sqrt(rng.chisquare(count, size=num_samples) / count)
        noise_mean = math.sqrt(2.0 / count) * math.exp(
            gammaln((count + 1) / 2) - gammaln(count / 2)
        )
        noise_std = math.sqrt(max(0.0, 1.0 - noise_mean**2))
        noise_p01, noise_p05, noise_p50, noise_p95, noise_p99 = np.sqrt(
            chi2.ppf([0.01, 0.05, 0.5, 0.95, 0.99], count) / count
        )
        equalization_factor = math.sqrt(pixels / count)
        row = {
            "band": band_index,
            "q_min": int(active_q[start]),
            "q_max": int(active_q[end]),
            "radius_min_pixels": float(math.sqrt(active_q[start])),
            "radius_max_pixels": float(math.sqrt(active_q[end])),
            "radius_min_cycles_per_pixel": float(math.sqrt(active_q[start]) / height),
            "radius_max_cycles_per_pixel": float(math.sqrt(active_q[end]) / height),
            "shell_count": int(end - start + 1),
            "frequency_count": count,
            "reference_ac_fraction": reference_fraction,
            "sequential_target_fraction": target,
            "reference_target_deviation": reference_fraction - target,
            "pooled_ac_fraction": float(raw_band_energy[band_index] / raw_ac_total),
            "total_image_energy_fraction": float(
                (raw_band_energy[band_index] / (num_samples * pixels)) / image_ms
            ),
            "raw_rms": float(raw_rms[band_index]),
            "sample_rms_mean": float(rms_values.mean()),
            "sample_rms_std": float(rms_values.std()),
            "normalized_sample_rms_mean": float(normalized_rms.mean()),
            "normalized_sample_rms_std": float(normalized_rms.std()),
            "pixel_exact_min": float(pixel_min[band_index]),
            "pixel_exact_max": float(pixel_max[band_index]),
            "pixel_exact_mean": float(pixel_sum[band_index] / total_band_values),
            "pixel_exact_abs_mean": float(pixel_abs_sum[band_index] / total_band_values),
            "pixel_positive_fraction": float(pixel_positive[band_index] / total_band_values),
            "pixel_negative_fraction": float(pixel_negative[band_index] / total_band_values),
            "pixel_near_zero_fraction_1e-8": float(pixel_near_zero[band_index] / total_band_values),
            "projected_white_noise_expected_rms": float(math.sqrt(count / pixels)),
            "noise_equalization_factor": equalization_factor,
            "unit_noise_rms_mean": noise_mean,
            "unit_noise_rms_std": noise_std,
            "unit_noise_rms_p01": float(noise_p01),
            "unit_noise_rms_p05": float(noise_p05),
            "unit_noise_rms_p50": float(noise_p50),
            "unit_noise_rms_p95": float(noise_p95),
            "unit_noise_rms_p99": float(noise_p99),
            "recommended_raw_noise_rms": float(raw_rms[band_index]),
            "recommended_white_noise_multiplier": float(raw_rms[band_index] * equalization_factor),
        }
        row.update(quantile_dict(rms_values, SAMPLE_QUANTILES, "sample_rms"))
        row.update(quantile_dict(normalized_rms, SAMPLE_QUANTILES, "normalized_sample_rms"))
        row.update(quantile_dict(energy_values, SAMPLE_QUANTILES, "sample_ac_fraction"))
        row.update(quantile_dict(samples, PIXEL_QUANTILES, "pixel_sample"))
        pixel_rows.append(row)

        for time in SNR_TIMES:
            signal_amplitude = time * normalized_rms
            noise_amplitude = (1 - time) * noise_rms_samples
            snr = 20 * np.log10(np.maximum(signal_amplitude, 1e-300) / noise_amplitude)
            finite = np.isfinite(snr)
            snr_rows.append(
                {
                    "band": band_index,
                    "time": time,
                    "snr_db_p05": float(np.quantile(snr[finite], 0.05)),
                    "snr_db_p50": float(np.quantile(snr[finite], 0.5)),
                    "snr_db_p95": float(np.quantile(snr[finite], 0.95)),
                    "finite_fraction": float(finite.mean()),
                }
            )

    train_scene_hash = canonical_hash(train_scene_ids)
    mask_hash = hashlib.sha256(
        masks.astype(np.uint8).tobytes()
        + train_scene_hash.encode()
        + (
            f"raw_dpm_01|mean_plus_{num_bands}_non_dc|ortho_fft|"
            "integer_q|sequential_remaining"
        ).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    figures.mkdir()

    write_csv(args.output_dir / "band_statistics.csv", pixel_rows)
    write_csv(args.output_dir / "snr_by_time.csv", snr_rows)
    np.savez_compressed(
        args.output_dir / "band_masks.npz",
        masks=masks,
        q=q,
        shell_q=shell_q,
        reference_spectrum=reference_spectrum,
        raw_rms=raw_rms,
    )
    np.savez_compressed(
        args.output_dir / "per_sample_statistics.npz",
        scene_id=sample_scene,
        tx_id=sample_tx,
        spatial_mean=sample_mean,
        total_rms=sample_total_rms,
        ac_rms=sample_ac_rms,
        band_rms=sample_band_rms,
        band_ac_energy_fraction=sample_energy_fraction,
    )
    np.savez_compressed(args.output_dir / "pixel_samples.npz", band_values=pixel_samples)
    (args.output_dir / "train_scene_ids.json").write_text(
        json.dumps(train_scene_ids, indent=2) + "\n", encoding="utf-8"
    )

    analysis_config = {
        "dataset_root": str(args.dataset_root.resolve()),
        "dpm_root": str(dpm_root.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "split_source": "authoritative_seed42_manifest",
        "training_scene_definition": "old_train + old_val",
        "train_scene_count": len(train_scene_ids),
        "train_sample_count": len(dataset),
        "train_scene_ids_hash": train_scene_hash,
        "excluded_from_statistics": "all authoritative old_test scenes (development validation/test and final test)",
        "gt_mode": "raw_dpm_01",
        "representation": f"spatial_mean_plus_{num_bands}_non_dc_radial_fft_bands",
        "fft_norm": "ortho",
        "frequency_coordinates": "integer q=ky^2+kx^2 in fftshift layout",
        "boundary_algorithm": "sequential remaining-energy target; indivisible radial shells",
        "num_bands": num_bands,
        "height": height,
        "width": width,
        "mask_hash": mask_hash,
        "device": str(device),
        "dtype": "float64/complex128",
        "pixel_distribution_sampling": {
            "seed": 20260927,
            "samples_per_batch_per_band": args.pixel_samples_per_batch,
            "samples_per_band": int(pixel_samples.shape[1]),
            "note": "min/max/moments/sign fractions are exact; quantiles use uniform sampled pixels",
        },
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "complete": True,
        "train_scene_count": len(train_scene_ids),
        "train_sample_count": len(dataset),
        "valid_non_dc_samples": valid_samples,
        "zero_non_dc_samples": zero_ac_samples,
        "image_rms": math.sqrt(image_ms),
        "image_mean_center": mean_center,
        "image_mean_scale": mean_scale,
        "mean_component_rms": math.sqrt(mean_ms),
        "mean_component_total_energy_fraction": mean_ms / image_ms,
        "non_dc_total_rms": math.sqrt(ac_ms),
        "non_dc_total_energy_fraction": ac_ms / image_ms,
        "parseval_absolute_error": parseval_error,
        "max_inverse_fft_imaginary": max_imaginary,
        "max_reconstruction_error_float64": max_reconstruction_error,
        "reconstruction_rmse_float64": reconstruction_rmse,
        "uint8_roundtrip_failure_pixels": uint8_failures,
        "mask_hash": mask_hash,
        "recommended_model_coordinates": "Z_k=B_k/s_raw[k]",
        "recommended_source_noise": "epsilon_k=P_k(g_k)*sqrt(HW/N_k) in model coordinates",
        "recommended_raw_source_noise": "s_raw[k]*P_k(g_k)*sqrt(HW/N_k)",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plot_masks(masks, pixel_rows, figures / "frequency_masks.png")
    plot_energy(pixel_rows, figures / "energy_rms_frequency_support.png")
    plot_rms_distributions(sample_band_rms.astype(np.float64), raw_rms, figures / "sample_rms_distributions.png")
    plot_pixel_distributions(pixel_samples.astype(np.float64), raw_rms, figures / "pixel_value_distributions.png")
    plot_noise(pixel_rows, figures / "noise_calibration.png")
    plot_snr(snr_rows, figures / "snr_heatmap.png")

    order = np.argsort(sample_ac_rms)
    representative_indices = [
        int(order[round((len(order) - 1) * quantile)]) for quantile in (0.05, 0.5, 0.95)
    ]
    for quantile, index in zip((5, 50, 95), representative_indices, strict=True):
        scene_id = int(sample_scene[index])
        tx_id = int(sample_tx[index])
        x = load_one(dpm_root / f"{scene_id}_{tx_id}.png", device)
        complex_bands, mean = decompose(x, masks_t)
        bands = complex_bands.real
        reconstruction = bands.sum(dim=1, keepdim=True) + mean
        plot_example(
            x[0, 0].cpu().numpy(),
            (x - mean)[0, 0].cpu().numpy(),
            bands[0].cpu().numpy(),
            reconstruction[0, 0].cpu().numpy(),
            pixel_rows,
            f"Training AC-RMS p{quantile:02d}: scene={scene_id}, tx={tx_id}, mean={mean.item():.5f}",
            figures / f"representative_ac_rms_p{quantile:02d}_scene{scene_id}_tx{tx_id}.png",
        )

    table_lines = [
        "| band | q range | N_k | ref AC | pooled AC | raw RMS s_k | sample RMS p05/p50/p95 | noise factor | unit-noise RMS p05/p50/p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pixel_rows:
        table_lines.append(
            "| {band:02d} | {q_min}-{q_max} | {frequency_count:,} | {reference_ac_fraction:.3%} | "
            "{pooled_ac_fraction:.3%} | {raw_rms:.6f} | {sample_rms_p050:.5f}/{sample_rms_p500:.5f}/"
            "{sample_rms_p950:.5f} | {noise_equalization_factor:.3f} | "
            "{unit_noise_rms_p05:.3f}/{unit_noise_rms_p50:.3f}/{unit_noise_rms_p95:.3f} |".format(**row)
        )

    band0 = pixel_rows[0]
    last_band = pixel_rows[-1]
    last_band_index = num_bands - 1
    report = f"""# RadioMapSeer 训练集{num_bands}频带统计与噪声分析

## 范围

- 只使用权威 seed-42 manifest 的旧训练500场景与旧验证100场景，共600个训练场景、48,000张标准DPM。
- 原始测试100场景完全未参与边界、尺度、分布或噪声统计。
- 输入为原始8-bit灰度DPM除以255；每图先分离空间均值，再把非DC频谱划分为{num_bands}个固定径向硬频带。
- 频带边界使用每张图先归一化的AC能量平均，并采用不可拆径向壳层的顺序剩余目标算法。

## 总体统计

- 原图 RMS：`{math.sqrt(image_ms):.9f}`。
- 空间均值中心/标准差：`{mean_center:.9f}` / `{mean_scale:.9f}`。
- 均值/DC分量占总平方能量：`{mean_ms / image_ms:.4%}`；{num_bands}个非DC带合计：`{ac_ms / image_ms:.4%}`。
- 全部48,000张训练图均有非零AC能量。
- float64最大虚部：`{max_imaginary:.3e}`；最大重构误差：`{max_reconstruction_error:.3e}`；RMSE：`{reconstruction_rmse:.3e}`。
- uint8往返恢复失败像素：`{uint8_failures}`。

## 频带统计

{chr(10).join(table_lines)}

`raw RMS s_k` 的定义为训练集中该频带所有样本、所有空间像素的总体平方均值开方。它不是逐图RMS平均，也不是逐图动态尺度。

## 主要特征

1. 低频AC高度集中。band 00只有 `{band0['frequency_count']}` 个频点，但占训练平均AC能量 `{band0['reference_ac_fraction']:.2%}`，原始RMS为 `{band0['raw_rms']:.6f}`。径向壳层不能拆分，因此它不一定接近名义{1 / num_bands:.2%}。
2. 最高频band {last_band_index:02d}包含 `{last_band['frequency_count']:,}` 个频点，占全部非DC频点的 `{last_band['frequency_count'] / (pixels - 1):.2%}`，但仅占池化AC能量 `{last_band['pooled_ac_fraction']:.2%}`。频点数与信号能量不是同一个量。
3. 所有空间子带都是有正有负、理论均值为零的实值图，不能裁剪到 `[0,1]`。精确空间均值和正负比例见 `band_statistics.csv`。
4. 固定RMS标准化后，训练总体上每带的平方RMS为1，但单张图仍有真实幅度差异；逐样本p05/p50/p95见上表和图 `sample_rms_distributions.png`。

## 噪声结论

对空间白噪声 `g_k` 做频带投影后，其预期空间平方RMS为 `N_k/(H*W)`。所以必须先按频点数补偿：

```text
epsilon_k = P_k(g_k) * sqrt(H*W/N_k)
```

该式令模型坐标中的 `E[RMS(epsilon_k)^2]=1`。由于低频带自由度很少，单次噪声RMS仍会明显波动：band 00 的单位噪声RMS p05/p50/p95为 `{band0['unit_noise_rms_p05']:.3f}/{band0['unit_noise_rms_p50']:.3f}/{band0['unit_noise_rms_p95']:.3f}`；band {last_band_index:02d}则为 `{last_band['unit_noise_rms_p05']:.3f}/{last_band['unit_noise_rms_p50']:.3f}/{last_band['unit_noise_rms_p95']:.3f}`。不能把每次噪声再强制归一到RMS=1，否则会改变高斯源分布。

推荐在模型坐标中使用：

```text
Z_k = B_k / s_k
epsilon_k = P_k(g_k) * sqrt(H*W/N_k)
z_t,k = (1-t_k) * epsilon_k + t_k * Z_k
```

等价的原始频带坐标源噪声为：

```text
noise_raw,k = s_k * P_k(g_k) * sqrt(H*W/N_k)
```

因此每带推荐的原始空间噪声RMS就是表中的固定 `s_k`；实际乘在投影白噪声上的系数是 `recommended_white_noise_multiplier`。不要沿用原空间patch的共同噪声幅度，也不要使用当前样本真实RMS动态调整噪声。

`snr_by_time.csv` 和 `snr_heatmap.png` 使用训练集逐样本信号RMS及对应自由度的高斯噪声RMS，给出不同时间的实际SNR分布。它们用于检查时间采样是否覆盖足够的高噪、中噪和精修状态，而不是反向按GT动态改噪声。

## 文件说明

- `band_statistics.csv`：每带完整几何、信号和噪声统计。
- `snr_by_time.csv`：每带在多个时间点的SNR p05/p50/p95。
- `per_sample_statistics.npz`：48,000张训练图的均值、总RMS、AC RMS及逐带统计。
- `pixel_samples.npz`：用于像素分位数的确定性均匀样本；精确min/max和矩统计仍来自全训练集。
- `band_masks.npz`：{num_bands}个固定mask、整数半径和参考频谱。
- `analysis_config.json`：数据来源、哈希、口径和采样说明。
- `summary.json`：总体数值验收与推荐公式。
- `figures/`：mask、能量、RMS、像素分布、噪声、SNR及代表样本图。
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
