"""RadioMapSeer DPM dataset using the authoritative seed-42 scene split."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


class RepeatedRandomSampler(Sampler[int]):
    """Finite stream of independently shuffled complete dataset passes."""

    def __init__(self, dataset: Dataset, num_samples: int, seed: int) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = int(seed)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        produced = 0
        while produced < self.num_samples:
            permutation = torch.randperm(len(self.dataset), generator=generator).tolist()
            take = min(len(permutation), self.num_samples - produced)
            yield from permutation[:take]
            produced += take

    def __len__(self) -> int:
        return self.num_samples


def _load_authoritative_splits(path: str | Path) -> dict[str, list[int]]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key, expected in (("train", 500), ("val", 100), ("test", 100)):
        if key not in manifest or len(manifest[key]) != expected:
            raise ValueError(f"Expected {expected} scene IDs in split {key!r}: {path}")
    train = [int(value) for value in manifest["train"] + manifest["val"]]
    held_out = [int(value) for value in manifest["test"]]
    if len(set(train + held_out)) != 700:
        raise ValueError("Authoritative scene splits must be disjoint and contain 700 scenes.")
    return {"train": train, "validation": held_out[:50], "test": held_out[50:]}


class RadioMapSeerDPM(Dataset):
    """DPM target, binary building mask, and one-pixel transmitter condition."""

    def __init__(
        self,
        root: str | Path,
        scene_ids: list[int],
        expected_tx_per_scene: int = 80,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.gain_dir = self.root / "gain" / "DPM"
        self.building_dir = self.root / "png" / "buildings_complete"
        self.antenna_dir = self.root / "png" / "antennas"
        self.scene_ids = [int(value) for value in scene_ids]
        self.samples = [
            (scene_id, tx_id)
            for scene_id in self.scene_ids
            for tx_id in range(int(expected_tx_per_scene))
        ]
        missing: list[str] = []
        for scene_id, tx_id in self.samples:
            candidates = (
                self.gain_dir / f"{scene_id}_{tx_id}.png",
                self.building_dir / f"{scene_id}.png",
                self.antenna_dir / f"{scene_id}_{tx_id}.png",
            )
            missing.extend(str(path) for path in candidates if not path.is_file())
            if len(missing) >= 5:
                break
        if missing:
            raise FileNotFoundError(f"Missing RadioMapSeer files; first entries: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_grayscale(path: Path) -> Tensor:
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (256, 256):
                image = image.convert("L")
                if image.size != (256, 256):
                    raise ValueError(f"Expected 256x256 image, got {image.size}: {path}")
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).unsqueeze(0).to(torch.float32) / 255.0

    @staticmethod
    def _one_pixel_tx(antenna: Tensor) -> Tensor:
        locations = torch.nonzero(antenna[0] > 0, as_tuple=False)
        if locations.numel() == 0:
            raise ValueError("Transmitter raster has no non-zero pixel.")
        center = locations.to(torch.float32).mean(dim=0).round().to(torch.int64)
        output = torch.zeros_like(antenna)
        output[0, center[0], center[1]] = 1.0
        return output

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        scene_id, tx_id = self.samples[index]
        image = self._read_grayscale(self.gain_dir / f"{scene_id}_{tx_id}.png")
        building = (
            self._read_grayscale(self.building_dir / f"{scene_id}.png") > 0
        ).to(torch.float32)
        antenna = self._read_grayscale(self.antenna_dir / f"{scene_id}_{tx_id}.png")
        return {
            "image": image,
            "building": building,
            "tx": self._one_pixel_tx(antenna),
            "scene_id": scene_id,
            "tx_id": tx_id,
        }


class RadioMapSeerDataModule(pl.LightningDataModule):
    """600-scene train, 50-scene development, and sealed 50-scene test."""

    def __init__(
        self,
        root: str | Path,
        split_manifest: str | Path,
        batch_size: int = 32,
        val_batch_size: int = 16,
        num_workers: int = 8,
        train_stream_samples: int = 400_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = str(root)
        self.split_manifest = str(split_manifest)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.train_stream_samples = int(train_stream_samples)
        self.seed = int(seed)
        self.datasets: dict[str, RadioMapSeerDPM] = {}

    def setup(self, stage=None) -> None:
        del stage
        splits = _load_authoritative_splits(self.split_manifest)
        self.datasets = {
            name: RadioMapSeerDPM(self.root, scene_ids)
            for name, scene_ids in splits.items()
        }

    def train_dataloader(self) -> DataLoader:
        dataset = self.datasets["train"]
        sampler = RepeatedRandomSampler(dataset, self.train_stream_samples, self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
