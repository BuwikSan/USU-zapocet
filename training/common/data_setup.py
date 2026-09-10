"""
Sestavení train/val DataLoaderů - JEDNO společné místo pro oba modely, aby
U-Net i UNETR garantovaně viděly identická data, stejný split, stejné
augmentace a stejný seed (podmínka kontrolovaného experimentu, viz
`devnotes/teorie/03_loss_metriky_a_trenink.md`, kap. 3.5).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.common import paths
from training.common.dataset import AtlasPatchDataset, load_pairs_from_csv
from training.common.seeding import make_generator, worker_init_fn
from training.common.transforms import build_eval_transforms, build_train_transforms

NUM_CLASSES = 7


def build_dataloaders(
    raw: dict[str, Any], limit_train: int | None = None, limit_val: int | None = None
) -> tuple[DataLoader, DataLoader]:
    data_cfg = raw["data"]
    seed = raw["run"]["seed"]
    patch_size = data_cfg["patch_size"]

    # Když jsou k dispozici předpočítané CLAHE snímky, čte se z nich a transformace
    # HistogramEqualizationd se do pipeline vůbec nepřidá (výsledek je totožný,
    # jen se nepočítá 8x na snímek za epochu). Viz scripts/precompute_heqv.py.
    use_precomputed = data_cfg.get("use_precomputed_heqv", False) and data_cfg["heqv"]
    if use_precomputed:
        if not paths.IMAGES_HEQV_DIR.exists():
            raise FileNotFoundError(
                f"use_precomputed_heqv je zapnuté, ale {paths.IMAGES_HEQV_DIR} neexistuje. "
                "Spusť nejdřív: py -3 scripts/precompute_heqv.py"
            )
        images_dir = paths.IMAGES_HEQV_DIR
        heqv_at_runtime = False
    else:
        images_dir = paths.IMAGES_DIR
        heqv_at_runtime = data_cfg["heqv"]

    train_pairs = load_pairs_from_csv(paths.FOLDS_DIR / "train.csv", images_dir, paths.MASKS_DIR)
    val_pairs = load_pairs_from_csv(paths.FOLDS_DIR / "val.csv", images_dir, paths.MASKS_DIR)
    if limit_train is not None:
        train_pairs = train_pairs[:limit_train]
    if limit_val is not None:
        val_pairs = val_pairs[:limit_val]

    train_transform = build_train_transforms(
        patch_size,
        heqv=heqv_at_runtime,
        rand_flip_prob=raw["transforms"]["rand_flip_prob"],
        rand_rotate90_prob=raw["transforms"]["rand_rotate90_prob"],
        rand_rotate90_max_k=raw["transforms"]["rand_rotate90_max_k"],
        rand_zoom_min=raw["transforms"]["rand_zoom_min"],
        rand_zoom_max=raw["transforms"]["rand_zoom_max"],
        rand_zoom_prob=raw["transforms"]["rand_zoom_prob"],
        rand_gaussian_noise_prob=raw["transforms"]["rand_gaussian_noise_prob"],
        seed=seed,
        crop_strategy=data_cfg.get("crop_strategy", "random"),
        pos_ratio=data_cfg.get("pos_ratio", 0.8),
    )
    eval_transform = build_eval_transforms(patch_size, heqv=heqv_at_runtime)

    train_dataset = AtlasPatchDataset(
        train_pairs, train_transform, NUM_CLASSES, patches_per_image=data_cfg["patches_per_image"]
    )
    # Validace vždy patches_per_image=1 (celý snímek, ne náhodný výřez) a batch_size=1,
    # protože snímky mají různé rozlišení a nejdou spolehlivě naskládat do jedné
    # dávky - sliding_window_inference v loop.py si s libovolnou velikostí poradí sama.
    val_dataset = AtlasPatchDataset(val_pairs, eval_transform, NUM_CLASSES, patches_per_image=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["train_batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=partial(worker_init_fn, base_seed=seed),
        generator=make_generator(seed),
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, data_cfg["num_workers"] // 2),
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
