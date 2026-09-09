"""Načtení YAML configu + jeho promítnutí do LoopConfig (sdíleno train_unet.py/train_unetr.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from training.common.loop import LoopConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loop_config(raw: dict[str, Any]) -> LoopConfig:
    return LoopConfig(
        run_name=raw["run"]["name"],
        num_classes=7,
        max_epochs=raw["run"]["max_epochs"],
        lr=raw["optimizer"]["lr"],
        seed=raw["run"]["seed"],
        patch_size=raw["data"]["patch_size"],
        use_amp=raw["amp"],
        sw_batch_size=raw["validation"]["sw_batch_size"],
        sw_overlap=raw["validation"]["sw_overlap"],
        full_val_every=raw["run"]["full_val_every"],
        max_minutes=raw["run"]["max_minutes"],
        scheduler_kind=raw["scheduler"]["kind"],
        t0=raw["scheduler"]["t0"],
        t_mult=raw["scheduler"]["t_mult"],
        eta_min=raw["scheduler"]["eta_min"],
        plateau_patience=raw["scheduler"]["plateau_patience"],
        plateau_ratio=raw["scheduler"]["plateau_ratio"],
    )
