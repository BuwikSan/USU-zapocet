"""
Pomocné funkce sdílené notebooky.

Notebooky v tomto projektu jsou záměrně "tenké" — veškerá logika žije ve
skriptech v `scripts/` a `training/`, notebook je jen spouští, vizualizuje
výsledky a vysvětluje, co se děje. Tenhle modul obsahuje to, co by se jinak
v každém notebooku opakovalo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INPUT_DIR = REPO_ROOT / "Inputdata"
IMAGES_DIR = INPUT_DIR / "datasets-PNG"
IMAGES_HEQV_DIR = INPUT_DIR / "datasets-PNG-heqv"
MASKS_DIR = INPUT_DIR / "datasets-MASK"
JSON_DIR = INPUT_DIR / "datasets-JSON"
FOLDS_DIR = INPUT_DIR / "folds" / "atlas_vertebra"
OUTPUTS_DIR = REPO_ROOT / "training_outputs"

CLASS_NAMES = {0: "pozadí", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7"}

# Táž paleta jako ve skriptech (RGB pořadí pro matplotlib).
CLASS_PALETTE_RGB = np.array(
    [[0, 0, 0], [64, 64, 255], [64, 220, 64], [255, 64, 64],
     [255, 220, 0], [220, 64, 255], [255, 255, 64]], dtype=np.uint8)


def run_script(*args: str, timeout: int = 7200) -> int:
    """
    Spustí skript projektu a proudem vypisuje jeho výstup do notebooku.

    Používá `py -3 -u`: `py -3` je na tomto stroji jediný Python s nainstalovanými
    balíčky, `-u` vypne bufferování, aby bylo v notebooku vidět průběžný postup
    a ne až všechno naráz na konci (viz devnotes/PROSTREDI_A_PASTI.md).
    """
    command = ["py", "-3", "-u", *args]
    print(f"$ {' '.join(command)}\n", flush=True)

    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        # Nezajímavý balast z knihoven, ať se v notebooku neztratí to podstatné.
        if any(noise in line for noise in ("oneDNN", "absl::InitializeLog", "UserWarning",
                                           "win_data =", "out[idx_zm]")):
            continue
        print(line, end="", flush=True)
    process.wait(timeout=timeout)
    print(f"\n[navratovy kod: {process.returncode}]")
    return process.returncode


def colorize(mask: np.ndarray) -> np.ndarray:
    """Maska tříd 0-6 -> barevný RGB obrázek."""
    return CLASS_PALETTE_RGB[mask]


def overlay(image_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Barevná maska prosvícená přes šedotónový snímek."""
    rgb = np.stack([image_gray] * 3, axis=-1).astype(np.float32)
    colored = colorize(mask).astype(np.float32)
    blended = np.where(mask[..., None] > 0, (1 - alpha) * rgb + alpha * colored, rgb)
    return blended.astype(np.uint8)


def show_row(panels: list[np.ndarray], titles: list[str], figsize=(16, 7), cmap=None) -> None:
    """Vykreslí obrázky vedle sebe s popisky."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap=cmap if panel.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def legend_patches():
    """Legenda tříd pro matplotlib."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=CLASS_PALETTE_RGB[i] / 255, label=CLASS_NAMES[i])
            for i in range(1, 7)]
