"""
Předpočítání ekvalizace histogramu (CLAHE) do souborů — optimalizace datové pipeline.

PROČ TENTO SKRIPT EXISTUJE
--------------------------
Měření na reálných datech (viz devnotes/DALSI_KROKY.md, krok 4) ukázalo, že
tréninková pipeline byla **omezená procesorem, ne grafikou**:

    cv2.imread snímku (1014x802x3)  ~5.1 ms
    cv2.imread masky                ~2.7 ms
    CLAHE na celém snímku (3 kanály) ~6.0 ms
    -------------------------------------
    CPU na jeden vzorek             ~13.7 ms
    -> dávka 8 vzorků               ~110 ms

    GPU krok U-Net (dávka 8)         21 ms
    GPU krok UNETR (dávka 8)        167 ms

U U-Netu tedy grafika 80 % času jen čekala na data (`nvidia-smi` hlásilo 0 %
vytížení). Navíc: protože `patches_per_image = 8`, načetl se a zekvalizoval
**tentýž snímek osmkrát za epochu** — pokaždé se stejným výsledkem.

CO S TÍM DĚLÁ TENTO SKRIPT
--------------------------
Spustí CLAHE na každý snímek **právě jednou** a uloží výsledek do
`Inputdata/datasets-PNG-heqv/`. Trénink pak už jen čte hotový soubor a
transformaci CLAHE úplně vynechá (`data.use_precomputed_heqv: true` v configu).

Ukládá se **jednokanálově** (šedotónově): originály jsou sice 3kanálové PNG, ale
všechny tři kanály jsou identické (jde o rentgen). Jednokanálový soubor se čte
rychleji a `AtlasPatchDataset` si ho stejně replikuje na 3 kanály, které model
očekává. Výsledek je tedy numericky totožný s CLAHE počítaným za běhu.

SÉMANTIKA ZŮSTÁVÁ STEJNÁ
------------------------
CLAHE je *adaptivní* — pracuje s mřížkou dlaždic přes celý snímek. Proto se musí
počítat na CELÉM snímku před vyříznutím patche, přesně jak to dělala transformace
za běhu. Kdyby se CLAHE pouštěl až na vyříznutý patch 256x256, dlaždice by
odpovídaly jinému výřezu a výsledek by byl jiný. Tento skript tedy nemění
experiment, jen odstraňuje opakovaný výpočet.

POUŽITÍ
-------
    python scripts/precompute_heqv.py
    python scripts/precompute_heqv.py --limit 20     # rychlá zkouška
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
DEFAULT_OUT_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG-heqv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("precompute_heqv")


def equalize(image_gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    Stejné parametry jako `HistogramEqualizationd` v training/common/transforms.py
    (clipLimit=2.0, tileGridSize=(8,8)) — kdyby se tam někdy měnily, musí se
    změnit i tady, jinak přestanou být obě cesty ekvivalentní.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(image_gray)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Zpracovat jen prvních N snímků (zkouška).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.src_dir.glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"Ve složce {args.src_dir} nejsou žádné PNG soubory.")

    done = skipped = 0
    for i, src_path in enumerate(paths):
        out_path = args.out_dir / src_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # Šedotónově: všechny tři kanály originálu jsou u rentgenu identické.
        image = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            log.warning("Nelze načíst %s, přeskakuji.", src_path.name)
            continue

        cv2.imwrite(str(out_path), equalize(image, args.clip_limit, args.tile_grid))
        done += 1

        if (i + 1) % 500 == 0:
            log.info("Zpracováno %d/%d...", i + 1, len(paths))

    log.info("Hotovo. Nově vytvořeno: %d, přeskočeno (už existovalo): %d.", done, skipped)
    log.info("Výstup: %s", args.out_dir)
    log.info("V configu nastav data.use_precomputed_heqv: true")


if __name__ == "__main__":
    main()
