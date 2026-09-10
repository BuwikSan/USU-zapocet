"""
Fáze C: inference TotalSegmentator2D (model TSXR) + filtrace a přemapování tříd.

ROLE V PROJEKTU
---------------
TotalSegmentator2D je třetí porovnávaný přístup — na rozdíl od U-Netu a UNETR se
netrénuje, jen se pouští předtrénovaný cizí model. Segmentuje ale **26 anatomických
struktur** (kost křížová, L1-L5, T1-T12, C1-C7), zatímco naše úloha má jen 7 tříd
(pozadí + C2-C7). Tenhle skript proto:

1. spustí inferenci na snímcích ze zvoleného splitu,
2. z výstupu **vyfiltruje jen krční obratle C2-C7** a přemapuje je na naše ID 1-6,
   všechno ostatní zahodí do pozadí,
3. uloží výsledek jako PNG masku 0-6 — tedy **přesně v tom formátu, jaký má ground
   truth i predikce z `training/predict.py`**, aby evaluace mohla všechny tři
   modely porovnávat úplně stejně.

PROČ SE MAPUJE PODLE JMEN, NE PODLE ČÍSEL
-----------------------------------------
V `dataset.json` daného modelu má `vertebrae-c2` hodnotu 25, `vertebrae-c7` hodnotu 20.
Zapsat si tahle čísla natvrdo by ale byla časovaná bomba: jiná revize modelu (nebo
jiný model z rodiny ts2d) může mít pořadí tříd jiné a maska by se tiše přemapovala
špatně. Skript proto čte **jména** tříd z metadat segmentace, kterou model vrátil,
a čísla si dohledá sám.

POZOR NA DOMÉNOVÝ POSUN
-----------------------
TSXR byl trénovaný na syntetických rentgenech (DiffDRR rekonstrukce z CT) celého
těla. My ho pouštíme na skutečné snímky krční páteře. Je zcela možné, že bude
segmentovat špatně nebo vůbec — **to je legitimní a očekávaný výsledek studie**
(přesně to předpovídá hypotéza v `plan.md`). Proto se doporučuje pustit nejdřív
`--limit 5 --preview` a podívat se na výsledek okem, ne až na čísla.

PROSTŘEDÍ — viz devnotes/PROSTREDI_A_PASTI.md §4.3-4.4:
- model se stahuje rozbitým `gdown`, proto musí být předem nakešovaný v ~/.ts2d/models
  (proto `use_remote=False`)
- CLI nástroje nebere PNG, používá se Python API
- SimpleITK v tomto prostředí neumí číst PNG, proto se jde přes cv2
- model chce PŘESNĚ 1 kanál (šedotón)

POUŽITÍ
-------
    python scripts/ts2d_inference.py --limit 5 --preview      # nejdřív tohle!
    python scripts/ts2d_inference.py --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Naše cílové třídy. Klíč = jméno tak, jak ho používá TSXR model, hodnota = naše ID.
TSXR_NAME_TO_OUR_CLASS = {
    "vertebrae-c2": 1,
    "vertebrae-c3": 2,
    "vertebrae-c4": 3,
    "vertebrae-c5": 4,
    "vertebrae-c6": 5,
    "vertebrae-c7": 6,
}

# Barvy shodné s rasterize_masks.py, aby šly náhledy porovnávat vedle sebe.
CLASS_PALETTE_BGR = np.array(
    [[0, 0, 0], [255, 64, 64], [64, 220, 64], [64, 64, 255],
     [0, 220, 255], [255, 64, 220], [64, 255, 255]], dtype=np.uint8)

DEFAULT_IMAGES_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
DEFAULT_FOLDS_DIR = REPO_ROOT / "Inputdata" / "folds" / "atlas_vertebra"
DEFAULT_OUT_DIR = REPO_ROOT / "training_outputs" / "predictions" / "ts2d"


def normalize_label_name(name: str) -> str:
    """TSXR používá 'vertebrae-c2', jiné modely můžou mít 'vertebrae_C2' apod."""
    return name.strip().lower().replace("_", "-")


def build_remap(segmentation: sitk.Image, verbose: bool = False) -> dict[int, int]:
    """
    Vrátí mapování {hodnota_v_masce_od_modelu: naše_ID}. Jména tříd se čtou přímo
    z metadat segmentace, kterou model vrátil.
    """
    from ts2d.core.util.meta import get_annotation_labels

    labels = get_annotation_labels(segmentation)
    remap: dict[int, int] = {}
    for raw_name, info in labels.items():
        our_class = TSXR_NAME_TO_OUR_CLASS.get(normalize_label_name(info.get("name", raw_name)))
        if our_class is not None:
            remap[int(info["value"])] = our_class

    if verbose:
        print(f"[mapovani] model vraci {len(labels)} trid, z toho relevantnich {len(remap)}:")
        for src, dst in sorted(remap.items(), key=lambda kv: kv[1]):
            print(f"    hodnota modelu {src:3d} -> nase trida {dst}")
        chybi = set(TSXR_NAME_TO_OUR_CLASS.values()) - set(remap.values())
        if chybi:
            print(f"[mapovani] VAROVANI: v teto masce nejsou tridy {sorted(chybi)} "
                  f"(model je na tomto snimku nenasel)")
    return remap


def segmentation_to_mask(segmentation: sitk.Image, remap: dict[int, int],
                         target_shape: tuple[int, int], verbose: bool = False) -> np.ndarray:
    """
    Převede výstup modelu na jednokanálovou masku 0-6 o rozměru původního snímku.

    Výstup umí přijít ve dvou podobách a obě se tu ošetřují:
    - jednokanálová mapa ID (typicky, když běžel jediný model)
    - vícekanálová (multilabel) maska, kde každý kanál je jedna struktura —
      TS2D podporuje překrývající se struktury, viz `combine_segmentations` v tool.py
    """
    array = sitk.GetArrayFromImage(segmentation)
    array = np.squeeze(array)

    if verbose:
        print(f"[maska] surovy vystup shape={array.shape} dtype={array.dtype}")

    mask = np.zeros(target_shape, dtype=np.uint8)

    if array.ndim == 2:
        for src_value, our_class in remap.items():
            mask[array[: target_shape[0], : target_shape[1]] == src_value] = our_class
    elif array.ndim == 3:
        # (H, W, kanaly) nebo (kanaly, H, W) — kanálová osa je ta nejkratší
        channel_axis = int(np.argmin(array.shape))
        array = np.moveaxis(array, channel_axis, 0)
        for src_value, our_class in remap.items():
            if src_value - 1 < array.shape[0]:
                channel = array[src_value - 1][: target_shape[0], : target_shape[1]]
                mask[channel > 0] = our_class
    else:
        raise RuntimeError(f"Necekany tvar vystupu segmentace: {array.shape}")

    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-key", type=str, default="tsxr_vertebrae")
    # Fyzické rozlišení vstupu. nnU-Net podle něj vstup převzorkovává na rozlišení,
    # se kterým model trénoval (1.5 mm/px) — má tedy přímý vliv na to, jak velké
    # struktury model "vidí". Viz scripts/ts2d_domain_check.py, kde se zkoumá,
    # jaká hodnota dává smysl. Výchozí 0.5 odpovídá velikostí tréninkovým datům.
    parser.add_argument("--spacing", type=float, default=0.5, help="Rozliseni vstupu v mm/px.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="Ulozit i barevny overlay pres original.")
    args = parser.parse_args()

    from ts2d import TS2D

    stems = [Path(n).stem for n in pd.read_csv(args.folds_dir / f"{args.split}.csv")["image"]]
    if args.limit is not None:
        stems = stems[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ts2d] startuji model '{args.model_key}' (nnU-Net startuje subproces, chvili to trva)...")
    t0 = time.perf_counter()
    model = TS2D(key=args.model_key, use_remote=False)
    print(f"[ts2d] model pripraven za {time.perf_counter() - t0:.1f} s")

    remap: dict[int, int] | None = None
    start = time.perf_counter()
    failed = 0

    try:
        for i, stem in enumerate(stems):
            image_path = args.images_dir / f"{stem}.png"
            image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"[ts2d] nelze nacist {image_path.name}, preskakuji")
                failed += 1
                continue

            try:
                sitk_image = sitk.GetImageFromArray(image_gray.astype("float32"))
                sitk_image.SetSpacing((args.spacing, args.spacing))
                result = model.predict(sitk_image)
                # TS2D.Result vystavuje segmentaci metodou, ne atributem.
                segmentation = result.get_segmentation()
            except Exception as exc:
                print(f"[ts2d] predikce selhala pro {stem}: {exc}")
                failed += 1
                continue

            first = remap is None
            if first:
                remap = build_remap(segmentation, verbose=True)

            mask = segmentation_to_mask(segmentation, remap, image_gray.shape, verbose=first)
            cv2.imwrite(str(args.out_dir / f"{stem}.png"), mask)

            if args.preview:
                original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                overlay = cv2.addWeighted(original, 0.6, CLASS_PALETTE_BGR[mask], 0.4, 0.0)
                cv2.imwrite(str(preview_dir / f"{stem}_ts2d.png"), overlay)

            if first:
                print(f"[ts2d] prvni maska hotova, nalezene tridy: {np.unique(mask).tolist()}")
            if (i + 1) % 25 == 0:
                elapsed = time.perf_counter() - start
                print(f"  {i + 1}/{len(stems)} ({elapsed / (i + 1):.2f} s/snimek)")
    finally:
        model.close()

    total = time.perf_counter() - start
    print(f"[ts2d] hotovo: {len(stems) - failed}/{len(stems)} masek za {total:.1f} s "
          f"({total / max(1, len(stems)):.2f} s/snimek), chyb: {failed}")
    print(f"[ts2d] vystup: {args.out_dir}")


if __name__ == "__main__":
    main()
