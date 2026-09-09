"""
Fáze B zápočtové studie: vlastní K-Fold rozdělení datasetu Atlas (prevence data leakage).

KONTEXT
-------
Původní (vypůjčený) U-Net repozitář (`Modely/vertebra_segmentation_and_keypoint_extraction-main`)
neobsahuje žádnou algoritmickou logiku K-Fold splitu - `Src/Utils/data_utils.py::get_split_files`
jen NAČÍTÁ hotové CSV soubory `train_<dataset>_fold_<i>.csv` / `val_<dataset>_fold_<i>.csv`
(sloupce "image", "label"), které byly vygenerované mimo tento repozitář a v repu chybí.
"Exaktní rekonstrukce" původního Fold 0 tedy není z dostupného kódu možná.

To ale nevadí: podle aktuálního zadání se U-Net i UNETR trénují OD NULY na vlastních
datech, takže je zapotřebí vlastní, transparentní a reprodukovatelný split - ne kopie
cizího. Tento skript ho vytváří nezávisle, ale ve STEJNÉM CSV formátu (`image,label`
s pouhými názvy souborů), aby šel v případě potřeby použít i s logikou z původního
repozitáře beze změny.

METODA
------
1. Najdou se všechny snímky, které mají zároveň originální PNG (`Inputdata/datasets-PNG`)
   i vyrasterizovanou masku z Fáze A (`Inputdata/datasets-MASK`).
2. Seznam se deterministicky zamíchá (seed, výchozí 42 - stejná hodnota jako
   `run.seed` v `atlas_multiclass_patch.yaml`, aby byl experiment reprodukovatelný
   napříč celou studií, viz `devnotes/teorie/03_loss_metriky_a_trenink.md`, kap. 3.7).
3. Zamíchaný seznam se rozdělí na `k` přibližně stejně velkých, vzájemně
   disjunktních částí (foldů) - `numpy.array_split`.
4. Pro každý fold `i` je "val" = fold `i`, "train" = zbylých `k - 1` foldů. Tím je
   zaručeno, že žádný snímek nikdy neskončí zároveň v train i val části TÉHOŽ foldu
   (základní podmínka proti data leakage, viz `plan.md`, kap. 3).

Datovou jednotkou splitu je zde CELÝ SNÍMEK (ne patch). Patche se z něj náhodně
vyříznou až za běhu tréninku (`patches_per_image` v configu) - takže i kdyby dva
patche z jednoho snímku skončily technicky v různých batchích, pořád pocházejí
jen ze snímků patřících do jedné množiny (train, nebo val), nikdy ne obojí.
Dataset neobsahuje sloupec s ID pacienta, proto se předpokládá 1 snímek = 1 nezávislý
pacient/expozice (stejný předpoklad, jaký dělá i formát CSV v původním repu).

Fold 0 je v této studii ten, jehož validační část slouží jako SDÍLENÝ, FIXNÍ
testovací set pro finální srovnání všech tří modelů (U-Net, UNETR, TotalSegmentator2D),
viz `devnotes/claude.md`, Fáze B.

DODATEK - PŘECHOD NA FIXNÍ 60/20/20 SPLIT (devnotes/dodatek k trénování.md)
----------------------------------------------------------------------------
Plná 5-Fold křížová validace (5x trénink obou modelů od nuly) je podle vlastního
odhadu v `dodatek k trénování.md` na dostupném hardwaru v řádu týdnů - není to za
daný deadline (`devnotes/constrains.md`, 13.9.) reálné. K-Fold větev výše se
přesto v kódu ponechává (a i spouští) jako doklad, že rigoróznější metodika byla
navržena, prakticky odbenchmarkována (viz `scripts/benchmark_training_step.py`)
a vědomě zamítnuta ve prospěch jednoho pevného, ale přísně odděleného splitu:

    60 % train / 20 % val / 20 % test

- train: se učí model (gradient).
- val: hlídá přetrénování a slouží k výběru "best" checkpointu během tréninku
  (early-stopping styl, viz `training/common/loop.py`) - model ho "vidí"
  nepřímo (ovlivňuje, který checkpoint se uloží jako finální).
- test: NIKDY se nepoužije během tréninku ani výběru modelu - otevře se až na
  úplný konec pro finální nestranné srovnání U-Net vs. UNETR vs. TotalSegmentator2D.
  Je to týž koncept jako "Fold 0 val" výše, jen s explicitně odděleným val/test,
  aby výběr checkpointu nekontaminoval finální srovnávací metriku.

Tato funkce je nezávislá na K-Fold větvi výše (jiné soubory, jiné názvy), obě
mohou bez konfliktu koexistovat ve stejném výstupním adresáři.

POUŽITÍ
-------
    python scripts/build_fold_split.py
        -> vygeneruje 5 foldů (výchozí) do Inputdata/folds/atlas_vertebra/

    python scripts/build_fold_split.py --k 4 --seed 42
        -> minimální počet foldů podle plan.md ("K-Fold min. 4 foldy")

    python scripts/build_fold_split.py --fixed-split
        -> vygeneruje train.csv/val.csv/test.csv (60/20/20) - toto skutečně
           spotřebovávají trénovací skripty v training/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PNG_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
DEFAULT_MASK_DIR = REPO_ROOT / "Inputdata" / "datasets-MASK"
DEFAULT_OUT_DIR = REPO_ROOT / "Inputdata" / "folds" / "atlas_vertebra"

DATASET_NAME = "atlas_vertebra"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_fold_split")


def find_paired_stems(png_dir: Path, mask_dir: Path) -> list[str]:
    """
    Vrátí setříděný seznam "stemů" (názvů bez přípony) snímků, které mají zároveň
    originální PNG a rasterizovanou masku. Setřídění před míchaním je důležité -
    `Path.glob` pořadí souborů negarantuje a bez pevného vstupního pořadí by seedované
    zamíchání nebylo mezi platformami/běhy stoprocentně reprodukovatelné.
    """
    png_stems = {p.stem for p in png_dir.glob("*.png")}
    mask_stems = {p.stem for p in mask_dir.glob("*.png")}

    missing_masks = png_stems - mask_stems
    if missing_masks:
        log.warning("%d snímků nemá vygenerovanou masku (přeskočeny). Spusťte nejdřív rasterize_masks.py.", len(missing_masks))

    return sorted(png_stems & mask_stems)


def make_kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """
    Vlastní, na scikit-learn nezávislá implementace K-Fold rozdělení indexů:
    deterministicky zamíchá pořadí 0..n-1 seedovaným generátorem a rozseká ho na
    `k` přibližně stejně velkých souvislých bloků. Návratová hodnota je seznam
    polí indexů náležících jednotlivým foldům (fold[i] = validační indexy pro i-tý běh).
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    return np.array_split(shuffled, k)


def write_fold_csvs(
    stems: list[str], fold_blocks: list[np.ndarray], out_dir: Path
) -> list[dict]:
    """Zapíše pro každý fold train/val CSV ve formátu kompatibilním s `data_utils.get_split_files`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for fold_id, val_idx in enumerate(fold_blocks):
        val_idx_set = set(val_idx.tolist())
        train_idx = [i for i in range(len(stems)) if i not in val_idx_set]

        def rows_for(indices: list[int]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "image": [f"{stems[i]}.png" for i in indices],
                    "label": [f"{stems[i]}.png" for i in indices],
                }
            )

        train_df = rows_for(train_idx)
        val_df = rows_for(sorted(val_idx.tolist()))

        train_df.to_csv(out_dir / f"train_{DATASET_NAME}_fold_{fold_id}.csv", index=False)
        val_df.to_csv(out_dir / f"val_{DATASET_NAME}_fold_{fold_id}.csv", index=False)

        summary.append({"fold": fold_id, "n_train": len(train_df), "n_val": len(val_df)})
        log.info("Fold %d: train=%d, val=%d", fold_id, len(train_df), len(val_df))

    return summary


def write_fold0_val_manifest(out_dir: Path) -> None:
    """
    Podle `devnotes/claude.md` (Fáze B) musí být výstupem i samostatný seznam
    validačních souborů pro Fold 0 - sdílený testovací set, na kterém se nakonec
    porovnají všechny tři modely (U-Net, UNETR, TotalSegmentator2D).
    """
    val_df = pd.read_csv(out_dir / f"val_{DATASET_NAME}_fold_0.csv")
    manifest = {
        "dataset_name": DATASET_NAME,
        "fold": 0,
        "role": "shared held-out test set for U-Net vs. UNETR vs. TotalSegmentator2D",
        "count": len(val_df),
        "stems": [Path(name).stem for name in val_df["image"]],
    }
    with (out_dir / "fold_0_test_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_fixed_split(
    stems: list[str], train_frac: float, val_frac: float, seed: int, out_dir: Path
) -> dict:
    """
    Vygeneruje jeden pevný, vzájemně disjunktní train/val/test split (výchozí 60/20/20).
    Použitý seed je STEJNÝ jako u K-Fold větve (řízeno jedním `--seed` argumentem),
    ale jde o jiné (nezávislé) zamíchání, protože se počítá samostatným voláním
    generátoru - k žádné kolizi/závislosti mezi oběma metodami split logiky nedochází.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(stems)).tolist()

    n_train = round(len(stems) * train_frac)
    n_val = round(len(stems) * val_frac)

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val :]

    def rows_for(indices: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image": [f"{stems[i]}.png" for i in indices],
                "label": [f"{stems[i]}.png" for i in indices],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    counts = {}
    for name, idx in splits.items():
        df = rows_for(idx)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        counts[name] = len(df)
        log.info("Fixní split - %s: %d snímků", name, len(df))

    # Pojistka proti data leakage: žádné dvě množiny se nesmí protnout a jejich
    # sjednocení musí sedět na celkový počet snímků.
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(stems)

    manifest = {
        "dataset_name": DATASET_NAME,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": round(1 - train_frac - val_frac, 6),
        "n_total": len(stems),
        "counts": counts,
        "role": {
            "train": "gradientový trénink",
            "val": "hlídání přetrénování + výběr best checkpointu během tréninku",
            "test": "nedotčený, sdílený test set pro finální srovnání U-Net / UNETR / TotalSegmentator2D",
        },
    }
    with (out_dir / "fixed_split_info.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=5, help="Počet foldů (plan.md požaduje minimálně 4).")
    parser.add_argument("--seed", type=int, default=42, help="Seed pro zamíchání (sdílený s run.seed v atlas_multiclass_patch.yaml).")
    parser.add_argument("--fixed-split", action="store_true", help="Navíc vygenerovat pevný train/val/test split (viz dodatek k trénování.md).")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    stems = find_paired_stems(args.png_dir, args.mask_dir)
    if not stems:
        raise FileNotFoundError(
            f"Nenalezeny žádné páry PNG+maska v {args.png_dir} / {args.mask_dir}. "
            "Nejdřív spusťte scripts/rasterize_masks.py."
        )
    log.info("Nalezeno %d snímků s kompletní maskou.", len(stems))

    fold_blocks = make_kfold_indices(len(stems), args.k, args.seed)
    summary = write_fold_csvs(stems, fold_blocks, args.out_dir)
    write_fold0_val_manifest(args.out_dir)

    with (args.out_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"dataset_name": DATASET_NAME, "k": args.k, "seed": args.seed, "n_total": len(stems), "folds": summary},
            f, ensure_ascii=False, indent=2,
        )

    if args.fixed_split:
        write_fixed_split(stems, args.train_frac, args.val_frac, args.seed, args.out_dir)

    log.info("Hotovo. CSV a manifesty uloženy do: %s", args.out_dir)


if __name__ == "__main__":
    main()
