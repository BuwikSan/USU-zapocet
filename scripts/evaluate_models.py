"""
Fáze E: vyhodnocení a srovnání všech tří modelů na sdíleném testovacím setu.

ROLE V PROJEKTU
---------------
Tohle je závěrečný krok celé studie. Bere hotové predikce (PNG masky 0-6) od
libovolného počtu modelů a proti ground-truth maskám počítá Dice a IoU.

Klíčová vlastnost: skript **vůbec neví, který model masku vyrobil**. Porovnává jen
PNG proti PNG. Díky tomu se do srovnání nemůže vloudit nespravedlnost tím, že by
se jeden model vyhodnocoval jinak než druhý — U-Net, UNETR i TotalSegmentator2D
projdou naprosto identickým výpočtem.

PROČ SE POZADÍ NEPOČÍTÁ DO PRŮMĚRU
----------------------------------
Obratle zabírají jen kolem 3 % plochy snímku. Kdyby se do průměrné metriky
započítalo i pozadí, dostal by i model, který predikuje výhradně pozadí, Dice
kolem 0,97/7 tříd — číslo, které vypadá skvěle a neznamená nic. Hlavní metrikou
je proto průměr přes třídy 1-6 (obratle C2-C7) BEZ pozadí. Podrobněji viz
`devnotes/teorie/03_loss_metriky_a_trenink.md`, kap. 3.4.

JAK SE ZACHÁZÍ S CHYBĚJÍCÍMI TŘÍDAMI
------------------------------------
Když daná třída není ani v ground truth, ani v predikci, není co měřit a Dice
není definovaný (0/0). Takový případ se zapíše jako `NaN` a průměruje se přes
`nanmean` — nezkresluje to výsledek ani nahoru (jako by se počítal jako 1,0),
ani dolů (jako 0,0). Stejný přístup používá i `Src/Utils/metrics.py` ve
vypůjčeném repozitáři.

Když ale třída V GROUND TRUTH je a model ji nenajde, je to skutečná chyba a
započítá se jako Dice 0 — takhle se pozná model, který prostě nic nepredikuje.

POUŽITÍ
-------
    # vyhodnotí všechny modely, které mají hotové predikce
    python scripts/evaluate_models.py

    # jen vybrané, s obrázkovým srovnáním
    python scripts/evaluate_models.py --models unet unetr --figures 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd

NUM_CLASSES = 7
CLASS_NAMES = {1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7"}

MASKS_DIR = REPO_ROOT / "Inputdata" / "datasets-MASK"
IMAGES_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
FOLDS_DIR = REPO_ROOT / "Inputdata" / "folds" / "atlas_vertebra"
PREDICTIONS_DIR = REPO_ROOT / "training_outputs" / "predictions"
EVAL_DIR = REPO_ROOT / "training_outputs" / "evaluation"

CLASS_PALETTE_BGR = np.array(
    [[0, 0, 0], [255, 64, 64], [64, 220, 64], [64, 64, 255],
     [0, 220, 255], [255, 64, 220], [64, 255, 255]], dtype=np.uint8)


def dice_and_iou(pred: np.ndarray, truth: np.ndarray, class_id: int) -> tuple[float, float]:
    """
    Dice a IoU pro jednu třídu jednoho snímku.

    Dice = 2|A n B| / (|A| + |B|),  IoU = |A n B| / |A u B|

    Obě metriky měří překryv, jen jinak váží. Dice je shovívavější k drobným
    odchylkám na hranici, IoU tvrdší — proto se uvádějí obě.
    Vrací (nan, nan), pokud třída není ani v jedné masce (viz docstring modulu).
    """
    pred_mask = pred == class_id
    truth_mask = truth == class_id

    pred_sum = int(pred_mask.sum())
    truth_sum = int(truth_mask.sum())

    if pred_sum == 0 and truth_sum == 0:
        return float("nan"), float("nan")

    intersection = int(np.logical_and(pred_mask, truth_mask).sum())
    union = pred_sum + truth_sum - intersection

    dice = 2.0 * intersection / (pred_sum + truth_sum)
    iou = intersection / union if union > 0 else float("nan")
    return dice, iou


def evaluate_model(model_name: str, pred_dir: Path, stems: list[str]) -> pd.DataFrame:
    """Vrátí tabulku po snímcích a třídách: sloupce stem, class_id, dice, iou."""
    rows = []
    missing = 0

    for stem in stems:
        pred_path = pred_dir / f"{stem}.png"
        truth_path = MASKS_DIR / f"{stem}.png"

        truth = cv2.imread(str(truth_path), cv2.IMREAD_GRAYSCALE)
        if truth is None:
            raise FileNotFoundError(f"Chybi ground truth maska: {truth_path}")

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            # Model pro tenhle snímek predikci nevyrobil. Tiše to přeskočit by
            # výsledek vylepšilo (počítalo by se z méně snímků), proto se to
            # počítá jako prázdná predikce — tedy Dice 0 tam, kde měl něco najít.
            missing += 1
            pred = np.zeros_like(truth)

        if pred.shape != truth.shape:
            raise ValueError(
                f"{model_name}/{stem}: rozmer predikce {pred.shape} != ground truth {truth.shape}"
            )

        for class_id in range(1, NUM_CLASSES):
            dice, iou = dice_and_iou(pred, truth, class_id)
            rows.append({"model": model_name, "stem": stem, "class_id": class_id,
                         "dice": dice, "iou": iou})

    if missing:
        print(f"  [{model_name}] VAROVANI: {missing}/{len(stems)} predikci chybi, "
              f"pocitaji se jako prazdne")
    return pd.DataFrame(rows)


def summarize(per_image: pd.DataFrame) -> dict:
    """Průměry přes snímky — po třídách i celkově (bez pozadí)."""
    summary = {}
    for class_id, name in CLASS_NAMES.items():
        subset = per_image[per_image["class_id"] == class_id]
        summary[f"dice_{name}"] = float(np.nanmean(subset["dice"])) if len(subset) else float("nan")
        summary[f"iou_{name}"] = float(np.nanmean(subset["iou"])) if len(subset) else float("nan")

    summary["mean_dice"] = float(np.nanmean(per_image["dice"]))
    summary["mean_iou"] = float(np.nanmean(per_image["iou"]))
    return summary


def save_comparison_figures(models: dict[str, Path], stems: list[str], out_dir: Path, count: int) -> None:
    """
    Uloží vedle sebe: originál | ground truth | predikce jednotlivých modelů.
    Vizuální kontrola odhalí věci, které se v průměrném čísle ztratí — třeba že
    model najde obratle správně, ale posunuté o jednu pozici.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems[:count]:
        original = cv2.imread(str(IMAGES_DIR / f"{stem}.png"), cv2.IMREAD_COLOR)
        truth = cv2.imread(str(MASKS_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        panels = [original, cv2.addWeighted(original, 0.6, CLASS_PALETTE_BGR[truth], 0.4, 0.0)]
        titles = ["original", "ground truth"]

        for model_name, pred_dir in models.items():
            pred = cv2.imread(str(pred_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if pred is None:
                pred = np.zeros_like(truth)
            panels.append(cv2.addWeighted(original, 0.6, CLASS_PALETTE_BGR[pred], 0.4, 0.0))
            titles.append(model_name)

        for panel, title in zip(panels, titles):
            cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(str(out_dir / f"{stem}_srovnani.png"), np.hstack(panels))

    print(f"  obrazkova srovnani: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Nazvy modelu = podadresare v training_outputs/predictions/. "
                             "Vychozi: vse, co tam je.")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--figures", type=int, default=3, help="Kolik obrazkovych srovnani ulozit.")
    args = parser.parse_args()

    stems = [Path(n).stem for n in pd.read_csv(FOLDS_DIR / f"{args.split}.csv")["image"]]
    if args.limit is not None:
        stems = stems[: args.limit]

    if args.models:
        model_names = args.models
    else:
        model_names = sorted(d.name for d in args.predictions_dir.iterdir() if d.is_dir()) \
            if args.predictions_dir.exists() else []

    if not model_names:
        raise FileNotFoundError(
            f"Nenalezeny zadne predikce v {args.predictions_dir}.\n"
            "Nejdriv vygeneruj predikce:\n"
            "  py -3 training/predict.py --model unet  --split test\n"
            "  py -3 training/predict.py --model unetr --split test\n"
            "  py -3 scripts/ts2d_inference.py --split test"
        )

    models = {name: args.predictions_dir / name for name in model_names}
    print(f"[eval] split={args.split}, snimku={len(stems)}, modelu={len(models)}: {', '.join(models)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_per_image = []
    summaries = {}
    for model_name, pred_dir in models.items():
        print(f"[eval] {model_name} ...")
        per_image = evaluate_model(model_name, pred_dir, stems)
        all_per_image.append(per_image)
        summaries[model_name] = summarize(per_image)

    per_image_df = pd.concat(all_per_image, ignore_index=True)
    per_image_df.to_csv(args.out_dir / f"per_image_{args.split}.csv", index=False)

    summary_df = pd.DataFrame(summaries).T
    summary_df.index.name = "model"
    summary_df = summary_df.sort_values("mean_dice", ascending=False)
    summary_df.to_csv(args.out_dir / f"summary_{args.split}.csv")

    with (args.out_dir / f"summary_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n_images": len(stems), "results": summaries},
                  f, ensure_ascii=False, indent=2)

    if args.figures > 0:
        save_comparison_figures(models, stems, args.out_dir / "srovnani", args.figures)

    # ---- výpis do konzole ----
    print()
    print("=" * 78)
    print(f"VYSLEDKY na splitu '{args.split}' ({len(stems)} snimku)")
    print("=" * 78)
    header = f"{'model':12s} {'Dice':>7s} {'IoU':>7s} | " + " ".join(f"{n:>6s}" for n in CLASS_NAMES.values())
    print(header)
    print("-" * len(header))
    for model_name, s in summary_df.iterrows():
        per_class = " ".join(f"{s[f'dice_{n}']:6.3f}" for n in CLASS_NAMES.values())
        print(f"{model_name:12s} {s['mean_dice']:7.4f} {s['mean_iou']:7.4f} | {per_class}")
    print("-" * len(header))
    print("Dice/IoU = prumer pres tridy C2-C7 BEZ pozadi. Sloupce vpravo = Dice po tridach.")
    print(f"\n[eval] vystup: {args.out_dir}")


if __name__ == "__main__":
    main()
