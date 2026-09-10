"""
Ověření, zda TotalSegmentator2D (TSXR) rozpoznává krční obratle — a pokud ne, proč.

PROČ TENTO SKRIPT EXISTUJE
--------------------------
První inference TSXR na našich datech vrátila místo krčních obratlů kost křížovou
a bederní obratle. To je závažné tvrzení — než ho lze zapsat do zprávy jako
"model na tuto doménu nepřenáší", je nutné vyloučit, že chyba není na NAŠÍ straně.
Vykázat cizí model jako selhávající kvůli vlastní chybě v předzpracování by byl
falešný závěr.

Tento skript proto systematicky prochází hypotézy, které mohly za špatný výsledek
stát, a nechává je projít nebo padnout. Je psaný tak, aby šel kdykoliv spustit
znovu a výsledek doložit.

CO SE OVĚŘUJE
-------------
1. **Měřítko (spacing).** nnU-Net převzorkovává vstup z jeho fyzického rozlišení
   na cílové, se kterým model trénoval. `plans.json` modelu uvádí cílové rozlišení
   1.5 mm/px a medián tréninkových snímků 226x239 px. Když se vstupu nenastaví
   spacing, SimpleITK použije 1.0 mm/px — náš snímek 745x578 px se pak převzorkuje
   na ~497x385 px, tedy zhruba dvojnásobek toho, na co je model zvyklý. Obratle mu
   proto připadají příliš velké. Skript zkouší řadu hodnot spacingu včetně té,
   která velikostně odpovídá tréninkovým datům.

2. **Polarita intenzit.** TSXR byl trénovaný na syntetických rentgenech (DiffDRR
   rekonstrukce z CT). Kdyby měly obrácenou polaritu než skutečný rentgen
   (kost světlá vs. tmavá), model by nic nepoznal. Zkouší se i invertovaný snímek.

3. **Funkčnost modelu jako takového.** Kdyby se špatně načetly váhy, model by
   vracel šum nebo prázdno. Skript proto vypisuje i to, jak velké a jak souvislé
   oblasti model našel — prostorově souvislá predikce svědčí o tom, že model
   funguje, jen se plete v identitě struktury.

INTERPRETACE VÝSLEDKU
---------------------
Pokud ani jedna kombinace nenajde krční obratle, zatímco model spolehlivě a
souvisle predikuje bederní oblast, je to doložený doménový posun: TSXR trénoval
na projekcích z CT datasetu TotalSegmentator, kde převažuje trup a břicho, a krční
páteř v úzkém výřezu nezná. To je legitimní a pro studii hodnotný negativní
výsledek — přesně to, co předpovídá hypotéza v `plan.md`.

POUŽITÍ
-------
    python scripts/ts2d_domain_check.py
    python scripts/ts2d_domain_check.py --stem 0001035 --n-images 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

CERVICAL_IDS = {20: "c7", 21: "c6", 22: "c5", 23: "c4", 24: "c3", 25: "c2"}

IMAGES_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
MASKS_DIR = REPO_ROOT / "Inputdata" / "datasets-MASK"
FOLDS_DIR = REPO_ROOT / "Inputdata" / "folds" / "atlas_vertebra"


def summarize(seg, image_size: int) -> tuple[int, int, list[str]]:
    """Vrátí (pixelů krčních obratlů, pixelů ostatních struktur, top nálezy)."""
    from ts2d.core.util.meta import get_annotation_labels, get_labels_voxels

    labels = get_annotation_labels(seg)
    counts = get_labels_voxels(seg)
    id_to_name = {int(info["value"]): info["name"] for info in labels.values()}

    cervical = sum(c for v, c in counts.items() if v in CERVICAL_IDS and c > 0)
    other = sum(c for v, c in counts.items() if v not in CERVICAL_IDS and c > 0)
    top = [f"{id_to_name.get(v, '?')}={c}"
           for c, v in sorted(((c, v) for v, c in counts.items() if c > 0), reverse=True)[:3]]
    return cervical, other, top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stem", type=str, default=None, help="Konkretni snimek. Jinak se vezme zacatek val splitu.")
    parser.add_argument("--n-images", type=int, default=2, help="Kolik snimku overit.")
    parser.add_argument("--spacings", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35, 0.25])
    args = parser.parse_args()

    from ts2d import TS2D

    if args.stem:
        stems = [args.stem]
    else:
        # Záměrně validační split — ladit cokoliv na testovacím setu by ho
        # znehodnotilo pro závěrečné srovnání (viz plan.md, kap. 3.2).
        stems = [Path(n).stem for n in pd.read_csv(FOLDS_DIR / "val.csv")["image"][: args.n_images]]

    print("=" * 78)
    print("OVERENI DOMENOVE PRENOSITELNOSTI TSXR NA KRCNI RENTGENY")
    print("=" * 78)
    print("Model ocekava (plans.json): rozliseni 1.5 mm/px, median trenink. snimku 226x239 px")
    print("Trenovaci intenzity (foreground): mean=17.6 std=16.7 p99.5=84.5")
    print()

    model = TS2D(key="tsxr_vertebrae", use_remote=False)
    try:
        for stem in stems:
            image = cv2.imread(str(IMAGES_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(str(MASKS_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            gt_px = int((gt > 0).sum())

            print("-" * 78)
            print(f"snimek {stem}: {image.shape}, intenzity mean={image.mean():.1f} "
                  f"std={image.std():.1f} max={image.max()}")
            print(f"  ground truth: obratle zabiraji {gt_px} px ({100 * gt_px / image.size:.2f} % plochy)")
            print()

            for variant, array in (("original", image.astype("float32")),
                                   ("invertovany", (255 - image).astype("float32"))):
                for spacing in args.spacings:
                    sitk_image = sitk.GetImageFromArray(array)
                    sitk_image.SetSpacing((spacing, spacing))
                    seg = model.predict(sitk_image).get_segmentation()
                    cervical, other, top = summarize(seg, image.size)

                    resampled = int(image.shape[0] * spacing / 1.5)
                    verdict = "KRCNI NALEZENY" if cervical > 0 else "krcni nenalezeny"
                    print(f"  {variant:11s} spacing={spacing:4.2f} "
                          f"(po prevzorkovani ~{resampled:4d} px) | "
                          f"krcni={cervical:6d} px ostatni={other:6d} px | {verdict}")
                    if top:
                        print(f"{'':>32}nejcastejsi: {', '.join(top)}")
            print()
    finally:
        model.close()

    print("=" * 78)
    print("ZAVER: pokud ani jedna kombinace nenasla krcni obratle, ale model souvisle")
    print("predikuje bederni/krizovou oblast, jde o dolozeny domenovy posun, ne o chybu")
    print("v nasem predzpracovani. Viz devnotes/STAV_PROJEKTU.md.")
    print("=" * 78)


if __name__ == "__main__":
    main()
