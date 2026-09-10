"""
Iterativní ladění hyperparametrů UNETR na krátkých bězích.

PROČ
----
Zkušební běh (`scripts/run_poc.py`) ukázal, že UNETR se s výchozím nastavením
učí zhruba desetkrát pomaleji než U-Net — v epoše 8 měl validační Dice 0,078
proti 0,699. Část toho je vlastnost architektury (Vision Transformer nemá
vestavěný induktivní bias konvoluce a musí se lokalitu naučit z dat), ale
v průběhu bylo vidět i konkrétní problém nastavení: kosinový scheduler
s `T_0 = 5` srazil learning rate do epochy 4 na ~1e-5 a trénink se prakticky
zastavil dřív, než se model stihl rozjet.

Než se na ostrý běh vsadí několik hodin nočního GPU okna, je levnější ověřit
úpravy na krátkých bězích. Tenhle skript to dělá systematicky: pustí sérii
variant za jinak identických podmínek a vypíše je vedle sebe.

METODIKA
--------
Všechny varianty běží na **stejných datech, se stejným seedem a stejný počet
epoch** — mění se jen laděné hyperparametry. Jinak by se výsledky nedaly
srovnávat.

Referenční hodnota z PoC (lr 1e-4, T_0 5, bez rozjezdu, 400 snímků, 10 epoch):
**val Dice 0,078**.

POUŽITÍ
-------
    python scripts/tune_unetr.py --list                 # co se bude zkoušet
    python scripts/tune_unetr.py                        # celá série
    python scripts/tune_unetr.py --only lr3e-4_t015     # jedna varianta
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_PATH = REPO_ROOT / "training_outputs" / "unetr_tuning.json"

NOISE = ("oneDNN", "absl::InitializeLog", "UserWarning", "win_data =", "out[idx_zm]",
         "WARNING: All log messages")

# Referenční bod z run_poc.py — stejná data, stejný počet epoch.
BASELINE = {"name": "baseline (PoC)", "val_dice": 0.078, "note": "lr 1e-4, T_0 5, bez rozjezdu"}

# Varianty k vyzkoušení. Pořadí je záměrné: od nejpravděpodobnější příčiny.
VARIANTS = [
    {
        "name": "lr3e-4_t015",
        "args": ["--lr", "3e-4", "--t0", "15"],
        "hypoteza": "LR byl moc nízký a scheduler ho navíc předčasně srazil. "
                    "Trojnásobný LR + cyklus delší než celý běh (tedy prakticky "
                    "monotónně klesající LR bez restartu).",
    },
    {
        "name": "lr3e-4_t015_warm2",
        "args": ["--lr", "3e-4", "--t0", "15", "--warmup-epochs", "2"],
        "hypoteza": "Totéž plus dvouepochový lineární rozjezd. U transformerů "
                    "standardní opatření — náhodně inicializovaná attention "
                    "dává zpočátku divoké gradienty.",
    },
    {
        "name": "lr1e-3_t015_warm2",
        "args": ["--lr", "1e-3", "--t0", "15", "--warmup-epochs", "2"],
        "hypoteza": "Agresivnější LR. Rozjezd by měl ohlídat, aby to hned "
                    "nezdivergovalo (1e-2 bez rozjezdu divergovalo).",
    },
    {
        "name": "posneg_lr3e-4_warm2",
        "args": ["--lr", "3e-4", "--t0", "15", "--warmup-epochs", "2",
                 "--crop-strategy", "pos_neg"],
        "hypoteza": "Útok na příčinu místo na learning rate: obratle zabírají "
                    "~3 % plochy, takže většina rovnoměrně náhodných výřezů "
                    "neobsahuje popředí a model se propadá do triviálního "
                    "řešení 'všechno je pozadí'. Cílené vzorkování výřezů na "
                    "popředí je v medicínské segmentaci standard.",
    },
]


def run_variant(variant: dict, train_images: int, val_images: int, epochs: int) -> dict:
    """Pustí jednu variantu a vytáhne z jejího výpisu průběh metrik."""
    run_name = f"tune-unetr-{variant['name']}"
    command = [
        "py", "-3", "-u", "training/train_unetr.py",
        "--run-name", run_name,
        "--max-epochs", str(epochs),
        "--limit-train", str(train_images),
        "--limit-val", str(val_images),
        "--full-val-every", "2",
        "--no-resume",
        *variant["args"],
    ]

    print(f"\n{'=' * 78}\nVARIANTA: {variant['name']}\n{'=' * 78}")
    print(f"hypoteza: {variant['hypoteza']}")
    print(f"$ {' '.join(command)}\n", flush=True)

    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", bufsize=1)
    assert process.stdout is not None

    epoch_re = re.compile(r"epoch (\d+): loss=([\d.naif]+) train_dice=([\d.naif]+) "
                          r"val_dice=([\d.naif-]+) lr=([\d.e+-]+)")
    history, diverged = [], False

    for line in process.stdout:
        if any(noise in line for noise in NOISE):
            continue
        print("  " + line.rstrip(), flush=True)
        if "PRERUSENO" in line or "diverged" in line:
            diverged = True
        m = epoch_re.search(line)
        if m:
            history.append({
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)) if m.group(2) not in ("nan", "inf") else None,
                "train_dice": float(m.group(3)) if m.group(3) not in ("nan", "inf") else None,
                "val_dice": float(m.group(4)) if m.group(4) not in ("-", "nan") else None,
            })
    process.wait()

    val_scores = [h["val_dice"] for h in history if h["val_dice"] is not None]
    return {
        "name": variant["name"],
        "hypoteza": variant["hypoteza"],
        "args": variant["args"],
        "returncode": process.returncode,
        "diverged": diverged,
        "minutes": (time.perf_counter() - start) / 60,
        "history": history,
        "best_val_dice": max(val_scores) if val_scores else 0.0,
        "final_val_dice": val_scores[-1] if val_scores else 0.0,
        "final_train_dice": history[-1]["train_dice"] if history else None,
    }


def print_summary(results: list[dict], epochs: int) -> None:
    print(f"\n{'=' * 78}\nSHRNUTI LADENI UNETR ({epochs} epoch, stejna data a seed)\n{'=' * 78}")
    header = f"{'varianta':24s} {'best val':>9s} {'konec val':>10s} {'train':>7s} {'min':>6s}"
    print(header)
    print("-" * len(header))
    print(f"{BASELINE['name']:24s} {BASELINE['val_dice']:9.4f} {'—':>10s} {'—':>7s} {'—':>6s}"
          f"   <- {BASELINE['note']}")
    for r in sorted(results, key=lambda x: -x["best_val_dice"]):
        flag = "  DIVERGOVALO" if r["diverged"] else ""
        train = f"{r['final_train_dice']:.4f}" if r["final_train_dice"] is not None else "—"
        print(f"{r['name']:24s} {r['best_val_dice']:9.4f} {r['final_val_dice']:10.4f} "
              f"{train:>7s} {r['minutes']:6.1f}{flag}")
    print("-" * len(header))

    best = max(results, key=lambda x: x["best_val_dice"], default=None)
    if best and best["best_val_dice"] > BASELINE["val_dice"]:
        factor = best["best_val_dice"] / max(BASELINE["val_dice"], 1e-9)
        print(f"\nNejlepsi: {best['name']} — val Dice {best['best_val_dice']:.4f} "
              f"({factor:.1f}x proti baseline)")
        print(f"Argumenty pro ostry beh: {' '.join(best['args'])}")
    else:
        print("\nZadna varianta baseline neprekonala.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-images", type=int, default=400)
    parser.add_argument("--val-images", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--only", type=str, default=None, help="Pustit jen variantu tohoto jmena.")
    parser.add_argument("--list", action="store_true", help="Jen vypsat varianty a skoncit.")
    args = parser.parse_args()

    variants = VARIANTS if args.only is None else [v for v in VARIANTS if v["name"] == args.only]
    if not variants:
        raise SystemExit(f"Varianta '{args.only}' neexistuje. K dispozici: "
                         f"{', '.join(v['name'] for v in VARIANTS)}")

    if args.list:
        for v in VARIANTS:
            print(f"{v['name']:24s} {' '.join(v['args'])}\n    {v['hypoteza']}\n")
        return

    print(f"Ladeni UNETR: {len(variants)} variant, {args.epochs} epoch, "
          f"{args.train_images} snimku. Baseline val Dice = {BASELINE['val_dice']}")

    results = [run_variant(v, args.train_images, args.val_images, args.epochs) for v in variants]

    print_summary(results, args.epochs)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    RESULTS_PATH.write_text(json.dumps(previous + results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\nPodrobnosti ulozeny: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
