"""
Proof of concept — zmenšený běh celé studie na jednom stroji.

K ČEMU TO JE
------------
Ostrý trénink je nevratná investice nočního okna na sdílené grafice. Než se do
něj pustíme, je potřeba mít ověřené, že **celý řetěz funguje a produkuje
smysluplná čísla** — ne jen že se nic nerozbije.

Tenhle skript projde úplně stejnou posloupnost kroků jako ostrý běh:

    trénink U-Net -> trénink UNETR -> predikce obou -> inference TS2D -> evaluace

...jen na zlomku dat a s málo epochami, aby se to vešlo do desítek minut na
6GB notebooku.

CO SE TÍM OVĚŘÍ
---------------
1. Oba modely se skutečně **učí** (loss klesá, Dice roste nad nulu). Kdyby Dice
   zůstal na nule, znamená to problém v pipeline nebo v hyperparametrech — a je
   lepší to vědět teď než po osmi hodinách.
2. Checkpointy jdou uložit, načíst a použít k predikci.
3. Evaluace projde a vyrobí srovnávací tabulku i obrázky.
4. Naměří se reálné časy, ze kterých jde spočítat délka ostrého běhu.

CO SE TÍM NEOVĚŘÍ
-----------------
**Kvalita modelů.** Na pár stovkách snímků a desítce epoch nemůže vzniknout
model použitelný do závěrečného srovnání. Výsledná čísla jsou orientační a do
zprávy nepatří — slouží jen k porovnání "učí se / neučí se".

ODDĚLENÍ OD OSTRÉHO BĚHU
------------------------
Všechny běhy mají vlastní `run_name` s předponou `poc-`, takže si checkpointy
ani TensorBoard logy nekolidují s ostrým tréninkem a `--resume` na ně omylem
nenaváže. Predikce jdou do `training_outputs/predictions-poc/`.

POUŽITÍ
-------
    python scripts/run_poc.py
    python scripts/run_poc.py --train-images 200 --epochs 8 --skip-ts2d
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

PREDICTIONS_DIR = REPO_ROOT / "training_outputs" / "predictions-poc"
EVAL_DIR = REPO_ROOT / "training_outputs" / "evaluation-poc"

NOISE = ("oneDNN", "absl::InitializeLog", "UserWarning", "win_data =", "out[idx_zm]",
         "WARNING: All log messages")


def run(*args: str) -> tuple[int, float]:
    """Spustí krok, proudem vypisuje jeho výstup a vrátí (návratový kód, trvání)."""
    command = ["py", "-3", "-u", *args]
    print(f"\n$ {' '.join(command)}", flush=True)
    start = time.perf_counter()

    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if not any(noise in line for noise in NOISE):
            print("  " + line.rstrip(), flush=True)
    process.wait()

    elapsed = time.perf_counter() - start
    print(f"  -> navratovy kod {process.returncode}, trvani {elapsed / 60:.1f} min", flush=True)
    return process.returncode, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-images", type=int, default=400, help="Kolik trenovacich snimku pouzit.")
    parser.add_argument("--val-images", type=int, default=40, help="Kolik validacnich snimku pouzit.")
    parser.add_argument("--test-images", type=int, default=60, help="Na kolika testovacich snimcich vyhodnotit.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--unetr-epochs", type=int, default=None,
                        help="UNETR ma drazsi krok; vychozi je stejny pocet jako --epochs.")
    parser.add_argument("--skip-unetr", action="store_true")
    parser.add_argument("--skip-ts2d", action="store_true")
    parser.add_argument("--prefix", type=str, default="poc")
    args = parser.parse_args()

    unetr_epochs = args.unetr_epochs if args.unetr_epochs is not None else args.epochs
    unet_run = f"{args.prefix}-unet"
    unetr_run = f"{args.prefix}-unetr"

    print("=" * 78)
    print("PROOF OF CONCEPT — zmenseny beh cele studie")
    print("=" * 78)
    print(f"trenovacich snimku : {args.train_images}")
    print(f"validacnich snimku : {args.val_images}")
    print(f"testovacich snimku : {args.test_images}")
    print(f"epoch U-Net/UNETR  : {args.epochs} / {unetr_epochs}")
    print(f"nazvy behu         : {unet_run}, {unetr_run}")
    print("\nPOZOR: vysledna cisla jsou orientacni, do zpravy nepatri.")

    timings: dict[str, float] = {}
    failed: list[str] = []

    def step(label: str, *command: str) -> bool:
        code, elapsed = run(*command)
        timings[label] = elapsed
        if code != 0:
            failed.append(label)
            print(f"  !! KROK '{label}' SELHAL")
            return False
        return True

    # ---- 1. trénink ----
    print("\n" + "=" * 78 + "\n1/4  TRENINK\n" + "=" * 78)
    unet_ok = step("trenink U-Net", "training/train_unet.py",
                   "--run-name", unet_run, "--max-epochs", str(args.epochs),
                   "--limit-train", str(args.train_images), "--limit-val", str(args.val_images),
                   "--full-val-every", "3", "--no-resume")

    unetr_ok = False
    if not args.skip_unetr:
        unetr_ok = step("trenink UNETR", "training/train_unetr.py",
                        "--run-name", unetr_run, "--max-epochs", str(unetr_epochs),
                        "--limit-train", str(args.train_images), "--limit-val", str(args.val_images),
                        "--full-val-every", "3", "--no-resume")

    # ---- 2. predikce natrénovaných modelů ----
    print("\n" + "=" * 78 + "\n2/4  PREDIKCE\n" + "=" * 78)
    if unet_ok:
        step("predikce U-Net", "training/predict.py", "--model", "unet",
             "--run-name", unet_run, "--split", "test", "--limit", str(args.test_images),
             "--out-dir", str(PREDICTIONS_DIR / "unet"), "--preview")
    if unetr_ok:
        step("predikce UNETR", "training/predict.py", "--model", "unetr",
             "--run-name", unetr_run, "--split", "test", "--limit", str(args.test_images),
             "--out-dir", str(PREDICTIONS_DIR / "unetr"), "--preview")

    # ---- 3. TotalSegmentator ----
    print("\n" + "=" * 78 + "\n3/4  TOTALSEGMENTATOR2D\n" + "=" * 78)
    if args.skip_ts2d:
        print("  (preskoceno)")
    else:
        step("inference TS2D", "scripts/ts2d_inference.py", "--split", "test",
             "--limit", str(args.test_images), "--out-dir", str(PREDICTIONS_DIR / "ts2d"))

    # ---- 4. evaluace ----
    print("\n" + "=" * 78 + "\n4/4  EVALUACE\n" + "=" * 78)
    step("evaluace", "scripts/evaluate_models.py", "--split", "test",
         "--limit", str(args.test_images), "--predictions-dir", str(PREDICTIONS_DIR),
         "--out-dir", str(EVAL_DIR), "--figures", "3")

    # ---- shrnutí ----
    print("\n" + "=" * 78)
    print("SHRNUTI")
    print("=" * 78)
    for label, seconds in timings.items():
        print(f"  {label:20s} {seconds / 60:6.1f} min")
    print(f"  {'CELKEM':20s} {sum(timings.values()) / 60:6.1f} min")

    if failed:
        print(f"\n  SELHALO: {', '.join(failed)}")
    else:
        print("\n  Vsechny kroky probehly.")

    # Extrapolace na ostrý běh — hlavní praktický přínos tohohle skriptu.
    full_train_images = 2978
    scale = full_train_images / max(1, args.train_images)
    print("\n  Odhad ostreho behu (prepocteno na cely trenovaci set):")
    for label in ("trenink U-Net", "trenink UNETR"):
        if label in timings:
            per_epoch = timings[label] / max(1, args.epochs) * scale
            print(f"    {label:16s} ~{per_epoch / 60:5.1f} min/epocha  "
                  f"-> 30 epoch {per_epoch * 30 / 3600:4.1f} h, 60 epoch {per_epoch * 60 / 3600:4.1f} h")

    print(f"\n  Vysledky: {EVAL_DIR}")
    print(f"  Predikce: {PREDICTIONS_DIR}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
