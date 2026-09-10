"""
Trénink UNETR od nuly na datasetu Atlas (Fáze D, revidovaný rozsah - viz
devnotes/claude.md a devnotes/dodatek k trénování.md).

Sdílí úplně stejnou trénovací smyčku, dataset, transformy a split jako
train_unet.py (training/common/) - liší se JEN samotný model, viz
devnotes/teorie/03_loss_metriky_a_trenink.md kap. 3.5 (proč je to důležité).

    python training/train_unetr.py --max-minutes 480   # noční okno na desktopu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Windows konzole (cp1252) neumí diakritiku v print() výstupu tréninkové smyčky -
# skript běží i bez dohledu přes noc, proto radši UTF-8 napevno.
sys.stdout.reconfigure(encoding="utf-8")

from monai.losses import DiceCELoss

from training.common import paths
from training.common.config import build_loop_config, load_yaml
from training.common.data_setup import build_dataloaders
from training.common.loop import run_training
from training.common.models import build_unetr
from training.common.seeding import set_seed

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "unetr.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--limit-train", type=int, default=None, help="Jen pro rychlý smoke-test: použít prvních N trénovacích snímků.")
    parser.add_argument("--limit-val", type=int, default=None, help="Jen pro rychlý smoke-test: použít prvních N validačních snímků.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Přebije run.name z configu. Odděluje checkpointy zkušebních běhů od ostrých.")
    parser.add_argument("--full-val-every", type=int, default=None, help="Přebije run.full_val_every.")
    # Přepínače pro ladění hyperparametrů bez zásahu do configu — ať jde pustit
    # sérii krátkých experimentů, aniž by se soubor pořád přepisoval.
    parser.add_argument("--lr", type=float, default=None, help="Přebije optimizer.lr.")
    parser.add_argument("--scheduler-kind", choices=["cosine", "plateau"], default=None)
    parser.add_argument("--t0", type=int, default=None,
                        help="Přebije scheduler.t0 (délka prvního kosinového cyklu v epochách).")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Počet epoch lineárního rozjezdu learning rate od nuly.")
    parser.add_argument("--crop-strategy", choices=["random", "pos_neg"], default=None,
                        help="'pos_neg' cílí výřezy na obratle místo rovnoměrně náhodné pozice.")
    parser.add_argument("--vit-size", choices=["base", "small", "tiny"], default=None,
                        help="Velikost Vision Transformeru uvnitř UNETR (viz models.py::VIT_SIZES).")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.max_epochs is not None:
        raw["run"]["max_epochs"] = args.max_epochs
    if args.max_minutes is not None:
        raw["run"]["max_minutes"] = args.max_minutes
    if args.run_name is not None:
        raw["run"]["name"] = args.run_name
    if args.full_val_every is not None:
        raw["run"]["full_val_every"] = args.full_val_every
    if args.lr is not None:
        raw["optimizer"]["lr"] = args.lr
    if args.scheduler_kind is not None:
        raw["scheduler"]["kind"] = args.scheduler_kind
    if args.t0 is not None:
        raw["scheduler"]["t0"] = args.t0
    if args.warmup_epochs is not None:
        raw["scheduler"]["warmup_epochs"] = args.warmup_epochs
    if args.crop_strategy is not None:
        raw["data"]["crop_strategy"] = args.crop_strategy
    if args.vit_size is not None:
        raw["model"] = {**raw.get("model", {}), "vit_size": args.vit_size}

    set_seed(raw["run"]["seed"])

    train_loader, val_loader = build_dataloaders(raw, limit_train=args.limit_train, limit_val=args.limit_val)
    vit_size = raw.get("model", {}).get("vit_size", "base")
    model = build_unetr(patch_size=raw["data"]["patch_size"], vit_size=vit_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNETR ViT-{vit_size}: {n_params / 1e6:.1f} M parametru")
    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    loop_cfg = build_loop_config(raw)

    print(f"[train_unetr] run={loop_cfg.run_name} train_batches/epoch={len(train_loader)} val_images={len(val_loader)}")
    result = run_training(
        model, train_loader, val_loader, loss_fn, loop_cfg,
        checkpoints_dir=paths.CHECKPOINTS_DIR, runs_dir=paths.RUNS_DIR, resume=args.resume,
    )
    print(f"[train_unetr] hotovo: {result}")


if __name__ == "__main__":
    main()
