"""
Trénink U-Net od nuly na datasetu Atlas (Fáze D, revidovaný rozsah - viz
devnotes/claude.md a devnotes/dodatek k trénování.md).

    python training/train_unet.py
    python training/train_unet.py --max-minutes 480          # noční okno na desktopu
    python training/train_unet.py --max-epochs 5 --no-resume  # rychlý sanity-check
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
from training.common.models import build_unet
from training.common.seeding import set_seed

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "unet.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-epochs", type=int, default=None, help="Přebije run.max_epochs z configu.")
    parser.add_argument("--max-minutes", type=float, default=None, help="Přebije run.max_minutes z configu (časový rozpočet jednoho spuštění).")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignorovat existující checkpoint a začít znovu od epochy 0.")
    parser.add_argument("--limit-train", type=int, default=None, help="Jen pro rychlý smoke-test: použít prvních N trénovacích snímků.")
    parser.add_argument("--limit-val", type=int, default=None, help="Jen pro rychlý smoke-test: použít prvních N validačních snímků.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Přebije run.name z configu. Odděluje checkpointy zkušebních běhů od ostrých.")
    parser.add_argument("--full-val-every", type=int, default=None, help="Přebije run.full_val_every.")
    parser.add_argument("--crop-strategy", choices=["random", "pos_neg"], default=None,
                        help="'pos_neg' cílí výřezy na obratle místo rovnoměrně náhodné pozice.")
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
    if args.crop_strategy is not None:
        raw["data"]["crop_strategy"] = args.crop_strategy

    set_seed(raw["run"]["seed"])

    train_loader, val_loader = build_dataloaders(raw, limit_train=args.limit_train, limit_val=args.limit_val)
    model = build_unet()
    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    loop_cfg = build_loop_config(raw)

    print(f"[train_unet] run={loop_cfg.run_name} train_batches/epoch={len(train_loader)} val_images={len(val_loader)}")
    result = run_training(
        model, train_loader, val_loader, loss_fn, loop_cfg,
        checkpoints_dir=paths.CHECKPOINTS_DIR, runs_dir=paths.RUNS_DIR, resume=args.resume,
    )
    print(f"[train_unet] hotovo: {result}")


if __name__ == "__main__":
    main()
