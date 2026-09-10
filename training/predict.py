"""
Generování predikcí natrénovaného modelu (U-Net nebo UNETR) na zvolené množině dat.

ROLE V PROJEKTU
---------------
Tohle je most mezi tréninkem a vyhodnocením. Bere hotový checkpoint a vyrábí z něj
**PNG masky ve stejném formátu, jaký má ground truth** (jednokanálové, hodnoty 0-6).

Proč zrovna takhle: všechny tři porovnávané modely (U-Net, UNETR,
TotalSegmentator2D) tím pádem produkují **identický typ artefaktu**. Evaluační
skript pak nemusí vůbec vědět, který model masku vyrobil — jen porovnává PNG proti
PNG. Díky tomu je srovnání férové a nejde do něj protlačit chybu tím, že by se
jeden model vyhodnocoval jinak než druhý.

Praktický důsledek: checkpoint natrénovaný kdekoliv (na desktopu přes noc, na
jiném stroji) stačí nakopírovat do `training_outputs/checkpoints/` a pustit tenhle
skript — nic dalšího se přenášet nemusí.

INFERENCE PŘES CELÝ SNÍMEK
--------------------------
Model je trénovaný na výřezech 256x256, ale predikuje se na celém snímku
v původním rozlišení (~800x1000 px). Používá se `sliding_window_inference`:
okno 256x256 se posouvá přes snímek s překryvem a výsledky se slévají gaussovským
vážením (okraje oken mají menší váhu než střed, takže na švech nevznikají hrany).
Viz devnotes/teorie/04_data_pipeline_a_inference.md, kap. 4.4.

POUŽITÍ
-------
    python training/predict.py --model unet  --split test
    python training/predict.py --model unetr --split test
    python training/predict.py --model unet --checkpoint cesta/k/vlastnimu.pth --limit 5 --preview
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
import torch
from monai.inferers import sliding_window_inference

from training.common import paths
from training.common.config import load_yaml
from training.common.dataset import load_pairs_from_csv
from training.common.models import build_unet, build_unetr
from training.common.transforms import build_eval_transforms

CONFIG_FOR_MODEL = {
    "unet": REPO_ROOT / "training" / "configs" / "unet.yaml",
    "unetr": REPO_ROOT / "training" / "configs" / "unetr.yaml",
}


def load_model(model_kind: str, patch_size: int, checkpoint_path: Path, device: torch.device,
               vit_size: str = "base"):
    """
    Postaví prázdnou architekturu a nalije do ní váhy z checkpointu.

    Checkpointy z `training/common/loop.py` jsou slovníky s klíčem
    `model_state_dict` (plus optimizer, scheduler atd.). Podporuje se ale i holý
    state_dict — kdyby checkpoint přišel odjinud nebo byl uložený jinak.
    """
    model = (build_unet() if model_kind == "unet"
             else build_unetr(patch_size=patch_size, vit_size=vit_size))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        trained_epochs = checkpoint.get("epoch", "?")
        best_dice = checkpoint.get("best_val_dice")
        info = f"epocha {trained_epochs}"
        if best_dice is not None:
            info += f", nejlepsi val Dice {best_dice:.4f}"
    else:
        state_dict = checkpoint
        info = "holy state_dict (bez metadat)"

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[model] {model_kind} nacten z {checkpoint_path.name} ({info})")
    return model


def predict_one(model, image_path: Path, transform, cfg_patch: int, cfg_sw_batch: int,
                cfg_overlap: float, use_amp: bool, device: torch.device) -> np.ndarray:
    """Vrátí masku tříd (H, W) o STEJNÉM rozměru jako vstupní snímek."""
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Nelze nacist snimek: {image_path}")
    original_h, original_w = image_gray.shape

    image_chw = np.repeat(image_gray[np.newaxis, :, :], 3, axis=0).astype(np.float32)
    # Transformace potřebuje i klíč "label"; při inferenci žádnou masku nemáme,
    # tak se podstrčí nulová – projde stejným paddingem jako obraz a zahodí se.
    data = transform({"image": image_chw, "label": np.zeros((1, original_h, original_w), dtype=np.float32)})
    inputs = data["image"].float().unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=(cfg_patch, cfg_patch),
                sw_batch_size=cfg_sw_batch,
                predictor=model,
                overlap=cfg_overlap,
                mode="gaussian",
            )
        prediction = torch.argmax(logits, dim=1)[0].to(torch.uint8).cpu().numpy()

    # Eval transformace jen doplňuje padding do minimální velikosti patche, takže
    # u malých snímků může být výstup větší než originál — ořízne se zpět, aby
    # maska pixel po pixelu odpovídala ground truth.
    return prediction[:original_h, :original_w]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["unet", "unetr"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Cesta k .pth. Když se neuvede, hleda se <run_name>-best.pth a pak -last.pth.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Kam ulozit masky. Vychozi: training_outputs/predictions/<model>/")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Jen prvnich N snimku (rychla kontrola).")
    parser.add_argument("--preview", action="store_true", help="Ulozit i barevny overlay pres original.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Prebije run.name z configu - urcuje, ktery checkpoint se hleda.")
    args = parser.parse_args()

    config_path = args.config or CONFIG_FOR_MODEL[args.model]
    raw = load_yaml(config_path)
    patch_size = raw["data"]["patch_size"]
    run_name = args.run_name or raw["run"]["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[hw] GPU: {torch.cuda.get_device_properties(0).name}")
    else:
        print("[hw] VAROVANI: bezi na CPU, inference bude pomala")

    # Výběr checkpointu: "best" (podle validačního Dice) má přednost před "last".
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        candidates = [paths.CHECKPOINTS_DIR / f"{run_name}-best.pth",
                      paths.CHECKPOINTS_DIR / f"{run_name}-last.pth"]
        checkpoint_path = next((c for c in candidates if c.exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Nenalezen zadny checkpoint pro '{run_name}' v {paths.CHECKPOINTS_DIR}.\n"
                f"Hledalo se: {', '.join(c.name for c in candidates)}\n"
                f"Natrenovany checkpoint sem nakopiruj, nebo uved --checkpoint."
            )

    model = load_model(args.model, patch_size, checkpoint_path, device,
                       vit_size=raw.get("model", {}).get("vit_size", "base"))

    # heqv se řídí stejným pravidlem jako při tréninku — model musí na vstupu
    # dostat data zpracovaná stejně, jinak by se testoval na jiné distribuci.
    use_precomputed = raw["data"].get("use_precomputed_heqv", False) and raw["data"]["heqv"]
    images_dir = paths.IMAGES_HEQV_DIR if use_precomputed else paths.IMAGES_DIR
    transform = build_eval_transforms(patch_size, heqv=raw["data"]["heqv"] and not use_precomputed)

    pairs = load_pairs_from_csv(paths.FOLDS_DIR / f"{args.split}.csv", images_dir, paths.MASKS_DIR)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    out_dir = args.out_dir or (paths.PREDICTIONS_DIR / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"[predict] model={args.model} split={args.split} snimku={len(pairs)} -> {out_dir}")

    start = time.perf_counter()
    for i, pair in enumerate(pairs):
        mask = predict_one(
            model, pair.image_path, transform, patch_size,
            raw["validation"]["sw_batch_size"], raw["validation"]["sw_overlap"],
            raw["amp"], device,
        )
        cv2.imwrite(str(out_dir / f"{pair.stem}.png"), mask)

        if args.preview:
            # Stejná paleta jako u ground-truth masek, aby šly obrázky porovnávat
            # vedle sebe bez přemýšlení, co která barva znamená.
            from scripts.rasterize_masks import CLASS_PALETTE_BGR
            original = cv2.imread(str(paths.IMAGES_DIR / f"{pair.stem}.png"), cv2.IMREAD_COLOR)
            overlay = cv2.addWeighted(original, 0.6, CLASS_PALETTE_BGR[mask], 0.4, 0.0)
            cv2.imwrite(str(preview_dir / f"{pair.stem}_pred.png"), overlay)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i + 1}/{len(pairs)} ({elapsed / (i + 1):.2f} s/snimek)")

    total = time.perf_counter() - start
    print(f"[predict] hotovo: {len(pairs)} masek za {total:.1f} s ({total / max(1, len(pairs)):.2f} s/snimek)")
    print(f"[predict] vystup: {out_dir}")


if __name__ == "__main__":
    main()
