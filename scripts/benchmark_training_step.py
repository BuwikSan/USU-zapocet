"""
Rychlý empirický benchmark rychlosti jednoho tréninkového kroku (forward + backward +
optimizer.step) pro MONAI UNet / UNETR na aktuální GPU.

PROČ TENTO SKRIPT EXISTUJE
--------------------------
`devnotes/dodatek k trénování.md` obsahuje odhad "30-45 minut na epochu" vycházející
z původního configu (patch 512, patches_per_image 20, batch 2, num_workers 0) na
desktopové RTX 3060 12GB. Vzhledem k tvrdému deadline (`devnotes/constrains.md`,
13.9.) a omezené dostupnosti desktopu (jen noční okna) je potřeba mít PŘED
rozhodnutím o finální konfiguraci (rozlišení patchů, batch size, počet epoch)
REÁLNĚ NAMĚŘENÁ čísla, ne odhad odhadu. Tento skript běží přímo na cílovém GPU
(ať už 6GB notebook, nebo 12GB desktop) a řekne, kolik sekund trvá jeden krok
při dané konfiguraci - z toho se pak snadno spočítá čas na epochu i na celý trénink.

Používá čistě syntetická (náhodná) data - neřeší se tu nic o obsahu snímků, jen o
výpočetní/paměťové náročnosti dané kombinace (architektura, rozlišení, batch size).

POUŽITÍ
-------
    python scripts/benchmark_training_step.py --model unet  --patch 256 --batch 8
    python scripts/benchmark_training_step.py --model unetr --patch 256 --batch 4
    python scripts/benchmark_training_step.py --model unetr --patch 512 --batch 2 --amp
"""

from __future__ import annotations

import argparse
import sys
import time

# Konzole na Windows (cp1252) neumí vytisknout diakritiku bez explicitního UTF-8 -
# tento skript se má spouštět i přímo na desktopu v noci bez dohledu, proto raději
# přepínáme kódování výstupu, než abychom se spoléhali na diakritiku-less texty.
sys.stdout.reconfigure(encoding="utf-8")

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR, UNet

NUM_CLASSES = 7
IN_CHANNELS = 3


def build_unet(patch: int) -> torch.nn.Module:
    # Stejná konfigurace jako v atlas_multiclass_patch.yaml (channels/strides/num_res_units) -
    # měníme jen rozlišení vstupu, ne architekturu, aby šlo srovnání interpretovat čistě
    # jako "co udělá zmenšení patchu", ne "jiná síť".
    return UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_unetr(patch: int) -> torch.nn.Module:
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=NUM_CLASSES,
        img_size=(patch, patch),
        spatial_dims=2,
        feature_size=16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["unet", "unetr"], required=True)
    parser.add_argument("--patch", type=int, default=512, help="Velikost čtvercového patchu (px).")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=15, help="Počet měřených kroků (po warmupu).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Zapnout automatic mixed precision (fp16).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA není dostupná - benchmark bez GPU nemá smysl (viz hardwarová omezení v claude.md).")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # tady chceme rychlost, ne determinismus (na rozdíl od skutečného tréninku)

    model = (build_unet(args.patch) if args.model == "unet" else build_unetr(args.patch)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Syntetická data: obraz [B,3,H,W], one-hot label [B,7,H,W] (přesně tvar, co produkuje
    # skutečný AtlasDataset po transformaci - viz training/common/dataset.py).
    x = torch.randn(args.batch, IN_CHANNELS, args.patch, args.patch, device=device)
    y_idx = torch.randint(0, NUM_CLASSES, (args.batch, args.patch, args.patch), device=device)
    y = torch.nn.functional.one_hot(y_idx, NUM_CLASSES).permute(0, 3, 1, 2).float()

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()

    peak_mem_before = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sec_per_step = elapsed / args.steps
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(f"model={args.model} patch={args.patch} batch={args.batch} amp={args.amp}")
    print(f"parametry: {n_params / 1e6:.1f} M")
    print(f"sekund/krok: {sec_per_step:.3f}")
    print(f"kroků/s: {1 / sec_per_step:.2f}")
    print(f"špička VRAM: {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
