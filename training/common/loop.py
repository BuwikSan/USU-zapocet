"""
Přerušitelná (resumable) trénovací smyčka sdílená U-Netem i UNETR.

Sdílená smyčka = metodologická nutnost (viz `devnotes/teorie/03_loss_metriky_a_trenink.md`,
kap. 3.5 - kontrolovaný experiment: liší se jen architektura, všechno ostatní je fixní).

PŘERUŠITELNOST (kvůli `devnotes/constrains.md` - GPU jen v nočních oknech)
---------------------------------------------------------------------------
- Checkpoint se ukládá po KAŽDÉ epoše (`<run_name>-last.pth`: váhy, optimizer,
  scheduler, scaler, číslo epochy, dosud nejlepší val Dice). Ctrl+C / pád /
  vypnutí stroje uprostřed epochy tedy stojí nanejvýš rozpracovanou epochu
  (řádově jednotky minut), ne celý noční běh.
- `--resume` (výchozí zapnuto v train_*.py) najde poslední checkpoint pro dané
  `run_name` a pokračuje přesně od další epochy - žádné ruční dopočítávání.
- `max_minutes` je měkký časový rozpočet pro JEDNO spuštění skriptu: po
  dokončení aktuální epochy se zkontroluje uplynulý čas, a pokud je limit
  překročen, smyčka se čistě ukončí (uloží checkpoint, vrátí stav). To umožňuje
  spustit trénink na začátku nočního okna s `--max-minutes 480` (8 hodin) a
  nebát se, že poběží přes den.

VALIDACE
--------
Plná `sliding_window_inference` (přes CELÝ, neořezaný validační snímek - snímky
mají různé rozlišení, proto validační DataLoader vždy batch_size=1) se počítá
jen každých `full_val_every` epoch - je řádově dražší než trénovací krok na
patchi (viz `devnotes/dodatek k trénování.md` + reálné benchmarky). Trénovací
Dice se naopak loguje KAŽDOU epochu prakticky zadarmo (spočte se z predikcí,
které už beztak vznikly při tréninkovém forward průchodu).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LoopConfig:
    run_name: str
    num_classes: int
    max_epochs: int
    lr: float
    seed: int
    patch_size: int
    use_amp: bool = True
    sw_batch_size: int = 4
    sw_overlap: float = 0.25
    full_val_every: int = 5
    max_minutes: float | None = None
    scheduler_kind: Literal["cosine", "plateau"] = "cosine"
    t0: int = 5
    t_mult: int = 2
    eta_min: float = 1e-8
    plateau_patience: int = 5
    plateau_ratio: float = 0.8
    grad_clip_norm: float = 1.0
    # Lineární rozjezd learning rate od nuly během prvních N epoch. U transformerů
    # je to běžná praxe: náhodně inicializovaná attention na začátku produkuje
    # nesmyslné gradienty a plný learning rate hned od prvního kroku model
    # rozhodí. U konvolučních sítí to potřeba není (0 = vypnuto).
    warmup_epochs: int = 0
    # Kolik nan/inf hodnot loss po sobě se toleruje, než se běh ukončí jako
    # zdivergovaný. Jednotlivé výpadky se přeskakují, série znamená rozpad tréninku.
    max_nan_streak: int = 20


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: LoopConfig):
    if cfg.scheduler_kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=cfg.plateau_ratio, patience=cfg.plateau_patience)
    return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min)


def _checkpoint_path(checkpoints_dir: Path, run_name: str, suffix: str) -> Path:
    return checkpoints_dir / f"{run_name}-{suffix}.pth"


def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, epoch: int, best_val_dice: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
        },
        path,
    )


def try_resume(path: Path, model, optimizer, scheduler, scaler, device) -> tuple[int, float]:
    """Vrátí (start_epoch, best_val_dice). Když checkpoint neexistuje, vrátí (0, 0.0)."""
    if not path.exists():
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"[resume] Nalezen checkpoint '{path.name}', pokračuji od epochy {ckpt['epoch'] + 1}.")
    return ckpt["epoch"] + 1, ckpt["best_val_dice"]


def _onehot_argmax(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred_class = torch.argmax(logits, dim=1)
    return F.one_hot(pred_class, num_classes=num_classes).permute(0, 3, 1, 2).float()


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: torch.nn.Module,
    cfg: LoopConfig,
    checkpoints_dir: Path,
    runs_dir: Path,
    resume: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Hlasitá diagnostika hardwaru. Trénink běží přes noc bez dohledu a tichý pád
    # na CPU (typicky po přeinstalaci torche na CPU-only build, viz
    # devnotes/PROSTREDI_A_PASTI.md §4.1) by znamenal ~100x pomalejší běh, který
    # by se odhalil až ráno. Radši ať to křičí hned na prvním řádku.
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[hw] GPU: {props.name} | VRAM {props.total_memory / 1e9:.2f} GB "
              f"| torch {torch.__version__} | AMP {'zapnuto' if cfg.use_amp else 'vypnuto'}")
    else:
        print("!" * 70)
        print("[hw] VAROVANI: CUDA NENI DOSTUPNA - trenink pobezi na CPU a bude radove pomalejsi!")
        print(f"[hw] torch = {torch.__version__} (build bez '+cu' = CPU-only)")
        print("[hw] Naprava: py -3 -m pip install --force-reinstall torch "
              "--index-url https://download.pytorch.org/whl/cu128")
        print("!" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = _build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    last_ckpt = _checkpoint_path(checkpoints_dir, cfg.run_name, "last")
    best_ckpt = _checkpoint_path(checkpoints_dir, cfg.run_name, "best")

    start_epoch, best_val_dice = (0, 0.0)
    if resume:
        start_epoch, best_val_dice = try_resume(last_ckpt, model, optimizer, scheduler, scaler, device)

    writer = SummaryWriter(runs_dir / cfg.run_name)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    run_start = time.perf_counter()
    stopped_reason = "max_epochs_reached"

    nan_streak = 0

    for epoch in range(start_epoch, cfg.max_epochs):
        epoch_start = time.perf_counter()

        # ---------------- WARMUP ----------------
        # Během rozjezdu se learning rate nastavuje ručně a scheduler se
        # nechá spát — jinak by si obě logiky přepisovaly hodnotu navzájem.
        in_warmup = cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs
        if in_warmup:
            warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr

        # ---------------- TRAIN ----------------
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches = 0
        dice_metric.reset()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                logits = model(images)
                loss = loss_function(logits, labels)

            # Pojistka proti rozpadu tréninku. Jakmile loss jednou skončí na nan/inf,
            # krok optimizeru by tu hodnotu rozlil do všech vah a od té chvíle už
            # model jen produkuje nan - další hodiny výpočtu jsou k ničemu.
            # Ojedinělý nan se přeskočí (může jít o jednu vadnou dávku), ale série
            # po sobě jdoucích nanů znamená divergenci (typicky moc vysoký learning
            # rate - přesně to potkalo UNETR s lr=1e-2, viz devnotes/STAV_PROJEKTU.md
            # §5.1) a je lepší běh rovnou zastavit s jasnou hláškou.
            if not torch.isfinite(loss):
                nan_streak += 1
                nan_batches += 1
                if nan_streak >= cfg.max_nan_streak:
                    print(f"[{cfg.run_name}] PRERUSENO: {nan_streak} nekonecnych/nan hodnot loss "
                          f"za sebou v epose {epoch}. Trenink diverguje - sniz learning rate "
                          f"(nyni {cfg.lr:.1e}).")
                    stopped_reason = "diverged_nan_loss"
                    break
                continue
            nan_streak = 0

            if cfg.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                # Gradienty se musí "odškálovat" PŘED clipováním, jinak by norm
                # počítal z uměle zvětšených (fp16 loss-scaled) hodnot a klip by
                # neodpovídal skutečné velikosti kroku. Standardní AMP idiom.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            dice_metric(_onehot_argmax(logits.detach(), cfg.num_classes), labels)

        if stopped_reason == "diverged_nan_loss":
            # Checkpoint se tu záměrně NEUKLÁDÁ - poslední uložený stav je z konce
            # předchozí epochy, tedy ještě před rozpadem, a je tak použitelnější.
            break

        train_loss = epoch_loss / max(1, n_batches)
        train_dice = float(dice_metric.aggregate().item())
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/train", train_dice, epoch)
        if nan_batches:
            print(f"[{cfg.run_name}] pozor: v epose {epoch} preskoceno {nan_batches} davek s nan/inf loss")

        # ---------------- VALIDACE (jen periodicky, plná sliding-window) ----------------
        run_full_val = ((epoch + 1) % cfg.full_val_every == 0) or (epoch == cfg.max_epochs - 1)
        val_dice = None
        train_seconds = time.perf_counter() - epoch_start
        val_seconds = 0.0
        if run_full_val:
            val_start = time.perf_counter()
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
                        logits = sliding_window_inference(
                            inputs=images,
                            roi_size=(cfg.patch_size, cfg.patch_size),
                            sw_batch_size=cfg.sw_batch_size,
                            predictor=model,
                            overlap=cfg.sw_overlap,
                            mode="gaussian",
                        )
                    dice_metric(_onehot_argmax(logits, cfg.num_classes), labels)
            val_dice = float(dice_metric.aggregate().item())
            val_seconds = time.perf_counter() - val_start
            writer.add_scalar("Dice/val", val_dice, epoch)

        # ---------------- SCHEDULER ----------------
        if in_warmup:
            current_lr = optimizer.param_groups[0]["lr"]
        elif cfg.scheduler_kind == "plateau":
            scheduler.step(train_loss if val_dice is None else -val_dice)
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            # Kosinový cyklus se počítá od konce rozjezdu, ne od epochy 0 —
            # jinak by warmup jen "snědl" začátek cyklu.
            scheduler.step(epoch - cfg.warmup_epochs)
            current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LR", current_lr, epoch)

        # ---------------- CHECKPOINTY ----------------
        save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler, epoch, best_val_dice)
        if val_dice is not None and val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(best_ckpt, model, optimizer, scheduler, scaler, epoch, best_val_dice)

        val_str = f"{val_dice:.4f}" if val_dice is not None else "-"
        # Časy se vypisují kvůli plánování: trénink běží v omezených oknech, takže
        # je potřeba vidět tempo a odhadnout, kolik epoch se do okna ještě vejde.
        time_str = f"{train_seconds / 60:.1f}min"
        if val_seconds:
            time_str += f"+{val_seconds / 60:.1f}min val"
        print(
            f"[{cfg.run_name}] epoch {epoch}: loss={train_loss:.4f} "
            f"train_dice={train_dice:.4f} val_dice={val_str} lr={current_lr:.2e} ({time_str})"
        )

        if cfg.max_minutes is not None:
            elapsed_min = (time.perf_counter() - run_start) / 60.0
            if elapsed_min >= cfg.max_minutes:
                stopped_reason = "time_budget_exceeded"
                print(f"[{cfg.run_name}] Časový rozpočet {cfg.max_minutes} min vyčerpán po epoše {epoch}. Ukončuji.")
                break

    writer.close()
    return {"best_val_dice": best_val_dice, "last_epoch": epoch, "stopped_reason": stopped_reason}
