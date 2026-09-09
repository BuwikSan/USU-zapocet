"""
Reprodukovatelnost - vlastní verze inspirovaná `Src/Utils/replicability.py`
z vypůjčeného U-Net repozitáře (stejný princip, nezávislý soubor, viz
`devnotes/teorie/03_loss_metriky_a_trenink.md`, kap. 3.7 pro vysvětlení PROČ
je každý z těchto kroků potřeba).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Nastaví seed všech zdrojů náhodnosti používaných v pipeline na jednu pevnou hodnotu."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismus cuDNN algoritmů - viz benchmark_training_step.py, kde je naopak
    # záměrně VYPNUTÝ (tam chceme rychlost, tady reprodukovatelnost skutečného tréninku).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """
    Každý DataLoader worker běží v samostatném procesu s vlastním stavem RNG -
    bez explicitního seedování by všichni workeři generovali STEJNOU náhodnou
    augmentaci (typická skrytá chyba). `base_seed + worker_id` dá každému worker
    procesu jiný, ale deterministický seed.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Seedovaný generátor pro `torch.utils.data.DataLoader(..., generator=...)` (řídí pořadí shuffle)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
