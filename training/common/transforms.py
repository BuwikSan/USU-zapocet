"""
MONAI transformační pipeline pro trénink a validaci - vlastní implementace, silně
inspirovaná přístupem v `Modely/vertebra_segmentation_and_keypoint_extraction-main/
Src/Atlas/training/atlas_model_multiclass_patch.py::get_train_transformation` /
`get_test_transformation`, ale přepsaná do nových souborů (originál se nemění,
viz `devnotes/claude.md`).

Vysvětlení JEDNOTLIVÝCH kroků (proč flip/rotate/zoom/šum, proč patch-based
trénink) je v `devnotes/teorie/04_data_pipeline_a_inference.md`, kap. 4.2-4.3.
"""

from __future__ import annotations

import cv2
import numpy as np
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    RandZoomd,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord,
)


class HistogramEqualizationd(MapTransform):
    """
    Adaptivní vyrovnání histogramu (CLAHE) na obrazovém kanálu - zvýrazní kontrast
    měkkých/kostních struktur na rentgenu. Vlastní implementace stejného principu,
    jaký používá vypůjčený repozitář (`Src/Utils/image_utils.py`), přepsaná do
    tohoto souboru, aby na starém repu nezávisela žádná spustitelná část nového kódu.
    """

    # POZOR: cv2.CLAHE objekt se NESMÍ ukládat jako atribut instance - na Windows
    # DataLoader s num_workers>0 používá "spawn" (ne "fork"), takže se celý transform
    # pickluje a posílá do worker procesů, a cv2.CLAHE (C++ wrapper) není picklovatelný
    # (padalo by to na "cannot pickle 'cv2.CLAHE' object"). Proto se vytváří nově při
    # každém volání - režie je zanedbatelná vůči vlastnímu výpočtu.
    def __call__(self, data):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if img.ndim == 2:
                d[key] = clahe.apply(img.astype(np.uint8))
            else:
                d[key] = np.stack([clahe.apply(c.astype(np.uint8)) for c in img])
        return d


def build_train_transforms(
    patch_size: int,
    *,
    heqv: bool,
    rand_flip_prob: float,
    rand_rotate90_prob: float,
    rand_rotate90_max_k: int,
    rand_zoom_min: float,
    rand_zoom_max: float,
    rand_zoom_prob: float,
    rand_gaussian_noise_prob: float,
    seed: int,
) -> Compose:
    """
    Trénovací pipeline: (volitelně HEQV) -> normalizace intenzity -> padding na
    alespoň `patch_size` -> náhodný výřez `patch_size`x`patch_size` (= "jeden patch"
    z `patches_per_image`, viz dataset.py) -> geometrické a šumové augmentace ->
    tenzory. Aplikuje se STEJNĚ na "image" i "label" tam, kde jde o geometrickou
    transformaci (jinak by anotace přestala sedět na obraz), šum jen na "image".
    """
    steps = []
    if heqv:
        steps.append(HistogramEqualizationd(keys=["image"]))
    steps += [
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(patch_size, patch_size)),
        RandSpatialCropd(keys=["image", "label"], roi_size=(patch_size, patch_size), random_size=False),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=rand_flip_prob),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=rand_flip_prob),
        RandRotate90d(keys=["image", "label"], prob=rand_rotate90_prob, max_k=rand_rotate90_max_k),
        # POZOR: label je diskrétní mapa tříd (0-6), ne spojitý obraz - se
        # zoomem se MUSÍ interpolovat metodou "nearest" (nejbližší soused),
        # jinak by bilineární interpolace na hranici dvou obratlů vytvořila
        # nesmyslné mezihodnoty (např. průměr tříd 2 a 3), které nejsou žádnou
        # skutečnou třídou. Obraz naopak "bilinear" zůstává (spojitá intenzita).
        RandZoomd(
            keys=["image", "label"],
            min_zoom=rand_zoom_min, max_zoom=rand_zoom_max, prob=rand_zoom_prob,
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=rand_gaussian_noise_prob),
        ToTensord(keys=["image", "label"]),
    ]
    transforms = Compose(steps)
    transforms.set_random_state(seed)
    return transforms


def build_eval_transforms(patch_size: int, *, heqv: bool) -> Compose:
    """
    Validační/testovací pipeline: BEZ náhodných augmentací (chceme stabilní,
    opakovatelné vyhodnocení) a BEZ ořezu na patch - jen padding na násobek
    `patch_size`, aby na výsledek šla pustit `sliding_window_inference` přes
    celý snímek (viz `devnotes/teorie/04_data_pipeline_a_inference.md`, kap. 4.4).
    """
    steps = []
    if heqv:
        steps.append(HistogramEqualizationd(keys=["image"]))
    steps += [
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(patch_size, patch_size)),
        ToTensord(keys=["image", "label"]),
    ]
    return Compose(steps)
