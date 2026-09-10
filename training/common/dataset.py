"""
Vlastní PyTorch Dataset pro Atlas dataset (obraz + jednokanálová maska tříd 0-6).

Nahrazuje chybějící `AtlasDataset` z vypůjčeného U-Net repozitáře (viz rozbor v
konverzaci - `Src/Atlas/data/` s reálnou implementací v repu chybí). Vstupem je
CSV soubor se sloupci "image","label" (jen názvy souborů) - stejná konvence, jakou
produkuje `scripts/build_fold_split.py` a jakou by četla i `data_utils.get_split_files`
z původního repa, kdyby existovala.

Datová jednotka __getitem__ je JEDEN PATCH, ne jeden snímek: `patches_per_image > 1`
znásobí efektivní délku datasetu tak, že se z každého snímku v jedné epoše náhodně
vyřízne víc patchů (nový náhodný výřez při každém volání, díky náhodnosti v transformu) -
viz `devnotes/teorie/04_data_pipeline_a_inference.md`, kap. 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.transforms import Compose
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    stem: str


def load_pairs_from_csv(csv_path: Path, images_dir: Path, masks_dir: Path) -> list[ImageMaskPair]:
    df = pd.read_csv(csv_path)
    pairs = []
    for _, row in df.iterrows():
        image_path = images_dir / row["image"]
        mask_path = masks_dir / row["label"]
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Chybí pár pro {row['image']}: {image_path} / {mask_path}")
        pairs.append(ImageMaskPair(image_path, mask_path, Path(row["image"]).stem))
    return pairs


class AtlasPatchDataset(Dataset):
    """
    `transform` dostane slovník {"image": (3,H,W) float32, "label": (1,H,W) int}
    a musí vrátit stejné klíče (viz `training/common/transforms.py`). Výstup
    `__getitem__` je (image_tensor[3,P,P], label_onehot_tensor[num_classes,P,P]) -
    label je už PŘEVEDENÝ NA ONE-HOT zde v datasetu (ne v transformu), protože
    ho v tomto tvaru očekává trénovací smyčka (`training/common/loop.py`), stejně
    jako v původním (vypůjčeném) `training_new.py` ("labels = labels.to(device) # one-hot").
    """

    def __init__(
        self,
        pairs: list[ImageMaskPair],
        transform: Compose,
        num_classes: int,
        patches_per_image: int = 1,
    ) -> None:
        self.pairs = pairs
        self.transform = transform
        self.num_classes = num_classes
        self.patches_per_image = max(1, patches_per_image)

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def _load_raw(self, pair: ImageMaskPair) -> dict:
        # Rentgen je fakticky šedotónový (i když originální PNG mají 3 identické
        # kanály), takže se čte jedním kanálem a replikuje se až tady. Ušetří to
        # dekódování dvou zbytečných kanálů oproti IMREAD_COLOR + cvtColor.
        # Model má na vstupu 3 kanály (viz models.py, IN_CHANNELS) kvůli shodě
        # s konfigurací původního repozitáře, proto se kanál ztrojí.
        image_gray = cv2.imread(str(pair.image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Nelze načíst snímek: {pair.image_path}")
        image_chw = np.repeat(image_gray[np.newaxis, :, :], 3, axis=0).astype(np.float32)  # (3,H,W)

        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Nelze načíst masku: {pair.mask_path}")
        mask_chw = mask[np.newaxis, :, :].astype(np.float32)  # (1,H,W)

        return {"image": image_chw, "label": mask_chw}

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx // self.patches_per_image]
        data = self._load_raw(pair)
        data = self.transform(data)

        # RandCropByPosNegLabeld (strategie "pos_neg") vrací SEZNAM výřezů, i když
        # se žádá jen o jeden — na rozdíl od ostatních transformací, které vracejí
        # slovník. Bereme první prvek, jinak by se dál pracovalo se seznamem.
        if isinstance(data, list):
            data = data[0]

        image_t = data["image"].float()
        label_idx = data["label"][0].round().long()
        # Pojistka: i po "nearest" interpolaci u zoomu se teoreticky můžeme ocitnout
        # těsně mimo platný rozsah tříd kvůli float zaokrouhlení na hranici - radši
        # explicitně oříznout, než aby F.one_hot spadl na indexový chybě uprostřed
        # mnohahodinového tréninku.
        label_idx = label_idx.clamp(0, self.num_classes - 1)
        label_onehot = F.one_hot(label_idx, num_classes=self.num_classes).permute(2, 0, 1).float()

        return image_t, label_onehot
