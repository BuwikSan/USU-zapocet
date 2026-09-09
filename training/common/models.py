"""
Tovární funkce pro oba porovnávané modely. Obě sítě mají STEJNÝ vstup/výstup
(in_channels=3, out_channels=7, čtvercový obraz) - to je podmínka pro férové
srovnání architektur, viz `devnotes/teorie/00_prehled.md`.

U-Net: konfigurace 1:1 podle `atlas_multiclass_patch.yaml` (channels/strides/
num_res_units) - viz `devnotes/teorie/01_konvoluce_a_unet.md`.

UNETR: `feature_size=16` (MONAI výchozí je 16) - benchmark (`scripts/
benchmark_training_step.py`) potvrdil, že na 256px patchi a batch 8 s AMP se
vejde do ~3 GB VRAM, tedy běží beze změny na 6GB notebooku i 12GB desktopu.
"""

from __future__ import annotations

from monai.networks.nets import UNETR, UNet

NUM_CLASSES = 7
IN_CHANNELS = 3


def build_unet(num_classes: int = NUM_CLASSES) -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_unetr(patch_size: int, num_classes: int = NUM_CLASSES) -> UNETR:
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=num_classes,
        img_size=(patch_size, patch_size),
        spatial_dims=2,
        feature_size=16,
    )
