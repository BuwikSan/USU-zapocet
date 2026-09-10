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


# Velikosti Vision Transformeru uvnitř UNETR: (hidden_size, mlp_dim, num_heads).
#
# Výchozí nastavení MONAI odpovídá ViT-Base (768/3072/12) — tedy velikosti
# z původní publikace, kde ale trénovali na řádově větších datech. Na našich
# 2978 snímcích to znamená 116 M parametrů proti 1,6 M u U-Netu, tedy 72násobek.
# Takto předimenzovaný model se v malém datovém režimu učí pomalu; zkušební
# běhy to potvrdily (viz devnotes/STAV_PROJEKTU.md §8.10).
#
# Menší varianty jsou proto legitimní přizpůsobení modelu velikosti dat, ne
# ochuzení experimentu — naopak přibližují kapacitu UNETR k U-Netu, takže je
# srovnání architektur poctivější.
VIT_SIZES: dict[str, tuple[int, int, int]] = {
    "base": (768, 3072, 12),   # ~116 M parametru, 166 ms/davka
    "small": (384, 1536, 6),   # ~30 M parametru,   89 ms/davka
    "tiny": (192, 768, 3),     # ~8.4 M parametru,  70 ms/davka
}


def build_unetr(patch_size: int, num_classes: int = NUM_CLASSES,
                vit_size: str = "base") -> UNETR:
    if vit_size not in VIT_SIZES:
        raise ValueError(f"Neznama velikost ViT: {vit_size!r} (k dispozici: {', '.join(VIT_SIZES)})")
    hidden_size, mlp_dim, num_heads = VIT_SIZES[vit_size]
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=num_classes,
        img_size=(patch_size, patch_size),
        spatial_dims=2,
        feature_size=16,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
    )
