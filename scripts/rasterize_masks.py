"""
Fáze A zápočtové studie: rasterizace anotací datasetu Atlas.

KONTEXT
-------
Anotace datasetu Atlas (`Inputdata/datasets-JSON/*.json`) NEJSOU klasické
"obtahové" polygony (desítky bodů kopírujících hranu obratle), ale
formát LabelMe s bodovými landmarky (`shape_type` = "point" nebo "polygon",
ale vždy jen s jedním bodem v poli "points"). Pro každý krční obratel jsou
zaznamenány 4 rohové body:

    "<kód obratle> top left"
    "<kód obratle> top right"
    "<kód obratle> bottom right"
    "<kód obratle> bottom left"

s výjimkou obratle C2 (axis), který má kvůli svému anatomicky odlišnému
tvaru (odontoidní výběžek) zaznamenané jen 3 body:

    "C2 bottom left"
    "C2 bottom right"
    "C2 centroid"

Tento skript tyto rohové landmarky pro každý obratel poskládá zpět do
polygonu (čtyřúhelník pro C3-C7, trojúhelník pro C2) a ten "vyplní"
(rasterizuje) do jednokanálové PNG masky, kde hodnota pixelu = index třídy:

    0 = pozadí, 1 = C2, 2 = C3, 3 = C4, 4 = C5, 5 = C6, 6 = C7

Tato maska je vstupem pro trénink segmentačních sítí (U-Net i UNETR) — je to
proto Fáze A z `devnotes/claude.md`. Hodnoty 0-6 jsou v běžném prohlížeči
prakticky nerozeznatelné od černé (jde jen o pár úrovní z 256), proto skript
vedle trénovací masky ukládá i barevnou verzi (jasné, vzájemně odlišné barvy
pro C2-C7) čistě pro vizuální kontrolu okem - ta se NEPOUŽÍVÁ k tréninku,
jen k tomu, aby šlo na první pohled ověřit, že rasterizace sedí na anatomii.

Přesné schéma JSON bylo ověřeno na
reálném vzorku (`examples/0001035.json`) i statisticky přes celý dataset
(všech 4963 souborů dodržuje stejnou strukturu: 6 obratlů C2-C7, 23 shapes,
kromě jednoho souboru s duplicitním landmarkem - viz `_group_shapes_by_vertebra`).

POUŽITÍ
-------
    python scripts/rasterize_masks.py --limit 20
        -> rychlý test na 20 snímcích (Fáze 1 z plan.md, vývoj na notebooku)

    python scripts/rasterize_masks.py
        -> zpracuje celý dataset (Fáze 3, ostrý běh na desktopu)

    python scripts/rasterize_masks.py --preview 3
        -> navíc uloží 3 kontrolní obrázky s barevně "prosvícenou" maskou
           přes originální snímek, pro vizuální ověření správnosti rasterizace
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Repo root = o jednu úroveň výš než tento soubor (scripts/..).
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_JSON_DIR = REPO_ROOT / "Inputdata" / "datasets-JSON"
DEFAULT_PNG_DIR = REPO_ROOT / "Inputdata" / "datasets-PNG"
DEFAULT_OUT_DIR = REPO_ROOT / "Inputdata" / "datasets-MASK"
DEFAULT_COLOR_OUT_DIR = REPO_ROOT / "Inputdata" / "datasets-MASK-color"

# Pořadí, ve kterém se obratle mapují na indexy tříd 1-6. Pozadí je vždy 0.
# Pořadí odpovídá anatomickému sledu C2 (nejvýš) -> C7 (nejníž), stejně jako
# v `plan.md` ("pozadí + 6 anatomických struktur").
VERTEBRA_ORDER: tuple[str, ...] = ("C2", "C3", "C4", "C5", "C6", "C7")
CLASS_ID: dict[str, int] = {code: idx + 1 for idx, code in enumerate(VERTEBRA_ORDER)}

# Rohy, které musí být přítomné, aby šlo pro daný obratel sestavit polygon.
# Pořadí v n-tici je zároveň pořadí vrcholů polygonu předávané do cv2.fillPoly
# - musí jít "po obvodu" (ne napříč), jinak by vznikl samoprotínající se tvar.
CORNERS_QUAD: tuple[str, ...] = ("top left", "top right", "bottom right", "bottom left")
CORNERS_C2: tuple[str, ...] = ("bottom left", "bottom right", "centroid")

# Jasná, vzájemně dobře odlišitelná paleta pro 6 tříd obratlů (index = BGR barva,
# index 0 = pozadí = černá). Používá se jak pro barevné masky určené k vizuální
# kontrole okem, tak pro X-ray overlay náhledy. Pořadí odpovídá CLASS_ID
# (index 1 = C2, ..., index 6 = C7), takže paleta[mask] přímo obarví masku.
CLASS_PALETTE_BGR: np.ndarray = np.array(
    [
        [0, 0, 0],        # 0 pozadí - černá
        [255, 64, 64],    # 1 C2 - jasně modrá
        [64, 220, 64],    # 2 C3 - jasně zelená
        [64, 64, 255],    # 3 C4 - jasně červená
        [0, 220, 255],    # 4 C5 - jasně žlutá
        [255, 64, 220],   # 5 C6 - jasně purpurová
        [64, 255, 255],   # 6 C7 - jasně tyrkysová
    ],
    dtype=np.uint8,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rasterize_masks")


@dataclass
class RasterizeStats:
    """Souhrnné počty pro závěrečné hlášení - kolik snímků/obratlů se povedlo."""

    processed_images: int = 0
    skipped_images: int = 0
    incomplete_vertebrae: int = 0
    duplicate_landmarks: int = 0


def _group_shapes_by_vertebra(
    shapes: list[dict], stats: RasterizeStats
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Převede plochý seznam LabelMe "shapes" na strukturu {kód_obratle: {roh: (x, y)}}.

    Label ve zdrojovém JSONu má tvar "<kód> <roh>", např. "C4 top left" nebo
    "C2 centroid" (roh může sám obsahovat mezeru, proto se dělí jen na první
    mezeře pomocí `str.partition`).

    V jednom souboru z celého datasetu (2548130.json) se vyskytuje duplicitní
    landmark ("C3 top right" dvakrát) - pravděpodobně chyba při ruční anotaci.
    Řešení: později uvedený bod přepíše dřívější (jednoduché přiřazení do
    slovníku) a případ se pouze zaloguje, aby nezastavil zpracování celého
    datasetu kvůli jednomu vadnému souboru.
    """
    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for shape in shapes:
        vertebra, _, corner = shape["label"].strip().partition(" ")
        if vertebra not in CLASS_ID:
            # Landmark mimo C2-C7 (v datasetu se nevyskytuje, ale radši
            # nespoléhat na to napevno - ignorujeme neznámé kódy).
            continue

        corner_points = grouped.setdefault(vertebra, {})
        if corner in corner_points:
            stats.duplicate_landmarks += 1
            log.debug("Duplicitní landmark '%s %s', přepisuji starší hodnotu.", vertebra, corner)

        # LabelMe ukládá bod jako [[x, y]] i pro shape_type "point" i "polygon"
        # (v tomto datasetu má "polygon" vždy jen jeden bod - viz docstring modulu).
        x, y = shape["points"][0]
        corner_points[corner] = (float(x), float(y))

    return grouped


def _vertebra_polygon(vertebra: str, corners: dict[str, tuple[float, float]]) -> np.ndarray | None:
    """
    Sestaví pole vrcholů polygonu pro daný obratel, nebo None, pokud v anotaci
    chybí některý z povinných rohů (neúplná anotace - vzácné, ale nekrachujeme
    na tom celý běh, jen daný obratel v daném snímku vynecháme).
    """
    required_corners = CORNERS_C2 if vertebra == "C2" else CORNERS_QUAD
    if not all(corner in corners for corner in required_corners):
        return None

    points = [corners[corner] for corner in required_corners]
    # cv2.fillPoly očekává integer souřadnice ve tvaru (N, 2).
    return np.round(np.array(points, dtype=np.float64)).astype(np.int32)


def rasterize_annotation(
    json_path: Path, image_shape: tuple[int, int], stats: RasterizeStats
) -> np.ndarray:
    """
    Načte jeden LabelMe JSON a vyrobí jednokanálovou masku `image_shape` (H, W)
    s hodnotami 0 (pozadí) až 6 (C7).

    Obratle se kreslí v pevném anatomickém pořadí C2 -> C7. Vertebrální těla se
    v korektní anotaci nepřekrývají, pořadí kreslení je zde spíš pojistka pro
    deterministický výsledek, kdyby se dva polygony přesto nepatrně překryly
    (např. kvůli nepřesnosti ručních landmarků) - poslední nakreslený vyhraje.
    """
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    grouped = _group_shapes_by_vertebra(data["shapes"], stats)

    for vertebra in VERTEBRA_ORDER:
        corners = grouped.get(vertebra)
        if corners is None:
            stats.incomplete_vertebrae += 1
            continue

        polygon = _vertebra_polygon(vertebra, corners)
        if polygon is None:
            stats.incomplete_vertebrae += 1
            log.warning(
                "%s: obratel %s má neúplné rohy (%s), přeskakuji.",
                json_path.name, vertebra, sorted(corners.keys()),
            )
            continue

        # fillPoly bere seznam polygonů, proto [polygon]. Trojúhelník (C2) i
        # čtyřúhelník (C3-C7) funguje stejnou funkcí - fillPoly nevyžaduje
        # pevný počet vrcholů.
        cv2.fillPoly(mask, [polygon], color=CLASS_ID[vertebra])

    return mask


def _save_class_mapping(out_dir: Path) -> None:
    """Uloží mapování třída -> index vedle vygenerovaných masek (dokumentace pro čtenáře výstupu)."""
    mapping = {0: "background", **{v: k for k, v in CLASS_ID.items()}}
    with (out_dir / "class_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Převede jednokanálovou masku tříd (0-6) na BGR obrázek podle CLASS_PALETTE_BGR.
    Čistě pro lidské oko - hodnoty 0-6 v šedotónovém PNG jsou v běžném prohlížeči
    prakticky nerozeznatelné od černé, barevná verze dělá obratle na první pohled
    zřetelné a vzájemně odlišné.
    """
    return CLASS_PALETTE_BGR[mask]


def _save_preview(image_path: Path, mask: np.ndarray, out_path: Path) -> None:
    """
    Uloží kontrolní snímek: originální RTG s barevně "prosvícenou" maskou přes
    obratle. Slouží jen k rychlé vizuální kontrole, že rohy sedí na správné
    místo - není součástí trénovacích dat.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    overlay = cv2.addWeighted(image, 0.6, _colorize_mask(mask), 0.4, 0.0)
    cv2.imwrite(str(out_path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR, help="Složka s Atlas JSON anotacemi.")
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR, help="Složka s originálními RTG snímky.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Kam uložit vygenerované trénovací masky (hodnoty pixelů 0-6).")
    parser.add_argument("--color-out-dir", type=Path, default=DEFAULT_COLOR_OUT_DIR, help="Kam uložit barevné (lidsky čitelné) verze masek. Prázdný řetězec vypne generování.")
    parser.add_argument("--limit", type=int, default=None, help="Zpracovat jen prvních N snímků (test na NTB, Fáze 1).")
    parser.add_argument("--preview", type=int, default=0, help="Počet kontrolních X-ray+overlay obrázků k uložení do <out-dir>/preview.")
    parser.add_argument("--overwrite", action="store_true", help="Přepsat už existující masky (jinak se přeskočí).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_class_mapping(args.out_dir)

    save_color = str(args.color_out_dir) != ""
    if save_color:
        args.color_out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(args.json_dir.glob("*.json"))
    if args.limit is not None:
        json_paths = json_paths[: args.limit]

    if not json_paths:
        raise FileNotFoundError(f"V {args.json_dir} nebyly nalezeny žádné .json soubory.")

    preview_dir = args.out_dir / "preview"
    if args.preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    stats = RasterizeStats()

    for i, json_path in enumerate(json_paths):
        stem = json_path.stem
        image_path = args.png_dir / f"{stem}.png"
        out_path = args.out_dir / f"{stem}.png"
        color_out_path = args.color_out_dir / f"{stem}.png" if save_color else None

        if not image_path.exists():
            stats.skipped_images += 1
            log.warning("Chybí párový snímek pro %s, přeskakuji.", json_path.name)
            continue

        already_done = out_path.exists() and (color_out_path is None or color_out_path.exists())
        if already_done and not args.overwrite:
            stats.processed_images += 1
            continue

        # Rozměry masky se berou z REÁLNÉHO souboru snímku (ne z hodnot
        # imageWidth/imageHeight zapsaných uvnitř JSONu), aby maska sedla
        # pixel na pixel na obraz, i kdyby se JSON metadata někdy rozešla
        # se skutečným souborem (např. po dodatečné kompresi/ořezu snímku).
        probe = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if probe is None:
            stats.skipped_images += 1
            log.warning("Snímek %s se nepodařilo načíst, přeskakuji.", image_path.name)
            continue

        mask = rasterize_annotation(json_path, probe.shape, stats)
        cv2.imwrite(str(out_path), mask)
        if color_out_path is not None:
            cv2.imwrite(str(color_out_path), _colorize_mask(mask))
        stats.processed_images += 1

        if i < args.preview:
            _save_preview(image_path, mask, preview_dir / f"{stem}_preview.png")

        if (i + 1) % 500 == 0:
            log.info("Zpracováno %d/%d snímků...", i + 1, len(json_paths))

    log.info(
        "Hotovo. Vygenerováno masek: %d, přeskočeno (chybějící pár): %d, "
        "neúplné anotace obratle: %d, duplicitní landmarky: %d.",
        stats.processed_images, stats.skipped_images,
        stats.incomplete_vertebrae, stats.duplicate_landmarks,
    )
    log.info("Trénovací masky (0-6) uloženy do: %s", args.out_dir)
    if save_color:
        log.info("Barevné masky pro vizuální kontrolu uloženy do: %s", args.color_out_dir)


if __name__ == "__main__":
    main()
