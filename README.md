# Srovnání architektur CNN a Vision Transformer pro 2D segmentaci krční páteře

Zápočtová práce, **Úvod do strojového učení**, UJEP — Aplikovaná informatika.

Cílem je porovnat tři přístupy k sémantické segmentaci obratlů **C2–C7**
z bočních rentgenových snímků krční páteře (dataset Atlas, 4963 snímků).
Úloha je formulována jako klasifikace **7 tříd** pro každý pixel:
pozadí + šest obratlů.

| # | Model | Přístup |
| :-- | :--- | :--- |
| 1 | **U-Net** | konvoluční síť, trénovaná od nuly |
| 2 | **UNETR** | Vision Transformer enkodér + konvoluční dekodér, trénovaný od nuly za identických podmínek |
| 3 | **TotalSegmentator2D (TSXR)** | hotový obecný model, pouze inference + post-processing |

**Hypotéza:** UNETR díky globálnímu kontextu z attention dosáhne srovnatelných
nebo lepších výsledků než U-Net; oba doménově trénované modely překonají obecný
TotalSegmentator2D.

---

## Struktura projektu

```
├── scripts/              příprava dat, inference cizího modelu, vyhodnocení
├── training/             vlastní trénovací pipeline (U-Net + UNETR)
│   ├── common/           dataset, transformace, modely, trénovací smyčka
│   └── configs/          hyperparametry (unet.yaml, unetr.yaml)
├── notebooks/            tenké notebooky, které volají skripty
├── devnotes/             dokumentace, teorie, plán (mimo git)
├── Inputdata/            data — vstupní i odvozená (mimo git)
├── Modely/               vypůjčené referenční repozitáře (neupravují se)
└── training_outputs/     checkpointy, predikce, výsledky (mimo git)
```

**Notebooky obsahují vysvětlující text a vizualizace, veškerá logika je ve
skriptech.** Notebooky skripty jen spouštějí.

---

## Instalace

Vyžaduje Python 3.13 a NVIDIA GPU s CUDA.

```bash
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
py -3 -m pip install monai opencv-python numpy pandas pyyaml tensorboard scikit-image scipy matplotlib nbformat jupyter
```

> ⚠️ **torch, torchvision a torchaudio musí být ze stejného CUDA buildu.**
> Nesoulad se projeví až za běhu jako `operator torchvision::nms does not exist`.

Ověření (musí projít celé):

```bash
py -3 -c "import torch, torchvision, torchaudio; import torchvision.ops; \
print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

Pro třetí model navíc `pip install -e Modely/Totalsegmentator2D/totalsegmentator2D-main`
— pozor, tenhle balíček si vynutí vlastní verzi torche a rozbije CUDA; po jeho
instalaci je nutné torch přeinstalovat příkazem výše. Podrobnosti a další
nástrahy jsou v `devnotes/PROSTREDI_A_PASTI.md`.

---

## Postup

### 1. Příprava dat

```bash
py -3 scripts/rasterize_masks.py       # anotace -> masky tříd 0–6
py -3 scripts/build_fold_split.py --fixed-split   # rozdělení 60/20/20
py -3 scripts/precompute_heqv.py       # předpočítaná ekvalizace histogramu
```

Anotace datasetu **nejsou obtahové polygony, ale bodové landmarky** — čtyři
rohové body na obratel (C2 jen tři). Rasterizace je skládá do čtyřúhelníků,
resp. trojúhelníku, a vyplňuje hodnotou třídy.

### 2. Trénink

```bash
py -3 -u training/train_unet.py  --max-minutes 180
py -3 -u training/train_unetr.py --max-minutes 360
```

Trénink je **přerušitelný**: checkpoint se ukládá po každé epoše, `--resume`
je výchozí a `--max-minutes` omezuje délku jednoho spuštění. Přerušení stojí
nanejvýš rozpracovanou epochu.

Průběh: `tensorboard --logdir training_outputs/runs`

### 3. Predikce a vyhodnocení

```bash
py -3 training/predict.py --model unet  --split test
py -3 training/predict.py --model unetr --split test
py -3 scripts/ts2d_inference.py --split test
py -3 scripts/evaluate_models.py --split test --figures 5
```

Všechny tři modely produkují **stejný typ artefaktu** — PNG masku 0–6 o rozměru
původního snímku. Evaluace pak neví, který model masku vyrobil, takže se do
srovnání nemůže vloudit nespravedlnost.

Výsledky: `training_outputs/evaluation/` (tabulky CSV/JSON + srovnávací obrázky).

> Natrénovaný model odjinud stačí nakopírovat jako `.pth` do
> `training_outputs/checkpoints/` — nic dalšího se nepřenáší.

---

## Metodika

| Parametr | Hodnota |
| :--- | :--- |
| Rozdělení dat | pevný split 60 / 20 / 20 (train / val / test), seed 42 |
| Velikost výřezu | 256 × 256 px, 8 výřezů na snímek |
| Batch size | 8, mixed precision (AMP) |
| Ztrátová funkce | Dice + Cross-Entropy |
| Optimalizátor | Adam — U-Net `1e-3`, UNETR `1e-4` |
| Scheduler | CosineAnnealingWarmRestarts |
| Metriky | Dice a IoU, průměr přes C2–C7 **bez pozadí** |
| Inference | sliding window 256 px, překryv 0,25, gaussovské vážení |

Oba trénované modely sdílejí **jednu trénovací smyčku, dataset, augmentace,
split i seed** — liší se pouze architektura a learning rate.

### Vědomé odchylky od původního návrhu

Všechny jsou zdůvodněné v `devnotes/plan.md` a `devnotes/STAV_PROJEKTU.md`:

1. **Místo 5-Fold křížové validace jeden pevný split.** Křížová validace by
   znamenala pětinásobný trénink obou architektur; na dostupném hardwaru řádově
   týdny. Kód pro K-Fold v projektu zůstal jako doklad, že varianta byla
   navržena a odbenchmarkována.
2. **Výřezy 256 px místo 512 px.** Cena self-attention roste kvadraticky
   s počtem tokenů — zmenšení dalo u UNETR pětinásobnou propustnost.
3. **Jiný learning rate než v referenčním repozitáři.** Hodnota `0.01`
   způsobila divergenci na `nan` u obou modelů.

---

## Dokumentace

| Soubor | Obsah |
| :--- | :--- |
| `devnotes/plan.md` | akademický plán studie |
| `devnotes/STAV_PROJEKTU.md` | co je hotové, jak to funguje a proč |
| `devnotes/DALSI_KROKY.md` | co zbývá |
| `devnotes/PROSTREDI_A_PASTI.md` | instalace, verze, nástrahy |

---

## Zdroje

Projekt vychází z veřejně dostupných prací, které jsou v `Modely/` uložené
v původní podobě a **nijak se neupravují** — slouží jen jako referenční vzor:

- **U-Net** — Ronneberger et al. (2015)
- **UNETR** — Hatamizadeh et al. (2021); implementace z knihovny MONAI
- **TotalSegmentator2D / TSXR** — Sabrowsky-Hirsch et al. (2025);
  Alshenoudy et al. (2025), *Leveraging Synthetic Data for Whole-Body
  Segmentation in X-Ray Images*
- **nnU-Net** — Isensee et al. (2021)
- repo https://github.com/JaroslavRadimsky/vertebra_segmentation_and_keypoint_extraction.git - odkud jsem čerpal inspiraci pro preprocessing dat a trening modelů
