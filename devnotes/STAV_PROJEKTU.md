# Stav projektu — co je hotové a proč (stav k 11. 9. 2026)

> ## 🏁 VÝSLEDKY OSTRÉHO BĚHU (11. 9. 2026)
>
> Testovací set, 992 snímků, nedotčený až do tohoto měření:
>
> | Model | Dice | IoU | C2 | C3 | C4 | C5 | C6 | C7 |
> | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
> | **U-Net** | **0,9412** | **0,8942** | 0,908 | 0,950 | 0,951 | 0,949 | 0,948 | 0,940 |
> | UNETR (ViT-Tiny) | 0,9229 | 0,8658 | 0,874 | 0,932 | 0,932 | 0,934 | 0,939 | 0,925 |
> | TotalSegmentator2D | 0,0004 | 0,0003 | 0,000 | 0,000 | 0,002 | 0,000 | 0,000 | 0,000 |
>
> Nejlepší validační Dice během tréninku: U-Net 0,9425 (epocha 34),
> UNETR 0,9147+ (epocha 39+). Podrobnosti v §9.

> **Aktualizace 10. 9.** Groundwork je hotový: celý řetěz
> *trénink → checkpoint → predikce → evaluace → srovnávací tabulka a obrázky*
> byl **ověřen end-to-end**. Zbývají notebooky a ostrý trénink.
> Nové v §8 níže: výkonnostní optimalizace datové pipeline, oprava divergence
> U-Netu a doložený doménový posun TotalSegmentatoru.

Tento dokument je **předávací zpráva**. Je psaný tak, aby ho mohl přečíst někdo
(nebo nová konverzace s asistentem), kdo o projektu nic neví, a byl schopen
okamžitě pokračovat, aniž by musel znovu objevovat, co už bylo zjištěno.

Doplňkové dokumenty:
- [DALSI_KROKY.md](DALSI_KROKY.md) — co zbývá udělat (detailní TODO)
- [PROSTREDI_A_PASTI.md](PROSTREDI_A_PASTI.md) — instalace, verze, a všechny nášlapné miny objevené za dneška
- [plan.md](plan.md) — akademický plán studie (aktualizovaný)
- [claude.md](claude.md) — zadání pro asistenta (aktualizované)
- [teorie/](teorie/) — teoretické vysvětlivky k modelům a technikám
- [constrains.md](constrains.md) — hardwarová a časová omezení
- [dodatek k trénování.md](dodatek%20k%20trénování.md) — odhad výpočetní náročnosti

---

## 1. Co je to za projekt (v jednom odstavci)

Zápočtová srovnávací studie pro předmět *Úvod do strojového učení* (UJEP, aplikovaná
informatika). Porovnávají se **tři přístupy k sémantické segmentaci krční páteře**
(7 tříd: pozadí + obratle C2–C7) na 2D rentgenových snímcích datasetu Atlas:

1. **U-Net** — klasická konvoluční architektura, trénovaná od nuly.
2. **UNETR** — Vision Transformer encoder + konvoluční dekodér, trénovaný od nuly
   za identických podmínek jako U-Net.
3. **TotalSegmentator2D (TSXR)** — cizí, obecný, předtrénovaný model; pouze inference
   + post-processing, žádný trénink.

Vše se nakonec vyhodnotí na **stejném, nedotčeném testovacím setu** metrikami Dice
a mIoU.

---

## 2. Zásadní změny oproti původnímu zadání (POZOR — čti dřív, než něco děláš)

Původní `plan.md`/`claude.md` byly za dnešek na několika místech **vědomě
revidovány**. Kdo bude pokračovat, musí vědět proč:

| Původně | Nově | Důvod |
| :--- | :--- | :--- |
| U-Net jen inference z předtrénovaných vah `Atlas-heqv-multi-patch-10000-fold-0-final.pth` | U-Net se **trénuje od nuly** | Rozhodnutí zadavatele — chce vlastní, kontrolovaný trénink obou modelů |
| 5-Fold křížová validace | **Jeden pevný split 60/20/20** (train/val/test) | 5-Fold by na dostupném HW trval týdny (viz `dodatek k trénování.md`); K-Fold kód ale zůstává a spouští se — je to doklad, že rigoróznější metodika byla navržena a vědomě zamítnuta |
| Patch 512×512 | **Patch 256×256** | Benchmark ukázal 5× vyšší propustnost u UNETR (attention škáluje kvadraticky s počtem tokenů). Odsouhlaseno zadavatelem jako nutný kompromis kvůli deadline 13. 9. |
| `patches_per_image: 20` | **8** | Zkrácení epochy; při 2978 trénovacích snímcích je to pořád ~24 tis. patchů na epochu |
| Kód psát do vypůjčených repozitářů | **Vše nové v `scripts/` a `training/`** | Zadavatel chce zachovat integritu vypůjčených repozitářů — ty slouží jen jako referenční vzor, nic se v nich nemění |

---

## 3. Adresářová struktura (co kde je a k čemu)

```
USU-zapocet/
├── Inputdata/                    # VŠECHNA data (mimo git, .gitignore)
│   ├── datasets-PNG/             # 4963 originálních RTG snímků (vstup, dodáno zadavatelem)
│   ├── datasets-JSON/            # 4963 LabelMe anotací (vstup, dodáno zadavatelem)
│   ├── datasets-MASK/            # ← VYGENEROVÁNO: masky tříd 0–6 pro trénink
│   ├── datasets-MASK-color/      # ← VYGENEROVÁNO: barevné masky pro kontrolu okem
│   └── folds/atlas_vertebra/     # ← VYGENEROVÁNO: definice splitů (CSV + JSON manifesty)
├── Modely/                       # VYPŮJČENÉ repozitáře — NEUPRAVOVAT, jen číst
│   ├── vertebra_segmentation_and_keypoint_extraction-main/   # U-Net referenční repo (částečně rozbité, viz §6)
│   └── Totalsegmentator2D/       # TS2D/TSXR nástroj (ts2d je odsud nainstalovaný editovatelně)
├── scripts/                      # ← NOVÉ: příprava dat + benchmark (odděleně od tréninku)
├── training/                     # ← NOVÉ: vlastní trénovací pipeline
├── training_outputs/             # ← VYGENEROVÁNO za běhu: checkpointy, predikce, výsledky (mimo git)
│                                 #    Před ostrým během je složka smazaná — vytvoří se sama.
├── devnotes/                     # dokumentace a plánování (mimo git)
└── examples/                     # ukázkový pár PNG+JSON (mimo git)
```

---

## 4. Hotové komponenty — soubor po souboru

### 4.1 `scripts/rasterize_masks.py` — Fáze A: rasterizace anotací ✅ HOTOVO

**Co dělá:** převádí LabelMe JSON anotace na jednokanálové PNG masky s hodnotami
pixelů `0` (pozadí) až `6` (C7).

**Vize / proč takhle:** segmentační síť potřebuje rastrovou masku, ne seznam bodů.
Klíčové zjištění dneška: **anotace NEJSOU obtahové polygony**, jak předpokládal
původní `plan.md`, ale **bodové landmarky** — pro každý obratel 4 rohy
(`"C4 top left"`, `"C4 top right"`, `"C4 bottom right"`, `"C4 bottom left"`),
s výjimkou C2, který má jen 3 body (`"C2 bottom left"`, `"C2 bottom right"`,
`"C2 centroid"`) kvůli své odlišné anatomii. Skript tyto rohy poskládá do polygonu
(čtyřúhelník pro C3–C7, trojúhelník pro C2) a vyplní pomocí `cv2.fillPoly`.

**Ověřený stav datasetu** (proskenováno všech 4963 souborů):
- 4963/4963 JSON má párové PNG, 0 chybějících
- vždy přesně třídy C2–C7, 23 shapes na soubor
- **1 anomálie:** `2548130.json` má 24 shapes — duplicitní landmark `"C3 top right"`.
  Skript to ošetřuje (pozdější bod přepíše dřívější) a jen zaloguje, nespadne.
- rozlišení snímků je proměnlivé (např. 802×1014, 568×740) — proto se rozměr masky
  bere z **reálného souboru snímku**, ne z `imageWidth/imageHeight` v JSONu

**Výstup:** `Inputdata/datasets-MASK/` (4963 masek, hodnoty 0–6) +
`Inputdata/datasets-MASK-color/` (tytéž masky v jasných barvách — hodnoty 0–6 jsou
v prohlížeči nerozeznatelné od černé, barevná verze slouží jen ke kontrole okem,
do tréninku NEVSTUPUJE) + `class_mapping.json`.

**Vizuálně ověřeno:** barevný overlay přes originální rentgen sedí přesně na
6 krčních obratlů (C2 trojúhelník nahoře, C3–C7 čtyřúhelníky dolů po páteři).

**Spuštění:**
```bash
py -3 scripts/rasterize_masks.py                    # celý dataset
py -3 scripts/rasterize_masks.py --limit 20 --preview 3   # rychlý test + overlay náhledy
```

---

### 4.2 `scripts/build_fold_split.py` — Fáze B: rozdělení dat ✅ HOTOVO

**Co dělá:** dvě nezávislé věci v jednom skriptu:
1. **K-Fold** (5 foldů) → `train/val_atlas_vertebra_fold_{0..4}.csv`
2. **Pevný split 60/20/20** (přepínač `--fixed-split`) → `train.csv`, `val.csv`, `test.csv`

**Vize / proč takhle:** klíčové zjištění — původní U-Net repozitář **žádnou logiku
K-Foldu neobsahuje**. Jeho `Src/Utils/data_utils.py::get_split_files` pouze *načítá*
hotové CSV soubory, které vznikly mimo repozitář a v repu chybí. „Exaktní
rekonstrukce původního Fold 0" (jak žádal původní `claude.md`) tedy **není z
dostupného kódu možná**. Protože se ale U-Net stejně trénuje od nuly, není to
překážka — děláme vlastní, transparentní split.

CSV formát (`image,label` s pouhými názvy souborů) je záměrně **stejný jako
konvence původního repa**, aby byl kompatibilní.

**Proč zůstal i K-Fold, když se nepoužije:** je to argumentační materiál do zprávy —
rigoróznější metodika byla navržena, prakticky odbenchmarkována a vědomě zamítnuta
kvůli výpočetnímu rozpočtu. To je legitimní a hodnotný bod k obhajobě.

**Reálné rozdělení (seed 42, ověřeno asserty i ručně):**
| Množina | Počet | Role |
| :--- | ---: | :--- |
| train | 2978 | gradientní trénink |
| val | 993 | hlídání přetrénování + výběr „best" checkpointu |
| test | 992 | **nedotčený** — otevře se až na konec pro finální srovnání všech 3 modelů |

Ověřeno: nulový průnik mezi množinami, sjednocení = 4963.

**Spuštění:**
```bash
py -3 scripts/build_fold_split.py --fixed-split
```

---

### 4.3 `scripts/benchmark_training_step.py` — empirický benchmark ✅ HOTOVO

**Co dělá:** změří reálný čas jednoho tréninkového kroku (forward + backward +
optimizer.step) pro U-Net / UNETR při dané kombinaci rozlišení, batch size a AMP,
na syntetických datech.

**Vize / proč vznikl:** `dodatek k trénování.md` odhadoval 30–45 min na epochu a z toho
plynulo, že projekt je časově neproveditelný. Než se na základě **odhadu** zahodí
polovina metodiky, bylo potřeba mít **naměřená čísla**. Ukázalo se, že odhad byl
silně pesimistický a že hlavní pákou není počet epoch, ale rozlišení patchů + AMP.

**Naměřeno na RTX 3060 Laptop 6 GB (tento notebook):**

| Model | Patch | Batch | AMP | s/krok | VRAM | vzorků/s |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| U-Net | 512 | 2 | ne | 0.022 | 0.40 GB | 91 |
| U-Net | 256 | 8 | ne | 0.021 | 0.40 GB | 381 |
| U-Net | 256 | 32 | ano | 0.059 | 0.98 GB | 542 |
| UNETR | 512 | 2 | ne | 0.435 | 4.60 GB | 4.6 |
| UNETR | 512 | 2 | ano | 0.211 | 4.47 GB | 9.5 |
| UNETR | 256 | 4 | ne | 0.188 | 2.50 GB | 21.3 |
| **UNETR** | **256** | **8** | **ano** | **0.167** | **3.09 GB** | **47.9** |
| UNETR | 256 | 16 | ano | 0.299 | 4.72 GB | 53.5 |
| UNETR | 256 | 32 | ano | 9.442 | 8.02 GB | 3.4 ← překročení VRAM, kolaps |

**Závěry, které z toho plynou (a proto je config takový, jaký je):**
- U-Net má jen **1.6 M parametrů**, UNETR **116 M** — U-Net není nikdy úzké hrdlo.
- Zmenšení patche 512→256 dá u UNETR **5× vyšší propustnost** (attention je
  O(n²) v počtu tokenů: 1024 tokenů při 512 px vs. 256 tokenů při 256 px).
- AMP (fp16) dá u UNETR dalších **~2×** (tensor cores na Ampere).
- Batch 8 @ 256 px se vejde do **~3.1 GB** → **identický config běží na 6GB
  notebooku i 12GB desktopu**, není nutné mezi stroji nic přepisovat.
- Batch 32 přeteče 6 GB a výkon se propadne 50× — nepoužívat na notebooku.

---

### 4.4 `training/` — vlastní trénovací pipeline ✅ HOTOVO (smoke-tested)

Celé je to **nový, nezávislý kód**, silně inspirovaný vypůjčeným U-Net repem
(struktura config → transformy → MONAI model → DiceCELoss → trénovací smyčka
s TensorBoardem a sliding-window validací), ale nic z něj za běhu neimportuje.

| Soubor | Co dělá | Vize |
| :--- | :--- | :--- |
| `common/paths.py` | Centralizované cesty | Záměrně nezávislé na `Src/project_paths.py` vypůjčeného repa |
| `common/seeding.py` | `set_seed`, worker seedy, generátor | Reprodukovatelnost — obdoba `replicability.py`, vlastní soubor |
| `common/transforms.py` | MONAI pipeline (CLAHE, ScaleIntensity, pad, náhodný crop, flip/rotate90/zoom/šum) | Augmentace 1:1 podle původního configu |
| `common/dataset.py` | `AtlasPatchDataset` | Náhrada za chybějící `AtlasDataset`. `__len__ = počet_snímků × patches_per_image` → z jednoho snímku se za epochu vyřízne víc náhodných patchů |
| `common/models.py` | `build_unet()`, `build_unetr()` | Oba modely mají identický vstup/výstup (3 kanály → 7 tříd) = podmínka férového srovnání |
| `common/loop.py` | **Přerušitelná** trénovací smyčka | Viz níže — nejdůležitější soubor |
| `common/data_setup.py` | Sestavení DataLoaderů | Jedno společné místo pro oba modely → garantovaně identická data |
| `common/config.py` | YAML → `LoopConfig` | — |
| `configs/unet.yaml`, `configs/unetr.yaml` | Hyperparametry | Vše shodné kromě `run.name`, `max_epochs` a `lr` (viz §5) |
| `train_unet.py`, `train_unetr.py` | Spouštěcí skripty | CLI: `--max-epochs`, `--max-minutes`, `--resume/--no-resume`, `--limit-train/--limit-val` |

#### Přerušitelnost tréninku (klíčový požadavek zadavatele)

Desktop s 12 GB je k dispozici jen v nočních oknech, notebook může běžet dlouho.
Smyčka je proto navržená takto:

- **Checkpoint se ukládá po KAŽDÉ epoše** do `<run_name>-last.pth` (váhy, optimizer,
  scheduler, AMP scaler, číslo epochy, dosud nejlepší val Dice). Přerušení
  (Ctrl+C, vypnutí, pád) stojí nanejvýš rozpracovanou epochu.
- **`--resume`** (výchozí zapnuto) automaticky najde poslední checkpoint a
  pokračuje od další epochy. Ověřeno v praxi.
- **`--max-minutes N`** = měkký časový rozpočet jednoho spuštění. Po dokončení
  epochy se zkontroluje čas a smyčka se čistě ukončí. Umožňuje pustit trénink na
  začátku nočního okna a nebát se, že poběží přes den.
- **`<run_name>-best.pth`** se ukládá zvlášť, když se zlepší validační Dice.

#### Validační strategie

Plná `sliding_window_inference` přes celý (neořezaný) snímek se dělá **jen každých
`full_val_every` epoch** (výchozí 5) a vždy v poslední epoše — je řádově dražší než
tréninkový krok. Validační DataLoader má vždy `batch_size=1`, protože snímky mají
různé rozlišení a nejdou naskládat do jedné dávky. Trénovací Dice se naopak loguje
každou epochu prakticky zadarmo.

#### Smoke testy — oba modely ověřeny ✅

```bash
py -3 training/train_unet.py  --max-epochs 2 --limit-train 6 --limit-val 2 --no-resume
py -3 training/train_unetr.py --max-epochs 3 --limit-train 4 --limit-val 2 --no-resume
```
Oba proběhly bez pádu, checkpointy se uložily, `--resume` ověřen (pokračoval
od správné epochy s obnoveným optimizerem i schedulerem).

---

### 4.5 `devnotes/teorie/` — vysvětlivky ✅ HOTOVO

Pět dokumentů psaných na úrovni studenta 2. ročníku Bc. aplikované informatiky,
navázaných na **reálné hodnoty z configu**, ne na obecné fráze:

- `00_prehled.md` — proč je studie postavená ze tří různých filozofií
- `01_konvoluce_a_unet.md` — konvoluce, receptive field, induktivní bias, encoder-decoder, skip connections
- `02_vision_transformer_a_unetr.md` — self-attention (Q/K/V), patch embedding, poziční embedding, architektura UNETR, tabulka CNN vs. Transformer
- `03_loss_metriky_a_trenink.md` — proč Dice + CE kombinace, Dice vs. IoU, K-Fold, data leakage, reprodukovatelnost
- `04_data_pipeline_a_inference.md` — rasterizace, patch trénink, augmentace, sliding window, TotalSegmentator

---

## 5. Objevené a opravené chyby (a proč na tom záleží)

Tyhle věci byly nalezené smoke testy — kdyby se to spustilo naostro přes noc,
přišlo by se na to až ráno se ztrátou celého okna.

### 5.1 UNETR divergoval na `loss = nan` ⚠️ OPRAVENO

S `lr = 0.01` (hodnota převzatá z původního configu, vyladěná pro **malý CNN**)
UNETR po první epoše zdiverguje na `nan`. Transformery s 116 M parametry jsou na
learning rate mnohem citlivější.

**Oprava:** `unetr.yaml` má `lr: 1.0e-4` (hodnota z originální UNETR publikace,
Hatamizadeh et al. 2021). U-Net si `lr: 0.01` ponechává. Po opravě loss klesá
zdravě: 3.32 → 3.23 → 3.15.

**Pozn. k férovosti srovnání:** rozdílný LR mezi architekturami je legitimní —
všechno ostatní (data, split, seed, loss, augmentace, smyčka) je identické. LR je
standardní per-architekturu laděný hyperparametr; použít pro transformer hodnotu
vyladěnou pro CNN by naopak bylo metodicky špatně (srovnávali bychom
zdivergovaný model).

### 5.2 Gradient clipping ⚠️ PŘIDÁNO

Do smyčky přidán `clip_grad_norm_(max_norm=1.0)` pro **oba** modely (u AMP větve
s korektním `scaler.unscale_()` před klipem). Standardní pojistka pro transformery,
u U-Netu neškodí.

### 5.3 Interpolace masek při zoomu ⚠️ OPRAVENO PREVENTIVNĚ

`RandZoomd` má nastavené `mode=("bilinear", "nearest")`. Bez toho by se maska tříd
interpolovala bilineárně a na hranici dvou obratlů by vznikly **neexistující
mezitřídy** (např. „2.5"). Tichá chyba, která by degradovala kvalitu anotací.

### 5.4 `cv2.CLAHE` nešel picklovat ⚠️ OPRAVENO

Na Windows používá DataLoader `spawn` (ne `fork`), takže se transformy picklují do
worker procesů. Uložený `cv2.CLAHE` objekt to shodilo. Nyní se vytváří až uvnitř
`__call__`.

### 5.5 Diakritika v konzoli ⚠️ OPRAVENO

Windows konzole (cp1252) shodila skript na `print()` s českou diakritikou.
Vstupní body mají `sys.stdout.reconfigure(encoding="utf-8")`.

---

## 6. Stav vypůjčených repozitářů

### 6.1 U-Net repo (`vertebra_segmentation_and_keypoint_extraction-main`) — ČÁSTEČNĚ ROZBITÉ

**Chybí celá složka `Src/Atlas/data/`** (README ji v adresářovém stromu uvádí,
fyzicky v repu není). Důsledky:

- `Src/Atlas/atlas_dataset.py` a `atlas_dataset_patch.py` → `ImportError` (jsou to
  jen re-export wrappery)
- `Src/Atlas/training/atlas_model_multiclass_patch.py:23` → rozbité (reálný trénink)
- `Src/Atlas/evaluation/atlas_test_multiclass_patch.py:14` → rozbité (fold evaluace)
- **NEROZBITÉ:** `Src/Atlas/inference/inference.py` a `single_inference.py` —
  pracují přímo se soubory přes glob, `AtlasDataset` nepotřebují

**Rozhodnutí:** neopravovat. Repo zůstává netknuté jako referenční vzor; vlastní
pipeline v `training/` je na něm nezávislá.

Užitečné soubory, ze kterých se čerpalo (jen ke čtení):
- `Src/Atlas/training/common.py` — transformační pipeline, `build_model`
- `Src/Atlas/training/atlas_model_multiclass_patch.py` — struktura tréninku
- `Src/Models/training_new.py` — trénovací smyčka, TensorBoard, SW validace, checkpointy
- `Src/Utils/replicability.py` — seedování
- `Src/Utils/metrics.py` — Dice/IoU per-channel
- `Src/Atlas/inference/extraction.py` — maska → LabelMe JSON (užitečné pro Fázi E)

### 6.2 TotalSegmentator2D — ROZPRACOVÁNO ⏳

Postup a zjištění dneška:

1. Nainstalováno přes `pip install -e .` z lokálního klonu. ⚠️ **Pozor:** tím se
   přeinstaloval torch na `2.7.1+cpu` a rozbila CUDA — viz `PROSTREDI_A_PASTI.md`.
2. **Stahování modelů je v nástroji rozbité.** `URLDataBase.copy()` používá knihovnu
   `gdown` (určenou pro Google Drive) na Zenodo URL — stáhne 763 bajtů místo 250 MB
   a spadne na `BadZipFile`. Přímý `curl` na tutéž URL funguje bez problému.
   **Obejito:** model stažen ručně curlem a rozbalen do
   `C:\Users\Administrator\.ts2d\models\` (přesně tam a v té struktuře, kterou
   nástroj očekává). Model `tsxr-v2-ep1000b2_vertebrae` je tam nyní nakešovaný.
3. **CLI nepřijímá PNG** — `_enumerate_cases` povoluje jen `nrrd/nii/nii.gz/mha/mhd`.
   **Řešení:** obejít CLI a použít Python API `TS2D.predict(sitk_image)`, které
   přijímá přímo `SimpleITK.Image` objekt.
4. **SimpleITK v tomto prostředí neumí číst PNG** (chybí zkompilovaný PNG IO reader).
   **Řešení:** načíst PNG přes `cv2` a převést `sitk.GetImageFromArray(arr)`.
5. **Model očekává přesně 1 kanál.** `dataset.json` uvádí `channel_names: {"0": "XR"}`.
   Vstup tedy musí být **šedotónový** (`cv2.IMREAD_GRAYSCALE`), ne 3kanálový —
   `_predict_model` jinak vyhodí chybu o neshodě počtu kanálů.
6. **Label mapping modelu je známý** (z `dataset.json`) — model segmentuje 26 struktur
   (sacrum, L1–L5, T1–T12, C1–C7). Pro nás relevantní:

   | TSXR label | ID | naše třída |
   | :--- | ---: | ---: |
   | vertebrae-c2 | 25 | 1 |
   | vertebrae-c3 | 24 | 2 |
   | vertebrae-c4 | 23 | 3 |
   | vertebrae-c5 | 22 | 4 |
   | vertebrae-c6 | 21 | 5 |
   | vertebrae-c7 | 20 | 6 |

   Vše ostatní → 0 (pozadí). **Tohle je přesně ta filtrace a přemapování, které
   žádá Fáze C** — a je to už vyřešené, stačí to napsat do skriptu.

**Co ještě není ověřeno:** samotné `predict()` na reálném snímku. Testovací běh byl
spuštěn, ale na CPU trval přes 10 minut a nedoběhl do konce sezení. Není jasné, zda
je jen pomalý, nebo se zasekl. **První úkol příště:** dotáhnout tenhle test s
funkční CUDA (torch je nyní zpět na `2.11.0+cu128`).

---

## 7. Rychlá orientace — co spustit, aby se ověřilo, že vše stojí

```bash
# 1) prostředí — musí projít CELÉ (viz past 4.1b: verze musí sedět všechny tři)
py -3 -c "import torch, torchvision, torchaudio; import torchvision.ops; \
print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
# očekávané: 2.11.0+cu128 0.26.0+cu128 2.11.0+cu128 True

# 2) data jsou připravená (nemusí se přegenerovávat)
ls Inputdata/datasets-MASK | wc -l          # 4964 (4963 masek + class_mapping.json)
ls Inputdata/datasets-PNG-heqv | wc -l      # 4963 (předpočítané CLAHE)
cat Inputdata/folds/atlas_vertebra/fixed_split_info.json

# 3) celý řetěz projde (~2 minuty)
py -3 training/train_unet.py --max-epochs 2 --limit-train 40 --limit-val 4 --no-resume
py -3 training/predict.py --model unet --split test --limit 3
py -3 scripts/ts2d_inference.py --limit 3
py -3 scripts/evaluate_models.py --limit 3
```

---

## 8. Co přibylo 10. 9. 2026

### 8.1 Datová pipeline byla omezená procesorem, ne grafikou ✅ VYŘEŠENO

`nvidia-smi` během tréninku hlásilo **0 % vytížení GPU**. Měření ukázalo proč:

| Operace na jeden vzorek | Čas |
| :--- | ---: |
| `cv2.imread` snímku (1014×802×3) | 5,05 ms |
| `cv2.imread` masky | 2,66 ms |
| **CLAHE na celém snímku** | **5,95 ms** |
| **CPU celkem** | **~13,7 ms** |
| GPU krok U-Net (dávka 8) | 21 ms |
| GPU krok UNETR (dávka 8) | 167 ms |

Dávka 8 vzorků znamenala ~110 ms práce na CPU proti 21 ms na GPU. Navíc se kvůli
`patches_per_image: 8` **tentýž snímek načítal a ekvalizoval osmkrát za epochu**
pokaždé se stejným výsledkem.

**Opravy:**
1. `scripts/precompute_heqv.py` — CLAHE se spočítá **jednou** dopředu do
   `Inputdata/datasets-PNG-heqv/`. Zapíná se `data.use_precomputed_heqv: true`;
   transformace za běhu se pak do pipeline vůbec nepřidá. Sémantika je totožná
   (CLAHE je adaptivní, musí se počítat na celém snímku před výřezem — což
   předpočítání zachovává).
2. Dataset čte snímek **jednokanálově** a kanál si ztrojí sám (originály mají 3
   identické kanály — je to rentgen).
3. `num_workers` 4 → 8.

### 8.2 Reálné časy tréninku (naměřeno na plných datech) 📊

Notebook RTX 3060 Laptop 6 GB, po optimalizaci:

| Model | ms/dávka | Epocha (2978 dávek) | 30 epoch | 60 epoch |
| :--- | ---: | ---: | ---: | ---: |
| U-Net | 52,3 | **2,6 min** | 1,3 h | **2,6 h** |
| UNETR | 187,0 | **9,3 min** | **4,6 h** | 9,3 h |

Srovnání s čistým GPU výpočtem (U-Net 21 ms, UNETR 167 ms): UNETR má už jen 12 %
režie, je prakticky GPU-bound. U-Net zůstává datově vázaný, ale je tak levný, že
to nevadí.

**Závěr: oba modely se vejdou do jedné noci (~7 h) i na notebooku.** Desktop bude
rychlejší. Původní odhad v `dodatek k trénování.md` mluvil o 25–30 dnech.

### 8.3 U-Net divergoval na `nan` ⚠️ OPRAVENO

S `lr = 0.01` (hodnota z původního repozitáře) U-Net na reálných datech po ~200
krocích spolehlivě zdiverguje na `loss = nan`. Ve včerejším smoke testu na
6 snímcích se to neprojevilo — bylo příliš málo kroků.

**Oprava:** `unet.yaml` má nyní `lr: 0.001` (standardní hodnota pro Adam).
Ověřeno stabilní.

**Poznámka:** dřívější závěr, že jde o specifikum transformerů (UNETR), byl
tedy jen částečně správný — `lr = 0.01` je příliš agresivní pro **oba** modely,
u UNETR se to jen projevilo dřív. UNETR má `1e-4`, U-Net `1e-3`.

### 8.4 Pojistka proti divergenci ✅ PŘIDÁNO

Trénovací smyčka nyní hlídá `nan`/`inf` v loss:
- ojedinělý výskyt → dávka se přeskočí (nepustí se krok optimizeru, který by
  `nan` rozlil do všech vah)
- 20 po sobě → běh se ukončí se stavem `diverged_nan_loss` a hláškou, ať se
  sníží learning rate; checkpoint se **neukládá**, aby zůstal použitelný stav
  z konce předchozí epochy

Bez téhle pojistky by osmihodinový noční běh vyprodukoval bezcenný model a
zjistilo by se to až ráno.

### 8.5 Hlasitá diagnostika hardwaru ✅ PŘIDÁNO

Trénink i predikce vypisují na prvním řádku:
```
[hw] GPU: NVIDIA GeForce RTX 3060 Laptop GPU | VRAM 6.44 GB | torch 2.11.0+cu128 | AMP zapnuto
```
Když CUDA není k dispozici, vypíše se nepřehlédnutelné varování s příkazem na
opravu — tichý propad na CPU by znamenal ~100× pomalejší běh.

### 8.6 TotalSegmentator2D — zprovozněno, ale nepřenáší se na tuto doménu 🔬

**Technicky funguje:** model naběhne za ~19 s, inference trvá **~1 s/snímek**,
skript `scripts/ts2d_inference.py` produkuje masky 0–6 ve stejném formátu jako
ostatní modely. Mapování tříd se čte podle **jmen** z metadat (ne natvrdo podle
ID), ověřeno: `vertebrae-c2`(25)→1 … `vertebrae-c7`(20)→6.

**Ale krční obratle nenajde.** Místo nich predikuje kost křížovou a bederní
obratle — na snímku krku.

Než to šlo označit za vlastnost modelu, bylo nutné vyloučit chybu na naší straně.
Skript `scripts/ts2d_domain_check.py` to ověřuje reprodukovatelně:

| Hypotéza | Test | Výsledek |
| :--- | :--- | :--- |
| Špatné volání API | `predict()` vrací strukturovaný výstup | ✗ vyloučeno |
| Špatné mapování kanálů | mapování podle jmen z metadat | ✗ vyloučeno |
| Špatné měřítko | spacing 1,0 / 0,7 / 0,5 / 0,35 / 0,25 / 0,15 mm/px | ✗ vyloučeno |
| Obrácená polarita | invertovaný snímek | ✗ vyloučeno |
| Rozbité váhy | predikce jsou prostorově souvislé | ✗ vyloučeno |

Model přitom očekává (`plans.json`) rozlišení **1,5 mm/px** a trénoval na
snímcích o mediánu **226×239 px**. Při spacingu 0,5 (což po převzorkování dá
248×193 px, nejblíž tréninkovým datům) predikuje nejvíc — ale pořád kost
křížovou a L3.

**Závěr:** TSXR trénoval na syntetických rentgenech (DiffDRR rekonstrukce)
z CT datasetu TotalSegmentator, kde převažuje trup. Úzký výřez krční páteře
je mimo jeho doménu. **Je to doložený negativní výsledek — přesně to, co
předpovídá hypotéza v `plan.md`**, a do zprávy patří i s tabulkou vyloučených
hypotéz výše (jinak by to vypadalo jako nedbalost).

Praktický důsledek: `--spacing` je v `ts2d_inference.py` parametr (výchozí 0,5),
protože se ukázalo, že na něm záleží.

### 8.7 Nové skripty ✅

| Soubor | Co dělá | Ověřeno |
| :--- | :--- | :--- |
| `scripts/precompute_heqv.py` | Jednorázový výpočet CLAHE do souborů | ✅ 4963/4963 |
| `training/predict.py` | Checkpoint → PNG masky 0–6 na zvoleném splitu | ✅ 0,22 s/snímek |
| `scripts/ts2d_inference.py` | TS2D inference + filtrace a přemapování tříd | ✅ 1 s/snímek |
| `scripts/ts2d_domain_check.py` | Reprodukovatelné ověření doménového posunu | ✅ |
| `scripts/evaluate_models.py` | Dice/IoU pro všechny modely + srovnávací obrázky | ✅ |

`training/predict.py` je navržený tak, aby stačilo **nakopírovat natrénovaný
checkpoint do `training_outputs/checkpoints/`** a spustit jeden příkaz — nic
jiného se mezi stroji přenášet nemusí.

### 8.9 Proof of concept — zmenšený běh celé studie ✅ (10. 9.)

`scripts/run_poc.py` projde úplně stejnou posloupnost jako ostrý běh
(trénink obou modelů → predikce → TS2D → evaluace), jen na zlomku dat.

Konfigurace běhu: 400 trénovacích snímků, 40 validačních, 60 testovacích,
U-Net 15 epoch, UNETR 10 epoch. **Celkem 27,4 minuty** na 6GB notebooku.

#### Výsledky (60 testovacích snímků)

| Model | Dice | IoU | C2 | C3 | C4 | C5 | C6 | C7 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **U-Net** | **0,8665** | 0,7886 | 0,769 | 0,905 | 0,868 | 0,873 | 0,896 | 0,888 |
| UNETR | 0,1057 | 0,0625 | 0,000 | 0,001 | 0,141 | 0,306 | 0,187 | 0,000 |
| TS2D | 0,0000 | 0,0000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |

> ⚠️ Čísla jsou **orientační** — na 400 snímcích a 15 epochách nevzniká model
> do závěrečného srovnání. Do zprávy patří až výsledky z ostrého běhu.

#### Co z toho plyne

**1. Pipeline funguje a U-Net konverguje velmi rychle.**

| Epocha | 0 | 2 | 5 | 8 | 11 | 14 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 1,289 | 1,022 | 0,790 | 0,739 | 0,654 | 0,604 |
| train Dice | 0,002 | 0,157 | 0,524 | 0,578 | 0,696 | 0,753 |
| val Dice | — | 0,297 | 0,694 | 0,699 | 0,833 | **0,883** |

Val Dice na konci pořád stoupá → s celým datasetem a víc epochami půjde výš.
Vizuálně U-Net segmentuje všech šest obratlů se správnými třídami a pozicemi.

**2. UNETR se učí řádově pomaleji.** V epoše 8 měl val Dice 0,078, zatímco
U-Net ve stejné epoše 0,699 — zhruba **desetinásobný rozdíl**. Vizuálně
trefuje správný pruh páteře, ale roztříštěně a s pomíchanými třídami.

Je to očekávané a je to **hodnotný materiál do zprávy**: Vision Transformer
nemá vestavěný induktivní bias konvoluce (lokalita, translační ekvivariance) a
musí se ho naučit z dat. Proto je náročnější na množství dat i na počet epoch.

**3. ⚠️ Riziko pro ostrý běh: 30 epoch nemusí UNETR stačit.**

Doporučení, která z PoC plynou (nutno rozhodnout před ostrým během):

- **Prodloužit cyklus scheduleru.** `CosineAnnealingWarmRestarts` má `T_0: 5`,
  takže learning rate klesne do epochy 4 na ~1e-5. U modelu, který se teprve
  rozjíždí, je to kontraproduktivní — v PoC bylo vidět, jak se UNETR
  v epochách 3–4 prakticky zastavil. Rozumnější `T_0: 10` až `15`.
- **Zvýšit UNETR learning rate.** `1e-2` divergovalo, `1e-4` je zjevně
  konzervativní. Stojí za zkoušku `3e-4`.
- **Víc epoch pro UNETR než pro U-Net.** U-Net je 3,5× levnější na epochu
  a konverguje rychleji — dát oběma stejný počet epoch by UNETR znevýhodnilo
  nad rámec toho, co je vlastností architektury.

#### Naměřené časy a extrapolace

| Krok | Čas |
| :--- | ---: |
| trénink U-Net (15 epoch, 400 snímků) | 9,0 min |
| trénink UNETR (10 epoch, 400 snímků) | 16,5 min |
| predikce U-Net (60 snímků) | 0,3 min |
| predikce UNETR (60 snímků) | 0,3 min |
| inference TS2D (60 snímků) | 1,2 min |
| evaluace | 0,1 min |
| **celkem** | **27,4 min** |

Přepočteno na celý trénovací set (2978 snímků): U-Net ~4,5 min/epocha,
UNETR ~8,2 min/epocha. Je to víc než čisté měření tréninkového kroku
(2,6 / 9,3 min) — do průměru se tu započítává i periodická validace a režie
startu. Pro plánování je tenhle odhad **realističtější**.

**Odhad ostrého běhu:** U-Net 60 epoch ≈ 4,5 h, UNETR 60 epoch ≈ 8,2 h.
Dohromady se to do jedné noci **nevejde** — bude to chtít dvě noci, nebo
nechat běžet notebook souběžně.

### 8.10 Ladění UNETR — iterace a vyhodnocení ✅ (10. 9.)

Nástroj: `scripts/tune_unetr.py` (krátké běhy za jinak identických podmínek,
400 snímků / 10 epoch, výsledky se kumulují do `training_outputs/unetr_tuning.json`).

Výchozí bod z PoC: **val Dice 0,078** (lr 1e-4, `T_0` 5, náhodné výřezy).

#### Iterace 1 — vyšší LR a delší cyklus scheduleru ❌

`lr 3e-4`, `T_0 15`. Hypotéza: learning rate byl nízký a scheduler ho navíc
předčasně srazil.

**Nepomohlo — bylo to horší než výchozí stav** (epocha 5: val 0,015 proti 0,035).
Loss přitom klesal *rychleji* než u baseline. To je typický příznak: model se
rychleji propadá do triviálního řešení „všechno je pozadí", které je při 3 %
popředí silné lokální minimum. Vyšší learning rate ho do něj dostal dřív.

Zbylé LR varianty (`+warmup`, `lr 1e-3`) byly na základě toho **přeskočeny** —
hypotéza o learning rate se ukázala jako mylná a nemělo smysl ji dál rozvíjet.

#### Iterace 2 — cílené vzorkování výřezů ✅

Útok na příčinu, ne na symptom. `RandCropByPosNegLabeld` vystředí 80 % výřezů
na pixel patřící obratli.

Nejdřív ověřeno, že to vůbec něco mění (320 výřezů):

| strategie | výřezů bez popředí | průměrné popředí |
| :--- | ---: | ---: |
| `random` | 15,6 % | 7,01 % |
| `pos_neg` | **4,4 %** | **11,01 %** |

Výsledek tréninku (`lr 3e-4`, `T_0 15`, rozjezd 2 epochy, `pos_neg`):

| epocha | 1 | 3 | 5 | 7 | 9 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| train Dice | 0,002 | 0,050 | 0,152 | 0,237 | 0,297 |
| val Dice | 0,001 | 0,030 | 0,092 | 0,103 | **0,182** |

**val Dice 0,182 = 2,3× proti výchozímu stavu**, a křivka na konci pořád strmě
stoupá (není to plató).

#### Iterace 3 — kontrola férovosti u U-Netu ✅

⚠️ Kdyby cílené výřezy dostal jen UNETR, přestalo by jít o srovnání architektur —
lišila by se datová pipeline. Proto se totéž ověřilo i na U-Netu, za stejných
podmínek jako v PoC (400 snímků, 15 epoch):

| U-Net | epocha 2 | epocha 5 | epocha 8 | epocha 11 | epocha 14 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `random` (PoC) | 0,297 | 0,694 | 0,699 | 0,833 | 0,883 |
| `pos_neg` | **0,537** | 0,812 | 0,791 | 0,857 | **0,897** |

Cílené vzorkování **pomáhá i U-Netu** (rychlejší rozjezd, o kousek lepší konec).
Je proto nasazené u **obou modelů** a srovnání zůstává korektní.

#### Přijaté nastavení

| Parametr | U-Net | UNETR | poznámka |
| :--- | :--- | :--- | :--- |
| `crop_strategy` | `pos_neg` | `pos_neg` | **musí být shodné** — je to datová pipeline |
| `pos_ratio` | 0,8 | 0,8 | shodné |
| `lr` | 1e-3 | 3e-4 | per-architekturu laděný hyperparametr |
| `t0` | 5 | 15 | UNETR se s krátkým cyklem nestihne rozjet |
| `warmup_epochs` | 0 | 2 | u transformerů standard |

#### Iterace 4 — zmenšení Vision Transformeru ✅✅ (největší páka)

Pozorování, které k tomu vedlo: UNETR má ve výchozím nastavení MONAI **ViT-Base
(116 M parametrů)**, což je velikost z původní publikace — jenže tam trénovali
na řádově větších datech. Proti U-Netu s 1,6 M parametru je to **72násobek na
dataset o 2978 snímcích**. Předimenzovaný model se v malém datovém režimu učí
pomalu; to nebyl handicap ve prospěch UNETR, ale proti němu.

Naměřené velikosti a rychlosti (dávka 8, 256 px, AMP):

| model | parametrů | ms/dávka | VRAM |
| :--- | ---: | ---: | ---: |
| U-Net | 1,6 M | 20,3 | 0,2 GB |
| UNETR ViT-Base | 116,0 M | 166,0 | 3,1 GB |
| UNETR ViT-Small | 30,1 M | 89,2 | — |
| **UNETR ViT-Tiny** | **8,4 M** | **69,9** | — |

Výsledek tréninku za jinak identických podmínek (400 snímků, 10 epoch,
`lr 3e-4`, `T_0 15`, rozjezd 2, `pos_neg`):

| epocha | 1 | 3 | 5 | 7 | 9 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ViT-Base val Dice | 0,001 | 0,030 | 0,092 | 0,103 | 0,182 |
| **ViT-Tiny val Dice** | 0,011 | 0,073 | 0,235 | 0,260 | **0,400** |

**ViT-Tiny je přes dvojnásobek ViT-Base — se čtrnáctinou parametrů.**
Křivka navíc na konci pořád strmě stoupá (train Dice 0,519).

#### Souhrn všech iterací

| varianta | parametrů | val Dice | vs výchozí |
| :--- | ---: | ---: | ---: |
| ViT-Base, `lr 1e-4`, `T_0 5`, náhodné výřezy | 116 M | 0,078 | — |
| ViT-Base, `lr 3e-4`, `T_0 15`, náhodné výřezy | 116 M | ~0,015 | **horší** |
| ViT-Base + `pos_neg` + rozjezd | 116 M | 0,182 | 2,3× |
| **ViT-Tiny + `pos_neg` + rozjezd** | **8,4 M** | **0,400** | **5,1×** |

Poučení, které patří do zprávy: hlavní páka **nebyl learning rate ani scheduler**
(ta hypotéza byla vyvrácena — samotné zvýšení LR výsledek zhoršilo), ale
**kapacita modelu vzhledem k objemu dat** a **způsob vzorkování výřezů**.

#### Vyhodnocení: vyplatí se UNETR dál tlačit?

**Stav po ladění:** UNETR (ViT-Tiny) má v epoše 9 val Dice 0,400, U-Net ve stejné
epoše zhruba 0,80. Odstup se ladicími iteracemi zmenšil ze **4× na 2×** a UNETR
je navíc po zmenšení ViT **2,4× rychlejší na dávku**, takže dostane víc epoch
za stejný čas. Křivka na konci pořád strmě stoupá — model není u stropu.

**Argumenty pro to pokračovat:**
- Bez UNETR není co s čím srovnávat — je to celý smysl studie.
- Křivka pořád strmě stoupá, model se **prokazatelně učí**, jen pomalu.
- Zkušební běhy jely na 400 snímcích. Ostrý běh má **7,4× víc dat na epochu**,
  a Vision Transformery jsou známé tím, že těží z objemu dat neúměrně víc než
  konvoluční sítě — právě tady by se to mělo projevit nejvíc.
- „UNETR potřebuje na tuto úlohu podstatně víc dat a výpočtu než U-Net" je
  **platný a literaturou podložený závěr**, ne selhání práce.

**Argumenty proti dalšímu ladění:**
- Velká páka (cílené vzorkování) je nalezená; další ladění má klesající výnos.
- Do odevzdání zbývá málo času a ostrý běh je potřeba stihnout.

**Závěr: ladění ukončit, nastavení zafixovat, a rozdíl řešit počtem epoch.**
Po čtyřech iteracích má UNETR reálnou šanci se v ostrém běhu (7,4× víc dat na
epochu) dostat do pásma, kde dává srovnání smysl. Vlastní hybridní architektura,
o které se uvažovalo, tak **není potřeba** — UNETR se podařilo zachránit
laděním.

Pokud UNETR ani tak U-Net nedožene, je to výsledek studie — musí se ale ve
zprávě **poctivě uvést jako výpočtem omezený**, ne prezentovat jako čistý závěr
o kvalitě architektury.

#### Přijaté nastavení UNETR (finální)

```yaml
model:   { vit_size: tiny }              # 8,4 M misto 116 M
data:    { crop_strategy: pos_neg, pos_ratio: 0.8 }
optimizer: { lr: 3.0e-4 }
scheduler: { t0: 15, warmup_epochs: 2 }
```

### 8.8 Ověření celého řetězu ✅

Proběhlo end-to-end na malém vzorku (mikro-trénink 2 epochy / 40 snímků,
predikce, evaluace). Výstupem je tabulka:

```
model           Dice     IoU |     C2     C3     C4     C5     C6     C7
ts2d          0.0000  0.0000 |  0.000  0.000  0.000  0.000  0.000  0.000
unet          0.0000  0.0000 |  0.000  0.000  0.000  0.000  0.000  0.000
```
plus obrázky *originál | ground truth | predikce jednotlivých modelů*.
Nuly jsou očekávané — šlo o test průchodnosti, ne o trénink.

---

## 9. Ostrý běh (10.–11. 9. 2026)

### 9.1 Průběh

Spuštěno jako jeden zřetězený běh (trénink → predikce → TS2D → evaluace) na
notebooku s RTX 3060 6 GB.

| Krok | Trvání |
| :--- | ---: |
| Trénink U-Net, 60 epoch | 3,7 h |
| Predikce U-Net (992 snímků) | 32 s |
| Trénink UNETR, 60 epoch | 7,4 h |
| Predikce UNETR (992 snímků) | ~1 min |
| Inference TotalSegmentator2D (992 snímků) | ~17 min |
| Evaluace | ~2 min |

Naměřené tempo: U-Net **3,5 min/epocha**, UNETR **7,0 min/epocha**, plná
sliding-window validace na 993 snímcích 1,1 min (U-Net) a 1,8 min (UNETR).

### 9.2 ⚠️ Přerušení: notebook usnul uprostřed tréninku

Ve 3:52 ráno, čtyři minuty po zápisu checkpointu epochy 43, přešel notebook do
režimu spánku. Systémový log:

```
11.09.2026 3:52:14   The system is entering sleep....
11.09.2026 11:01:24  The system has returned from a low power state....
```

Proces uspání **přežil** (nadále existoval), ale rozbil se mu CUDA kontext
a DataLoader workery. Sedm hodin pak visel, aniž by cokoliv počítal nebo zapsal
do logu. Poznalo se to podle toho, že se soubor logu ani checkpoint sedm hodin
nezměnily, zatímco epocha trvá 6,7 minuty.

**Ztráta: jedna rozpracovaná epocha (~7 minut).** Přerušitelnost smyčky
(checkpoint po každé epoše + `--resume`) zafungovala přesně tak, jak byla
navržena — po ukončení zaseknutého procesu se běh obnovil od epochy 44.

**Opatření:** `scripts/keep_awake.ps1` drží systém vzhůru po dobu běhu voláním
`SetThreadExecutionState` (stejný mechanismus jako přehrávače videa). Nemění
trvale nastavení systému. **Nechrání před zavřením víka.**

### 9.3 Průběh učení

U-Net (validační Dice):

| epocha | 4 | 14 | 24 | **34** | 44 | 54 | 59 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val Dice | 0,916 | 0,937 | 0,927 | **0,943** | 0,921 | 0,935 | 0,940 |

UNETR (validační Dice):

| epocha | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val Dice | 0,639 | 0,798 | 0,868 | 0,779 | 0,851 | 0,870 | 0,894 | 0,915 |

Dva postřehy pro zprávu:

1. **U-Net byl na stropě zhruba od 15. epochy** — dál už jen kolísal podle fáze
   kosinového cyklu. Šedesát epoch bylo zbytečně mnoho; stačilo by ~25.
   Mechanismus `-best.pth` ale správně vybral nejlepší stav (epocha 34).
2. **UNETR stoupal až do konce** a při warm restartu scheduleru měl hlubší
   propad než U-Net (epocha 14 → 19: 0,868 → 0,779). Transformer je na skok
   learning rate citlivější — konzistentní s tím, proč vůbec potřeboval warmup.
   **Při delším tréninku by se rozdíl mohl dál zmenšit.**

### 9.4 Interpretace výsledků

Hypotéza z `plan.md` měla dvě části a potvrdila se jen jedna:

| Předpoklad | Výsledek |
| :--- | :--- |
| UNETR bude srovnatelný nebo lepší než U-Net | **zčásti** — srovnatelný (rozdíl 0,018), ale ne lepší |
| Oba trénované modely překonají TotalSegmentator2D | **potvrzeno drtivě** |

- **Rozdíl U-Net vs. UNETR je konzistentních ~0,02 napříč všemi třídami.**
  UNETR tedy neselhává na něčem konkrétním, je systematicky mírně horší.
  Očekávaná výhoda globálního kontextu se neprojevila — úloha je silně lokální.
- **C2 je pro oba nejtěžší** (0,908 / 0,874 proti 0,93–0,95 u ostatních).
  Je to strop daný anotací: C2 má jen tři landmarky, takže jeho maska je
  trojúhelníková aproximace.
- **Poměr cena/výkon vychází pro U-Net**: 5× méně parametrů, 2× rychlejší
  trénink, a přesto lepší výsledek.
- **Cesta UNETR z 0,106 na 0,923** je nejcennější poznatek práce — viz §8.10.
  Závěr nezní „transformer nestačí", ale „transformer je konkurenceschopný,
  pokud se naškáluje na objem dostupných dat".

### 9.5 Kde jsou výstupy

```
training_outputs/
├── checkpoints/          unet-atlas-v1-best.pth (18 MB), unetr-atlas-v1-best.pth (82 MB)
├── checkpoints_backup/   záloha pořízená před přegenerováním notebooků
├── predictions/          unet/, unetr/, ts2d/ — po 992 maskách
├── evaluation/           summary_test.csv/json, per_image_test.csv, srovnani/
└── runs/                 TensorBoard logy obou běhů
```
