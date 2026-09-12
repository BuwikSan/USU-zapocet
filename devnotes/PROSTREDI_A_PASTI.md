# Prostředí, verze a nášlapné miny

Praktická příručka. Všechno níže je **ověřené na reálném stroji**, ne opsané z
dokumentace. Kdo bude pokračovat, ušetří tím několik hodin hledání.

---

## 1. Stroje

| Stroj | GPU | VRAM | Dostupnost | Role |
| :--- | :--- | ---: | :--- | :--- |
| Notebook (i5 11. gen) | RTX 3060 **Laptop** | 6 GB | prakticky nepřetržitě, klidně 24/7 na 2 dny | vývoj, smoke testy, dlouhé pomalé tréninky |
| Desktop (Ryzen 5 3600) | RTX 3060 | 12 GB | 8–10 h/noc, 2–3 noci | ostrý trénink v nočních oknech |

**Deadline: 13. 9.** (viz `constrains.md`)

Vývoj a dosavadní práce probíhaly na **notebooku**. Benchmarky v
`STAV_PROJEKTU.md` §4.3 jsou naměřené tam — desktop bude o něco rychlejší
(plnotučný čip místo laptopového), takže odhady jsou spíš konzervativní.

Config je nastavený tak (patch 256, batch 8, AMP → **~3.1 GB VRAM**), aby
**běžel beze změny na obou strojích**. Není nutné nic přepínat.

---

## 2. Python a interpret

⚠️ **Na tomto stroji jsou dva Pythony.** V Git Bash je `python3` namapovaný na
MSYS build **bez pipu a bez balíčků** — s ním nic nefunguje.

**Vždy používej Windows launcher:**
```bash
py -3 ...           # správně   → Python 3.13.7, C:\Users\Administrator\AppData\Local\Programs\Python\Python313
python3 ...         # ŠPATNĚ    → MSYS build, "No module named pip", chybí cv2/torch
```

---

## 3. Nainstalované balíčky (klíčové verze)

| Balíček | Verze | Poznámka |
| :--- | :--- | :--- |
| torch | **2.11.0+cu128** | CUDA build, viz past níže |
| monai | aktuální | `pip install monai` |
| opencv-python | 4.13.0 | |
| numpy | 2.4.4 | |
| pandas | 3.0.2 | |
| SimpleITK | 2.5.6 | ⚠️ **bez PNG readeru** |
| ts2d | editable z `Modely/Totalsegmentator2D/totalsegmentator2D-main` | |
| + tensorboard, scikit-image, scipy, pyyaml, gdown | | |

---

## 4. PASTI — přečti dřív, než začneš instalovat nebo debugovat

### 4.1 ⚠️ NEJVĚTŠÍ PAST: `pip install -e .` pro ts2d rozbije CUDA

Balíček `ts2d` má v závislostech **pevně připnutý `torch==2.7.1`**. Jakákoliv pip
operace, která znovu řeší jeho závislosti, **stáhne CPU-only torch a přepíše
tvůj CUDA build**. Projeví se to tím, že:
- `torch.cuda.is_available()` začne vracet `False`
- ts2d vypíše „CUDA is not available in the installed Pytorch package!"
- vlastní trénink v `training/` spadne na CPU (100× pomalejší) nebo neběží vůbec

**Náprava:**
```bash
py -3 -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
py -3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Pravidlo:** ts2d je už nainstalované. **Nespouštěj znovu `pip install -e .`**
v jeho složce. Kdyby to bylo nutné, hned potom přeinstaluj CUDA torch a ověř.

### 4.1b ⚠️⚠️ torch, torchvision a torchaudio MUSÍ být ze stejného buildu

**Tohle byla příčina toho, proč TS2D „visel" a nešel rozchodit.** Po
přeinstalaci torche na `2.11.0+cu128` zůstal torchvision zkompilovaný proti
torch 2.7.1. Binárně to nesedí a projeví se to až hluboko uvnitř nnU-Netu:

```
RuntimeError: operator torchvision::nms does not exist
```

Chyba přitom vypadá, jako by šlo o problém v ts2d nebo v modelu — ve
skutečnosti je to nesoulad verzí. Vždy přeinstalovat **všechny tři najednou**
ze stejného indexu:

```bash
py -3 -m pip install --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Ověření (musí projít celé):
```bash
py -3 -c "import torch, torchvision, torchaudio; import torchvision.ops; \
print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
# ocekavane: 2.11.0+cu128 0.26.0+cu128 2.11.0+cu128 True
```

Pip bude hlásit `ts2d 1.2.0 requires torch~=2.7.0, but you have torch 2.11.0+cu128`
— to je jen varování, ts2d s novějším torchem funguje.

### 4.2 ⚠️ Wheel pro CUDA existuje jen pod tagem `cu128`

Python 3.13 + torch 2.11: indexy `cu121`, `cu124` **nemají žádný build**
(`ERROR: Could not find a version that satisfies the requirement torch`).
Funguje **`cu128`**. Ověření dostupnosti bez instalace:
```bash
py -3 -m pip index versions torch --index-url https://download.pytorch.org/whl/cu128
```
Navíc: `pip install torch --index-url ...` **bez** `--force-reinstall` nic neudělá —
pip vidí, že „torch" už je splněný, a CPU build tam nechá. Vždy `--force-reinstall`.

### 4.3 ⚠️ Stahování modelů TS2D je rozbité (chyba v cizím nástroji)

`ts2d/core/inference/database.py:213` volá `gdown.download()` — knihovnu pro Google
Drive — na **Zenodo** URL. Výsledek: stáhne se 763 bajtů (chybová stránka) místo
250 MB a spadne to na `zipfile.BadZipFile: File is not a zip file`.

**Obejití (funguje, model už je nakešovaný):**
```bash
# 1) stáhnout ručně — curl funguje bez problému
curl -sL -o /tmp/vert.zip "https://zenodo.org/records/17052912/files/tsxr-v2-ep1000b2_vertebrae.zip?download=1"

# 2) rozbalit do cache adresáře, který nástroj očekává
py -3 -c "import zipfile; zipfile.ZipFile(r'<windows_cesta_k_zipu>').extractall(r'C:\Users\Administrator\.ts2d\models')"

# 3) spouštět s use_remote=False, ať se o stažení vůbec nepokouší
```
Cílová struktura (`~/.ts2d/models/<model_key>/<revize>/...`) je **už v tom zipu**,
takže stačí rozbalit do `models/`.

URL ostatních modelů jsou v
`Modely/Totalsegmentator2D/totalsegmentator2D-main/ts2d/data/shared.json`.

### 4.4 ⚠️ TS2D CLI nepřijímá PNG → použij Python API

`ts2d/main.py::_enumerate_cases` povoluje jen `nrrd, nii, nii.gz, mha, mhd`.
Řešení je obejít CLI:

```python
import cv2, SimpleITK as sitk
from ts2d import TS2D

# POZOR: model chce PŘESNĚ 1 kanál (dataset.json: channel_names {"0": "XR"})
arr = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE).astype("float32")
img = sitk.GetImageFromArray(arr)          # 2D, 1 komponenta na pixel

with TS2D(key="tsxr_vertebrae", use_remote=False) as model:
    result = model.predict(img)
    seg = result.segmentation
```

Proč zrovna takhle:
- `sitk.ReadImage(png)` **nefunguje** — tenhle build SimpleITK nemá PNG IO reader
  (`Unable to determine ImageIO reader`). Proto `cv2` → `GetImageFromArray`.
- 3kanálový vstup vyhodí `RuntimeError: The number of channels in the input image
  does not match the models channel definition (1 vs 3)`.
- `key="tsxr_vertebrae"` načte **jen** model pro obratle. Klíč `"tsxr"` by se
  pokusil natáhnout i cardiac/muscles/organs/ribs, které nakešované nejsou.
- `use_remote=False` zabrání pokusu o (rozbité) stahování.

### 4.5 ⚠️ Windows konzole shodí skript na diakritice

Výchozí cp1252 vyhodí `UnicodeEncodeError` na českém `print()`. Každý nový
spouštěcí skript musí mít hned nahoře:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```
Už je to v `training/train_unet.py`, `train_unetr.py`,
`scripts/benchmark_training_step.py`.

### 4.6 ⚠️ DataLoader na Windows používá `spawn` → vše musí být picklovatelné

Cokoliv uložené jako atribut transformu se pickluje do worker procesů. `cv2.CLAHE`
picklovatelný **není** (`TypeError: cannot pickle 'cv2.CLAHE' object`). Proto se
v `training/common/transforms.py` vytváří až uvnitř `__call__`. Stejný pozor si
dej u čehokoliv dalšího s C++ backendem (SimpleITK objekty, otevřené file handly).

### 4.7 ⚠️ Python na pozadí bez `-u` nic nevypíše

Když se dlouhý běh pustí na pozadí s přesměrovaným výstupem, Python stdout
**bufferuje** a v logu není nic ani po desítkách minut — nejde pak rozeznat
„počítá" od „zaseklo se". Vždy `py -3 -u ...` (nebo `PYTHONUNBUFFERED=1`).
Stálo to dnes jedno zbytečně ukončené sezení s TS2D.

### 4.7b ⚠️ Skript s DataLoaderem musí mít `if __name__ == "__main__":`

Windows spouští workery přes `spawn`, což znamená, že si každý worker
**znovu naimportuje hlavní modul**. Bez guardu se celý skript spustí tolikrát,
kolik je workerů — pozná se to podle toho, že se výpis několikrát zopakuje a
skript nikdy nedoběhne. Trénovací skripty guard mají; myslet na to u každého
nového ad-hoc skriptu, který staví DataLoader.

### 4.7c ⚠️ Python bere `sys.path[0]` podle umístění SKRIPTU, ne podle cwd

Skript uložený v Temp a spuštěný z adresáře projektu neuvidí balíčky projektu
(`ImportError: cannot import name 'TS2D' from 'ts2d' (unknown location)`).
Pomocné skripty proto ukládat do `scripts/`, ne do dočasného adresáře.

### 4.8 ⚠️ Batch 32 na 6GB kartě = 50× zpomalení, ne OOM

UNETR @ 256 px, batch 32 si řekne o 8 GB. Karta má 6 GB → **nespadne to na OOM**,
ale zdegraduje na 9.4 s/krok (místo 0.167 s). Snadno se přehlédne. Na notebooku
nepřekračovat batch 16.

---

## 5. Poznámky k datům

- `Inputdata/` je **kompletní a připravené** — masky i splity jsou vygenerované,
  není potřeba nic přegenerovávat.
- Celý `Inputdata/`, `training_outputs/`, `devnotes/` i `examples/` jsou
  v `.gitignore` — do gitu jde jen kód.
- Anotace mají zvláštnost: `shape_type` je u části bodů `"point"` a u části
  `"polygon"`, ale **vždy jde o jediný bod** v poli `points`. Neplést s obtahovým
  polygonem — rozhoduje label (`"C4 top left"` apod.), ne `shape_type`.
