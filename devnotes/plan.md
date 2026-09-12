# Návrh zápočtové studie: Porovnání architektur CNN a Vision Transformer pro 2D segmentaci páteře

**Autor:** Matěj Bureš  
**Instituce:** Univerzita Jana Evangelisty Purkyně v Ústí nad Labem, Aplikovaná informatika  
**Předmět:** Úvod do strojového učení

> **Revize k 9. 9. 2026.** Dokument byl aktualizován po prvním dni implementace.
> Změny oproti původnímu návrhu jsou vyznačené a **zdůvodněné** — nejsou to úhybné
> manévry, ale doložená rozhodnutí (viz [STAV_PROJEKTU.md](STAV_PROJEKTU.md) a
> [DALSI_KROKY.md](DALSI_KROKY.md)).

---

## 1. Abstrakt a cíle práce

Cílem práce je vyhodnotit a porovnat různé přístupy k sémantické segmentaci 2D
rentgenových snímků krční páteře na datasetu Atlas. Úloha je formulována jako
klasifikace **7 tříd** (pozadí + obratle C2–C7) pro každý pixel.

**Porovnávané modely:**

1. **Lokální konvoluční U-Net** — reprezentant tradičního CNN přístupu,
   **trénovaný od nuly** *(revize: původně se počítalo s pouhou inferencí z
   předtrénovaných vah; vlastní trénink umožňuje kontrolované srovnání za
   identických podmínek)*.
2. **UNETR (UNet Transformers)** — moderní architektura s Vision Transformer
   enkodérem, trénovaná od nuly za identických podmínek jako U-Net.
3. **TotalSegmentator2D / TSXR** — robustní generalizovaný nástroj, využitý
   **pouze pro inferenci** s následným post-processingem výstupních tříd.

**Hypotéza:** UNETR, který díky attention mechanismu pracuje s globálním kontextem
obrazu od první vrstvy, dosáhne srovnatelných nebo lepších výsledků v metrice Dice
než lokální konvoluční U-Net, přičemž oba doménově trénované modely překonají
generalizovaný TotalSegmentator2D, který nebyl na tuto anatomii ani modalitu
specificky trénován.

### ✅ Výsledek (testovací set, 992 snímků)

| Model | Dice | IoU |
| :--- | ---: | ---: |
| **U-Net** | **0,9412** | **0,8942** |
| UNETR (ViT-Tiny) | 0,9229 | 0,8658 |
| TotalSegmentator2D | 0,0004 | 0,0003 |

**Hypotéza se potvrdila jen zčásti.** Výsledky obou trénovaných modelů jsou
srovnatelné (rozdíl 0,018 Dice), ale UNETR U-Net **nepřekonal** — očekávaná
výhoda globálního kontextu se neprojevila, protože úloha je silně lokální.
Druhá část hypotézy (převaha doménově trénovaných modelů nad generalizovaným
nástrojem) se potvrdila drtivě. Podrobný rozbor je v
[STAV_PROJEKTU.md](STAV_PROJEKTU.md) §9.

**Vedlejší (metodický) cíl:** doložit, jak výpočetní rozpočet reálně omezuje
volbu experimentálního designu — a jak se dá metodika zredukovat, aniž by ztratila
vypovídací hodnotu.

---

## 2. Architektura a hardwarové zdroje

| Fáze | Hardware | VRAM | Dostupnost | Účel |
| :--- | :--- | :--- | :--- | :--- |
| Příprava a vývoj | Notebook (i5 11. gen, RTX 3060 Laptop) | 6 GB | nepřetržitě | zpracování dat, vývoj, smoke testy, dlouhé pomalé běhy |
| Ostrý trénink | Desktop (Ryzen 5 3600, RTX 3060) | 12 GB | 8–10 h/noc, 2–3 noci | finální trénink obou modelů |

**Termín odevzdání: 13. 9. 2026.**

**Technologický stack:** PyTorch 2.11 (CUDA 12.8), MONAI, OpenCV, SimpleITK, TensorBoard.

**Konfigurace je navržena tak, aby beze změny běžela na obou strojích** — špičková
spotřeba VRAM je u obou modelů pod 3,1 GB (viz naměřená tabulka v `STAV_PROJEKTU.md` §4.3).

---

## 3. Zpracování dat (Data Pipeline)

Dataset tvoří **4963 párů** PNG snímek + JSON anotace.

### 3.1 Rasterizace anotací ✅ hotovo

**Revize oproti původnímu návrhu:** anotace **nejsou obtahové polygony**, jak se
původně předpokládalo, ale **bodové landmarky** ve formátu LabelMe. Každý obratel
C3–C7 je popsán 4 rohovými body (`top left`, `top right`, `bottom right`,
`bottom left`), obratel C2 kvůli odlišné anatomii pouze 3 body (`bottom left`,
`bottom right`, `centroid`).

Rasterizace tedy spočívá ve složení těchto rohů do polygonu (čtyřúhelník, resp.
trojúhelník) a jeho vyplnění pomocí `cv2.fillPoly` hodnotou třídy. Výsledkem jsou
jednokanálové PNG masky s hodnotami `0` (pozadí) až `6` (C7).

**Metodický důsledek, který patří do zprávy:** maska je **čtyřúhelníková aproximace**
obratlového těla, ne jeho přesný obrys. To omezuje horní hranici dosažitelného Dice —
ale omezuje ji **stejně pro všechny tři modely**, takže srovnání zůstává platné.

### 3.2 Rozdělení dat — prevence data leakage ✅ hotovo

**Revize:** místo rotující 5-Fold křížové validace se používá **jeden pevný split**:

| Množina | Podíl | Počet | Role |
| :--- | ---: | ---: | :--- |
| train | 60 % | 2978 | gradientní trénink |
| val | 20 % | 993 | hlídání přetrénování, výběr „best" checkpointu |
| test | 20 % | 992 | **nedotčený** — otevře se až na závěr pro srovnání všech tří modelů |

**Zdůvodnění:** plná 5-Fold CV znamená 5× trénink obou architektur od nuly, což na
dostupném hardwaru vychází na řády týdnů (viz [dodatek k trénování.md](dodatek%20k%20trénování.md)).
Kód pro generování K-Foldu je v repozitáři **ponechán a spouští se** — rigoróznější
metodika tedy byla navržena, prakticky odbenchmarkována a **vědomě zamítnuta**,
což je samo o sobě výsledek hodný zmínky.

Oddělení `val` a `test` je zde přísnější než v původním návrhu: výběr checkpointu
podle validační sady nesmí kontaminovat finální srovnávací metriku.

Zamíchání je řízeno seedem 42; disjunktnost množin je ověřena asserty.

### 3.3 Patch-based trénink

Z každého snímku se za epochu náhodně vyřízne **8 patchů o velikosti 256×256**
(*revize: původně 20 patchů 512×512*), s augmentacemi (překlopení, rotace o násobky
90°, zoom 0,9–1,1×, gaussovský šum) a volitelnou ekvalizací histogramu (CLAHE).
Batch size 8.

**Zdůvodnění zmenšení patche:** empirický benchmark (`scripts/benchmark_training_step.py`)
ukázal, že u UNETR dá přechod 512 → 256 px **pětinásobnou propustnost**, protože
výpočetní náročnost self-attention roste kvadraticky s počtem tokenů
(1024 tokenů při 512 px vs. 256 tokenů při 256 px). Spolu s mixed precision (AMP)
je to rozdíl mezi „neproveditelné" a „vejde se do jedné noci".

---

## 4. Metodika trénování a vyhodnocení

Oba trénované modely sdílejí **jednu společnou trénovací smyčku, dataset,
augmentace, split i seed** (`training/common/`), aby byl rozdíl ve výsledku
přičitatelný architektuře, ne experimentální nekázni.

* **Optimalizátor:** Adam.
* **Learning rate:** U-Net `1e-2`, **UNETR `1e-4`**.
  *Odchylka je záměrná a nutná:* s hodnotou `1e-2`, vyladěnou pro malý CNN, UNETR
  po první epoše diverguje na `loss = nan` (ověřeno). `1e-4` je hodnota z originální
  publikace UNETR. LR je standardní per-architekturu laděný hyperparametr —
  srovnávat zdivergovaný model by bylo metodicky horší než použít rozdílný LR.
* **Gradient clipping:** norma 1,0 pro oba modely (pojistka proti explozi gradientu).
* **Scheduler:** CosineAnnealingWarmRestarts (`T_0=5`, `T_mult=2`).
* **Ztrátová funkce:** kombinace Dice a Cross-Entropy (`lambda_dice: 1.0`,
  `lambda_ce: 1.0`) — Dice řeší silnou nevyváženost tříd (pozadí vs. drobné obratle),
  CE dává stabilní gradienty od začátku tréninku.
* **Mixed precision (AMP):** zapnuto — na Ampere GPU zhruba dvojnásobné zrychlení.
* **Validace:** `sliding_window_inference` (výřezy 256×256, překryv 0,25,
  gaussovské vážení) přes celé snímky v původním rozlišení. Kvůli ceně se počítá
  **každou 5. epochu**, ne každou; trénovací Dice se loguje každou epochu.
* **Metriky:** **Mean Dice Score bez pozadí** (hlavní) a Mean IoU.
* **Reprodukovatelnost:** seed 42 pro Python, NumPy i PyTorch, deterministické cuDNN,
  seedované DataLoader workery.
* **Přerušitelnost:** checkpoint po každé epoše (včetně stavu optimizeru, scheduleru
  a AMP scaleru), automatický `--resume` a měkký časový rozpočet `--max-minutes` —
  přímý důsledek toho, že desktop je k dispozici jen v nočních oknech.

---

## 5. Harmonogram a stav prací

### ✅ Fáze 1: Příprava dat — HOTOVO
* [x] Analýza formátu anotací (zjištěno: landmarky, ne polygony)
* [x] Skript pro rasterizaci JSON → PNG masky 0–6 (`scripts/rasterize_masks.py`)
* [x] Vygenerování masek pro celý dataset (4963/4963, 0 chyb)
* [x] Barevná varianta masek pro vizuální kontrolu
* [x] Vlastní K-Fold i pevný split 60/20/20 (`scripts/build_fold_split.py`)

### ✅ Fáze 2: Integrace architektur — HOTOVO
* [x] Empirický benchmark rychlosti obou architektur (`scripts/benchmark_training_step.py`)
* [x] Vlastní dataset, transformace, seedování (`training/common/`)
* [x] Sdílená přerušitelná trénovací smyčka s TensorBoardem
* [x] U-Net i UNETR ověřeny end-to-end na malém vzorku (smoke testy)
* [x] Ověřeno, že se oba vejdou do 6 GB VRAM

### ⏳ Fáze 3: Zprovoznění třetího modelu a evaluace — PROBÍHÁ
* [x] TSXR model stažen a nakešován, zjištěn formát vstupu i mapování tříd
* [ ] Ověřit inferenci TS2D na reálném snímku (GPU)
* [ ] Skript pro dávkovou inferenci TS2D + filtraci/přemapování tříd
* [ ] Skript pro predikce U-Net/UNETR na test setu
* [ ] Evaluační skript (Dice, mIoU, srovnávací tabulka)

### ✅ Fáze 3b: Ladění UNETR — HOTOVO
* [x] Čtyři iterace krátkých běhů (`scripts/tune_unetr.py`)
* [x] Vyvrácena hypotéza o learning rate (samotné zvýšení výsledek zhoršilo)
* [x] Nalezeny skutečné příčiny: vzorkování výřezů a předimenzování modelu
* [x] Zlepšení z val Dice 0,078 na 0,400 (5,1×) na zkušebních datech

### ✅ Fáze 4: Ostrý trénink — HOTOVO
* [x] Změřena reálná délka epochy (U-Net 3,5 min, UNETR 7,0 min)
* [x] Trénink U-Net, 60 epoch (3,7 h) — nejlepší val Dice 0,9425
* [x] Trénink UNETR, 60 epoch (7,4 h) — nejlepší val Dice 0,915+
* [x] Zvládnuto přerušení (notebook usnul) díky resume mechanismu

### ✅ Fáze 5: Syntéza — TÉMĚŘ HOTOVO
* [x] Notebooky (příprava dat, oba tréninky, TS2D, evaluace)
* [x] Finální metriky nad testovacím setem pro všechny tři modely
* [x] Vizuální srovnání predikcí vs. ground truth
* [ ] Export masek zpět do JSON polygonů *(volitelné)*
* [ ] **Sepsání zprávy**
