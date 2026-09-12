"""Generátor notebooků. Po vygenerování se dá smazat — notebooky jsou samostatné."""
import sys
from pathlib import Path
import nbformat as nbf

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent

HEADER = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() if Path.cwd().name == "notebooks" else Path.cwd() / "notebooks"))
from _common import *
import cv2, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
print("repo:", REPO_ROOT)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
               for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.13"}}
    return n


# =====================================================================
# 01 — data a příprava
# =====================================================================
n01 = nb([
("md", """# 01 — Dataset a příprava dat

Tenhle notebook ukazuje, **co jsou vstupní data a jak se z nich vyrobí to, co
potřebuje segmentační síť**. Odpovídá Fázi A a B z `devnotes/plan.md`.

Veškerá logika je ve skriptech v `scripts/` — notebook je jen volá a ukazuje
výsledky. Vysvětlení pojmů je v [`devnotes/teorie/`](../devnotes/teorie/)."""),

("code", HEADER),

("md", """## 1. Co je v datasetu

Dataset Atlas obsahuje **4963 párů**: rentgenový snímek krční páteře (PNG) a
k němu anotace ve formátu LabelMe (JSON)."""),

("code", """stems = sorted(p.stem for p in JSON_DIR.glob("*.json"))
print(f"snímků: {len(list(IMAGES_DIR.glob('*.png')))}, anotací: {len(stems)}")

example = stems[0]
with open(JSON_DIR / f"{example}.json", encoding="utf-8") as f:
    ann = json.load(f)

print(f"\\nklíče JSONu: {list(ann.keys())}")
print(f"rozlišení: {ann['imageWidth']} x {ann['imageHeight']}")
print(f"počet anotovaných útvarů: {len(ann['shapes'])}")"""),

("md", """## 2. Klíčové zjištění: anotace NEJSOU polygony

Původní návrh práce předpokládal, že anotace obsahují obtahové polygony —
tedy desítky bodů kopírujících hranu obratle. Ve skutečnosti jde o **bodové
landmarky**: každý útvar má v poli `points` jediný bod a rozhoduje jeho `label`.

Schéma je:
- obratle **C3–C7**: 4 rohové body (`top left`, `top right`, `bottom right`, `bottom left`)
- obratel **C2**: jen 3 body (`bottom left`, `bottom right`, `centroid`) — má
  anatomicky odlišný tvar kvůli zubu čepovce

Pozor na past: `shape_type` je u části bodů `"point"` a u části `"polygon"`,
ale vždycky jde o jediný bod. Řídit se tedy podle `label`, ne podle `shape_type`."""),

("code", """for s in ann["shapes"][:8]:
    print(f"{s['label']:20s} shape_type={s['shape_type']:8s} bodů={len(s['points'])}  {s['points'][0]}")
print("...")
print(f"\\ncelkem útvarů: {len(ann['shapes'])}  (C2: 3 body + C3..C7: 5x4 body = 23)")"""),

("code", """# Landmarky vykreslené na snímku.
# Body se barví podle obratle (legenda místo popisků u každého bodu — ty by se
# při 23 bodech překrývaly a byly by nečitelné). Tvar značky rozlišuje roh.
img = cv2.imread(str(IMAGES_DIR / f"{example}.png"), cv2.IMREAD_GRAYSCALE)
code_to_class = {f"C{i}": i - 1 for i in range(2, 8)}     # C2->1 ... C7->6
corner_marker = {"top left": "^", "top right": ">", "bottom right": "v",
                 "bottom left": "<", "centroid": "*"}

plt.figure(figsize=(8, 10))
plt.imshow(img, cmap="gray")
for s in ann["shapes"]:
    code, _, corner = s["label"].strip().partition(" ")
    x, y = s["points"][0]
    color = CLASS_PALETTE_RGB[code_to_class[code]] / 255
    plt.plot(x, y, corner_marker.get(corner, "o"), ms=11, color=color,
             markeredgecolor="black", markeredgewidth=0.6)

plt.legend(handles=legend_patches(), loc="upper right", fontsize=9, title="obratel")
plt.title(f"Landmarky v anotaci — snímek {example}\\n"
          f"tvar značky = roh (▲ top-left, ▶ top-right, ▼ bottom-right, ◀ bottom-left, ★ centroid)",
          fontsize=10)
plt.axis("off"); plt.tight_layout(); plt.show()"""),

("md", """## 3. Rasterizace — z landmarků na masku

Síť potřebuje masku, kde má každý pixel přiřazenou třídu. Skript
`scripts/rasterize_masks.py` proto rohové body poskládá do polygonu
(čtyřúhelník pro C3–C7, trojúhelník pro C2) a vyplní ho hodnotou třídy:

| hodnota | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| třída | pozadí | C2 | C3 | C4 | C5 | C6 | C7 |

**Metodický důsledek, který patří do zprávy:** maska je *čtyřúhelníková
aproximace* obratlového těla, ne jeho přesný obrys. Tím je omezená horní hranice
dosažitelného Dice — ale omezená **stejně pro všechny tři porovnávané modely**,
takže srovnání zůstává platné."""),

("code", """# Data už jsou vygenerovaná; přegenerování by trvalo pár minut.
# Odkomentuj, pokud je potřeba je vyrobit znovu:
# run_script("scripts/rasterize_masks.py")

n_masks = len(list(MASKS_DIR.glob("*.png")))
print(f"hotových masek: {n_masks}")"""),

("code", """mask = cv2.imread(str(MASKS_DIR / f"{example}.png"), cv2.IMREAD_GRAYSCALE)
print("hodnoty v masce:", np.unique(mask))
print("rozměr masky:", mask.shape, "| rozměr snímku:", img.shape)

show_row([img, colorize(mask), overlay(img, mask)],
         ["originál", "maska (0–6)", "maska přes snímek"], cmap="gray")
plt.figure(figsize=(7, 0.6)); plt.legend(handles=legend_patches(), ncol=6, loc="center")
plt.axis("off"); plt.show()"""),

("md", """### Proč existují dvě verze masek

Maska pro trénink má hodnoty 0–6. To je z 256 úrovní šedi tak málo, že v běžném
prohlížeči vypadá jako černý obrázek. Skript proto vedle ní ukládá i barevnou
verzi do `datasets-MASK-color/` — slouží **jen ke kontrole okem**, do tréninku
nevstupuje."""),

("md", """## 4. Rozdělení dat — prevence data leakage

Skript `scripts/build_fold_split.py` dělá dvě věci:

1. **K-Fold** (5 foldů) — původně plánovaná rigoróznější metodika
2. **Pevný split 60/20/20** — to, co se skutečně používá

Proč se od K-Foldu ustoupilo: znamenal by 5× trénink obou architektur od nuly,
což na dostupném hardwaru vychází na týdny (viz `devnotes/dodatek k trénování.md`).
Kód pro K-Fold ale v projektu **zůstal a spouští se** — je doklad, že
rigoróznější varianta byla navržena, odbenchmarkována a vědomě zamítnuta.

Rozdělení na tři části je přísnější než pouhé train/val:

| množina | role |
|---|---|
| **train** | učí se na ní model (gradient) |
| **val** | hlídá přetrénování, vybírá se podle ní nejlepší checkpoint |
| **test** | **nedotčená** — otevře se až na konec pro srovnání všech tří modelů |

Kdyby se checkpoint vybíral podle téže sady, na které se pak měří výsledek,
bylo by finální číslo nadhodnocené."""),

("code", """info = json.loads((FOLDS_DIR / "fixed_split_info.json").read_text(encoding="utf-8"))
print(json.dumps(info, ensure_ascii=False, indent=2))"""),

("code", """# Kontrola, že se množiny nepřekrývají (to je celá pointa)
sets = {name: set(pd.read_csv(FOLDS_DIR / f"{name}.csv")["image"]) for name in ["train", "val", "test"]}
for a in sets:
    for b in sets:
        if a < b:
            print(f"průnik {a} ∩ {b}: {len(sets[a] & sets[b])}")
print("sjednocení:", len(set().union(*sets.values())), "z celkem", len(stems))"""),

("md", """## 5. Předpočítaná ekvalizace histogramu (CLAHE)

CLAHE zvýrazní kontrast kostních struktur. Původně se počítala za běhu tréninku,
což se ukázalo jako **úzké hrdlo celé pipeline**: měření ukázalo ~13,7 ms práce
na CPU na jeden vzorek proti 21 ms výpočtu na GPU pro celou dávku osmi vzorků —
grafika tedy z 80 % času jen čekala.

Navíc se kvůli `patches_per_image: 8` tentýž snímek zpracovával **osmkrát za
epochu** se stejným výsledkem.

Skript `scripts/precompute_heqv.py` proto CLAHE spočítá jednou dopředu.
Sémantika zůstává stejná — CLAHE je adaptivní a musí se počítat na celém snímku
před vyříznutím výřezu, což předpočítání zachovává."""),

("code", """heqv = cv2.imread(str(IMAGES_HEQV_DIR / f"{example}.png"), cv2.IMREAD_GRAYSCALE)
show_row([img, heqv], ["originál", "po CLAHE"], figsize=(11, 7), cmap="gray")

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].hist(img.ravel(), bins=64, color="gray"); ax[0].set_title("histogram — originál")
ax[1].hist(heqv.ravel(), bins=64, color="steelblue"); ax[1].set_title("histogram — po CLAHE")
plt.tight_layout(); plt.show()"""),

("md", """## 6. Statistika datasetu

Kolik plochy vlastně obratle zabírají? Odpověď vysvětluje, proč se metriky
počítají **bez pozadí** — a proč se používá Dice loss."""),

("code", """rng = np.random.default_rng(42)
sample = rng.choice(stems, size=100, replace=False)

shares, sizes = [], []
for st in sample:
    m = cv2.imread(str(MASKS_DIR / f"{st}.png"), cv2.IMREAD_GRAYSCALE)
    shares.append((m > 0).sum() / m.size * 100)
    sizes.append(m.shape)

print(f"podíl obratlů na ploše: průměr {np.mean(shares):.2f} %, "
      f"rozsah {np.min(shares):.2f}–{np.max(shares):.2f} %")
print(f"rozlišení snímků: výška {min(s[0] for s in sizes)}–{max(s[0] for s in sizes)}, "
      f"šířka {min(s[1] for s in sizes)}–{max(s[1] for s in sizes)}")

plt.figure(figsize=(7, 3))
plt.hist(shares, bins=25, color="steelblue")
plt.xlabel("podíl obratlů na ploše snímku [%]"); plt.ylabel("počet snímků")
plt.title("Nevyváženost tříd (vzorek 100 snímků)"); plt.show()"""),

("md", """**Závěr:** obratle zabírají kolem 3 % plochy. Model, který by predikoval
výhradně pozadí, by měl přes 96 % pixelů správně — proto se jako hlavní metrika
používá **Dice průměrovaný přes třídy C2–C7 bez pozadí** a jako ztrátová funkce
kombinace Dice + Cross-Entropy.

Rozlišení snímků se navíc liší, což je důvod, proč se trénuje na výřezech pevné
velikosti a při vyhodnocení se používá sliding-window inference.

➡️ Dál: [`02_trenink_unet.ipynb`](02_trenink_unet.ipynb)"""),
])

# =====================================================================
# 02 — trénink U-Net
# =====================================================================
n02 = nb([
("md", """# 02 — Trénink U-Net

U-Net je **konvoluční baseline** celé studie. Trénuje se od nuly na datech
připravených v notebooku 01.

Teorie: [`devnotes/teorie/01_konvoluce_a_unet.md`](../devnotes/teorie/01_konvoluce_a_unet.md)"""),

("code", HEADER),

("md", """## 1. Konfigurace

Všechny hyperparametry jsou v `training/configs/unet.yaml`. UNETR má vlastní
config se **stejnými** hodnotami všude kromě názvu běhu, počtu epoch a learning
rate — to je podmínka kontrolovaného experimentu: liší se jen architektura."""),

("code", """import yaml
cfg = yaml.safe_load((REPO_ROOT / "training/configs/unet.yaml").read_text(encoding="utf-8"))
print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))"""),

("md", """### Poznámka k learning rate

Původní vypůjčený repozitář používal `lr = 0.01`. Při ověřování se ukázalo, že
s naší konfigurací (patch 256, batch 8, mixed precision) **U-Net po zhruba 200
krocích zdiverguje na `loss = nan`**. Hodnota je proto snížená na `1e-3`
(standard pro optimalizátor Adam).

Je to praktická ilustrace toho, že hyperparametry se nedají přebírat mezi
konfiguracemi bez ověření. Trénovací smyčka má kvůli tomu i pojistku, která
divergenci pozná a běh zastaví s jasnou hláškou — bez ní by osmihodinový noční
běh vyprodukoval bezcenný model a zjistilo by se to až ráno."""),

("md", """## 2. Architektura

Vstup má 3 kanály (rentgen je šedotónový, kanál se ztrojí kvůli shodě
s konfigurací původního repozitáře), výstup 7 kanálů — jeden na třídu."""),

("code", """import torch
from training.common.models import build_unet

model = build_unet()
n_params = sum(p.numel() for p in model.parameters())
print(f"U-Net — počet parametrů: {n_params/1e6:.2f} M")

x = torch.randn(1, 3, 256, 256)
print("vstup :", tuple(x.shape))
print("výstup:", tuple(model(x).shape), "  (7 kanálů = 7 tříd)")"""),

("code", """# Kontrola hardwaru — bez GPU by trénink trval řádově déle
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, VRAM {p.total_memory/1e9:.2f} GB")
    print("torch:", torch.__version__)"""),

("md", """## 3. Rychlý test průchodnosti

Než se pustí několikahodinový běh, ověříme na hrstce snímků, že celý řetěz
funguje. Trvá to zhruba minutu.

⚠️ Test běží pod **vlastním jménem** (`--run-name smoke-unet`). Bez toho by
zapisoval do checkpointů ostrého běhu a v kombinaci s `--no-resume` by je
přepsal — tedy zahodil několikahodinový trénink."""),

("code", """run_script("training/train_unet.py", "--run-name", "smoke-unet",
           "--max-epochs", "2", "--limit-train", "40", "--limit-val", "4", "--no-resume")"""),

("md", """## 4. Ostrý trénink

Naměřeno na plných datech: **3,5 min na epochu** + 1,1 min na plnou
sliding-window validaci (každou 5. epochu). 60 epoch tedy vyjde na **~3,7 h**.

Trénink je **přerušitelný**:
- checkpoint se ukládá po každé epoše, takže přerušení stojí nanejvýš
  rozpracovanou epochu
- `--resume` (výchozí) najde poslední checkpoint a naváže
- `--max-minutes` je časový rozpočet jednoho spuštění — hodí se, když je grafika
  k dispozici jen v nočním okně

Buňka je záměrně zakomentovaná, aby se dlouhý běh nespustil omylem."""),

("code", """# ODKOMENTUJ pro ostrý trénink:
# run_script("training/train_unet.py", "--max-minutes", "180")

print("Spouštět raději z terminálu, ať notebook nedrží běh:")
print("  py -3 -u training/train_unet.py --max-minutes 180")"""),

("md", """## 5. Průběh tréninku

Smyčka zapisuje metriky do TensorBoardu. Načteme je přímo, ať je průběh vidět
v notebooku.

Validační Dice se počítá jen každou 5. epochu — plná sliding-window inference
přes celé snímky je řádově dražší než tréninkový krok na výřezu."""),

("code", """from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_curves(run_name):
    run_dir = OUTPUTS_DIR / "runs" / run_name
    if not run_dir.exists():
        print(f"Log {run_dir} zatím neexistuje — nejdřív trénink."); return {}
    acc = EventAccumulator(str(run_dir)); acc.Reload()
    return {tag: ([e.step for e in acc.Scalars(tag)], [e.value for e in acc.Scalars(tag)])
            for tag in acc.Tags()["scalars"]}

curves = load_curves("unet-atlas-v1")
print("dostupné metriky:", list(curves))"""),

("code", """if curves:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    if "Loss/train" in curves:
        ax[0].plot(*curves["Loss/train"]); ax[0].set_title("Loss (train)")
    if "Dice/train" in curves:
        ax[1].plot(*curves["Dice/train"], label="train")
    if "Dice/val" in curves:
        ax[1].plot(*curves["Dice/val"], "o-", label="val")
    ax[1].set_title("Dice"); ax[1].legend()
    if "LR" in curves:
        ax[2].plot(*curves["LR"]); ax[2].set_title("Learning rate")
    for a in ax: a.set_xlabel("epocha"); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()"""),

("md", """**Jak číst křivky:**
- *Loss klesá, Dice roste* → model se učí
- *train Dice roste, val Dice stagnuje nebo klesá* → přetrénování; použije se
  checkpoint `-best.pth`, který se ukládá podle nejlepšího validačního Dice
- *pilovité LR* → scheduler `CosineAnnealingWarmRestarts` periodicky restartuje
  learning rate, což pomáhá vyskočit z mělkých lokálních minim

➡️ Dál: [`03_trenink_unetr.ipynb`](03_trenink_unetr.ipynb)"""),
])

# =====================================================================
# 03 — trénink UNETR
# =====================================================================
n03 = nb([
("md", """# 03 — Trénink UNETR

UNETR nahrazuje konvoluční enkodér **Vision Transformerem**. Trénuje se za
naprosto stejných podmínek jako U-Net — stejná data, stejný split, stejné
augmentace, stejný seed, stejná trénovací smyčka. Liší se jen architektura
(a learning rate, viz níže).

Teorie: [`devnotes/teorie/02_vision_transformer_a_unetr.md`](../devnotes/teorie/02_vision_transformer_a_unetr.md)"""),

("code", HEADER),

("md", """## 1. Porovnání konfigurací

Ukážeme si rozdíly proti U-Netu explicitně — u srovnávací studie je zásadní umět
doložit, že se neliší nic jiného než to, co se lišit má."""),

("code", """import yaml
u = yaml.safe_load((REPO_ROOT / "training/configs/unet.yaml").read_text(encoding="utf-8"))
t = yaml.safe_load((REPO_ROOT / "training/configs/unetr.yaml").read_text(encoding="utf-8"))

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{pre}{k}.")) if isinstance(v, dict) else out.update({f"{pre}{k}": v})
    return out

fu, ft = flat(u), flat(t)
rows = [{"parametr": k, "U-Net": fu.get(k), "UNETR": ft.get(k)}
        for k in sorted(set(fu) | set(ft)) if fu.get(k) != ft.get(k)]
print("ROZDÍLY:"); display(pd.DataFrame(rows))
print(f"\\nshodných parametrů: {sum(1 for k in fu if fu.get(k) == ft.get(k))}")"""),

("md", """### Proč má UNETR jiný learning rate

S hodnotou `1e-2` UNETR **diverguje na `nan` hned po první epoše**. `1e-4` je
hodnota z originální publikace UNETR (Hatamizadeh et al., 2021) a je to běžný
řád pro Vision Transformery.

Rozdílný learning rate mezi architekturami je legitimní — je to standardní
per-architekturu laděný hyperparametr. Použít pro transformer hodnotu vyladěnou
pro CNN by naopak bylo metodicky horší: srovnávali bychom zdivergovaný model."""),

("md", """## 2. Architektura a její cena

UNETR má i po zmenšení víc parametrů než U-Net. Velikost použitého Vision
Transformeru se bere **z configu**, ne z výchozí hodnoty MONAI — proč, to
vysvětluje kapitola 3."""),

("code", """import torch
from training.common.models import build_unet, build_unetr, VIT_SIZES

vit_size = t.get("model", {}).get("vit_size", "base")
unet, unetr = build_unet(), build_unetr(patch_size=256, vit_size=vit_size)
pu = sum(p.numel() for p in unet.parameters())
pt = sum(p.numel() for p in unetr.parameters())
print(f"U-Net                  {pu/1e6:7.2f} M parametrů")
print(f"UNETR (ViT-{vit_size:<5s})      {pt/1e6:7.2f} M parametrů   ({pt/pu:.0f}x více)")

print("\\nvšechny dostupné velikosti ViT (hidden, mlp, heads):")
for name, cfg in VIT_SIZES.items():
    mark = "  <- použitá" if name == vit_size else ""
    print(f"  {name:6s} {cfg}{mark}")

x = torch.randn(1, 3, 256, 256)
print("\\nvýstup UNETR:", tuple(unetr(x).shape))"""),

("md", """## 3. Proč zmenšený transformer — výsledky ladění

Ve zkušebních bězích se ukázalo, že UNETR s výchozím nastavením MONAI
(ViT-Base, 116 M parametrů) se na našich datech prakticky **nerozjede**.
Následovaly čtyři iterace ladění, všechny za identických podmínek
(400 snímků, 10 epoch) — podrobnosti v
[`devnotes/STAV_PROJEKTU.md`](../devnotes/STAV_PROJEKTU.md) §8.10.

| varianta | parametrů | val Dice | vs výchozí |
|---|---:|---:|---:|
| ViT-Base, `lr 1e-4`, `T_0 5`, náhodné výřezy | 116 M | 0,078 | — |
| ViT-Base, `lr 3e-4`, `T_0 15`, náhodné výřezy | 116 M | ~0,015 | **horší** |
| ViT-Base + cílené výřezy + rozjezd LR | 116 M | 0,182 | 2,3× |
| **ViT-Tiny + cílené výřezy + rozjezd LR** | **8,4 M** | **0,400** | **5,1×** |

Nejcennější poučení: **první hypotéza byla mylná.** Sázelo se na learning rate
a scheduler, jenže samotné zvýšení LR výsledek naopak *zhoršilo*. Skutečné
příčiny byly dvě jiné:

1. **Vzorkování výřezů.** Obratle zabírají jen ~3 % plochy snímku, takže 15,6 %
   rovnoměrně náhodných výřezů neobsahovalo vůbec žádné popředí. Model dostával
   převážně signál „všechno je pozadí" a triviální řešení je pro něj silné
   lokální minimum. Cílené vzorkování (80 % výřezů vystředěných na obratel)
   snížilo podíl prázdných na 4,4 %.
2. **Kapacita vzhledem k objemu dat.** 116 M parametrů na 2978 snímcích je
   72násobek U-Netu. To nebyl handicap *ve prospěch* UNETR, ale proti němu.

Obojí je nasazené i u U-Netu tam, kde jde o datovou pipeline — cílené
vzorkování musí být u obou modelů stejné, jinak by se nesrovnávaly architektury,
ale předzpracování. Ověřeno, že U-Netu taky pomohlo (0,883 → 0,897)."""),

("md", """### Proč zrovna 256 px

Self-attention porovnává každý token s každým, takže její cena roste
**kvadraticky s počtem tokenů**. Při dělení obrazu na dlaždice 16×16 px:

| velikost výřezu | tokenů | dvojic (∝ cena attention) |
|---|---|---|
| 512×512 | 32×32 = 1024 | ~1 050 000 |
| 256×256 | 16×16 = 256 | ~65 500 |

Naměřená propustnost to potvrzuje — tabulka je v
[`devnotes/STAV_PROJEKTU.md`](../devnotes/STAV_PROJEKTU.md) §4.3. Přechod na
256 px dal u UNETR **pětinásobnou propustnost**, mixed precision další zhruba
dvojnásobek.

U-Net je proti tomu tak levný, že by mu 512 px nevadilo — ale musí mít **stejné
podmínky** jako UNETR, jinak by srovnání nic neznamenalo."""),

("md", """## 4. Test průchodnosti

⚠️ Stejně jako u U-Netu běží pod vlastním jménem (`--run-name smoke-unetr`),
aby nepřepsal checkpointy ostrého běhu."""),

("code", """run_script("training/train_unetr.py", "--run-name", "smoke-unetr",
           "--max-epochs", "2", "--limit-train", "16", "--limit-val", "2", "--no-resume")"""),

("md", """## 5. Ostrý trénink

Naměřeno na plných datech: **7,0 min na epochu** + 1,8 min na plnou
sliding-window validaci (běží každou 5. epochu). 60 epoch tedy vyjde
zhruba na **7,4 h**."""),

("code", """# ODKOMENTUJ pro ostrý trénink:
# run_script("training/train_unetr.py", "--max-minutes", "360")

print("Spouštět raději z terminálu:")
print("  py -3 -u training/train_unetr.py --max-minutes 360")"""),

("md", """## 6. Srovnání průběhu obou modelů

Křivky se načítají z TensorBoard logů ostrého běhu (`unet-atlas-v1`,
`unetr-atlas-v1`), ne z testu průchodnosti výše."""),

("code", """from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_curves(run_name):
    run_dir = OUTPUTS_DIR / "runs" / run_name
    if not run_dir.exists(): return {}
    acc = EventAccumulator(str(run_dir)); acc.Reload()
    return {tag: ([e.step for e in acc.Scalars(tag)], [e.value for e in acc.Scalars(tag)])
            for tag in acc.Tags()["scalars"]}

cu, ct = load_curves("unet-atlas-v1"), load_curves("unetr-atlas-v1")

if cu or ct:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for c, name in ((cu, "U-Net"), (ct, "UNETR")):
        if "Loss/train" in c: ax[0].plot(*c["Loss/train"], label=name)
        if "Dice/val" in c:   ax[1].plot(*c["Dice/val"], "o-", label=name)
    ax[0].set_title("Loss (train)"); ax[1].set_title("Dice (validace)")
    for a in ax: a.set_xlabel("epocha"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.show()
else:
    print("Zatím nejsou logy — nejdřív trénink.")"""),

("md", """➡️ Dál: [`04_totalsegmentator.ipynb`](04_totalsegmentator.ipynb)"""),
])

# =====================================================================
# 04 — TotalSegmentator2D
# =====================================================================
n04 = nb([
("md", """# 04 — TotalSegmentator2D (TSXR)

Třetí porovnávaný přístup. Na rozdíl od U-Netu a UNETR se **netrénuje** — jde
o hotový, obecný, předtrénovaný model, který se jen pustí na naše data.

Reprezentuje otázku: *„Vyplatí se trénovat vlastní model, když existuje hotové
obecné řešení?"*"""),

("code", HEADER),

("md", """## 1. Co model umí a co potřebujeme my

TSXR segmentuje **26 anatomických struktur** (kost křížová, L1–L5, T1–T12,
C1–C7). Nás z toho zajímá jen šest krčních obratlů.

Skript `scripts/ts2d_inference.py` proto výstup filtruje a přemapovává na naši
škálu 0–6. Mapování se dělá **podle jmen tříd**, ne podle natvrdo zapsaných
čísel — jiná revize modelu by mohla mít jiné pořadí a maska by se tiše
přemapovala špatně."""),

("code", """model_dir = Path.home() / ".ts2d" / "models" / "tsxr-v2-ep1000b2_vertebrae" / "r001"
ds = list(model_dir.rglob("dataset.json"))
if ds:
    meta = json.loads(ds[0].read_text(encoding="utf-8"))
    print("kanály na vstupu:", meta["channel_names"])
    print(f"tříd celkem: {len(meta['labels'])}")
    print("\\nkrční obratle a jejich ID v modelu:")
    for name, v in meta["labels"].items():
        if name.startswith("vertebrae-c"):
            print(f"  {name:16s} -> {v}")
else:
    print("Model není nakešovaný — viz devnotes/PROSTREDI_A_PASTI.md §4.3")"""),

("md", """### Poznámky k rozchození (stálo to nejvíc času)

Postupně se narazilo na čtyři nezávislé překážky — všechny jsou popsané
v [`devnotes/PROSTREDI_A_PASTI.md`](../devnotes/PROSTREDI_A_PASTI.md):

1. **Stahování modelů v nástroji je rozbité** — používá knihovnu `gdown`
   (určenou pro Google Drive) na Zenodo URL; stáhne 763 bajtů místo 250 MB.
   Model se musel stáhnout ručně přes `curl`.
2. **`torchvision` nesměl zůstat ze starého buildu** — jinak inference padá
   hluboko uvnitř nnU-Netu na `operator torchvision::nms does not exist`.
   Torch, torchvision i torchaudio musí být ze stejného CUDA buildu.
3. **CLI nepřijímá PNG** (jen `nrrd/nii/mha/mhd`) → obchází se Python API.
4. **Vstup musí být jednokanálový** a se **správným fyzickým rozlišením** —
   nnU-Net podle něj vstup převzorkovává."""),

("md", """## 2. Inference

Nejdřív pár snímků, ať je vidět, co model vrací."""),

("code", """run_script("scripts/ts2d_inference.py", "--split", "test", "--limit", "3", "--preview")"""),

("code", """ts2d_dir = OUTPUTS_DIR / "predictions" / "ts2d"
preds = sorted(ts2d_dir.glob("*.png"))
print(f"vyrobených masek: {len(preds)}")

if preds:
    stem = preds[0].stem
    img = cv2.imread(str(IMAGES_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(MASKS_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(str(preds[0]), cv2.IMREAD_GRAYSCALE)
    print("třídy v ground truth:", np.unique(gt))
    print("třídy v predikci TS2D:", np.unique(pr))
    show_row([img, overlay(img, gt), overlay(img, pr)],
             ["originál", "ground truth", "TS2D"], cmap="gray")"""),

("md", """## 3. Model krční obratle nenajde

Predikce je prázdná. Model **nenajde ani jeden krční obratel** — místo nich
predikuje kost křížovou a bederní obratle. Na snímku krku.

To je závažné tvrzení. Než ho lze zapsat do zprávy jako *„model na tuto doménu
nepřenáší"*, je nutné vyloučit, že chyba není na naší straně: vykázat cizí model
jako selhávající kvůli vlastní chybě v předzpracování by byl falešný závěr.

Skript `scripts/ts2d_domain_check.py` proto systematicky prochází hypotézy."""),

("code", """# Trvá pár minut — každá kombinace je jedna inference.
run_script("scripts/ts2d_domain_check.py", "--n-images", "1")"""),

("md", """### Vyloučené hypotézy

| Hypotéza | Jak se testovala | Výsledek |
|---|---|---|
| Špatné volání API | `predict()` vrací strukturovaný výstup | ✗ vyloučeno |
| Špatné mapování kanálů | mapování podle jmen z metadat modelu | ✗ vyloučeno |
| Špatné měřítko | spacing 1,0 / 0,7 / 0,5 / 0,35 / 0,25 / 0,15 mm/px | ✗ vyloučeno |
| Obrácená polarita | invertovaný snímek | ✗ vyloučeno |
| Rozbité váhy | predikce jsou prostorově souvislé, ne šum | ✗ vyloučeno |

Model podle svého `plans.json` očekává rozlišení **1,5 mm/px** a trénoval na
snímcích o mediánu **226×239 px**. Při spacingu 0,5 (což po převzorkování dá
248×193 px, tedy nejblíž tréninkovým datům) predikuje nejvíc — ale pořád kost
křížovou a L3.

### Závěr

TSXR trénoval na syntetických rentgenech (DiffDRR rekonstrukce) z CT datasetu
TotalSegmentator, kde převažuje trup a břicho. **Úzký výřez krční páteře je mimo
jeho doménu.**

Je to **doložený negativní výsledek** — a přesně to, co předpovídá hypotéza
v [`devnotes/plan.md`](../devnotes/plan.md). Do zprávy patří i s tabulkou
vyloučených hypotéz, jinak by to vypadalo jako nedbalost při použití nástroje."""),

("md", """## 4. Inference na celém testovacím setu

~1 s na snímek, tedy zhruba 17 minut pro 992 snímků."""),

("code", """# ODKOMENTUJ pro plný běh:
# run_script("scripts/ts2d_inference.py", "--split", "test")

print("Nebo z terminálu:  py -3 -u scripts/ts2d_inference.py --split test")"""),

("md", """➡️ Dál: [`05_vyhodnoceni.ipynb`](05_vyhodnoceni.ipynb)"""),
])

# =====================================================================
# 05 — vyhodnocení
# =====================================================================
n05 = nb([
("md", """# 05 — Vyhodnocení a srovnání

Závěrečný krok studie. Všechny tři modely se porovnají na **stejném, dosud
nedotčeném testovacím setu**.

Teorie k metrikám: [`devnotes/teorie/03_loss_metriky_a_trenink.md`](../devnotes/teorie/03_loss_metriky_a_trenink.md)"""),

("code", HEADER),

("md", """## 1. Jak je srovnání zařízené, aby bylo férové

Všechny tři modely produkují **identický typ artefaktu**: PNG masku s hodnotami
0–6 o rozměru původního snímku.

| model | čím vzniká |
|---|---|
| U-Net | `training/predict.py --model unet` |
| UNETR | `training/predict.py --model unetr` |
| TotalSegmentator2D | `scripts/ts2d_inference.py` |

Evaluační skript pak **vůbec neví, který model masku vyrobil** — porovnává jen
PNG proti PNG. Do srovnání se tak nemůže vloudit nespravedlnost tím, že by se
jeden model vyhodnocoval jinak než druhý."""),

("md", """## 2. Generování predikcí

Vyžaduje natrénované checkpointy v `training_outputs/checkpoints/`. Když se
model trénoval jinde, stačí sem zkopírovat `.pth` soubor — nic jiného se
přenášet nemusí."""),

("code", """ckpt_dir = OUTPUTS_DIR / "checkpoints"
found = sorted(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
print("nalezené checkpointy:")
for c in found:
    print(f"  {c.name}  ({c.stat().st_size/1e6:.0f} MB)")
if not found:
    print("  žádné — nejdřív trénink (notebooky 02 a 03)")"""),

("code", """# Predikce už jsou vygenerované ostrým během. Kdyby bylo potřeba je vyrobit
# znovu (jiný checkpoint, jiný split), stačí odkomentovat — každý běh trvá
# desítky sekund až minuty:
# run_script("training/predict.py", "--model", "unet",  "--split", "test")
# run_script("training/predict.py", "--model", "unetr", "--split", "test")
# run_script("scripts/ts2d_inference.py", "--split", "test")

pred_dir = OUTPUTS_DIR / "predictions"
if pred_dir.exists():
    for d in sorted(pred_dir.iterdir()):
        if d.is_dir():
            print(f"{d.name:10s}: {len(list(d.glob('*.png')))} masek")"""),

("md", """## 3. Metriky

Počítají se dvě: **Dice** a **IoU**. Obě měří překryv predikce a skutečnosti,
ale jinak váží — IoU je přísnější, Dice shovívavější k drobným odchylkám na
hranici. Proto se uvádějí obě.

$$\\text{Dice} = \\frac{2|A \\cap B|}{|A| + |B|} \\qquad
  \\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}$$

**Pozadí se do průměru nepočítá.** Obratle zabírají kolem 3 % plochy, takže
model predikující výhradně pozadí by s pozadím v průměru vypadal skvěle a
neznamenalo by to nic.

Když třída není ani v ground truth, ani v predikci, není co měřit — zapíše se
`NaN` a průměruje se přes `nanmean`. Když ale třída v ground truth **je** a
model ji nenajde, je to skutečná chyba a počítá se jako Dice 0."""),

("code", """run_script("scripts/evaluate_models.py", "--split", "test", "--figures", "4")"""),

("md", """## 4. Výsledky"""),

("code", """summary_path = OUTPUTS_DIR / "evaluation" / "summary_test.csv"
if summary_path.exists():
    df = pd.read_csv(summary_path, index_col="model")
    display(df[["mean_dice", "mean_iou"]].style.format("{:.4f}")
              .background_gradient(cmap="RdYlGn", vmin=0, vmax=1))
else:
    print("Nejdřív spusť evaluaci.")"""),

("code", """if summary_path.exists():
    df = pd.read_csv(summary_path, index_col="model")
    classes = ["C2", "C3", "C4", "C5", "C6", "C7"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    df["mean_dice"].plot.bar(ax=ax[0], color="steelblue", rot=0)
    ax[0].set_title("Průměrný Dice (bez pozadí)"); ax[0].set_ylim(0, 1); ax[0].grid(axis="y", alpha=.3)

    for m in df.index:
        ax[1].plot(classes, [df.loc[m, f"dice_{c}"] for c in classes], "o-", label=m)
    ax[1].set_title("Dice po jednotlivých obratlích"); ax[1].set_ylim(0, 1)
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()"""),

("md", """### Rozptyl mezi snímky

Průměr sám o sobě zakrývá, jestli model funguje stabilně, nebo je na některých
snímcích výborný a na jiných úplně mimo."""),

("code", """per_image_path = OUTPUTS_DIR / "evaluation" / "per_image_test.csv"
if per_image_path.exists():
    pi = pd.read_csv(per_image_path)
    per_img = pi.groupby(["model", "stem"])["dice"].mean().reset_index()
    models = per_img["model"].unique()

    plt.figure(figsize=(8, 4))
    plt.boxplot([per_img[per_img["model"] == m]["dice"].dropna() for m in models], labels=models)
    plt.ylabel("Dice na snímek"); plt.title("Rozptyl mezi snímky"); plt.grid(axis="y", alpha=.3)
    plt.show()"""),

("md", """## 5. Vizuální srovnání

Čísla neukážou všechno. Vizuální kontrola odhalí věci, které se v průměru
ztratí — třeba že model najde obratle správně, ale posunuté o jednu pozici."""),

("code", """fig_dir = OUTPUTS_DIR / "evaluation" / "srovnani"
figs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
for f in figs[:3]:
    im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(17, 7)); plt.imshow(im); plt.axis("off")
    plt.title(f.stem); plt.show()
if not figs:
    print("Nejdřív spusť evaluaci.")"""),

("md", """## 6. Závěr

### Naměřené výsledky (testovací set, 992 snímků)

| Model | Dice | IoU |
|---|---:|---:|
| **U-Net** | **0,9412** | **0,8942** |
| UNETR (ViT-Tiny) | 0,9229 | 0,8658 |
| TotalSegmentator2D | 0,0004 | 0,0003 |

### Vyhodnocení hypotézy

Hypotéza z [`devnotes/plan.md`](../devnotes/plan.md) měla dvě části a
**potvrdila se jen jedna**:

1. *„UNETR dosáhne srovnatelných nebo lepších výsledků než U-Net"* —
   **zčásti**. Výsledky jsou srovnatelné (rozdíl 0,018 Dice), ale UNETR
   U-Net nepřekonal. Očekávaná výhoda globálního kontextu z attention se
   neprojevila; úloha je totiž silně lokální — obratle jsou kompaktní útvary
   a konvoluční induktivní bias na ně sedí lépe.
2. *„Oba doménově trénované modely překonají TotalSegmentator2D"* —
   **potvrzeno drtivě** (0,94 a 0,92 proti 0,0004).

### Co stojí za pozornost

**Rozdíl mezi U-Netem a UNETR je konzistentní napříč všemi třídami**
(~0,02 u každého obratle). Není to tedy tak, že by UNETR selhával na něčem
konkrétním — je systematicky o kousek horší. To je čistší zjištění, než kdyby
měl výpadek na jedné třídě.

**C2 je pro oba modely nejtěžší** (0,908 a 0,874 proti 0,93–0,95 u ostatních).
Souvisí to s anotací: C2 má jen tři landmarky místo čtyř, takže jeho maska je
trojúhelníková aproximace. Je to strop daný daty, ne slabina modelů.

**Cesta UNETR z 0,106 na 0,923** je nejsilnější poznatek celé práce. Ve
zkušebním běhu vypadal jako selhání a uvažovalo se o jeho nahrazení. Rozhodly
dvě věci, obě doložené měřením (viz notebook 03 a
[`devnotes/STAV_PROJEKTU.md`](../devnotes/STAV_PROJEKTU.md) §8.10):
zmenšení Vision Transformeru ze 116 M na 8,4 M parametrů a plný dataset místo
400 snímků. Závěr tedy nezní „transformer na tuto úlohu nestačí", ale
**„transformer je konkurenceschopný, pokud se naškáluje na objem dostupných dat"**.

**Poměr cena/výkon** přesto vychází ve prospěch U-Netu: má 5× méně parametrů,
trénuje se 2× rychleji (3,5 vs 7,0 min/epocha) a je o 0,018 Dice lepší.

### Omezení práce

Je poctivé uvést:

- Maska je **čtyřúhelníková aproximace** obratle, ne přesný obrys — strop pro
  dosažitelný Dice, stejný pro všechny modely.
- **Jeden pevný split** místo křížové validace (výpočetní rozpočet), takže
  výsledky jsou zatížené variabilitou konkrétního rozdělení.
- **Výřezy 256 px** místo 512 px kvůli kvadratické ceně attention.
- U-Net byl na stropě zhruba od 15. epochy, zatímco UNETR stoupal až do konce —
  **při delším tréninku by se rozdíl mohl dál zmenšit**.
- TotalSegmentator2D nebyl na tuto doménu trénován; jeho výsledek nevypovídá
  o kvalitě nástroje, jen o přenositelnosti mimo trénovací doménu."""),
])

ALL = [("01_data_a_priprava", n01), ("02_trenink_unet", n02),
       ("03_trenink_unetr", n03), ("04_totalsegmentator", n04),
       ("05_vyhodnoceni", n05)]

# Volitelný filtr: `py -3 _build.py 01 04` prepise jen vybrane notebooky.
# Bez argumentu se prepisou vsechny - pozor, prepis zahodi ulozene vystupy bunek.
wanted = sys.argv[1:]
for name, notebook in ALL:
    if wanted and not any(name.startswith(w) for w in wanted):
        continue
    path = OUT / f"{name}.ipynb"
    nbf.write(notebook, str(path))
    print(f"zapsano: {path.name} ({len(notebook.cells)} bunek)")
