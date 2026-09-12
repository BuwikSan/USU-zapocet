# Další kroky — co zbývá dodělat

Stav k **11. 9. 2026**, deadline **13. 9.**

> ## 🏁 Trénink i evaluace jsou hotové
>
> | Model | Dice | IoU |
> | :--- | ---: | ---: |
> | **U-Net** | **0,9412** | **0,8942** |
> | UNETR (ViT-Tiny) | 0,9229 | 0,8658 |
> | TotalSegmentator2D | 0,0004 | 0,0003 |
>
> Podrobný rozbor v [STAV_PROJEKTU.md](STAV_PROJEKTU.md) §9.

---

## Co zbývá

| # | Krok | Odhad | Stav |
| :-- | :--- | :--- | :--- |
| 1 | **Zpráva** | 3–4 h | ⏳ **jediná podstatná věc** |
| 2 | (volitelně) Export masek do JSON polygonů | 1 h | ⏳ |

Všechno ostatní — příprava dat, trénink, ladění, predikce, evaluace, notebooky,
dokumentace — je hotové a ověřené.

---

## 1. Zpráva

Podklady jsou nachystané, jde hlavně o sepsání. Níže je osnova a ke každému
bodu odkaz, kde jsou čísla.

### Osnova

**1. Úvod a cíl.** Segmentace obratlů C2–C7 z bočních rentgenů, 7 tříd
(pozadí + 6 obratlů), dataset Atlas (4963 snímků). Tři porovnávané přístupy.
→ `plan.md` kap. 1

**2. Data a předzpracování.**
- Anotace **nejsou polygony, ale bodové landmarky** — 4 rohy na obratel,
  C2 jen 3. Rasterizace je skládá do čtyřúhelníků/trojúhelníku.
- Důsledek: maska je **aproximace**, ne přesný obrys → strop pro dosažitelný
  Dice, stejný pro všechny modely.
- Split 60/20/20 s odděleným, nedotčeným testovacím setem.
→ `STAV_PROJEKTU.md` §4.1–4.2, notebook 01

**3. Metodika.** Sdílená trénovací smyčka, dataset, augmentace i seed pro oba
modely. Liší se **jen 6 parametrů** (architektura, LR, scheduler) — notebook 03
to dokládá automaticky generovanou tabulkou rozdílů (26 parametrů shodných).
→ `plan.md` kap. 4

**4. Výsledky.** Tabulka výše + per-class Dice + vizuální srovnání.
→ `training_outputs/evaluation/`, notebook 05

**5. Diskuse.** Viz sekce níže.

**6. Závěr a omezení.**

### Body do diskuse (nejsilnější materiál)

Každý je podložený vlastním měřením, ne citací:

1. **Anotace jako strop.** C2 má nejnižší Dice u obou modelů (0,908 / 0,874
   proti 0,93–0,95 u ostatních) — protože má jen tři landmarky. Není to slabina
   modelů, ale dat.

2. **Zamítnutá 5-Fold CV.** Byla navržena, odbenchmarkována a vědomě zamítnuta
   kvůli výpočetnímu rozpočtu. Kód v repozitáři zůstal.
   → `STAV_PROJEKTU.md` §4.2

3. **Patch 256 místo 512.** S naměřenou tabulkou propustnosti — attention je
   O(n²) v počtu tokenů, zmenšení dalo u UNETR 5× vyšší propustnost.
   → `STAV_PROJEKTU.md` §4.3

4. **Learning rate z referenčního repozitáře divergoval u obou modelů.**
   UNETR hned, U-Net po ~200 krocích. Praktická ilustrace, proč se
   hyperparametry nedají přebírat bez ověření.
   → `STAV_PROJEKTU.md` §5.1, §8.3

5. **Datová pipeline byla úzkým hrdlem, ne GPU.** 110 ms CPU proti 21 ms GPU
   na dávku; GPU čekalo 80 % času. Řešení: předpočítané CLAHE.
   Ukazuje, že „pomalý trénink" neznamená „slabá grafika".
   → `STAV_PROJEKTU.md` §8.1

6. **Ladění UNETR — včetně vyvrácené hypotézy.** První domněnka (learning rate)
   byla mylná; samotné zvýšení LR výsledek *zhoršilo*. Skutečné příčiny:
   vzorkování výřezů a předimenzování modelu (116 M parametrů na 3000 snímcích).
   Cesta z 0,106 na 0,923. **Nejcennější část práce** — doložená diagnóza,
   ne náhodný úspěch.
   → `STAV_PROJEKTU.md` §8.10, notebook 03

7. **TotalSegmentator2D je mimo doménu.** S tabulkou vyloučených hypotéz
   (API, mapování tříd, měřítko, polarita, funkčnost vah) — aby nešlo namítnout,
   že jsme nástroj jen špatně použili.
   → `STAV_PROJEKTU.md` §8.6, notebook 04

8. **Přerušitelnost se vyplatila.** Notebook uprostřed tréninku usnul a proces
   sedm hodin visel s rozbitým CUDA kontextem. Ztráta: jedna epocha.
   → `STAV_PROJEKTU.md` §9.2

9. **U-Net byl na stropě od ~15. epochy, UNETR stoupal až do konce.**
   Při delším tréninku by se rozdíl mohl dál zmenšit — poctivé to uvést.
   → `STAV_PROJEKTU.md` §9.3

---

## 2. Volitelné: export masek do JSON polygonů

`cv2.findContours` na každou třídu → polygon → LabelMe JSON ve formátu původního
datasetu. Inspirace:
`Modely/vertebra_segmentation_and_keypoint_extraction-main/Src/Atlas/inference/extraction.py`.

Cílový soubor: `scripts/masks_to_json.py`. Na srovnání modelů to nemá vliv,
je to jen doplněk podle původního zadání (Fáze E).

---

## Užitečné příkazy

```bash
# přehrát evaluaci (predikce už existují)
py -3 scripts/evaluate_models.py --split test --figures 6

# křivky tréninku
tensorboard --logdir training_outputs/runs

# znovu vygenerovat notebooky (POZOR: zahodí uložené výstupy buněk)
cd notebooks && py -3 _build.py          # vsechny
cd notebooks && py -3 _build.py 03       # jen jeden
```
