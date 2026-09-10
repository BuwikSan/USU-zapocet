"""
Centralizované cesty pro vlastní (novou) trénovací pipeline této zápočtové studie.

Záměrně NEZÁVISÍ na `Modely/vertebra_segmentation_and_keypoint_extraction-main/Src/project_paths.py`
- ten patří k vypůjčenému, netknutému referenčnímu repozitáři (viz `devnotes/claude.md`,
  požadavek zachovat jeho integritu). Naše vlastní soubory mají vlastní, nezávislý strom cest.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Vstupní data (raw snímky, JSON anotace, vyrasterizované masky, foldy) - vše mimo
# git (viz .gitignore), vyrobené skripty ve scripts/ (Fáze A a B).
INPUT_DATA_DIR = REPO_ROOT / "Inputdata"
IMAGES_DIR = INPUT_DATA_DIR / "datasets-PNG"
# Tytéž snímky s předem spočítanou ekvalizací histogramu (scripts/precompute_heqv.py).
# Používá se, když je v configu data.use_precomputed_heqv: true — odstraňuje to
# opakovaný výpočet CLAHE za běhu, který byl úzkým hrdlem tréninku.
IMAGES_HEQV_DIR = INPUT_DATA_DIR / "datasets-PNG-heqv"
MASKS_DIR = INPUT_DATA_DIR / "datasets-MASK"
FOLDS_DIR = INPUT_DATA_DIR / "folds" / "atlas_vertebra"

# Výstupy vlastního tréninku - checkpointy, TensorBoard logy, predikce, evaluace.
# Také mimo git (velké binární soubory).
OUTPUTS_DIR = REPO_ROOT / "training_outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RUNS_DIR = OUTPUTS_DIR / "runs"           # TensorBoard
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
