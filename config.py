"""
config.py
=========
Central configuration file for the Urdu Few-Shot ABSA project.

WHY THIS FILE EXISTS:
Every later script (cleaning, balancing, splitting, training, evaluation)
will `import config` and read paths/settings from here, instead of
re-typing paths or magic numbers in every script. This is standard
practice in real ML projects: one source of truth for settings, so
if a path or a hyperparameter changes, you edit it in ONE place.

HOW TO USE:
    from config import CFG
    print(CFG.SEED)
    print(CFG.RAW_DATA_DIR)
"""

import os
import random
import numpy as np


class Config:
    # ------------------------------------------------------------
    # 1. PROJECT FOLDER STRUCTURE (auto-detects Kaggle / Colab / local)
    # ------------------------------------------------------------
    # On Kaggle: input datasets are READ-ONLY and live under
    # /kaggle/input/<dataset-name>/, while anything you want to save
    # (clean data, models, results) MUST go under /kaggle/working/.
    #
    # On Colab: we mount Google Drive so your data and outputs persist
    # between sessions (Colab's local disk is wiped when the runtime
    # disconnects). Everything lives under one folder in your Drive.
    #
    # On a local machine: everything lives relative to this config.py
    # file.
    # IS_KAGGLE must be checked using an environment variable that ONLY
    # exists inside an actual Kaggle notebook kernel -- checking for the
    # /kaggle folder itself is NOT reliable, since Colab has started
    # including a /kaggle path too (for its own dataset-caching
    # features), which caused a false positive.
    IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

    try:
        import google.colab  # noqa: F401
        IS_COLAB = True
    except ImportError:
        IS_COLAB = False

    # Colab is checked BEFORE Kaggle below, since IS_COLAB (via the
    # google.colab import) is the more reliable signal of the two.

    if IS_COLAB:
        # Everything lives under one folder in Google Drive so it
        # survives between Colab sessions. Run the Drive-mount cell
        # (see instructions) BEFORE importing this config.
        DRIVE_ROOT = "/content/drive/MyDrive/urdu_absa_project"
        RAW_DATA_DIR = os.path.join(DRIVE_ROOT, "data", "raw")
        CLEAN_DATA_DIR = os.path.join(DRIVE_ROOT, "data", "clean")
        SPLITS_DIR = os.path.join(DRIVE_ROOT, "data", "splits")
        RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
        MODELS_DIR = os.path.join(DRIVE_ROOT, "saved_models")
        LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")

    elif IS_KAGGLE:
        # NOTE: replace "urdu-absa-data" with the EXACT name shown
        # under /kaggle/input/ after you attach your dataset.
        RAW_DATA_DIR = "/kaggle/input/urdu-absa-data"
        CLEAN_DATA_DIR = "/kaggle/working/data/clean"
        SPLITS_DIR = "/kaggle/working/data/splits"
        RESULTS_DIR = "/kaggle/working/results"
        MODELS_DIR = "/kaggle/working/saved_models"
        LOGS_DIR = "/kaggle/working/logs"

    else:
        # Local machine. Use __file__ if available (running a .py file);
        # fall back to the current working directory if this code was
        # instead pasted directly into a notebook cell (where __file__
        # doesn't exist).
        try:
            PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            PROJECT_ROOT = os.getcwd()

        RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
        CLEAN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean")
        SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
        RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
        MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
        LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

    # Raw file names (put your 3 original files inside RAW_DATA_DIR)
    RAW_FOOD_FILE = "Food_Reviews_annotated.xlsx"
    RAW_SPORTS_FILE = "Sports.csv"
    RAW_NEWS_FILE = "News_Politcs__Areas.csv"

    # ------------------------------------------------------------
    # 2. REPRODUCIBILITY
    # ------------------------------------------------------------
    # Using ONE fixed seed everywhere means: every time you re-run
    # the pipeline (cleaning -> splitting -> training), you get the
    # SAME train/val/test split and the SAME shot samples.
    # This is essential for a thesis -- your results must be
    # reproducible, and comparisons across models/shots are only
    # fair if everyone sees the exact same data.
    SEED = 42

    # Multiple seeds for the 5/10/20-shot experiments (variance matters
    # a lot in few-shot learning -- report mean +/- std, not one number).
    # Zero-shot doesn't need multiple seeds since nothing is sampled or
    # trained differently between runs.
    SEEDS = [42, 123, 2024]

    # ------------------------------------------------------------
    # 3. DOMAIN / LABEL DEFINITIONS
    # ------------------------------------------------------------
    DOMAINS = ["food", "sports", "news_politics"]
    LABELS = ["negative", "neutral", "positive"]
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

    # ------------------------------------------------------------
    # 4. FEW-SHOT SETTINGS (per your approved proposal)
    # ------------------------------------------------------------
    # These are examples PER CLASS PER DOMAIN, nested
    # (5-shot subset of 10-shot subset of 20-shot).
    SHOT_SETTINGS = [0, 5, 10, 20]   # 0 = zero-shot
    # Optional stretch setting -- only add 15 if time permits later.
    # SHOT_SETTINGS = [0, 5, 10, 15, 20]

    # ------------------------------------------------------------
    # 5. SPLIT RATIOS (applied AFTER balancing, BEFORE shot sampling)
    # ------------------------------------------------------------
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ------------------------------------------------------------
    # 6. MODEL CHECKPOINTS (Hugging Face hub names)
    # ------------------------------------------------------------
    MODEL_CHECKPOINTS = {
        "mbert": "bert-base-multilingual-cased",
        "xlmr": "xlm-roberta-base",
        "mt5": "google/mt5-small",   # mt5-small first -- fast to iterate,
                                       # can upgrade to mt5-base later if
                                       # time/compute allows.
    }

    MAX_SEQ_LENGTH = 128  # covers your observed sentence lengths
                           # (max ~548 chars in Food, most sentences
                           # much shorter) -- we'll confirm the exact
                           # token length in the tokenization step.


def set_global_seed(seed: int = None):
    """
    Fixes randomness across python, numpy, and torch (if installed),
    so every run of the pipeline is reproducible.
    Call this at the START of every script.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not installed yet -- fine at this stage, will matter
        # from Step 5 (tokenization) onwards.
        pass


def make_project_folders():
    """
    Creates the folder structure defined above if it doesn't exist yet.
    Safe to run multiple times (won't overwrite existing data).
    """
    # On Kaggle, RAW_DATA_DIR is the read-only input dataset -- never
    # try to create/write it. Everything else is writable output.
    writable_folders = [
        Config.CLEAN_DATA_DIR,
        Config.SPLITS_DIR,
        Config.RESULTS_DIR,
        Config.MODELS_DIR,
        Config.LOGS_DIR,
    ]
    for folder in writable_folders:
        os.makedirs(folder, exist_ok=True)

    print("Project folders ready:")
    for folder in writable_folders:
        print(f"  - {folder}")

    if Config.IS_KAGGLE:
        print(f"  - {Config.RAW_DATA_DIR}  (read-only Kaggle input dataset)")
    else:
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        print(f"  - {Config.RAW_DATA_DIR}")
        if Config.IS_COLAB:
            print("    (living inside your mounted Google Drive -- "
                  "persists across Colab sessions)")


# Single shared instance -- other scripts do: `from config import CFG`
CFG = Config()


if __name__ == "__main__":
    # Running this file directly (`python config.py`) just sets up
    # the folders and confirms the seed -- a quick sanity check.
    make_project_folders()
    set_global_seed()
    print(f"\nSeed fixed at: {CFG.SEED}")
    print(f"Domains: {CFG.DOMAINS}")
    print(f"Labels: {CFG.LABELS}")
    print(f"Shot settings: {CFG.SHOT_SETTINGS}")
