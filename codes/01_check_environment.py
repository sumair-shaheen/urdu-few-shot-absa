"""
check_environment.py
=====================
STEP 1 of the implementation roadmap: Environment & Project Setup.

WHAT THIS SCRIPT DOES:
1. Confirms your Python version is compatible.
2. Checks whether a GPU (CUDA) is available -- this matters A LOT,
   since fine-tuning transformers on CPU only is extremely slow
   (could turn a 10-minute training run into several hours).
3. Prints the installed versions of the key libraries, so we can
   catch version mismatches EARLY, before they cause confusing
   errors deep into training.
4. Creates the project folder structure (data/raw, data/clean, etc.)
   via config.py, so everything has a home from the start.

HOW TO RUN:
    python check_environment.py

WHAT TO DO WITH THE OUTPUT:
Copy the full printed output and send it back to me. Based on what
it shows (especially GPU availability), I'll decide how to write the
training code in later steps (e.g. batch size, whether to recommend
Google Colab if no GPU is found).
"""

import sys
import platform


def check_python_version():
    print("=" * 60)
    print("1. PYTHON VERSION")
    print("=" * 60)
    print(f"Python version : {platform.python_version()}")
    print(f"Platform       : {platform.platform()}")
    major, minor = sys.version_info[0], sys.version_info[1]
    if (major, minor) < (3, 9):
        print("⚠️  WARNING: Python 3.9+ is recommended for current "
              "transformers/peft versions. Consider upgrading.")
    else:
        print("✅ Python version looks fine.")
    print()


def check_gpu():
    print("=" * 60)
    print("2. GPU / CUDA AVAILABILITY")
    print("=" * 60)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"Torch version   : {torch.__version__}")
        print(f"CUDA available  : {cuda_available}")
        if cuda_available:
            print(f"GPU device name : {torch.cuda.get_device_name(0)}")
            print(f"GPU count       : {torch.cuda.device_count()}")
            print("✅ You have GPU acceleration. Training will be fast.")
        else:
            print("⚠️  No GPU detected. Fine-tuning transformers on CPU "
                  "is possible but SLOW, especially across multiple "
                  "models x shot-settings x PEFT/no-PEFT combinations.")
            print("    Recommendation: if you don't have a local GPU, "
                  "use Google Colab (free GPU tier) for the training "
                  "steps (6, 7).")
    except ImportError:
        print("❌ PyTorch is not installed yet. Run:")
        print("   pip install -r requirements.txt")
    print()


def check_key_packages():
    print("=" * 60)
    print("3. KEY PACKAGE VERSIONS")
    print("=" * 60)
    packages_to_check = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "evaluate",
        "pandas",
        "sklearn",   # import name differs from pip name (scikit-learn)
        "numpy",
    ]
    for pkg_name in packages_to_check:
        try:
            module = __import__(pkg_name)
            version = getattr(module, "__version__", "unknown version")
            print(f"✅ {pkg_name:<15} -> {version}")
        except ImportError:
            print(f"❌ {pkg_name:<15} -> NOT INSTALLED "
                  f"(run: pip install -r requirements.txt)")
    print()


def setup_project_folders():
    print("=" * 60)
    print("4. PROJECT FOLDER STRUCTURE")
    print("=" * 60)
    try:
        from config import make_project_folders, set_global_seed, CFG
        make_project_folders()
        set_global_seed()
        print(f"\nRandom seed fixed at: {CFG.SEED}")
        print("✅ Folders created / verified, seed fixed.")
    except ImportError:
        print("❌ Could not import config.py -- make sure config.py "
              "is in the SAME folder as this script.")
    print()


if __name__ == "__main__":
    print("\n### ENVIRONMENT CHECK -- Urdu Few-Shot ABSA Project ###\n")
    check_python_version()
    check_gpu()
    check_key_packages()
    setup_project_folders()
    print("=" * 60)
    print("DONE. Copy everything above and send it back so we can "
          "confirm the environment before moving to Step 2 (data "
          "loading & cleaning).")
    print("=" * 60)
