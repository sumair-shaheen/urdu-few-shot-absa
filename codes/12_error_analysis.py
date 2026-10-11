"""
12_error_analysis.py
======================
COMMITTEE FEEDBACK RESPONSE (Point 10): "No qualitative analysis of
what kinds of errors the models make."

WHAT THIS SCRIPT DOES:
Reads the per-example prediction files written by 07_baseline_finetune.py
(v2), 08_peft_finetune.py (v2), and 07c_mt5_incontext_prompting.py, and
produces, for each requested (model, method) configuration:
  1. A confusion matrix (raw counts + row-normalized), plotted and
     saved as a .png, aggregated across all 3 seeds at the 20-shot
     setting (your best-performing, most defensible configuration).
  2. A CSV of every misclassified test example for that configuration.
  3. A curated sample of misclassified examples, auto-tagged with
     simple heuristic flags (negation present, code-mixed with Latin
     script, sentence length) to help you quickly find qualitatively
     interesting cases (negation errors, sarcasm, code-mixing) to
     discuss by hand in Chapter 4/5 -- the heuristics are a STARTING
     POINT for you to manually curate from, not a replacement for
     actually reading the examples yourself.

PREREQUISITE:
This script requires results/predictions_baseline.csv and/or
results/predictions_peft.csv to exist, which only happens if you have
run the v2 versions of 07_baseline_finetune.py / 08_peft_finetune.py
(the ones that save per-example predictions). If you only have the
original v1 scripts' output (metrics only, no predictions), this
script will tell you which files are missing and exit cleanly instead
of crashing.

HOW TO RUN (after 07/08/07c have produced their prediction CSVs):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/12_error_analysis.py"

OUTPUT (all under results/error_analysis/):
    confusion_matrix_<model>_<method>.png
    misclassified_<model>_<method>.csv
    error_analysis_summary.md   -- a ready-to-paste summary table

WHAT TO DO WITH THE OUTPUT:
Open misclassified_*.csv for your best model (XLM-R, 20-shot) and hand
-pick 5-10 genuinely interesting errors -- ideally ones involving
negation, sarcasm, or code-mixed Urdu/English -- to discuss explicitly
in your results chapter. Do not just describe the confusion matrix;
show 2-3 real Urdu sentences and explain in one sentence each why you
think the model got it wrong.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from config import CFG, make_project_folders


ERROR_ANALYSIS_DIR = os.path.join(CFG.RESULTS_DIR, "error_analysis")

# Configurations to analyze: (label_for_filenames, predictions_csv_filename,
# model_key_to_filter, method_name). shot is fixed at 20 -- your best,
# most-defensible setting, and the one the committee is most likely to
# ask about. Edit this list if you want other shot settings analyzed too.
CONFIGS_TO_ANALYZE = [
    ("mbert_full",       "predictions_baseline.csv",       "mbert", "Full Fine-Tune"),
    ("mbert_lora",       "predictions_peft.csv",           "mbert", "LoRA"),
    ("xlmr_full",        "predictions_baseline.csv",       "xlmr",  "Full Fine-Tune"),
    ("xlmr_lora",        "predictions_peft.csv",           "xlmr",  "LoRA"),
    ("mt5_full",         "predictions_baseline.csv",       "mt5",   "Full Fine-Tune"),
    ("mt5_lora",         "predictions_peft.csv",           "mt5",   "LoRA"),
    ("mt5_incontext",    "predictions_mt5_incontext.csv",  "mt5_incontext", "In-Context Prompting"),
]
ANALYSIS_SHOT = 20

# Simple heuristics to flag potentially interesting misclassifications
# for manual review -- these are NOT a formal linguistic error taxonomy,
# just a fast way to surface candidates worth reading by hand.
NEGATION_MARKERS = ["نہیں", "نہ ", "کوئی نہیں", "ناں"]


def has_negation(text: str) -> bool:
    return any(marker in str(text) for marker in NEGATION_MARKERS)


def has_code_mixing(text: str) -> bool:
    """Flags sentences containing Latin-script (English/Roman Urdu) characters
    alongside Urdu script -- a simple proxy for code-mixed content."""
    return bool(re.search(r"[A-Za-z]{2,}", str(text)))


def load_predictions_file(filename: str) -> pd.DataFrame:
    path = os.path.join(CFG.RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, matrix, fmt, subtitle in [
        (axes[0], cm, "d", "Raw counts"),
        (axes[1], cm_normalized, ".2f", "Row-normalized (recall per true class)"),
    ]:
        im = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(subtitle, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                text_str = f"{value:d}" if fmt == "d" else f"{value:.2f}"
                text_color = "white" if value > matrix.max() * 0.5 else "black"
                ax.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cm, cm_normalized


def analyze_one_config(label, pred_filename, model_key, method_name, shot):
    df = load_predictions_file(pred_filename)
    if df is None:
        print(f"[{label}] SKIPPED -- {pred_filename} not found. "
              f"Run the v2 training script that produces it first.")
        return None

    subset = df[(df["model"] == model_key) & (df["shot"] == shot)].copy()
    if len(subset) == 0:
        print(f"[{label}] SKIPPED -- no rows found for model={model_key}, shot={shot} "
              f"in {pred_filename}.")
        return None

    print(f"[{label}] Found {len(subset)} predictions across "
          f"{subset['seed'].nunique()} seed(s). Aggregating for confusion matrix.")

    y_true = subset["true_label"].tolist()
    y_pred = subset["predicted_label"].tolist()

    cm_path = os.path.join(ERROR_ANALYSIS_DIR, f"confusion_matrix_{label}.png")
    cm, cm_norm = plot_confusion_matrix(
        y_true, y_pred, CFG.LABELS,
        title=f"{method_name} -- {model_key.upper()} ({shot}-shot, all seeds pooled)",
        save_path=cm_path,
    )

    # Misclassified examples, with heuristic error-type flags
    misclassified = subset[subset["true_label"] != subset["predicted_label"]].copy()
    misclassified["has_negation"] = misclassified["Sentence"].apply(has_negation)
    misclassified["has_code_mixing"] = misclassified["Sentence"].apply(has_code_mixing)
    misclassified["sentence_length_chars"] = misclassified["Sentence"].str.len()
    misclassified = misclassified.sort_values(["has_negation", "has_code_mixing"], ascending=False)

    mis_path = os.path.join(ERROR_ANALYSIS_DIR, f"misclassified_{label}.csv")
    misclassified.to_csv(mis_path, index=False, encoding="utf-8-sig")

    accuracy = (subset["true_label"] == subset["predicted_label"]).mean()
    n_negation_errors = misclassified["has_negation"].sum()
    n_codemix_errors = misclassified["has_code_mixing"].sum()

    print(f"  Accuracy (pooled across seeds): {accuracy:.4f}")
    print(f"  Total misclassified: {len(misclassified)} / {len(subset)}")
    print(f"  ...of which flagged 'possible negation involved': {n_negation_errors}")
    print(f"  ...of which flagged 'possible code-mixing involved': {n_codemix_errors}")
    print(f"  Confusion matrix saved to: {cm_path}")
    print(f"  Misclassified examples saved to: {mis_path}")

    return {
        "label": label, "model": model_key, "method": method_name, "shot": shot,
        "n_examples": len(subset), "accuracy": accuracy,
        "n_misclassified": len(misclassified),
        "n_negation_flagged": int(n_negation_errors),
        "n_codemix_flagged": int(n_codemix_errors),
    }


def write_summary_markdown(summary_rows):
    path = os.path.join(ERROR_ANALYSIS_DIR, "error_analysis_summary.md")
    lines = [
        "# Error Analysis Summary\n",
        f"All results pooled across 3 seeds at {ANALYSIS_SHOT}-shot.\n",
        "| Model | Method | N examples | Accuracy | N misclassified | Negation-flagged | Code-mix-flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['n_examples']} | "
            f"{row['accuracy']:.4f} | {row['n_misclassified']} | "
            f"{row['n_negation_flagged']} | {row['n_codemix_flagged']} |"
        )
    lines.append(
        "\n*Negation/code-mixing flags are simple keyword heuristics meant to "
        "surface candidate examples for manual reading -- they are not a "
        "validated linguistic error taxonomy. Always read the actual sentence "
        "before citing it as an example of a specific error type.*"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary table written to: {path}")


def main():
    make_project_folders()
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    print("=" * 60)
    print(f"ERROR ANALYSIS (Point 10) -- confusion matrices + misclassified examples")
    print(f"Analyzing shot={ANALYSIS_SHOT} for every configuration below")
    print("=" * 60)

    summary_rows = []
    for label, pred_filename, model_key, method_name in CONFIGS_TO_ANALYZE:
        result = analyze_one_config(label, pred_filename, model_key, method_name, ANALYSIS_SHOT)
        if result is not None:
            summary_rows.append(result)
        print()

    if len(summary_rows) == 0:
        print("No configurations could be analyzed -- none of the required prediction "
              "files were found. Run the v2 training scripts first.")
        return

    write_summary_markdown(summary_rows)
    print("\nDONE. Open results/error_analysis/misclassified_xlmr_lora.csv (or "
          "whichever config is your best model) and hand-pick 5-10 genuinely "
          "interesting misclassifications -- ones involving negation, sarcasm, "
          "or code-mixing -- to discuss explicitly, with the actual Urdu "
          "sentence quoted, in your results chapter.")


if __name__ == "__main__":
    main()
