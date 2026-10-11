"""
09_evaluation_harness.py
==========================
STEP 8 (of our roadmap): Evaluation Harness.

WHAT THIS SCRIPT DOES:
Pulls together results/baseline_results.csv (Step 6) and
results/peft_results.csv (Step 7) into the final comparison tables
and charts you'll actually paste into your thesis's results chapter.

OUTPUTS (all saved into results/final_evaluation/):
  1. combined_results.csv        -- every run from both files, tagged
                                     with a "method" column (baseline/peft)
  2. summary_overall.csv         -- mean +/- std accuracy/F1 per
                                     (model, method, shot), overall scope
  3. summary_per_domain.csv      -- same, broken down by domain too
  4. learning_curves.png         -- accuracy vs. shot count, one line
                                     per (model, method) -- the main
                                     "does more data help" figure
  5. baseline_vs_peft.png        -- grouped bar chart comparing
                                     baseline vs. PEFT at each shot
                                     setting, one panel per model
  6. summary_table.md            -- a clean Markdown table (mean +/- std)
                                     ready to paste directly into your
                                     thesis document

HOW TO RUN (in Colab, AFTER both 07_baseline_finetune.py and
08_peft_finetune.py have been run):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/09_evaluation_harness.py"

WHAT TO DO WITH THE OUTPUT:
Everything in results/final_evaluation/ is meant to go straight into
your thesis. Send me the printed summary so we can sanity-check it
together before you paste anything in.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from config import CFG, set_global_seed, make_project_folders


def get_output_dir():
    out_dir = os.path.join(CFG.RESULTS_DIR, "final_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_combined_results() -> pd.DataFrame:
    baseline_path = os.path.join(CFG.RESULTS_DIR, "baseline_results.csv")
    peft_path = os.path.join(CFG.RESULTS_DIR, "peft_results.csv")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"baseline_results.csv not found at {baseline_path} -- run 07_baseline_finetune.py first.")
    if not os.path.exists(peft_path):
        raise FileNotFoundError(f"peft_results.csv not found at {peft_path} -- run 08_peft_finetune.py first.")

    baseline_df = pd.read_csv(baseline_path)
    peft_df = pd.read_csv(peft_path)

    baseline_df["method"] = "baseline"
    peft_df["method"] = "peft"

    combined = pd.concat([baseline_df, peft_df], ignore_index=True)
    return combined


def build_summary_tables(combined: pd.DataFrame, out_dir: str):
    overall = combined[combined["scope"] == "overall"]

    # Mean +/- std per (model, method, shot) -- overall scope
    summary_overall = overall.groupby(["model", "method", "shot"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n_runs=("accuracy", "count"),
    ).reset_index()
    summary_overall.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60)
    print("SUMMARY -- OVERALL SCOPE (mean +/- std across seeds)")
    print("=" * 60)
    print(summary_overall.to_string(index=False))

    # Per-domain breakdown
    per_domain = combined[combined["scope"] != "overall"].rename(columns={"scope": "domain"})
    summary_domain = per_domain.groupby(["model", "method", "shot", "domain"]).agg(
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
    ).reset_index()
    summary_domain.to_csv(os.path.join(out_dir, "summary_per_domain.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 60)
    print("SUMMARY -- PER-DOMAIN BREAKDOWN (mean across seeds)")
    print("=" * 60)
    print(summary_domain.to_string(index=False))

    return summary_overall, summary_domain


def plot_learning_curves(summary_overall: pd.DataFrame, out_dir: str):
    """One figure, one line per (model, method), accuracy vs shot count."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"mbert": "tab:blue", "xlmr": "tab:orange", "mt5": "tab:green"}
    linestyles = {"baseline": "-", "peft": "--"}
    markers = {"baseline": "o", "peft": "s"}

    for model in summary_overall["model"].unique():
        for method in summary_overall["method"].unique():
            subset = summary_overall[(summary_overall["model"] == model) & (summary_overall["method"] == method)]
            subset = subset.sort_values("shot")
            if len(subset) == 0:
                continue
            ax.errorbar(
                subset["shot"], subset["accuracy_mean"], yerr=subset["accuracy_std"].fillna(0),
                label=f"{model} ({method})",
                color=colors.get(model, "gray"),
                linestyle=linestyles.get(method, "-"),
                marker=markers.get(method, "o"),
                capsize=4,
            )

    ax.axhline(y=1/3, color="red", linestyle=":", alpha=0.5, label="Random chance (33.3%)")
    ax.set_xlabel("Number of shots (training examples per class)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Few-Shot Learning Curves: Accuracy vs. Number of Shots")
    ax.set_xticks(sorted(summary_overall["shot"].unique()))
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart: {save_path}")


def plot_baseline_vs_peft(summary_overall: pd.DataFrame, out_dir: str):
    """Grouped bar chart, one panel per model, baseline vs PEFT at each shot setting."""
    models = summary_overall["model"].unique()
    shots = sorted([s for s in summary_overall["shot"].unique() if s > 0])  # PEFT has no shot=0

    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    bar_width = 0.35
    x = range(len(shots))

    for ax, model in zip(axes, models):
        baseline_vals = []
        peft_vals = []
        for shot in shots:
            b = summary_overall[(summary_overall["model"] == model) & (summary_overall["method"] == "baseline") & (summary_overall["shot"] == shot)]
            p = summary_overall[(summary_overall["model"] == model) & (summary_overall["method"] == "peft") & (summary_overall["shot"] == shot)]
            baseline_vals.append(b["accuracy_mean"].values[0] if len(b) else 0)
            peft_vals.append(p["accuracy_mean"].values[0] if len(p) else 0)

        ax.bar([i - bar_width/2 for i in x], baseline_vals, width=bar_width, label="Baseline (full fine-tune)")
        ax.bar([i + bar_width/2 for i in x], peft_vals, width=bar_width, label="PEFT (LoRA)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{s}-shot" for s in shots])
        ax.set_title(model)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "baseline_vs_peft.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {save_path}")


def export_markdown_table(summary_overall: pd.DataFrame, out_dir: str):
    """Builds a clean Markdown table, ready to paste directly into the thesis document."""
    lines = ["| Model | Method | Shot | Accuracy (mean ± std) | F1 (mean ± std) |",
             "|---|---|---|---|---|"]

    summary_sorted = summary_overall.sort_values(["model", "method", "shot"])
    for _, row in summary_sorted.iterrows():
        acc_std = row["accuracy_std"] if pd.notna(row["accuracy_std"]) else 0
        f1_std = row["f1_std"] if pd.notna(row["f1_std"]) else 0
        shot_label = "zero-shot" if row["shot"] == 0 else f"{int(row['shot'])}-shot"
        lines.append(
            f"| {row['model']} | {row['method']} | {shot_label} | "
            f"{row['accuracy_mean']*100:.1f}% ± {acc_std*100:.1f}% | "
            f"{row['f1_mean']:.3f} ± {f1_std:.3f} |"
        )

    save_path = os.path.join(out_dir, "summary_table.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved Markdown table: {save_path}")
    print("\n" + "\n".join(lines))


def main():
    set_global_seed()
    make_project_folders()

    out_dir = get_output_dir()

    print("=" * 60)
    print("STEP 8: EVALUATION HARNESS")
    print("=" * 60)

    combined = load_combined_results()
    combined_path = os.path.join(out_dir, "combined_results.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved combined results ({len(combined)} rows) to: {combined_path}")

    summary_overall, summary_domain = build_summary_tables(combined, out_dir)
    plot_learning_curves(summary_overall, out_dir)
    plot_baseline_vs_peft(summary_overall, out_dir)
    export_markdown_table(summary_overall, out_dir)

    print("\nDONE. Everything in results/final_evaluation/ is ready to")
    print("paste into your thesis. Send me the printed summary so we can")
    print("sanity-check it together first.")


if __name__ == "__main__":
    main()
