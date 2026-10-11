"""
13_significance_tests.py
==========================
COMMITTEE FEEDBACK RESPONSE (Point 7): "Results reported as mean +/-
std across 3 seeds, but no statistical tests (t-tests, ANOVA) to
determine if differences are significant."

WHAT THIS SCRIPT DOES:
Reads results/baseline_results.csv (full fine-tuning) and
results/peft_results.csv (LoRA), which already contain per-seed
accuracy and F1 for every (model, shot) configuration. For every
(model, shot) pair, runs a paired comparison between the 3 baseline
seeds and the 3 LoRA seeds:
  - Paired t-test (scipy.stats.ttest_rel) -- the standard parametric
    choice when the SAME seeds (42, 123, 2024) are used for both
    conditions, so runs are naturally paired by seed.
  - Wilcoxon signed-rank test as a non-parametric cross-check, since
    n=3 is very small and t-test normality assumptions are shaky at
    that sample size -- reporting both is more honest than picking
    whichever one gives a "nicer" p-value.

IMPORTANT CAVEAT (state this explicitly in your thesis too): with only
n=3 seeds per condition, statistical power is inherently very low. A
non-significant result here does NOT prove the two methods perform
identically -- it means the noise level given only 3 samples per group
is too high to distinguish a real difference of this size from chance.
Report p-values honestly alongside this caveat, not as a substitute for
having more seeds if you have time to add them later.

NO GPU / NO TRAINING REQUIRED -- this runs entirely on your existing
results CSVs.

HOW TO RUN (locally or in Colab, any time after 07 and 08 have
produced their results files):
    python 13_significance_tests.py
    (or the Colab %run equivalent)

OUTPUT:
    results/significance_tests.csv   -- one row per (model, shot) pair
    results/significance_tests_summary.md   -- ready-to-paste table +
                                                interpretation notes
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from config import CFG, make_project_folders


def load_results(filename: str) -> pd.DataFrame:
    path = os.path.join(CFG.RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 07_baseline_finetune.py and 08_peft_finetune.py first."
        )
    df = pd.read_csv(path)
    return df[df["scope"] == "overall"].copy()


def get_per_seed_values(df: pd.DataFrame, model_key: str, shot: int, metric: str) -> np.ndarray:
    subset = df[(df["model"] == model_key) & (df["shot"] == shot)].sort_values("seed")
    return subset[metric].to_numpy()


def run_tests_for_metric(baseline_df, peft_df, model_key, shot, metric: str) -> dict:
    baseline_vals = get_per_seed_values(baseline_df, model_key, shot, metric)
    peft_vals = get_per_seed_values(peft_df, model_key, shot, metric)

    if len(baseline_vals) < 2 or len(peft_vals) < 2 or len(baseline_vals) != len(peft_vals):
        return {
            "metric": metric, "n_seeds": min(len(baseline_vals), len(peft_vals)),
            "baseline_mean": np.mean(baseline_vals) if len(baseline_vals) else np.nan,
            "peft_mean": np.mean(peft_vals) if len(peft_vals) else np.nan,
            "diff": np.nan, "ttest_p": np.nan, "wilcoxon_p": np.nan,
            "note": "insufficient or mismatched paired samples -- test skipped",
        }

    diff = peft_vals.mean() - baseline_vals.mean()

    # Paired t-test: valid pairing since both conditions used the SAME
    # 3 seeds (42, 123, 2024), so baseline_vals[i] and peft_vals[i]
    # correspond to the same seed.
    t_stat, t_p = stats.ttest_rel(peft_vals, baseline_vals)

    # Wilcoxon signed-rank as a non-parametric cross-check. With n=3,
    # Wilcoxon has very limited resolution (few possible p-values), but
    # it doesn't assume normality, which the t-test does -- reporting
    # both is more transparent than relying on the t-test alone at this
    # sample size.
    try:
        w_stat, w_p = stats.wilcoxon(peft_vals, baseline_vals)
    except ValueError:
        # Wilcoxon fails if all differences are zero or n is too small
        # for the exact distribution -- report as NaN with a note rather
        # than crashing the whole script.
        w_p = np.nan

    return {
        "metric": metric, "n_seeds": len(baseline_vals),
        "baseline_mean": baseline_vals.mean(), "peft_mean": peft_vals.mean(),
        "diff": diff, "ttest_p": t_p, "wilcoxon_p": w_p,
        "note": "significant at alpha=0.05" if (not np.isnan(t_p) and t_p < 0.05) else "not significant at alpha=0.05",
    }


def main():
    make_project_folders()

    print("=" * 60)
    print("SIGNIFICANCE TESTING (Point 7): Full Fine-Tuning vs. LoRA")
    print("=" * 60)

    baseline_df = load_results("baseline_results.csv")
    peft_df = load_results("peft_results.csv")

    rows = []
    for model_key in CFG.MODEL_CHECKPOINTS.keys():
        for shot in [s for s in CFG.SHOT_SETTINGS if s > 0]:
            for metric in ["accuracy", "f1"]:
                result = run_tests_for_metric(baseline_df, peft_df, model_key, shot, metric)
                result["model"] = model_key
                result["shot"] = shot
                rows.append(result)

                if not np.isnan(result["ttest_p"]):
                    print(f"[{model_key}, shot={shot}, {metric}] "
                          f"baseline={result['baseline_mean']:.4f} vs LoRA={result['peft_mean']:.4f} "
                          f"(diff={result['diff']:+.4f}) -- "
                          f"paired t-test p={result['ttest_p']:.4f}, "
                          f"Wilcoxon p={result['wilcoxon_p']:.4f} -- {result['note']}")
                else:
                    print(f"[{model_key}, shot={shot}, {metric}] {result['note']}")

    results_df = pd.DataFrame(rows)[
        ["model", "shot", "metric", "n_seeds", "baseline_mean", "peft_mean", "diff",
         "ttest_p", "wilcoxon_p", "note"]
    ]
    csv_path = os.path.join(CFG.RESULTS_DIR, "significance_tests.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # Markdown summary, ready to paste into Chapter 4
    # ------------------------------------------------------------
    md_path = os.path.join(CFG.RESULTS_DIR, "significance_tests_summary.md")
    lines = [
        "# Statistical Significance: Full Fine-Tuning vs. LoRA\n",
        "Paired comparison across the same 3 seeds (42, 123, 2024) used for both "
        "conditions. Paired t-test is the primary test; Wilcoxon signed-rank is "
        "reported as a non-parametric cross-check given the small sample size.\n",
        "**Caveat: with only n=3 seeds per condition, statistical power is low. "
        "A 'not significant' result here means the difference could not be "
        "distinguished from noise at this sample size -- it does not prove the "
        "two methods are equivalent.**\n",
        "| Model | Shot | Metric | Baseline Mean | LoRA Mean | Diff | t-test p | Wilcoxon p | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in results_df.iterrows():
        if pd.isna(row["ttest_p"]):
            lines.append(f"| {row['model']} | {row['shot']} | {row['metric']} | - | - | - | - | - | {row['note']} |")
        else:
            lines.append(
                f"| {row['model']} | {row['shot']} | {row['metric']} | "
                f"{row['baseline_mean']:.4f} | {row['peft_mean']:.4f} | {row['diff']:+.4f} | "
                f"{row['ttest_p']:.4f} | {row['wilcoxon_p']:.4f} | {row['note']} |"
            )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n_significant = (results_df["ttest_p"] < 0.05).sum()
    n_tested = results_df["ttest_p"].notna().sum()
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_significant} / {n_tested} tested (model, shot, metric) "
          f"combinations show a statistically significant difference (p < 0.05) "
          f"between full fine-tuning and LoRA.")
    print("=" * 60)
    print(f"\nFull results saved to: {csv_path}")
    print(f"Markdown summary saved to: {md_path}")
    print("\nDONE. Paste the markdown table into Chapter 4, and explicitly state")
    print("the n=3 statistical power caveat alongside it.")


if __name__ == "__main__":
    main()
