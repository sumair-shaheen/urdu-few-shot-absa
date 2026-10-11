"""
07c_mt5_incontext_prompting.py
================================
COMMITTEE FEEDBACK RESPONSE (Point 5): "mT5 failure not explored
deeply... no alternative prompting strategies... comparison is unfair."

WHAT THIS SCRIPT DOES:
07_baseline_finetune.py already tried two things with mt5-small:
  (a) zero-shot direct generation -- collapsed to sentinel-token output
      (see results/mt5_diagnostic/ from 07b_mt5_diagnostic.py)
  (b) gradient-based fine-tuning on the 5/10/20-shot pools

This script tries a THIRD, fundamentally different strategy that
requires NO fine-tuning at all: IN-CONTEXT FEW-SHOT PROMPTING. Instead
of updating any model weights, we prepend a small number of labeled
demonstration examples directly into the prompt text, then ask the
pretrained mt5-small checkpoint to generate a label for the target
example, conditioned on those demonstrations. This is the standard
"in-context learning" strategy popularized by GPT-3-style models, and
it is a genuinely different mechanism from both (a) and (b) above --
if mt5-small still fails here, that is much stronger evidence that the
problem is the checkpoint itself (too small / not instruction-tuned),
not our specific fine-tuning recipe.

DESIGN DECISION -- WHY NOT ALL 45/90/180 SHOT-POOL EXAMPLES AS DEMOS:
Feeding the ENTIRE shot pool as in-context demonstrations is not
feasible: mt5-small's practical context window and our MAX_SEQ_LENGTH
budget cannot fit 45-180 Urdu sentences into one prompt, and doing so
would also not match how in-context learning is normally evaluated in
the literature (a handful of demonstrations per prompt, not the entire
training set). Instead, for each shot setting K, we sample up to
MAX_DEMOS_PER_CLASS demonstration examples PER CLASS (balanced across
negative/neutral/positive, pooled across all 3 domains) from that
seed's existing shot pool -- the same underlying data used for
fine-tuning elsewhere, just used differently here. This keeps the
prompt short enough to fit reliably while still giving the model
genuine labeled context for all 3 possible answers. This design choice
is stated explicitly here (and should be restated in the thesis) so it
is transparent and defensible, not silently arbitrary.

HOW TO RUN (in Colab, after 06_tokenize.py; does NOT require
07_baseline_finetune.py to have finished first, since this script does
not depend on any of its outputs -- only on the shot pools from
05_split_and_shots.py):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/07c_mt5_incontext_prompting.py"

OUTPUT:
    results/mt5_incontext_results.csv       -- accuracy/F1 per (shot, seed)
    results/predictions_mt5_incontext.csv   -- per-example predictions,
                                                 same schema as the other
                                                 predictions files, so
                                                 12_error_analysis.py can
                                                 include this condition too.

WHAT TO DO WITH THE OUTPUT:
Compare mt5_incontext_results.csv against the mt5 rows in
baseline_results.csv (fine-tuned) and the zero-shot mt5 diagnostic
results. Report all three mt5 strategies side by side in Chapter 4 --
this directly answers the committee's request for "alternative
prompting strategies" and lets you say explicitly whether in-context
prompting closes any of the gap, rather than leaving mt5's failure
unaddressed.
"""

import os
import gc
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import CFG, set_global_seed, make_project_folders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cap on demonstrations per class, per the design decision explained in
# the module docstring above. 3 per class x 3 classes = 9 demonstrations
# total per prompt, which comfortably fits mt5-small's practical context
# budget alongside the target sentence itself.
MAX_DEMOS_PER_CLASS = 3

RESULT_COLUMNS = ["model", "shot", "seed", "scope", "accuracy", "precision", "recall", "f1",
                   "num_demos_used", "train_seconds", "peak_gpu_memory_mb"]
PREDICTION_COLUMNS = ["model", "shot", "seed", "Sentence", "Aspect", "domain", "true_label", "predicted_label"]


# ============================================================
# 1. METRICS (identical to 07/08, kept consistent for fair comparison)
# ============================================================

def compute_all_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(test_df: pd.DataFrame, predictions: list) -> dict:
    y_true = test_df["Sentiment"].tolist()
    results = {"overall": compute_all_metrics(y_true, predictions)}
    test_df = test_df.copy()
    test_df["_prediction"] = predictions
    for domain in CFG.DOMAINS:
        mask = test_df["domain"] == domain
        if mask.sum() == 0:
            continue
        results[domain] = compute_all_metrics(
            test_df.loc[mask, "Sentiment"].tolist(), test_df.loc[mask, "_prediction"].tolist()
        )
    return results


# ============================================================
# 2. RESULTS / PREDICTIONS LOGGING (same resumable pattern as 07/08)
# ============================================================

def get_results_path():
    return os.path.join(CFG.RESULTS_DIR, "mt5_incontext_results.csv")


def get_predictions_path():
    return os.path.join(CFG.RESULTS_DIR, "predictions_mt5_incontext.csv")


def load_existing_results() -> pd.DataFrame:
    path = get_results_path()
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def already_done(existing_df, shot, seed) -> bool:
    if len(existing_df) == 0:
        return False
    return len(existing_df[(existing_df["shot"] == shot) & (existing_df["seed"] == seed)]) > 0


def append_results(shot, seed, metrics_dict, num_demos_used, train_seconds, peak_gpu_memory_mb):
    rows = []
    for scope, metrics in metrics_dict.items():
        rows.append({
            "model": "mt5_incontext", "shot": shot, "seed": seed, "scope": scope,
            "accuracy": metrics["accuracy"], "precision": metrics["precision"],
            "recall": metrics["recall"], "f1": metrics["f1"],
            "num_demos_used": num_demos_used, "train_seconds": train_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        })
    new_df = pd.DataFrame(rows)
    combined = pd.concat([load_existing_results(), new_df], ignore_index=True)
    combined.to_csv(get_results_path(), index=False, encoding="utf-8-sig")
    print(f"  Appended in-context results for shot={shot}, seed={seed} "
          f"(num_demos_used={num_demos_used}, inference_time={train_seconds:.1f}s)")


def append_predictions(shot, seed, test_df: pd.DataFrame, predictions: list):
    pred_df = test_df[["Sentence", "Aspect", "domain", "Sentiment"]].copy()
    pred_df = pred_df.rename(columns={"Sentiment": "true_label"})
    pred_df["predicted_label"] = predictions
    pred_df["model"] = "mt5_incontext"
    pred_df["shot"] = shot
    pred_df["seed"] = seed
    pred_df = pred_df[PREDICTION_COLUMNS]

    path = get_predictions_path()
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[~((existing["shot"] == shot) & (existing["seed"] == seed))]
        combined = pd.concat([existing, pred_df], ignore_index=True)
    else:
        combined = pred_df
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ============================================================
# 3. DEMONSTRATION SELECTION AND PROMPT CONSTRUCTION
# ============================================================

def select_balanced_demonstrations(shot_pool_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Selects up to MAX_DEMOS_PER_CLASS examples PER CLASS from the shot
    pool (pooled across domains), shuffled deterministically by `seed`
    so the demo set is reproducible. See module docstring for why we
    cap demo count rather than using the entire shot pool.
    """
    rng = np.random.RandomState(seed)
    demo_rows = []
    for label in CFG.LABELS:
        class_rows = shot_pool_df[shot_pool_df["Sentiment"] == label]
        if len(class_rows) == 0:
            continue
        n_take = min(MAX_DEMOS_PER_CLASS, len(class_rows))
        chosen_idx = rng.choice(class_rows.index, size=n_take, replace=False)
        demo_rows.append(class_rows.loc[chosen_idx])
    demos = pd.concat(demo_rows, ignore_index=True)
    # Shuffle the combined demo set so labels aren't grouped in a
    # predictable block order (which could itself bias generation).
    demos = demos.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return demos


def build_incontext_prompt(demos: pd.DataFrame, sentence: str, aspect: str) -> str:
    """
    Builds a single prompt string containing labeled demonstrations
    followed by the target example, e.g.:

        sentiment classification examples:
        sentence: <demo 1 sentence> aspect: <demo 1 aspect> sentiment: positive
        sentence: <demo 2 sentence> aspect: <demo 2 aspect> sentiment: negative
        ...
        sentence: <target sentence> aspect: <target aspect> sentiment:
    """
    lines = ["sentiment classification examples:"]
    for _, row in demos.iterrows():
        lines.append(f"sentence: {row['Sentence']} aspect: {row['Aspect']} sentiment: {row['Sentiment']}")
    lines.append(f"sentence: {sentence} aspect: {aspect} sentiment:")
    return "\n".join(lines)


def parse_mt5_prediction(decoded_text: str) -> str:
    decoded_lower = decoded_text.strip().lower()
    for label in CFG.LABELS:
        if label in decoded_lower:
            return label
    return "neutral"


# ============================================================
# 4. IN-CONTEXT PREDICTION (NO TRAINING -- weights are never updated)
# ============================================================

def predict_incontext(model, tokenizer, demos: pd.DataFrame, test_df: pd.DataFrame,
                       max_length: int = 512) -> list:
    model.eval()
    model.to(DEVICE)
    predictions = []
    batch_size = 16  # smaller batch than 07/08 since in-context prompts are much longer

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i + batch_size]
            prompts = [
                build_incontext_prompt(demos, s, a)
                for s, a in zip(batch["Sentence"], batch["Aspect"])
            ]
            encoded = tokenizer(
                prompts, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            ).to(DEVICE)
            generated_ids = model.generate(**encoded, max_new_tokens=8)
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend([parse_mt5_prediction(d) for d in decoded])

    return predictions


# ============================================================
# 5. MAIN EXPERIMENT LOOP
# ============================================================

def load_data():
    return pd.read_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"))


def load_shot_pool(seed: int, shot: int) -> pd.DataFrame:
    path = os.path.join(CFG.SPLITS_DIR, "fewshot", f"seed{seed}", f"shot{shot}.csv")
    return pd.read_csv(path)


def main():
    set_global_seed()
    make_project_folders()

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no GPU detected -- inference will be slow (though still "
              "far faster than training, since no gradients are computed).")

    test_df = load_data()
    checkpoint = CFG.MODEL_CHECKPOINTS["mt5"]

    print("\n" + "=" * 60)
    print("POINT 5 RESPONSE: mT5 IN-CONTEXT FEW-SHOT PROMPTING (no fine-tuning)")
    print(f"Max demonstrations per class: {MAX_DEMOS_PER_CLASS}")
    print("=" * 60)

    print("\nLoading pretrained mt5-small checkpoint (weights are NEVER "
          "updated in this script -- pure in-context prompting)...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    for shot in [s for s in CFG.SHOT_SETTINGS if s > 0]:
        for seed in CFG.SEEDS:
            existing_results = load_existing_results()
            if already_done(existing_results, shot, seed):
                print(f"shot={shot} seed={seed} already done -- skipping.")
                continue

            print(f"\nBuilding demonstrations from shot={shot} seed={seed} pool...")
            shot_pool = load_shot_pool(seed, shot)
            demos = select_balanced_demonstrations(shot_pool, seed)
            print(f"  Selected {len(demos)} demonstrations "
                  f"({demos['Sentiment'].value_counts().to_dict()})")

            reset_gpu_memory_stats()
            start = time.time()
            preds = predict_incontext(model, tokenizer, demos, test_df)
            elapsed = time.time() - start
            peak_mem = get_peak_gpu_memory_mb()

            metrics = evaluate_predictions(test_df, preds)
            append_results(shot, seed, metrics, len(demos), elapsed, peak_mem)
            append_predictions(shot, seed, test_df, preds)
            print(f"  [mt5_incontext] shot={shot} seed={seed} overall accuracy: "
                  f"{metrics['overall']['accuracy']:.4f} (inference time: {elapsed:.1f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final summary + direct comparison against fine-tuned mT5
    # ------------------------------------------------------------
    final_df = load_existing_results()
    print("\n" + "=" * 60)
    print("FINAL IN-CONTEXT RESULTS SUMMARY (overall scope, mean +/- std across seeds)")
    print("=" * 60)
    overall_df = final_df[final_df["scope"] == "overall"]
    summary = overall_df.groupby("shot")[["accuracy", "f1"]].agg(["mean", "std"])
    print(summary)

    baseline_path = os.path.join(CFG.RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        mt5_finetuned = baseline_df[(baseline_df["model"] == "mt5") & (baseline_df["scope"] == "overall")]
        if len(mt5_finetuned) > 0:
            print("\nFor direct comparison -- fine-tuned mT5 (from 07_baseline_finetune.py):")
            print(mt5_finetuned.groupby("shot")[["accuracy", "f1"]].agg(["mean", "std"]))

    print(f"\nFull results saved to: {get_results_path()}")
    print(f"Per-example predictions saved to: {get_predictions_path()}")
    print("\nDONE. Report this alongside the fine-tuned mT5 numbers and the")
    print("zero-shot diagnostic in Chapter 4 as the three mT5 strategies tried.")


if __name__ == "__main__":
    main()
