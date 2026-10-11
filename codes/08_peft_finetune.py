"""
08_peft_finetune.py  (v2)
====================
STEP 7 (of our roadmap): PEFT / LoRA Fine-Tuning.

CHANGELOG (v2 -- added in response to committee feedback):
  - Every individual prediction is now saved to results/predictions_peft.csv,
    same schema as predictions_baseline.csv, so 12_error_analysis.py can
    build confusion matrices and compare error patterns between full
    fine-tuning and LoRA directly.
  - Wall-clock training time (seconds) and peak GPU memory (MB) are now
    measured for every training run and saved as extra columns in
    peft_results.csv. Combined with baseline_results.csv (which now has
    the same columns), this gives you a real, measured training-cost
    comparison between full fine-tuning and LoRA -- not just the
    trainable-parameter-count argument alone.
  - Trainable / total parameter counts are logged per run (previously
    only printed to console, now persisted).

WHAT THIS SCRIPT DOES:
Same experiment grid as Step 6 (07_baseline_finetune.py), but instead
of updating ALL of the model's weights, we FREEZE the pretrained model
and only train small LoRA adapter matrices inserted into the attention
layers. This is the direct comparison your research objective #4 asks
for: full fine-tuning (Step 6) vs. parameter-efficient fine-tuning
(this step), same models, same shots, same seeds, same test set.

WHY NO ZERO-SHOT HERE:
LoRA adapters start at zero contribution (by design) until trained, so
"zero-shot LoRA" would be mathematically IDENTICAL to the plain
pretrained model. We already have that number from Step 6 -- no need
to duplicate it here. This script only covers 5/10/20-shot.

LORA CONFIGURATION:
  - mBERT / XLM-R: LoRA applied to the "query" and "value" attention
    projections (the standard target for BERT-style architectures).
  - mT5: LoRA applied to the "q" and "v" attention projections (T5's
    naming convention for the same thing).
  - r=8, alpha=16, dropout=0.1 -- standard small-model LoRA defaults.

INCREMENTAL SAVING: same resumable design as Step 6 -- results append
to peft_results.csv after every run, and completed (model, shot, seed)
combinations are automatically skipped if you have to restart.

HOW TO RUN (in Colab, AFTER 07_baseline_finetune.py, GPU on):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/08_peft_finetune.py"

IMPORTANT -- IF RE-RUNNING FROM SCRATCH:
This version adds new columns to peft_results.csv. If you have an OLD
peft_results.csv from before this update, DELETE it (or rename it)
before running this script.

WHAT TO DO WITH THE OUTPUT:
Send me the final results table (results/peft_results.csv), the
predictions file (results/predictions_peft.csv), and the printed
summary. We'll compare it side-by-side against baseline_results.csv.
"""

import os
import gc
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from config import CFG, set_global_seed, make_project_folders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_COLUMNS = [
    "model", "shot", "seed", "scope", "accuracy", "precision", "recall", "f1",
    "train_seconds", "peak_gpu_memory_mb", "trainable_params", "total_params",
]

PREDICTION_COLUMNS = [
    "model", "shot", "seed", "Sentence", "Aspect", "domain", "true_label", "predicted_label",
]


# ============================================================
# 1. METRICS (identical to Step 6 -- kept consistent for fair comparison)
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
        domain_mask = test_df["domain"] == domain
        if domain_mask.sum() == 0:
            continue
        domain_true = test_df.loc[domain_mask, "Sentiment"].tolist()
        domain_pred = test_df.loc[domain_mask, "_prediction"].tolist()
        results[domain] = compute_all_metrics(domain_true, domain_pred)

    return results


# ============================================================
# 2. RESULTS LOGGING (separate file from baseline, same resumable design)
# ============================================================

def get_results_path():
    return os.path.join(CFG.RESULTS_DIR, "peft_results.csv")


def get_predictions_path():
    return os.path.join(CFG.RESULTS_DIR, "predictions_peft.csv")


def load_existing_results() -> pd.DataFrame:
    path = get_results_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in RESULT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[RESULT_COLUMNS]
    return pd.DataFrame(columns=RESULT_COLUMNS)


def already_done(existing_df, model_key, shot, seed) -> bool:
    if len(existing_df) == 0:
        return False
    match = existing_df[
        (existing_df["model"] == model_key)
        & (existing_df["shot"] == shot)
        & (existing_df["seed"] == seed)
    ]
    return len(match) > 0


def append_results(model_key, shot, seed, metrics_dict, train_seconds, peak_gpu_memory_mb,
                    trainable_params, total_params):
    rows = []
    for scope, metrics in metrics_dict.items():
        rows.append({
            "model": model_key, "shot": shot, "seed": seed, "scope": scope,
            "accuracy": metrics["accuracy"], "precision": metrics["precision"],
            "recall": metrics["recall"], "f1": metrics["f1"],
            "train_seconds": train_seconds, "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "trainable_params": trainable_params, "total_params": total_params,
        })
    new_df = pd.DataFrame(rows)
    existing_df = load_existing_results()
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(get_results_path(), index=False, encoding="utf-8-sig")
    print(f"  Appended results for model={model_key}, shot={shot}, seed={seed} "
          f"(train_seconds={train_seconds:.1f}, peak_gpu_memory_mb={peak_gpu_memory_mb:.1f})")


def append_predictions(model_key, shot, seed, test_df: pd.DataFrame, predictions: list):
    pred_df = test_df[["Sentence", "Aspect", "domain", "Sentiment"]].copy()
    pred_df = pred_df.rename(columns={"Sentiment": "true_label"})
    pred_df["predicted_label"] = predictions
    pred_df["model"] = model_key
    pred_df["shot"] = shot
    pred_df["seed"] = seed
    pred_df = pred_df[PREDICTION_COLUMNS]

    path = get_predictions_path()
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[
            ~((existing["model"] == model_key) & (existing["shot"] == shot) & (existing["seed"] == seed))
        ]
        combined = pd.concat([existing, pred_df], ignore_index=True)
    else:
        combined = pred_df
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def count_parameters(model) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


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
# 3. LORA CONFIGS PER MODEL FAMILY
# ============================================================

def get_lora_config_encoder() -> LoraConfig:
    """For mBERT/XLM-R classification -- targets the attention query/value projections."""
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )


def get_lora_config_mt5() -> LoraConfig:
    """For mT5 -- targets T5's attention q/v projections (same idea, different naming)."""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        bias="none",
    )


def print_trainable_params(model, model_key: str):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  [{model_key}] Trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")


# ============================================================
# 4. ENCODER MODELS (mBERT, XLM-R) -- LoRA FINE-TUNING PATH
# ============================================================

def finetune_encoder_lora(checkpoint: str, train_df: pd.DataFrame, seed: int, model_key: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(CFG.LABELS))

    lora_config = get_lora_config_encoder()
    model = get_peft_model(base_model, lora_config)
    print_trainable_params(model, model_key)

    def tokenize_fn(batch):
        return tokenizer(batch["Sentence"], batch["Aspect"], truncation=True, max_length=CFG.MAX_SEQ_LENGTH)

    train_df = train_df.copy()
    train_df["label"] = train_df["Sentiment"].map(CFG.LABEL2ID)
    train_dataset = Dataset.from_pandas(train_df[["Sentence", "Aspect", "label"]])
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # LoRA can tolerate (and often benefits from) a slightly higher
    # learning rate than full fine-tuning, since far fewer parameters
    # are being updated -- 2e-4 is a common default for LoRA on small data.
    training_args = TrainingArguments(
        output_dir="/tmp/peft_encoder_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    trainer.train()

    return model, tokenizer


def predict_encoder(model, tokenizer, test_df: pd.DataFrame) -> list:
    model.eval()
    model.to(DEVICE)
    predictions = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i + batch_size]
            encoded = tokenizer(
                list(batch["Sentence"]), list(batch["Aspect"]),
                truncation=True, max_length=CFG.MAX_SEQ_LENGTH,
                padding=True, return_tensors="pt",
            ).to(DEVICE)
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend([CFG.ID2LABEL[p] for p in pred_ids])

    return predictions


# ============================================================
# 5. mT5 -- LoRA FINE-TUNING PATH
# ============================================================

def build_mt5_prompt(sentence: str, aspect: str) -> str:
    return f"sentiment classification: sentence: {sentence} aspect: {aspect}"


def finetune_mt5_lora(checkpoint: str, train_df: pd.DataFrame, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    lora_config = get_lora_config_mt5()
    model = get_peft_model(base_model, lora_config)
    print_trainable_params(model, "mt5")

    def tokenize_fn(batch):
        inputs = [build_mt5_prompt(s, a) for s, a in zip(batch["Sentence"], batch["Aspect"])]
        model_inputs = tokenizer(inputs, truncation=True, max_length=CFG.MAX_SEQ_LENGTH)
        labels = tokenizer(batch["Sentiment"], truncation=True, max_length=8)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_pandas(train_df[["Sentence", "Aspect", "Sentiment"]])
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/peft_mt5_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=1e-3,  # LoRA on T5-style models typically wants an even higher LR
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=True,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    trainer.train()

    return model, tokenizer


def parse_mt5_prediction(decoded_text: str) -> str:
    decoded_lower = decoded_text.strip().lower()
    for label in CFG.LABELS:
        if label in decoded_lower:
            return label
    return "neutral"


def predict_mt5(model, tokenizer, test_df: pd.DataFrame) -> list:
    model.eval()
    model.to(DEVICE)
    predictions = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i + batch_size]
            inputs = [build_mt5_prompt(s, a) for s, a in zip(batch["Sentence"], batch["Aspect"])]
            encoded = tokenizer(inputs, truncation=True, max_length=CFG.MAX_SEQ_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(**encoded, max_new_tokens=8)
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend([parse_mt5_prediction(d) for d in decoded])

    return predictions


# ============================================================
# 6. MAIN EXPERIMENT LOOP
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
        print("WARNING: no GPU detected -- training will be very slow.")

    test_df = load_data()
    existing_results = load_existing_results()

    print("\n" + "=" * 60)
    print("STEP 7: PEFT / LoRA FINE-TUNING (v2 -- with timing/memory/predictions)")
    print("=" * 60)
    print("\nNOTE: no zero-shot here -- LoRA has nothing to add without")
    print("training, so use Step 6's zero-shot numbers as the shared baseline.")

    for model_key, checkpoint in CFG.MODEL_CHECKPOINTS.items():
        for shot in [s for s in CFG.SHOT_SETTINGS if s > 0]:
            for seed in CFG.SEEDS:
                existing_results = load_existing_results()
                if already_done(existing_results, model_key, shot, seed):
                    print(f"[{model_key}] shot={shot} seed={seed} already done -- skipping.")
                    continue

                print(f"\n[{model_key}] LoRA training: shot={shot}, seed={seed}...")
                train_df = load_shot_pool(seed, shot)

                reset_gpu_memory_stats()
                start = time.time()
                if model_key == "mt5":
                    model, tokenizer = finetune_mt5_lora(checkpoint, train_df, seed)
                    preds = predict_mt5(model, tokenizer, test_df)
                else:
                    model, tokenizer = finetune_encoder_lora(checkpoint, train_df, seed, model_key)
                    preds = predict_encoder(model, tokenizer, test_df)
                elapsed = time.time() - start
                peak_mem = get_peak_gpu_memory_mb()
                trainable_params, total_params = count_parameters(model)

                metrics = evaluate_predictions(test_df, preds)
                append_results(model_key, shot, seed, metrics, elapsed, peak_mem,
                                trainable_params, total_params)
                append_predictions(model_key, shot, seed, test_df, preds)
                print(f"  [{model_key}] shot={shot} seed={seed} overall accuracy: "
                      f"{metrics['overall']['accuracy']:.4f} "
                      f"(train time: {elapsed:.1f}s, peak GPU mem: {peak_mem:.1f} MB)")

                del model
                gc.collect()
                torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    final_df = load_existing_results()
    print("\n" + "=" * 60)
    print("FINAL PEFT RESULTS SUMMARY (overall scope, mean +/- std across seeds)")
    print("=" * 60)
    overall_df = final_df[final_df["scope"] == "overall"]
    summary = overall_df.groupby(["model", "shot"])[["accuracy", "f1", "train_seconds", "peak_gpu_memory_mb"]].agg(["mean", "std"])
    print(summary)

    print(f"\nFull results saved to: {get_results_path()}")
    print(f"Per-example predictions saved to: {get_predictions_path()}")
    print("\nDONE. Send me peft_results.csv, predictions_peft.csv (or the printed")
    print("summary) so we can compare it against baseline_results.csv before")
    print("Step 8 (the evaluation harness that combines both).")


if __name__ == "__main__":
    main()
