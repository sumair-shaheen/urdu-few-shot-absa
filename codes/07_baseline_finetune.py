"""
07_baseline_finetune.py  (v2)
========================
STEP 6 (of our roadmap): Baseline Fine-Tuning (no PEFT yet -- that's
Step 7). This is your FIRST real model training run.

CHANGELOG (v2 -- added in response to committee feedback):
  - Every individual prediction (sentence, aspect, true label, predicted
    label, domain) is now saved to results/predictions_baseline.csv.
    This is required for confusion matrices and qualitative error
    analysis (12_error_analysis.py reads this file).
  - Wall-clock training time (seconds) and peak GPU memory (MB) are now
    measured for every training run and saved as extra columns in
    baseline_results.csv. This is required for the training-cost /
    efficiency comparison against LoRA (08_peft_finetune.py).
  - Trainable / total parameter counts are also logged per run, so the
    parameter-efficiency comparison doesn't depend on reading printed
    console output.

DESIGN DECISIONS (unchanged from v1 -- read this before running):

1. TRAINING SCOPE: for each (model, shot_setting, seed), we train ONE
   model on the COMBINED cross-domain shot pool (e.g. 5-shot = 45 rows
   spanning all 3 domains x 3 classes x 5 examples each) -- NOT 3
   separate per-domain models. We then evaluate on the full test set,
   broken down BOTH combined AND per-domain.

2. ZERO-SHOT HANDLING:
     - mBERT / XLM-R: CLOZE-STYLE PROMPTING against the model's
       pretrained Masked-Language-Model head (no training at all).
     - mT5: direct generative prompting (no special handling needed).

3. MULTI-SEED: zero-shot runs ONCE per model. 5/10/20-shot runs ONCE
   PER SEED (3 seeds), so you get mean +/- std in your results table.

4. INCREMENTAL SAVING: results (and now predictions) are appended after
   EVERY individual run, so if Colab disconnects partway through, you
   just re-run this script and it SKIPS any (model, shot, seed)
   combination already present in the results file.

HOW TO RUN (in Colab, AFTER 06_tokenize.py has been run, WITH GPU
enabled):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/07_baseline_finetune.py"

IMPORTANT -- IF RE-RUNNING FROM SCRATCH:
This version adds new columns (train_seconds, peak_gpu_memory_mb,
trainable_params, total_params) to baseline_results.csv. If you have an
OLD baseline_results.csv from before this update, DELETE it (or rename
it) before running this script, so the schema doesn't get mixed. The
script will recreate it cleanly on first run.

WHAT TO DO WITH THE OUTPUT:
Send me the final results table (results/baseline_results.csv), the
predictions file (results/predictions_baseline.csv), and the printed
summary.
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
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from config import CFG, set_global_seed, make_project_folders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# English verbalizer words used for the zero-shot cloze prompt (see
# module docstring point 2 for why English words, even though the
# sentence/aspect content is Urdu -- multilingual encoders share
# embedding space, and English label words tokenize more reliably).
VERBALIZERS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}

RESULT_COLUMNS = [
    "model", "shot", "seed", "scope", "accuracy", "precision", "recall", "f1",
    "train_seconds", "peak_gpu_memory_mb", "trainable_params", "total_params",
]

PREDICTION_COLUMNS = [
    "model", "shot", "seed", "Sentence", "Aspect", "domain", "true_label", "predicted_label",
]


# ============================================================
# 1. METRICS
# ============================================================

def compute_all_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(test_df: pd.DataFrame, predictions: list) -> dict:
    """
    Computes overall metrics AND per-domain metrics, given a list of
    predicted label strings aligned with test_df rows.
    """
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
# 2. RESULTS LOGGING (incremental, resumable)
# ============================================================

def get_results_path():
    return os.path.join(CFG.RESULTS_DIR, "baseline_results.csv")


def get_predictions_path():
    return os.path.join(CFG.RESULTS_DIR, "predictions_baseline.csv")


def load_existing_results() -> pd.DataFrame:
    path = get_results_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Robustness: if resuming from an OLDER results file that predates
        # the timing/memory/param columns, add them as NaN rather than
        # crashing on a schema mismatch.
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
    """metrics_dict: {"overall": {...}, "food": {...}, "sports": {...}, "news_politics": {...}}"""
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
    """
    Saves ONE ROW PER TEST EXAMPLE with the true label and the model's
    predicted label, so that 12_error_analysis.py can build confusion
    matrices and pull out real misclassified examples for qualitative
    discussion. This is written incrementally, same resumable pattern
    as the metrics file.
    """
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
        # Drop any old rows for this exact (model, shot, seed) combo before
        # appending fresh ones, so re-running a single config never
        # duplicates prediction rows.
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
# 3. ENCODER MODELS (mBERT, XLM-R) -- FINE-TUNING PATH
# ============================================================

def finetune_encoder(checkpoint: str, train_df: pd.DataFrame, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(CFG.LABELS))

    def tokenize_fn(batch):
        return tokenizer(batch["Sentence"], batch["Aspect"], truncation=True, max_length=CFG.MAX_SEQ_LENGTH)

    train_df = train_df.copy()
    train_df["label"] = train_df["Sentiment"].map(CFG.LABEL2ID)
    train_dataset = Dataset.from_pandas(train_df[["Sentence", "Aspect", "label"]])
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Few-shot regime -- tiny data, so more epochs + small batch size,
    # with a conservative learning rate to avoid catastrophic
    # overfitting on so few examples.
    training_args = TrainingArguments(
        output_dir="/tmp/encoder_finetune_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
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
# 4. ENCODER MODELS -- ZERO-SHOT CLOZE PATH (no training)
# ============================================================

def build_cloze_prompt(sentence: str, aspect: str, mask_token: str) -> str:
    return f"{sentence} Regarding {aspect}, the sentiment is {mask_token}."


def get_verbalizer_token_ids(tokenizer, word: str) -> list:
    """Tokenizes a verbalizer word WITHOUT special tokens. May return
    more than one subword id -- handled by multi-mask scoring below."""
    return tokenizer.encode(word, add_special_tokens=False)


def zero_shot_cloze(checkpoint: str, test_df: pd.DataFrame) -> list:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model.eval()
    model.to(DEVICE)

    mask_token = tokenizer.mask_token
    verbalizer_ids = {label: get_verbalizer_token_ids(tokenizer, word) for label, word in VERBALIZERS.items()}
    max_verbalizer_len = max(len(ids) for ids in verbalizer_ids.values())

    predictions = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            # Use enough mask tokens to cover the LONGEST verbalizer,
            # so every candidate label can be scored fairly even if
            # some verbalizer words split into multiple subwords.
            masks = " ".join([mask_token] * max_verbalizer_len)
            prompt = build_cloze_prompt(row["Sentence"], row["Aspect"], masks)

            encoded = tokenizer(prompt, truncation=True, max_length=CFG.MAX_SEQ_LENGTH, return_tensors="pt").to(DEVICE)
            mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

            logits = model(**encoded).logits[0]  # (seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

            scores = {}
            for label, ids in verbalizer_ids.items():
                # Sum log-probs of each subword id at its corresponding
                # mask position (approximation: treats subwords as
                # independent, which is standard practice in simple
                # DIY cloze/PET implementations).
                score = 0.0
                for j, token_id in enumerate(ids):
                    if j < len(mask_positions):
                        pos = mask_positions[j]
                        score += log_probs[pos, token_id].item()
                scores[label] = score

            best_label = max(scores, key=scores.get)
            predictions.append(best_label)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


# ============================================================
# 5. mT5 -- FINE-TUNING PATH
# ============================================================

def build_mt5_prompt(sentence: str, aspect: str) -> str:
    return f"sentiment classification: sentence: {sentence} aspect: {aspect}"


def finetune_mt5(checkpoint: str, train_df: pd.DataFrame, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

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
        output_dir="/tmp/mt5_finetune_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=3e-4,  # T5-style models typically use a higher LR than BERT-style
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=True,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    return model, tokenizer


def parse_mt5_prediction(decoded_text: str) -> str:
    """Maps mT5's generated text to the closest of our 3 valid labels."""
    decoded_lower = decoded_text.strip().lower()
    for label in CFG.LABELS:
        if label in decoded_lower:
            return label
    return "neutral"  # fallback if generation doesn't match any known label


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


def zero_shot_mt5(checkpoint: str, test_df: pd.DataFrame) -> list:
    """mT5 zero-shot: no training at all, just prompt the pretrained model directly."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    predictions = predict_mt5(model, tokenizer, test_df)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


# ============================================================
# 6. MAIN EXPERIMENT LOOP
# ============================================================

def load_data():
    test_df = pd.read_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"))
    return test_df


def load_shot_pool(seed: int, shot: int) -> pd.DataFrame:
    path = os.path.join(CFG.SPLITS_DIR, "fewshot", f"seed{seed}", f"shot{shot}.csv")
    return pd.read_csv(path)


def main():
    set_global_seed()
    make_project_folders()

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no GPU detected -- training will be very slow. "
              "Check Runtime > Change runtime type > GPU in Colab.")

    test_df = load_data()
    existing_results = load_existing_results()

    print("\n" + "=" * 60)
    print("STEP 6: BASELINE FINE-TUNING (v2 -- with timing/memory/predictions)")
    print("=" * 60)

    for model_key, checkpoint in CFG.MODEL_CHECKPOINTS.items():

        # ---------------- ZERO-SHOT (runs once, no seed loop) ----------------
        if not already_done(existing_results, model_key, 0, -1):
            print(f"\n[{model_key}] Zero-shot evaluation...")
            reset_gpu_memory_stats()
            start = time.time()
            if model_key == "mt5":
                preds = zero_shot_mt5(checkpoint, test_df)
            else:
                preds = zero_shot_cloze(checkpoint, test_df)
            elapsed = time.time() - start
            peak_mem = get_peak_gpu_memory_mb()

            metrics = evaluate_predictions(test_df, preds)
            # Zero-shot trains nothing, so trainable_params=0 by definition;
            # total_params isn't tracked here since no model object survives
            # past the predict function (it's deleted inside zero_shot_* for
            # memory hygiene) -- left as NaN, which is correct and honest.
            append_results(model_key, 0, -1, metrics, elapsed, peak_mem,
                            trainable_params=0, total_params=np.nan)
            append_predictions(model_key, 0, -1, test_df, preds)
            print(f"  [{model_key}] zero-shot overall accuracy: {metrics['overall']['accuracy']:.4f} "
                  f"(inference time: {elapsed:.1f}s)")
        else:
            print(f"\n[{model_key}] Zero-shot already done -- skipping.")

        # ---------------- 5/10/20-SHOT (runs per seed) ----------------
        for shot in [s for s in CFG.SHOT_SETTINGS if s > 0]:
            for seed in CFG.SEEDS:
                existing_results = load_existing_results()  # refresh after each save
                if already_done(existing_results, model_key, shot, seed):
                    print(f"[{model_key}] shot={shot} seed={seed} already done -- skipping.")
                    continue

                print(f"\n[{model_key}] Training: shot={shot}, seed={seed}...")
                train_df = load_shot_pool(seed, shot)

                reset_gpu_memory_stats()
                start = time.time()
                if model_key == "mt5":
                    model, tokenizer = finetune_mt5(checkpoint, train_df, seed)
                    preds = predict_mt5(model, tokenizer, test_df)
                else:
                    model, tokenizer = finetune_encoder(checkpoint, train_df, seed)
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
    print("FINAL RESULTS SUMMARY (overall scope, mean +/- std across seeds)")
    print("=" * 60)
    overall_df = final_df[final_df["scope"] == "overall"]
    summary = overall_df.groupby(["model", "shot"])[["accuracy", "f1", "train_seconds", "peak_gpu_memory_mb"]].agg(["mean", "std"])
    print(summary)

    print(f"\nFull results saved to: {get_results_path()}")
    print(f"Per-example predictions saved to: {get_predictions_path()}")
    print("\nDONE. Send me baseline_results.csv, predictions_baseline.csv (or the")
    print("printed summary) so we can review real numbers before Step 7 (PEFT/LoRA).")


if __name__ == "__main__":
    main()
