"""
07b_mt5_diagnostic.py
======================
DIAGNOSTIC addendum to Step 6 (not a new roadmap step) -- investigates
WHY mT5's zero-shot accuracy/F1 looked like a mathematical signature of
"always predicting the same class" (accuracy=0.3333, F1=0.1667 exactly
matches that pattern on a balanced 3-class test set).

WHAT THIS SCRIPT DOES:
1. Runs mT5 ZERO-SHOT on the full test set and saves the RAW generated
   text (before any parsing/label-matching) for every single example --
   so you can see exactly what the un-fine-tuned model actually outputs,
   with nothing hidden.
2. Prints the most frequent raw outputs, so you can see at a glance
   whether the model is repeating the same text over and over.
3. Re-trains mT5 at 20-shot for TWO contrasting seeds:
     - seed 42  (the one that reached 0.4287 accuracy -- "worked")
     - seed 123 (the one that landed at 0.3339 -- suspected collapse)
   and saves their raw generated text too, so you can directly compare
   a "working" run against a "collapsed" run side by side.
4. Reports the full distribution of PARSED predictions for each run
   (how many times "positive"/"negative"/"neutral" was predicted),
   which makes class collapse immediately visible as a number, not
   just a suspicion.

WHY THIS IS SELF-CONTAINED (not importing from 07_baseline_finetune.py):
Python module names can't start with a digit, so importing
"07_baseline_finetune" directly isn't straightforward -- rather than
fight with importlib workarounds, the small set of functions needed
are duplicated here directly. This keeps the diagnostic simple and
independently runnable.

HOW TO RUN (in Colab, GPU on):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/07b_mt5_diagnostic.py"

WHAT TO DO WITH THE OUTPUT:
Send me the printed summary (frequency tables + prediction
distributions). The full raw-text CSVs are saved in results/ for you
to inspect yourself and quote directly in your thesis if useful.
"""

import os
import gc

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from config import CFG, set_global_seed, make_project_folders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MT5_CHECKPOINT = CFG.MODEL_CHECKPOINTS["mt5"]


def build_mt5_prompt(sentence: str, aspect: str) -> str:
    return f"sentiment classification: sentence: {sentence} aspect: {aspect}"


def parse_mt5_prediction(decoded_text: str) -> str:
    decoded_lower = decoded_text.strip().lower()
    for label in CFG.LABELS:
        if label in decoded_lower:
            return label
    return "neutral"  # same fallback as the main script -- kept identical on purpose


def generate_raw(model, tokenizer, test_df: pd.DataFrame) -> pd.DataFrame:
    """Generates raw text for every test example, WITHOUT parsing, and
    returns a dataframe with sentence/aspect/true label/raw text/parsed label."""
    model.eval()
    model.to(DEVICE)
    raw_outputs = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i:i + batch_size]
            inputs = [build_mt5_prompt(s, a) for s, a in zip(batch["Sentence"], batch["Aspect"])]
            encoded = tokenizer(inputs, truncation=True, max_length=CFG.MAX_SEQ_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(**encoded, max_new_tokens=8)
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            raw_outputs.extend(decoded)

    result_df = test_df[["Sentence", "Aspect", "Sentiment", "domain"]].copy()
    result_df["raw_generated_text"] = raw_outputs
    result_df["parsed_prediction"] = result_df["raw_generated_text"].apply(parse_mt5_prediction)
    return result_df


def finetune_mt5(train_df: pd.DataFrame, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(MT5_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT5_CHECKPOINT)

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
        output_dir="/tmp/mt5_diagnostic_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
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


def report_run(label: str, result_df: pd.DataFrame, eda_dir: str):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    print("\nTop 15 most frequent RAW generated outputs (before parsing):")
    print(result_df["raw_generated_text"].value_counts().head(15))

    n_unique_raw = result_df["raw_generated_text"].nunique()
    print(f"\nNumber of DISTINCT raw outputs across {len(result_df)} test examples: {n_unique_raw}")
    if n_unique_raw <= 3:
        print("  ^ WARNING: extremely few distinct outputs -- strong sign of class collapse.")

    print("\nParsed prediction distribution:")
    print(result_df["parsed_prediction"].value_counts())

    print("\nTrue label distribution (for comparison -- test set is balanced):")
    print(result_df["Sentiment"].value_counts())

    safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
    save_path = os.path.join(eda_dir, f"mt5_raw_generations_{safe_name}.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved full raw-text output ({len(result_df)} rows) to: {save_path}")


def main():
    set_global_seed()
    make_project_folders()

    diag_dir = os.path.join(CFG.RESULTS_DIR, "mt5_diagnostic")
    os.makedirs(diag_dir, exist_ok=True)

    test_df = pd.read_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"))

    print("=" * 60)
    print("mT5 DIAGNOSTIC: RAW GENERATED TEXT INSPECTION")
    print("=" * 60)

    # ---------------- 1. Zero-shot (no training) ----------------
    print("\nLoading mT5 zero-shot (no fine-tuning)...")
    tokenizer = AutoTokenizer.from_pretrained(MT5_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT5_CHECKPOINT)
    zero_shot_df = generate_raw(model, tokenizer, test_df)
    report_run("zero-shot (no training)", zero_shot_df, diag_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- 2. 20-shot, seed 42 (the run that "worked") ----------------
    print("\n\nTraining mT5 at 20-shot, seed=42 (this run previously reached 42.9% accuracy)...")
    shot_path_42 = os.path.join(CFG.SPLITS_DIR, "fewshot", "seed42", "shot20.csv")
    train_df_42 = pd.read_csv(shot_path_42)
    model_42, tokenizer_42 = finetune_mt5(train_df_42, seed=42)
    result_42 = generate_raw(model_42, tokenizer_42, test_df)
    report_run("20-shot seed=42 (previously worked, 42.9% accuracy)", result_42, diag_dir)
    del model_42
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- 3. 20-shot, seed 123 (the run that "collapsed") ----------------
    print("\n\nTraining mT5 at 20-shot, seed=123 (this run previously landed at 33.4% -- suspected collapse)...")
    shot_path_123 = os.path.join(CFG.SPLITS_DIR, "fewshot", "seed123", "shot20.csv")
    train_df_123 = pd.read_csv(shot_path_123)
    model_123, tokenizer_123 = finetune_mt5(train_df_123, seed=123)
    result_123 = generate_raw(model_123, tokenizer_123, test_df)
    report_run("20-shot seed=123 (previously suspected collapse, 33.4% accuracy)", result_123, diag_dir)
    del model_123
    gc.collect()
    torch.cuda.empty_cache()

    print("\n\nDONE. Copy the full printed output above and send it back.")
    print("The 3 CSV files in results/mt5_diagnostic/ contain every single")
    print("raw generation -- nothing is hidden or summarized away there.")


if __name__ == "__main__":
    main()
