"""
10b_save_fewshot_demo_model.py
================================
Trains and SAVES the actual few-shot demo model: XLM-R, LoRA, 20-shot,
seed=42 -- this is your thesis's headline result (57.9% accuracy), the
number the whole "few-shot" argument is built on.

WHY THIS SCRIPT EXISTS:
07_baseline_finetune.py and 08_peft_finetune.py both delete their model
after logging metrics -- neither ever calls save_pretrained(). Only
10_full_data_and_save.py saves a model, and that one is trained on the
FULL 7,878-row training set (72.8% accuracy) -- NOT the 20-shot model
this thesis is actually about. If you demo xlmr_full_data live, you are
not demoing the few-shot framework you're defending.

This script trains ONE specific, already-evaluated configuration again
(XLM-R, LoRA, 20-shot, seed=42) and this time SAVES it, so
11_inference_demo.py can load your real few-shot model instead.

IMPORTANT -- LoRA MERGING:
A raw PEFT adapter checkpoint can't be loaded with a plain
AutoModelForSequenceClassification.from_pretrained() call (that's how
11_inference_demo.py loads things, to keep the demo simple). So after
training, we call merge_and_unload() to bake the LoRA weights into the
base model permanently, then save that merged model like any normal
Hugging Face model.

HOW TO RUN (in Colab, GPU on -- after 05_split_and_shots.py):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/10b_save_fewshot_demo_model.py"

WHAT TO DO WITH THE OUTPUT:
Once it prints "Saved model + tokenizer to: ...xlmr_20shot_lora", point
11_inference_demo.py at that folder instead of xlmr_full_data (see the
one-line change noted at the bottom of this file) and re-run your demo.
"""

import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from config import CFG, set_global_seed, make_project_folders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The exact configuration behind your thesis's headline number.
MODEL_KEY = "xlmr"
SHOT = 20
SEED = 42  # first of your three seeds -- pick one fixed run to save & demo
CHECKPOINT = CFG.MODEL_CHECKPOINTS[MODEL_KEY]


def compute_all_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


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
                truncation=True, max_length=CFG.MAX_SEQ_LENGTH, padding=True, return_tensors="pt",
            ).to(DEVICE)
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend([CFG.ID2LABEL[p] for p in pred_ids])
    return predictions


def main():
    set_global_seed()
    make_project_folders()

    print(f"Device: {DEVICE}")
    print(f"Re-training {MODEL_KEY} with LoRA, shot={SHOT}, seed={SEED} "
          f"(the exact run behind your 57.9% headline number)...")

    shot_path = os.path.join(CFG.SPLITS_DIR, "fewshot", f"seed{SEED}", f"shot{SHOT}.csv")
    train_df = pd.read_csv(shot_path)
    test_df = pd.read_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"))
    print(f"Few-shot train pool: {len(train_df)} rows | Test set: {len(test_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    base_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(CFG.LABELS))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    def tokenize_fn(batch):
        return tokenizer(batch["Sentence"], batch["Aspect"], truncation=True, max_length=CFG.MAX_SEQ_LENGTH)

    train_df = train_df.copy()
    train_df["label"] = train_df["Sentiment"].map(CFG.LABEL2ID)
    train_dataset = Dataset.from_pandas(train_df[["Sentence", "Aspect", "label"]])
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="/tmp/fewshot_demo_tmp",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=SEED,
        disable_tqdm=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    trainer.train()

    print("\nEvaluating on test set (sanity check against your logged peft_results.csv)...")
    predictions = predict_encoder(model, tokenizer, test_df)
    metrics = compute_all_metrics(test_df["Sentiment"].tolist(), predictions)
    print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
    print("  (this should be close to -- not necessarily identical to -- the seed=42 row")
    print("   for xlmr/peft/shot=20 in your peft_results.csv; small differences can come")
    print("   from library version drift between runs, not a bug.)")

    # ------------------------------------------------------------
    # Merge LoRA weights into the base model so it can be loaded with
    # a plain AutoModelForSequenceClassification.from_pretrained() call,
    # then save it for the inference demo.
    # ------------------------------------------------------------
    print("\nMerging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    save_path = os.path.join(CFG.MODELS_DIR, f"{MODEL_KEY}_20shot_lora")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved model + tokenizer to: {save_path}")
    print("This is your REAL few-shot (20-shot LoRA) demo model.")

    print("\nDONE. Next step: in 11_inference_demo.py, change:")
    print('    model_path = os.path.join(CFG.MODELS_DIR, "xlmr_full_data")')
    print("to:")
    print(f'    model_path = os.path.join(CFG.MODELS_DIR, "{MODEL_KEY}_20shot_lora")')
    print("then re-run the demo -- it will now show your actual thesis result,")
    print("not the full-data reference number.")


if __name__ == "__main__":
    main()
