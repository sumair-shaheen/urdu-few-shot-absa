"""
10_full_data_and_save.py
==========================
STEP (bonus, not in original numbered roadmap): Full-Data Reference
Training + Model Saving.

WHY THIS EXISTS:
Two purposes at once:
1. Trains XLM-R (your best-performing model) on the FULL training set
   (7,878 rows -- not just a k-shot subset) to give you an UPPER-BOUND
   reference number. This is what turns "our few-shot accuracy is only
   58%" into "here's the ceiling with full data (X%), here's zero-shot
   (37%), and here's how fast we close that gap with just 20 examples"
   -- a much stronger defense answer, as discussed.
2. SAVES the trained model + tokenizer to disk (saved_models/), since
   every previous script deleted its model after evaluating it. This
   saved model is what the next script (11_inference_demo.py) loads
   to let you type in a sentence + aspect and see a live prediction.

HOW TO RUN (in Colab, GPU on -- can run any time after 05_split_and_shots.py):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/10_full_data_and_save.py"

WHAT TO DO WITH THE OUTPUT:
Send me the printed accuracy/F1. Once confirmed, run
11_inference_demo.py to try your own sentences.
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
from datasets import Dataset

from config import CFG, set_global_seed, make_project_folders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# XLM-R was the clear best performer across all your experiments --
# this is the model we save for the interactive demo.
MODEL_KEY = "xlmr"
CHECKPOINT = CFG.MODEL_CHECKPOINTS[MODEL_KEY]


def compute_all_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
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
        results[domain] = compute_all_metrics(test_df.loc[mask, "Sentiment"].tolist(), test_df.loc[mask, "_prediction"].tolist())
    return results


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
    print(f"Training {MODEL_KEY} ({CHECKPOINT}) on the FULL training set...")

    train_df = pd.read_csv(os.path.join(CFG.SPLITS_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"))
    print(f"Full train set: {len(train_df)} rows | Test set: {len(test_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=len(CFG.LABELS))

    def tokenize_fn(batch):
        return tokenizer(batch["Sentence"], batch["Aspect"], truncation=True, max_length=CFG.MAX_SEQ_LENGTH)

    train_df = train_df.copy()
    train_df["label"] = train_df["Sentiment"].map(CFG.LABEL2ID)
    train_dataset = Dataset.from_pandas(train_df[["Sentence", "Aspect", "label"]])
    train_dataset = train_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Full data has MUCH more signal than few-shot -- far fewer epochs
    # needed (3, not 30), and a bigger batch size since we have plenty
    # of examples now.
    training_args = TrainingArguments(
        output_dir="/tmp/full_data_tmp",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=CFG.SEED,
        disable_tqdm=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    trainer.train()

    print("\nEvaluating on test set...")
    predictions = predict_encoder(model, tokenizer, test_df)
    metrics = evaluate_predictions(test_df, predictions)

    print("\n" + "=" * 60)
    print("FULL-DATA REFERENCE RESULTS")
    print("=" * 60)
    for scope, m in metrics.items():
        print(f"  {scope:15} accuracy={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    # Save this result alongside your other results, tagged shot="full"
    # so it's easy to add to your comparison table/chart later.
    results_path = os.path.join(CFG.RESULTS_DIR, "full_data_results.csv")
    rows = [{"model": MODEL_KEY, "shot": "full", "seed": CFG.SEED, "scope": scope,
             "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
            for scope, m in metrics.items()]
    pd.DataFrame(rows).to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved results to: {results_path}")

    # ------------------------------------------------------------
    # Save the model + tokenizer to disk for the inference demo.
    # ------------------------------------------------------------
    save_path = os.path.join(CFG.MODELS_DIR, f"{MODEL_KEY}_full_data")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved model + tokenizer to: {save_path}")
    print("This is what 11_inference_demo.py will load.")

    print("\nDONE. Send me the printed accuracy/F1 above. Once confirmed,")
    print("run 11_inference_demo.py to try your own sentences.")


if __name__ == "__main__":
    main()
