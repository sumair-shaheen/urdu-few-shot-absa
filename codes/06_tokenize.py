"""
06_tokenize.py
==============
STEP 5 (of our roadmap): Tokenization.

WHAT THIS SCRIPT DOES:
1. Defines the tokenization logic for BOTH model families used in
   this project:
     - ENCODER models (mBERT, XLM-R): classification-style input
         "sentence [SEP] aspect"  ->  label id (0/1/2)
     - ENCODER-DECODER model (mT5): generation-style input
         "sentiment classification: sentence: ... aspect: ..."
         -> generated text ("positive" / "negative" / "neutral")
2. Runs a DRY RUN on your actual test.csv and one shot pool, for ALL
   3 model checkpoints, and prints token length statistics. This
   confirms whether MAX_SEQ_LENGTH=128 (set in config.py) is actually
   enough for your data, BEFORE we commit to it in real training
   (if sequences get truncated, sentiment-bearing words could be cut
   off -- better to catch this now than after a long training run).
3. Saves a small tokenized preview (first 3 examples) to the results
   folder so you can visually sanity-check the tokenization looks
   correct on real examples.

NOTE ON THE MODELS THEMSELVES:
This script does NOT download the full pretrained model weights yet --
only the (much smaller) TOKENIZERS, since that's all tokenization
needs. The full model weights get downloaded in Step 6 (baseline
fine-tuning), which is next.

HOW TO RUN (in Colab, AFTER 05_split_and_shots.py has been run):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/06_tokenize.py"

WHAT TO DO WITH THE OUTPUT:
Copy the full printed summary and send it back so we can confirm
MAX_SEQ_LENGTH is appropriate before moving to Step 6 (baseline
fine-tuning -- the first real model training).
"""

import os

import pandas as pd
from transformers import AutoTokenizer

from config import CFG, set_global_seed, make_project_folders


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_test_and_shot_pool():
    test_path = os.path.join(CFG.SPLITS_DIR, "test.csv")
    shot_path = os.path.join(CFG.SPLITS_DIR, "fewshot", f"seed{CFG.SEED}", "shot20.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path} -- run 05_split_and_shots.py first.")

    test_df = pd.read_csv(test_path)
    shot_df = pd.read_csv(shot_path) if os.path.exists(shot_path) else None

    return test_df, shot_df


# ============================================================
# 2. ENCODER-STYLE INPUT (mBERT, XLM-R)
# ============================================================

def build_encoder_input(sentence: str, aspect: str) -> str:
    """
    For mBERT/XLM-R: the tokenizer itself handles joining two texts
    with [SEP] when given as a pair (see tokenize_encoder_batch below)
    -- we just need to pass sentence and aspect as two separate
    strings to the tokenizer, not concatenate them manually.
    """
    return sentence, aspect  # returned as a pair, used directly by the tokenizer


def tokenize_encoder_batch(tokenizer, sentences, aspects, max_length):
    """
    Tokenizes (sentence, aspect) pairs the way BERT-style models
    expect: [CLS] sentence [SEP] aspect [SEP]. The tokenizer handles
    the [SEP] insertion automatically when given text pairs.
    """
    return tokenizer(
        list(sentences),
        list(aspects),
        truncation=True,
        max_length=max_length,
        padding=False,  # padding is applied later, per-batch, during training
    )


# ============================================================
# 3. GENERATION-STYLE INPUT (mT5)
# ============================================================

def build_mt5_input(sentence: str, aspect: str) -> str:
    """
    For mT5: everything must be ONE text string (no [SEP] pairs --
    T5-style models are trained on plain text-to-text prompts).
    The exact wording of this prompt template matters for few-shot
    prompting quality -- keep it consistent across all experiments.
    """
    return f"sentiment classification: sentence: {sentence} aspect: {aspect}"


def tokenize_mt5_batch(tokenizer, sentences, aspects, labels, max_length):
    """
    Tokenizes the mT5 input (prompt) and target (label word) as plain
    text-to-text pairs.
    """
    inputs = [build_mt5_input(s, a) for s, a in zip(sentences, aspects)]
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_length, padding=False)

    # Targets are tokenized separately -- these become the labels mT5
    # is trained to GENERATE (e.g. the literal word "positive").
    label_encodings = tokenizer(list(labels), truncation=True, max_length=8, padding=False)
    model_inputs["labels"] = label_encodings["input_ids"]

    return model_inputs


# ============================================================
# 4. TOKEN LENGTH DIAGNOSTICS
# ============================================================

def report_token_lengths(tokenizer, df: pd.DataFrame, model_name: str, is_mt5: bool):
    print(f"\n{'-' * 60}\nToken length check -- {model_name}\n{'-' * 60}")

    if is_mt5:
        texts = [build_mt5_input(s, a) for s, a in zip(df["Sentence"], df["Aspect"])]
        lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    else:
        encoded = tokenizer(list(df["Sentence"]), list(df["Aspect"]), truncation=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]

    lengths_series = pd.Series(lengths)
    print(lengths_series.describe())

    over_limit = (lengths_series > CFG.MAX_SEQ_LENGTH).sum()
    pct_over = 100 * over_limit / len(lengths_series)
    print(f"\nExamples exceeding MAX_SEQ_LENGTH={CFG.MAX_SEQ_LENGTH}: {over_limit} ({pct_over:.2f}%)")

    if pct_over > 5:
        print(f"  WARNING: more than 5% of examples would be truncated. "
              f"Consider raising MAX_SEQ_LENGTH in config.py.")
    else:
        print(f"  OK: MAX_SEQ_LENGTH={CFG.MAX_SEQ_LENGTH} covers the vast majority of examples.")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    set_global_seed()
    make_project_folders()

    print("=" * 60)
    print("STEP 5: TOKENIZATION")
    print("=" * 60)

    test_df, shot_df = load_test_and_shot_pool()
    print(f"\nLoaded test.csv: {len(test_df)} rows")
    if shot_df is not None:
        print(f"Loaded shot20.csv (seed {CFG.SEED}): {len(shot_df)} rows")

    preview_rows = []

    for model_key, checkpoint in CFG.MODEL_CHECKPOINTS.items():
        print(f"\n{'=' * 60}\nLoading tokenizer: {model_key} ({checkpoint})\n{'=' * 60}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        is_mt5 = (model_key == "mt5")
        report_token_lengths(tokenizer, test_df, model_key, is_mt5=is_mt5)

        # Build a small preview of the first 3 tokenized examples, for
        # a manual sanity check.
        for i in range(min(3, len(test_df))):
            row = test_df.iloc[i]
            if is_mt5:
                text_in = build_mt5_input(row["Sentence"], row["Aspect"])
                token_ids = tokenizer(text_in)["input_ids"]
                decoded = tokenizer.decode(token_ids)
            else:
                encoded = tokenizer(row["Sentence"], row["Aspect"])
                token_ids = encoded["input_ids"]
                decoded = tokenizer.decode(token_ids)

            preview_rows.append({
                "model": model_key,
                "sentence": row["Sentence"],
                "aspect": row["Aspect"],
                "sentiment": row["Sentiment"],
                "num_tokens": len(token_ids),
                "decoded_check": decoded,
            })

    preview_df = pd.DataFrame(preview_rows)
    preview_path = os.path.join(CFG.RESULTS_DIR, "tokenization_preview.csv")
    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved tokenization preview (first 3 examples x 3 models) to: {preview_path}")

    print("\nDONE. Copy this full output and send it back so we can")
    print("confirm MAX_SEQ_LENGTH before moving to Step 6 (baseline")
    print("fine-tuning -- the first real model training run).")


if __name__ == "__main__":
    main()
