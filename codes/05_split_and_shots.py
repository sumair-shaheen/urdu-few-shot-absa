"""
05_split_and_shots.py
======================
STEP 4 (of our roadmap): Train/Val/Test Split + Few-Shot Pool Construction.

WHAT THIS SCRIPT DOES:
1. Loads the balanced dataset (11,256 rows, equal classes).
2. Splits it into train/val/test (70/15/15), STRATIFIED BY
   (domain, sentiment) -- meaning every domain+class combination gets
   its own fair 70/15/15 share, so val/test remain representative of
   all 3 domains and all 3 classes, not skewed toward whichever is
   largest.
3. From the TRAIN portion only, builds NESTED few-shot support sets:
   for each domain and each class, the 5-shot examples are a subset of
   the 10-shot examples, which are a subset of the 20-shot examples.
   This isolates "effect of more examples" from "effect of which
   random examples got picked" -- standard practice in few-shot
   literature.
4. Repeats step 3 for EACH of 3 seeds (42, 123, 2024), so you get 3
   independent nested chains -- this is what lets you report
   mean +/- std across seeds later, instead of a single lucky/unlucky
   number.
5. Val and test are built ONCE (not per-seed) and stay fixed across
   every experiment, so every model/shot-setting/seed combination is
   evaluated on the exact same test examples -- this is what makes
   your results table actually comparable.

FOLDER STRUCTURE CREATED (inside CFG.SPLITS_DIR):
    data/splits/
        train.csv              <- full training pool (not used directly
                                   for few-shot training -- shot pools
                                   below are sampled FROM this)
        val.csv                <- fixed validation set
        test.csv                <- fixed test set (used for ALL evaluations)
        fewshot/
            seed42/
                shot5.csv       <- 5 examples per (domain, class)
                shot10.csv      <- 10 examples per (domain, class)
                shot20.csv      <- 20 examples per (domain, class)
            seed123/
                shot5.csv, shot10.csv, shot20.csv
            seed2024/
                shot5.csv, shot10.csv, shot20.csv

HOW TO RUN (in Colab, AFTER 04_balance_data.py has been run):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/05_split_and_shots.py"

WHAT TO DO WITH THE OUTPUT:
Copy the full printed summary and send it back so we can confirm the
split sizes and shot-pool sizes before moving to Step 5 (tokenization).
"""

import os

import pandas as pd

from config import CFG, set_global_seed, make_project_folders


def load_balanced_data() -> pd.DataFrame:
    balanced_path = os.path.join(CFG.CLEAN_DATA_DIR, "balanced_dataset.csv")
    if not os.path.exists(balanced_path):
        raise FileNotFoundError(
            f"balanced_dataset.csv not found at {balanced_path} -- "
            "run 04_balance_data.py first."
        )
    return pd.read_csv(balanced_path)


def stratified_split(df: pd.DataFrame, seed: int):
    """
    Splits each (domain, Sentiment) group independently into
    train/val/test (70/15/15), then concatenates all groups back
    together. This guarantees every domain+class combination is
    represented proportionally in every split.
    """
    train_parts, val_parts, test_parts = [], [], []

    for (domain, sentiment), group in df.groupby(["domain", "Sentiment"]):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_test = round(n * CFG.TEST_RATIO)
        n_val = round(n * CFG.VAL_RATIO)
        n_train = n - n_val - n_test

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


def build_nested_shot_pools(train_df: pd.DataFrame, seed: int) -> dict:
    """
    For a given seed, builds nested 5/10/20-shot support sets per
    (domain, class). Returns {shot_count: DataFrame}.
    """
    shuffled_groups = {}
    for (domain, sentiment), group in train_df.groupby(["domain", "Sentiment"]):
        shuffled_groups[(domain, sentiment)] = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    shot_pools = {}
    for shot_count in CFG.SHOT_SETTINGS:
        if shot_count == 0:
            continue  # zero-shot uses no training examples at all
        parts = []
        for (domain, sentiment), shuffled in shuffled_groups.items():
            available = len(shuffled)
            take = min(shot_count, available)
            if take < shot_count:
                print(f"  WARNING: seed {seed}, shot {shot_count}, ({domain}, {sentiment}) "
                      f"only has {available} training examples available -- taking all {take}.")
            parts.append(shuffled.iloc[:take])
        shot_pools[shot_count] = pd.concat(parts, ignore_index=True)

    return shot_pools


def main():
    set_global_seed()
    make_project_folders()

    print("=" * 60)
    print("STEP 4: TRAIN/VAL/TEST SPLIT + FEW-SHOT POOL CONSTRUCTION")
    print("=" * 60)

    df = load_balanced_data()

    # ------------------------------------------------------------
    # Fixed train/val/test split (uses the main CFG.SEED, NOT the
    # multi-seed list -- this split happens ONCE and stays fixed
    # across every experiment).
    # ------------------------------------------------------------
    train_df, val_df, test_df = stratified_split(df, seed=CFG.SEED)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("\nTrain set distribution (domain x class):")
    print(train_df.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0))
    print("\nVal set distribution (domain x class):")
    print(val_df.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0))
    print("\nTest set distribution (domain x class):")
    print(test_df.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0))

    train_df.to_csv(os.path.join(CFG.SPLITS_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(CFG.SPLITS_DIR, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(CFG.SPLITS_DIR, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"\nSaved train.csv, val.csv, test.csv to: {CFG.SPLITS_DIR}")

    # ------------------------------------------------------------
    # Nested few-shot pools, one independent chain per seed.
    # ------------------------------------------------------------
    fewshot_dir = os.path.join(CFG.SPLITS_DIR, "fewshot")
    os.makedirs(fewshot_dir, exist_ok=True)

    for seed in CFG.SEEDS:
        print(f"\n{'-' * 60}\nBuilding few-shot pools for seed {seed}\n{'-' * 60}")
        seed_dir = os.path.join(fewshot_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        shot_pools = build_nested_shot_pools(train_df, seed=seed)

        for shot_count, pool_df in shot_pools.items():
            save_path = os.path.join(seed_dir, f"shot{shot_count}.csv")
            pool_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"  shot{shot_count}: {len(pool_df)} rows -> {save_path}")

        # Sanity check: confirm nesting actually holds (5-shot IDs are
        # a subset of 10-shot IDs, which are a subset of 20-shot IDs).
        if all(k in shot_pools for k in [5, 10, 20]):
            ids_5 = set(shot_pools[5].index) if "Sentence" not in shot_pools[5] else set(shot_pools[5]["Sentence"] + "||" + shot_pools[5]["Aspect"])
            ids_10 = set(shot_pools[10]["Sentence"] + "||" + shot_pools[10]["Aspect"])
            ids_20 = set(shot_pools[20]["Sentence"] + "||" + shot_pools[20]["Aspect"])
            nested_ok = ids_5.issubset(ids_10) and ids_10.issubset(ids_20)
            print(f"  Nesting check (5-shot subset of 10-shot subset of 20-shot): {'PASSED' if nested_ok else 'FAILED'}")

    print("\nDONE. Copy this full output and send it back so we can")
    print("confirm the split and shot-pool sizes before moving to")
    print("Step 5 (tokenization).")


if __name__ == "__main__":
    main()
