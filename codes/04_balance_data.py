"""
04_balance_data.py
===================
STEP 3 (of our roadmap) of the pipeline: Domain-Proportional Balancing.

WHAT THIS SCRIPT DOES:
Takes the cleaned combined dataset (15,133 rows: negative=5,283,
neutral=3,752, positive=6,098) and downsamples negative/positive down
toward the neutral count (~3,752), so all 3 sentiment classes end up
roughly equal -- WITHOUT randomly deleting rows in a way that erases
domain diversity.

THE LOGIC (per your decision earlier in our conversation):
  1. Target count per class = the SMALLEST existing class total
     (neutral, ~3,752) -- we downsample toward it rather than
     upsampling/duplicating data.
  2. Food's negative (28) and positive (29) counts are a HARD FLOOR --
     they are NEVER downsampled further, since that data doesn't exist
     to spare. Food's neutral (342) is also left untouched.
  3. For negative and positive, after reserving Food's small fixed
     count, the REMAINING target is distributed proportionally between
     Sports and News/Politics based on their existing share of that
     class -- so we don't accidentally end up mostly-News data with a
     token amount of Sports, which would quietly undermine the
     cross-domain comparison your thesis objectives need.
  4. Neutral is left as-is in every domain (already at/near target,
     no downsampling needed there).
  5. Random sampling uses the FIXED SEED from config.py, so this is
     100% reproducible -- running this script twice gives identical
     results.

HOW TO RUN (in Colab, AFTER 03_clean_data.py has been run):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/04_balance_data.py"

WHAT TO DO WITH THE OUTPUT:
Copy the full printed summary and send it back so we can confirm the
balanced numbers before moving to Step 4 (train/val/test split +
few-shot pool construction).
"""

import os

import pandas as pd

from config import CFG, set_global_seed, make_project_folders


def load_clean_data() -> pd.DataFrame:
    clean_path = os.path.join(CFG.CLEAN_DATA_DIR, "combined_clean.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"combined_clean.csv not found at {clean_path} -- "
            "run 03_clean_data.py first."
        )
    return pd.read_csv(clean_path)


def compute_balance_plan(df: pd.DataFrame) -> dict:
    """
    Works out exactly how many rows to keep from each (domain, class)
    combination. Returns a dict: {(domain, sentiment): target_count}.
    """
    counts = df.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0)
    print("Current counts (before balancing):")
    print(counts)

    class_totals = df["Sentiment"].value_counts()
    target_total = class_totals.min()
    target_class = class_totals.idxmin()
    print(f"\nTarget per-class total: {target_total} (matching the smallest class: '{target_class}')")

    plan = {}

    for sentiment in CFG.LABELS:
        food_count = counts.loc["food", sentiment] if "food" in counts.index else 0

        if sentiment == target_class:
            # This IS the smallest class -- keep everything as-is,
            # no downsampling needed anywhere.
            for domain in CFG.DOMAINS:
                plan[(domain, sentiment)] = counts.loc[domain, sentiment] if domain in counts.index else 0
            continue

        # Food is a hard floor: always keep all of Food's rows for
        # this class, never downsample it further.
        plan[("food", sentiment)] = food_count

        # Remaining target to distribute between sports and news,
        # proportional to their EXISTING share of this class.
        remaining_target = max(target_total - food_count, 0)
        other_domains = [d for d in CFG.DOMAINS if d != "food"]
        other_counts = {d: counts.loc[d, sentiment] for d in other_domains}
        other_total = sum(other_counts.values())

        allocated = 0
        for i, domain in enumerate(other_domains):
            if i == len(other_domains) - 1:
                # Last domain absorbs any rounding remainder, so the
                # totals add up EXACTLY to remaining_target.
                domain_target = remaining_target - allocated
            else:
                share = other_counts[domain] / other_total if other_total > 0 else 0
                domain_target = round(remaining_target * share)
                allocated += domain_target
            plan[(domain, sentiment)] = max(domain_target, 0)

    return plan


def apply_balance_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Randomly samples the planned number of rows from each (domain, class) group."""
    sampled_parts = []
    for (domain, sentiment), target_count in plan.items():
        subset = df[(df["domain"] == domain) & (df["Sentiment"] == sentiment)]
        actual_count = min(target_count, len(subset))
        if actual_count < target_count:
            print(f"  WARNING: wanted {target_count} rows for ({domain}, {sentiment}) "
                  f"but only {len(subset)} exist -- taking all {actual_count}.")
        sampled = subset.sample(n=actual_count, random_state=CFG.SEED)
        sampled_parts.append(sampled)

    balanced_df = pd.concat(sampled_parts, ignore_index=True)
    # Shuffle the final combined result so rows aren't grouped by
    # domain/class in file order (matters later for training).
    balanced_df = balanced_df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    return balanced_df


def main():
    set_global_seed()
    make_project_folders()

    print("=" * 60)
    print("STEP 3: DOMAIN-PROPORTIONAL BALANCING")
    print("=" * 60)

    df = load_clean_data()

    plan = compute_balance_plan(df)

    print("\nBalancing plan (target row counts per domain x class):")
    for (domain, sentiment), count in sorted(plan.items()):
        print(f"  {domain:<15} {sentiment:<10} -> {count}")

    balanced_df = apply_balance_plan(df, plan)

    # Save combined balanced file
    balanced_path = os.path.join(CFG.CLEAN_DATA_DIR, "balanced_dataset.csv")
    balanced_df.to_csv(balanced_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved balanced combined dataset to: {balanced_path}")

    # Save per-domain balanced files too, for convenience
    for domain in CFG.DOMAINS:
        domain_df = balanced_df[balanced_df["domain"] == domain]
        domain_path = os.path.join(CFG.CLEAN_DATA_DIR, f"{domain}_balanced.csv")
        domain_df.to_csv(domain_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("FINAL BALANCED DISTRIBUTION")
    print("=" * 60)
    final_counts = balanced_df.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0)
    print(final_counts)

    print("\nTotal combined sentiment distribution (balanced):")
    print(balanced_df["Sentiment"].value_counts())

    print(f"\nTotal rows before balancing : {len(df)}")
    print(f"Total rows after balancing  : {len(balanced_df)}")

    print("\nDONE. Copy this full output and send it back so we can")
    print("confirm the balanced numbers before moving to Step 4")
    print("(train/val/test split + few-shot pool construction).")


if __name__ == "__main__":
    main()
