"""
02_clean_data.py
=================
STEP 2 of the implementation roadmap: Data Loading & Cleaning.

WHAT THIS SCRIPT DOES (in order):
1. Loads the 3 raw files (Food, Sports, News/Politics).
2. Cleans each column:
     - strips leading/trailing whitespace from Sentence/Aspect/Sentiment
     - normalizes Unicode (NFC form) and fixes Arabic/Urdu look-alike
       character issues (e.g. Arabic yeh vs Urdu yeh) -- this is a
       common silent bug in Urdu NLP where two characters LOOK
       identical but are different Unicode code points, which would
       otherwise confuse the tokenizer.
     - standardizes Sentiment values to exactly: negative / neutral /
       positive (lowercase, no stray spaces).
3. Drops rows with missing Sentence/Aspect/Sentiment (there were 4 in
   the News file).
4. Removes EXACT duplicate rows (same Sentence + Aspect + Sentiment).
5. Resolves CONFLICTING duplicates: same Sentence + same Aspect, but
   DIFFERENT Sentiment across rows. Resolved by majority vote; if
   there's a tie, the row is dropped (can't be resolved automatically).
   NOTE: this does NOT touch legitimate multi-aspect sentences (same
   sentence, different aspects) -- those are correct ABSA data and are
   left untouched.
6. Tags every row with its `domain` (food / sports / news_politics).
7. Saves:
     - one cleaned file per domain
     - one combined cleaned file (all domains, with domain column)
   into CFG.CLEAN_DATA_DIR, plus prints a full summary so you can
   verify nothing unexpected happened.

HOW TO RUN (in Colab):
    Upload this file into your Drive project folder (same folder as
    config.py), then in a Colab cell:

        import sys
        sys.path.append("/content/drive/MyDrive/urdu_absa_project")
        %run "/content/drive/MyDrive/urdu_absa_project/02_clean_data.py"

WHAT TO DO WITH THE OUTPUT:
Copy the full printed summary and send it back so we can confirm the
numbers match expectations before moving to Step 3 (domain-proportional
balancing).
"""

import os
import re
import unicodedata

import pandas as pd

from config import CFG, set_global_seed, make_project_folders


# ============================================================
# 1. UNICODE / CHARACTER NORMALIZATION HELPERS
# ============================================================

# Common Arabic-range characters that LOOK identical to their Urdu
# counterparts but are different Unicode code points. Left uncorrected,
# a tokenizer may treat "same-looking" words as different tokens purely
# because of which code point was used when the text was typed/scraped.
ARABIC_TO_URDU_MAP = {
    "\u064A": "\u06CC",  # Arabic yeh  -> Urdu yeh
    "\u0643": "\u06A9",  # Arabic kaf  -> Urdu keheh
    "\u0629": "\u06C1",  # Arabic teh marbuta -> Urdu heh (common in
                          # scraped text; approximate but standard fix)
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def normalize_text(text: str) -> str:
    """
    Applies Unicode NFC normalization, fixes Arabic/Urdu look-alike
    characters, strips URLs, and collapses extra whitespace.
    Used on both Sentence and Aspect columns.
    """
    if not isinstance(text, str):
        return text

    # Unicode NFC normalization -- combines decomposed characters into
    # their canonical composed form (important for consistent tokenization).
    text = unicodedata.normalize("NFC", text)

    # Fix Arabic/Urdu look-alike characters.
    for arabic_char, urdu_char in ARABIC_TO_URDU_MAP.items():
        text = text.replace(arabic_char, urdu_char)

    # Remove URLs (rare in this data, but cheap to handle defensively).
    text = URL_PATTERN.sub("", text)

    # Collapse multiple spaces/tabs/newlines into a single space, then
    # strip leading/trailing whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_label(label: str) -> str:
    """Standardizes a Sentiment value: lowercase + strip whitespace."""
    if not isinstance(label, str):
        return label
    return label.strip().lower()


# ============================================================
# 2. LOAD RAW FILES
# ============================================================

def load_raw_datasets():
    """Loads the 3 raw files and tags each with its domain name."""
    food_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_FOOD_FILE)
    sports_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_SPORTS_FILE)
    news_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_NEWS_FILE)

    print("Loading raw files from:", CFG.RAW_DATA_DIR)

    food_df = pd.read_excel(food_path)
    sports_df = pd.read_csv(sports_path)
    news_df = pd.read_csv(news_path)

    food_df["domain"] = "food"
    sports_df["domain"] = "sports"
    news_df["domain"] = "news_politics"

    print(f"  Food           : {len(food_df)} rows loaded")
    print(f"  Sports         : {len(sports_df)} rows loaded")
    print(f"  News/Politics  : {len(news_df)} rows loaded")

    return food_df, sports_df, news_df


# ============================================================
# 3. CLEAN A SINGLE DATAFRAME
# ============================================================

def clean_dataframe(df: pd.DataFrame, domain_name: str) -> pd.DataFrame:
    """
    Applies all cleaning steps to one domain's dataframe:
    drop missing values, normalize text/labels, drop exact duplicates.
    Returns the cleaned dataframe (conflicting-duplicate resolution is
    handled separately, AFTER combining all domains, so it can also
    catch conflicts across domain boundaries if any exist).
    """
    before = len(df)

    # Drop rows with missing Sentence/Aspect/Sentiment.
    df = df.dropna(subset=["Sentence", "Aspect", "Sentiment"]).copy()
    dropped_missing = before - len(df)

    # Normalize text columns.
    df["Sentence"] = df["Sentence"].apply(normalize_text)
    df["Aspect"] = df["Aspect"].apply(normalize_text)
    df["Sentiment"] = df["Sentiment"].apply(normalize_label)

    # Keep only the 3 valid sentiment labels -- drops any stray/invalid
    # label that normalization didn't catch (defensive safety net).
    valid_labels = set(CFG.LABELS)
    before_label_filter = len(df)
    df = df[df["Sentiment"].isin(valid_labels)].copy()
    dropped_invalid_labels = before_label_filter - len(df)

    # Drop rows where Sentence or Aspect became empty after cleaning.
    before_empty_filter = len(df)
    df = df[(df["Sentence"].str.len() > 0) & (df["Aspect"].str.len() > 0)]
    dropped_empty = before_empty_filter - len(df)

    # Remove EXACT duplicate rows (same Sentence + Aspect + Sentiment).
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Sentence", "Aspect", "Sentiment"])
    dropped_exact_dupes = before_dedup - len(df)

    print(f"\n[{domain_name}] cleaning summary:")
    print(f"  Missing values dropped        : {dropped_missing}")
    print(f"  Invalid labels dropped        : {dropped_invalid_labels}")
    print(f"  Empty after cleaning dropped  : {dropped_empty}")
    print(f"  Exact duplicates dropped      : {dropped_exact_dupes}")
    print(f"  Remaining rows                : {len(df)}")

    return df


# ============================================================
# 4. RESOLVE CONFLICTING DUPLICATES (same Sentence+Aspect, diff Sentiment)
# ============================================================

def resolve_conflicts(df: pd.DataFrame):
    """
    Finds groups with the SAME (Sentence, Aspect) but DIFFERENT
    Sentiment values, and resolves each group by majority vote.
    If there's a tie (e.g. exactly 1 positive vs 1 negative), the
    group is set aside into `conflicts_df` for manual review instead
    of being silently discarded.

    IMPORTANT: this does NOT affect legitimate multi-aspect sentences
    (same Sentence, DIFFERENT Aspect) -- those are correct ABSA data
    and are grouped separately, so they pass through untouched.

    Returns:
        resolved_df   -- the clean, conflict-free dataframe to use
                          going forward in the pipeline.
        conflicts_df  -- ALL rows involved in unresolved ties, saved
                          separately so you can manually inspect and
                          decide on each one later if you have time.
    """
    grouped = df.groupby(["Sentence", "Aspect"])

    resolved_rows = []
    conflict_rows = []
    conflict_groups = 0
    dropped_tie_groups = 0

    for (sentence, aspect), group in grouped:
        unique_sentiments = group["Sentiment"].unique()

        if len(unique_sentiments) == 1:
            # No conflict -- keep just one representative row.
            resolved_rows.append(group.iloc[0])
        else:
            conflict_groups += 1
            sentiment_counts = group["Sentiment"].value_counts()
            top_count = sentiment_counts.iloc[0]
            is_tie = (sentiment_counts == top_count).sum() > 1

            if is_tie:
                dropped_tie_groups += 1
                # Keep ALL rows from this tied group for manual review,
                # instead of silently discarding them.
                for _, row in group.iterrows():
                    conflict_rows.append(row)
                continue

            majority_sentiment = sentiment_counts.idxmax()
            resolved_row = group.iloc[0].copy()
            resolved_row["Sentiment"] = majority_sentiment
            resolved_rows.append(resolved_row)

    resolved_df = pd.DataFrame(resolved_rows).reset_index(drop=True)
    conflicts_df = pd.DataFrame(conflict_rows).reset_index(drop=True)

    print("\nConflict resolution summary:")
    print(f"  (Sentence, Aspect) groups with conflicting Sentiment : {conflict_groups}")
    print(f"  Resolved via majority vote                            : {conflict_groups - dropped_tie_groups}")
    print(f"  Set aside for manual review (unresolvable tie)        : {dropped_tie_groups}")
    print(f"  Rows before conflict resolution                       : {len(df)}")
    print(f"  Rows kept (clean, conflict-free)                      : {len(resolved_df)}")
    print(f"  Rows set aside for review                             : {len(conflicts_df)}")

    return resolved_df, conflicts_df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    set_global_seed()
    make_project_folders()

    print("\n" + "=" * 60)
    print("STEP 2: DATA LOADING & CLEANING")
    print("=" * 60)

    food_df, sports_df, news_df = load_raw_datasets()

    food_clean = clean_dataframe(food_df, "Food")
    sports_clean = clean_dataframe(sports_df, "Sports")
    news_clean = clean_dataframe(news_df, "News/Politics")

    # Combine all 3 domains BEFORE resolving conflicts, so conflicts
    # are resolved consistently across the whole dataset.
    combined = pd.concat([food_clean, sports_clean, news_clean], ignore_index=True)
    print(f"\nCombined rows before conflict resolution: {len(combined)}")

    combined_resolved, conflicts_df = resolve_conflicts(combined)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    combined_path = os.path.join(CFG.CLEAN_DATA_DIR, "combined_clean.csv")
    combined_resolved.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved combined cleaned file to: {combined_path}")

    conflicts_path = os.path.join(CFG.CLEAN_DATA_DIR, "conflicts_for_manual_review.csv")
    conflicts_df.to_csv(conflicts_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(conflicts_df)} conflicting rows for manual review to: {conflicts_path}")
    print("  (These rows share the same Sentence+Aspect but have contradicting")
    print("   Sentiment labels with no clear majority. They are EXCLUDED from")
    print("   the clean dataset used going forward. Review this file whenever")
    print("   you have spare time -- if you manually pick the correct label for")
    print("   any row, you can add it back into combined_clean.csv later.)")

    for domain_name in CFG.DOMAINS:
        domain_df = combined_resolved[combined_resolved["domain"] == domain_name]
        domain_path = os.path.join(CFG.CLEAN_DATA_DIR, f"{domain_name}_clean.csv")
        domain_df.to_csv(domain_path, index=False, encoding="utf-8-sig")
        print(f"Saved {domain_name} cleaned file to: {domain_path}  ({len(domain_df)} rows)")

    # ------------------------------------------------------------
    # Final summary -- sentiment distribution per domain, post-cleaning
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SENTIMENT DISTRIBUTION (after cleaning)")
    print("=" * 60)
    summary = combined_resolved.groupby(["domain", "Sentiment"]).size().unstack(fill_value=0)
    print(summary)

    print("\n" + "=" * 60)
    print("TOTAL COMBINED SENTIMENT DISTRIBUTION")
    print("=" * 60)
    print(combined_resolved["Sentiment"].value_counts())

    print("\nDONE. Copy this full output and send it back so we can")
    print("confirm the numbers before moving to Step 3 (balancing).")


if __name__ == "__main__":
    main()
