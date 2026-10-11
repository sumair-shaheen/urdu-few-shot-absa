"""
02b_eda.py
==========
EDA (Exploratory Data Analysis) -- should have come right after loading
the raw data and BEFORE the cleaning decisions in 02_clean_data.py were
made. We're running it now, retroactively, covering both:

  (A) the RAW data  -- this justifies WHY the cleaning steps were
      needed (duplicates, label inconsistencies, etc.) -- useful for
      your thesis's "data preprocessing" section, since you can show
      the "before" picture.
  (B) the CLEANED data -- confirms cleaning worked, and gives you the
      real distributions/statistics you'll actually build your
      few-shot splits from.

WHAT THIS SCRIPT PRODUCES:
  1. Printed statistics (row counts, missing values, duplicates,
     label distributions, sentence length stats, aspect frequency,
     whitespace issues, aspect-not-in-sentence mismatches).
  2. Saved chart images (.png) in CFG.RESULTS_DIR/eda/:
       - sentiment_distribution_raw.png
       - sentiment_distribution_clean.png
       - sentence_length_distribution.png
       - top_aspects_per_domain.png

HOW TO RUN (in Colab, after Step 2 has already been run once):
    import sys
    sys.path.append("/content/drive/MyDrive/urdu_absa_project")
    %run "/content/drive/MyDrive/urdu_absa_project/02b_eda.py"

WHAT TO DO WITH THE OUTPUT:
Copy the printed summary AND look at the saved chart images (open them
from the Colab file browser, or from Drive directly) -- these charts
are usable directly in your thesis's EDA section. Send me the printed
summary so we can confirm everything looks reasonable before Step 3.
"""

import os
import subprocess
import sys as _sys
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import CFG, set_global_seed, make_project_folders


# ============================================================
# 0. SETUP -- Urdu font + text shaping
# ============================================================
# WHY THIS IS NEEDED: matplotlib's default font (DejaVu Sans) has NO
# Urdu/Arabic glyphs, so any chart with Urdu text labels would render
# as empty boxes. On top of that, Arabic-script text needs "reshaping"
# (letters change shape depending on position in a word) and
# "bidi" reordering (right-to-left display) -- matplotlib does NEITHER
# of these automatically. We fix both issues here, once, at the top.

def ensure_shaping_libraries():
    """Installs arabic-reshaper and python-bidi if not already present."""
    for pkg, import_name in [("arabic-reshaper", "arabic_reshaper"), ("python-bidi", "bidi")]:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {pkg} (needed for correct Urdu chart rendering)...")
            subprocess.run([_sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def get_urdu_font(eda_dir: str) -> fm.FontProperties:
    """
    Downloads Noto Naskh Arabic (a free Google font that renders Urdu
    correctly) into the results/eda folder if not already present, and
    registers it with matplotlib. Returns a FontProperties object to
    apply to any chart text that contains Urdu.
    """
    font_path = os.path.join(eda_dir, "NotoNaskhArabic.ttf")
    if not os.path.exists(font_path):
        print("Downloading Urdu-capable font (one-time, ~300KB)...")
        font_url = (
            "https://raw.githubusercontent.com/google/fonts/main/ofl/"
            "notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf"
        )
        urllib.request.urlretrieve(font_url, font_path)
        print(f"Saved font to: {font_path}")

    fm.fontManager.addfont(font_path)
    return fm.FontProperties(fname=font_path)


def fix_urdu_text(text: str) -> str:
    """
    Reshapes Urdu/Arabic text into correct joined letterforms and
    reorders it for right-to-left display. Call this on any Urdu
    string BEFORE putting it into a chart label.
    """
    import arabic_reshaper
    from bidi.algorithm import get_display
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)


def get_eda_output_dir():
    eda_dir = os.path.join(CFG.RESULTS_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    return eda_dir




def load_raw():
    food_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_FOOD_FILE)
    sports_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_SPORTS_FILE)
    news_path = os.path.join(CFG.RAW_DATA_DIR, CFG.RAW_NEWS_FILE)

    food_df = pd.read_excel(food_path)
    sports_df = pd.read_csv(sports_path)
    news_df = pd.read_csv(news_path)

    food_df["domain"] = "food"
    sports_df["domain"] = "sports"
    news_df["domain"] = "news_politics"

    return pd.concat([food_df, sports_df, news_df], ignore_index=True)


def load_clean():
    clean_path = os.path.join(CFG.CLEAN_DATA_DIR, "combined_clean.csv")
    if not os.path.exists(clean_path):
        print("NOTE: combined_clean.csv not found -- run 02_clean_data.py first "
              "if you want the 'after cleaning' comparison. Continuing with raw-only EDA.")
        return None
    return pd.read_csv(clean_path)


# ============================================================
# 2. STRUCTURAL / INTEGRITY CHECKS
# ============================================================

def structural_checks(df: pd.DataFrame, label: str):
    print(f"\n{'=' * 60}\nSTRUCTURAL CHECKS -- {label}\n{'=' * 60}")
    print(f"Total rows: {len(df)}")

    print("\nMissing values per column:")
    print(df[["Sentence", "Aspect", "Sentiment"]].isna().sum())

    exact_dupes = df.duplicated(subset=["Sentence", "Aspect", "Sentiment"]).sum()
    print(f"\nExact duplicate rows (Sentence+Aspect+Sentiment): {exact_dupes}")

    # Whitespace issues in Aspect column
    aspect_str = df["Aspect"].astype(str)
    whitespace_issues = (aspect_str != aspect_str.str.strip()).sum()
    print(f"Aspect values with leading/trailing whitespace: {whitespace_issues}")

    # Aspect not found verbatim inside Sentence
    mismatch_count = sum(
        not (str(a).strip() in str(s))
        for a, s in zip(df["Aspect"], df["Sentence"])
    )
    print(f"Rows where Aspect text is NOT found inside Sentence: {mismatch_count}")


# ============================================================
# 3. LABEL DISTRIBUTION CHECKS
# ============================================================

def label_distribution(df: pd.DataFrame, label: str, eda_dir: str, filename: str):
    print(f"\n{'=' * 60}\nLABEL DISTRIBUTION -- {label}\n{'=' * 60}")

    # Show RAW unique values first -- this reveals inconsistencies like
    # "positive " (trailing space) or "Positive" (different case) before
    # any cleaning is applied.
    print("Raw unique Sentiment values found:", df["Sentiment"].astype(str).unique())

    print("\nPer-domain sentiment counts:")
    counts = df.groupby(["domain", df["Sentiment"].astype(str).str.strip().str.lower()]).size().unstack(fill_value=0)
    print(counts)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Sentiment Distribution per Domain -- {label}")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of rows")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(eda_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {save_path}")


# ============================================================
# 4. TEXT CHARACTERISTICS
# ============================================================

def text_characteristics(df: pd.DataFrame, eda_dir: str):
    print(f"\n{'=' * 60}\nSENTENCE LENGTH STATISTICS (character count)\n{'=' * 60}")

    df = df.copy()
    df["sentence_length"] = df["Sentence"].astype(str).str.len()

    print(df.groupby("domain")["sentence_length"].describe())

    fig, ax = plt.subplots(figsize=(8, 5))
    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]["sentence_length"]
        ax.hist(subset, bins=30, alpha=0.5, label=domain)
    ax.set_title("Sentence Length Distribution per Domain")
    ax.set_xlabel("Sentence length (characters)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(eda_dir, "sentence_length_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {save_path}")


# ============================================================
# 5. ASPECT FREQUENCY ANALYSIS
# ============================================================

def aspect_frequency(df: pd.DataFrame, eda_dir: str, urdu_font: fm.FontProperties):
    print(f"\n{'=' * 60}\nASPECT FREQUENCY ANALYSIS\n{'=' * 60}")

    for domain in df["domain"].unique():
        subset = df[df["domain"] == domain]
        n_unique = subset["Aspect"].nunique()
        n_single = (subset["Aspect"].value_counts() == 1).sum()
        print(f"\n[{domain}] unique aspects: {n_unique} | appearing only once: {n_single}")
        print(f"[{domain}] top 10 most frequent aspects:")
        print(subset["Aspect"].value_counts().head(10))

    # Combined chart: top 10 aspects per domain, side by side.
    # Urdu labels are reshaped + bidi-reordered, and rendered with the
    # Urdu-capable font, so they display as proper joined Urdu script
    # instead of broken boxes.
    domains = df["domain"].unique()
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5))
    if len(domains) == 1:
        axes = [axes]
    for ax, domain in zip(axes, domains):
        top10 = df[df["domain"] == domain]["Aspect"].value_counts().head(10)
        labels = [fix_urdu_text(a) for a in top10.index[::-1]]
        ax.barh(labels, top10.values[::-1])
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontproperties(urdu_font)
            tick_label.set_fontsize(12)
        ax.set_title(f"Top 10 Aspects -- {domain}")
        ax.set_xlabel("Count")
    plt.tight_layout()
    save_path = os.path.join(eda_dir, "top_aspects_per_domain.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart: {save_path}")


# ============================================================
# 6. MULTI-ASPECT SENTENCE ANALYSIS
# ============================================================

def multi_aspect_analysis(df: pd.DataFrame):
    print(f"\n{'=' * 60}\nMULTI-ASPECT SENTENCE ANALYSIS\n{'=' * 60}")

    dup_sentences = df[df.duplicated(subset=["Sentence"], keep=False)]
    print(f"Rows sharing a Sentence with at least one other row: {len(dup_sentences)}")
    print(f"Unique such sentences: {dup_sentences['Sentence'].nunique()}")

    multi_aspect_count = 0
    conflicting_sentiment_count = 0
    for sentence, group in dup_sentences.groupby("Sentence"):
        if group["Aspect"].nunique() > 1:
            multi_aspect_count += 1
            if group["Sentiment"].nunique() > 1:
                conflicting_sentiment_count += 1

    print(f"Sentences with multiple DIFFERENT aspects (legitimate multi-aspect ABSA data): {multi_aspect_count}")
    print(f"  ...of which sentiment also differs across aspects (expected/correct): {conflicting_sentiment_count}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    set_global_seed()
    make_project_folders()
    eda_dir = get_eda_output_dir()

    ensure_shaping_libraries()
    urdu_font = get_urdu_font(eda_dir)

    print("\n" + "#" * 60)
    print("# PART A: EDA ON RAW DATA (before cleaning)")
    print("#" * 60)

    raw_df = load_raw()
    structural_checks(raw_df, "RAW")
    label_distribution(raw_df, "RAW", eda_dir, "sentiment_distribution_raw.png")
    multi_aspect_analysis(raw_df)

    print("\n" + "#" * 60)
    print("# PART B: EDA ON CLEANED DATA (after 02_clean_data.py)")
    print("#" * 60)

    clean_df = load_clean()
    if clean_df is not None:
        structural_checks(clean_df, "CLEAN")
        label_distribution(clean_df, "CLEAN", eda_dir, "sentiment_distribution_clean.png")
        text_characteristics(clean_df, eda_dir)
        aspect_frequency(clean_df, eda_dir, urdu_font)

    print("\nDONE. Charts saved in:", eda_dir)
    print("Copy the full printed output above and send it back, and take a")
    print("look at the saved chart images too -- these are usable directly")
    print("in your thesis's EDA section.")


if __name__ == "__main__":
    main()
