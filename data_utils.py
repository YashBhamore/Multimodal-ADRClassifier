import re
import pandas as pd
from typing import List
from utils.logger import log

ADR_KEYWORDS = [
    "rash", "hives", "urticaria", "eczema",
    "itch", "itching", "redness", "pruritus",
    "swelling", "bumps"
]


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted symbols."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_mentions_simple(text: str) -> List[str]:
    """
    Basic keyword-based mention extractor.
    Detects ADR-related terms from text using substring matches.
    """
    low = str(text).lower()
    mentions = sorted({kw for kw in ADR_KEYWORDS if kw in low})
    return mentions


def normalize_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Normalize a DataFrame by standardizing text and label columns.
    """
    if text_col not in df.columns:
        raise ValueError(f"Missing column '{text_col}' in DataFrame.")
    if label_col not in df.columns:
        log(f"Label column '{label_col}' not found; creating blank labels.", level="warning")
        df[label_col] = ""

    df[text_col] = df[text_col].astype(str).map(clean_text)
    df[label_col] = df[label_col].astype(str).fillna("")
    return df[[label_col, text_col]]


def stratified_split(df: pd.DataFrame, label_col: str = "label", train_ratio: float = 0.8):
    """
    Perform a stratified 80/10/10 split (train/dev/test).
    """
    from sklearn.model_selection import train_test_split

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    has_labels = df[label_col].notna().any() and df[label_col].str.len().gt(0).any()

    try:
        if has_labels:
            train_df, rest = train_test_split(df, test_size=(1 - train_ratio),
                                              stratify=df[label_col].fillna(""), random_state=42)
            dev_df, test_df = train_test_split(rest, test_size=0.5,
                                               stratify=rest[label_col].fillna(""), random_state=42)
        else:
            train_df, rest = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
            dev_df, test_df = train_test_split(rest, test_size=0.5, random_state=42)
    except Exception as e:
        log(f"Stratified split failed ({e}); performing random split.", level="warning")
        train_df = df.sample(frac=0.8, random_state=42)
        rest = df.drop(train_df.index)
        dev_df = rest.sample(frac=0.5, random_state=42)
        test_df = rest.drop(dev_df.index)

    log(f"Split sizes → Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df
