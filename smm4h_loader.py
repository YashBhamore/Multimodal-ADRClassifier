import pandas as pd
from pathlib import Path
from utils.logger import log
from utils.io_utils import ensure_dir
from utils.data_utils import normalize_dataframe, stratified_split


def load_smm4h(cfg) -> pd.DataFrame:
    """
    Load SMM4H TSV files and prepare a unified dataset.
    Supports ade_train.tsv / ade_dev.tsv / ade_test.tsv, or smm4h.tsv with split column.
    """
    base = Path(cfg["base_dir"]) / cfg["raw_subdir"] / "text" / "smm4h"
    ensure_dir(base)

    train_path = base / "ade_train.tsv"
    dev_path = base / "ade_dev.tsv"
    test_path = base / "ade_test.tsv"
    mono_path = base / "smm4h.tsv"

    if all(p.exists() for p in [train_path, dev_path, test_path]):
        log("Found explicit train/dev/test TSVs under SMM4H.")
        train_df = pd.read_csv(train_path, sep="\t", names=["label", "text"], header=None)
        dev_df = pd.read_csv(dev_path, sep="\t", names=["label", "text"], header=None)
        test_df = pd.read_csv(test_path, sep="\t", names=["label", "text"], header=None)
    elif mono_path.exists():
        log("Found single smm4h.tsv; performing split by 'split' column or stratified fallback.")
        df = pd.read_csv(mono_path, sep="\t", dtype=str)
        if "split" in df.columns:
            train_df = df[df["split"] == "train"]
            dev_df = df[df["split"] == "dev"]
            test_df = df[df["split"] == "test"]
        else:
            train_df, dev_df, test_df = stratified_split(df)
    else:
        log("SMM4H data not found. Returning empty DataFrame.", level="warning")
        return pd.DataFrame(columns=["label", "text"])

    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    all_df = normalize_dataframe(all_df)
    log(f"Loaded SMM4H ({len(all_df)} rows total).")
    return all_df
