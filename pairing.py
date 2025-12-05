import random
import pandas as pd
from utils.logger import log
from utils.data_utils import extract_mentions_simple


def create_text_image_pairs(cadec_df: pd.DataFrame, smm4h_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Align text-based mentions with representative image paths (if available).
    This is a proxy alignment heuristic used in RxFusion.
    """
    # Merge both datasets
    text_df = pd.concat([cadec_df, smm4h_df], ignore_index=True)
    if "mention" not in text_df.columns:
        text_df["mention"] = text_df["text"].map(extract_mentions_simple)

    # Example image sets (to extend with your collectors)
    eczema_paths = []
    urticaria_paths = []
    generic_paths = []

    rows = []
    for _, r in text_df.iterrows():
        mention = str(r["mention"]).lower()
        chosen, conf = None, 0.3
        if any(k in mention for k in ["eczema", "dermatitis"]) and eczema_paths:
            chosen, conf = random.choice(eczema_paths), 0.9
        elif any(k in mention for k in ["hives", "urticaria"]) and urticaria_paths:
            chosen, conf = random.choice(urticaria_paths), 0.9
        elif generic_paths:
            chosen, conf = random.choice(generic_paths), 0.5

        rows.append({
            "text": r["text"],
            "mention": r["mention"],
            "label": r.get("label", ""),
            "image_path": chosen,
            "alignment_confidence": conf
        })

    out_df = pd.DataFrame(rows)
    log(f"Created {len(out_df)} text–image proxy pairs.")
    return out_df
