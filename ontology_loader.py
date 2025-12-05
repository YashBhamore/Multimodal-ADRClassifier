import re
import torch
import pandas as pd
from pathlib import Path
from utils.logger import log
from utils.io_utils import ensure_dir
from models.encoders import TextEncoder


def load_meddra(cfg):
    """
    Load or create a local MedDRA-like ontology (concept ID, PT name, synonyms).
    Returns a dict containing:
        - DataFrame with all terms
        - Concept embeddings (tensor)
        - Text list of concept definitions
    """
    data_dir = Path(cfg["base_dir"]) / cfg["data_dir"]
    ont_dir = ensure_dir(data_dir / "ontologies")
    meddra_path = ont_dir / "meddra.tsv"

    if meddra_path.exists():
        df = pd.read_csv(meddra_path, sep="\t")
        log(f"Loaded MedDRA TSV: {len(df)} concepts.")
    else:
        log("No MedDRA TSV found; creating proxy ontology.")
        df = pd.DataFrame([
            {"concept_id": "271807003", "pt_name": "Rash",
             "synonyms": "exanthem, skin rash, eruption"},
            {"concept_id": "126485001", "pt_name": "Urticaria",
             "synonyms": "hives, wheals, urticarial"},
            {"concept_id": "43116000", "pt_name": "Eczema",
             "synonyms": "dermatitis, atopic dermatitis"},
            {"concept_id": "418363000", "pt_name": "Pruritus",
             "synonyms": "itching, itchy skin"},
            {"concept_id": "386661006", "pt_name": "Erythema",
             "synonyms": "redness of skin, skin redness"}
        ])

    # Expand synonyms
    def explode_terms(row):
        toks = []
        for col in ["pt_name", "synonyms"]:
            for tok in re.split(r"[;,]\s*|\|\s*|\s{2,}", str(row.get(col, ""))):
                if tok.strip():
                    toks.append(tok.strip())
        return sorted(set(toks))

    df["all_terms"] = df.apply(explode_terms, axis=1)
    concept_texts = [
        f"{r.pt_name}. Synonyms: {', '.join(r.all_terms)}"
        for _, r in df.iterrows()
    ]

    # Encode concepts
    log("Encoding ontology concepts using text encoder...")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    encoder = TextEncoder(cfg["models"]["text_model"], device)
    emb = encoder.encode(concept_texts)

    return {
        "df": df,
        "embeddings": emb,
        "texts": concept_texts,
        "ids": df["concept_id"].tolist(),
        "names": df["pt_name"].tolist()
    }
