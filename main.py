import json
import torch
import pandas as pd
from pathlib import Path

from utils.logger import log
from data_loaders.cadec_loader import load_cadec
from data_loaders.smm4h_loader import load_smm4h
from data_loaders.ontology_loader import load_meddra
from data_loaders.skincap_loader import load_skincap
from data_loaders.pairing import create_text_image_pairs
from models.encoders import TextEncoder, ImageEncoder
from models.rankers import LateFusionRanker, EarlyFusionRanker, GatedFusionRanker
from models.datasets import make_dataloaders
from models.trainer import train_ranker
from models.evaluators import evaluate_all
from api.normalize_api import normalize_query


def main():
    # 1️⃣ Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # 2️⃣ Load data
    log("Loading datasets...")
    # cadec_df = load_cadec(cfg)
    cadec_df = pd.read_csv(r"C:\Users\dp1622\OneDrive - UNT System\Desktop\info5731_project\data\cadec\cadec_dataset_grouped.csv")
    # smm4h_df = load_smm4h(cfg)
    # ontology = load_meddra(cfg)
    # skincap_df = load_skincap(cfg)
    skincap_df = pd.read_excel(r"C:\Users\dp1622\OneDrive - UNT System\Desktop\info5731_project\data\skincap_v240715.xlsx")
    # pairs_df = create_text_image_pairs(cadec_df, smm4h_df, cfg)
    # log(f"Prepared {len(pairs_df)} text–image pairs.")
    concat_df = pd.concat([cadec_df, skincap_df], ignore_index=True)

    concat_df["label"] = (
    concat_df["label"]
    .astype(str)                # ensure string type
    .str.strip()                # remove leading/trailing spaces
    .replace(["", "nan", "None", "NaN"], None)  # normalize empties
    .fillna("Other")            # fill missing values
    .str.lower()                # convert to lowercase
)
    concat_df = concat_df.sample(200)
    # 3️⃣ Initialize encoders and concept embeddings
    log("Initializing encoders...")
    text_enc = TextEncoder(cfg["text_model"], device)
    img_enc = ImageEncoder(cfg["image_model"], True, device)
    # concept_emb, concept_texts = ontology["embeddings"], ontology["texts"]

    # 4️⃣ Make loaders
    train_loader, val_loader, test_loader = make_dataloaders(concat_df, cfg, text_enc, img_enc)

    unique_labels = concat_df["label"].unique().tolist()
    concept_emb = text_enc.encode(unique_labels)
    d_text, d_img = text_enc.output_dim, img_enc.output_dim
    d_concept = concept_emb.shape[-1]
    late_ranker = LateFusionRanker(d_text, d_img, d_concept).to(device)
    early_ranker = EarlyFusionRanker(d_text, d_img, d_concept).to(device)
    gated_ranker = GatedFusionRanker(d_text, d_img, d_concept).to(device)

    # 6️⃣ Train and evaluate
    log("Training and evaluating models...")
    train_ranker(late_ranker, train_loader, val_loader, concept_emb, cfg)
    train_ranker(early_ranker, train_loader, val_loader, concept_emb, cfg)
    train_ranker(gated_ranker, train_loader, val_loader, concept_emb, cfg)

    results = evaluate_all([late_ranker, early_ranker, gated_ranker], test_loader, concept_emb, cfg)
    log("Evaluation complete.")
    print(results)

    # 7️⃣ Example real-time normalization
    result = normalize_query("Patient developed severe itching and rash", cfg, text_enc, img_enc, ontology)
    print("\nExample normalization:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
