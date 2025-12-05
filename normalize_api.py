import os
import torch
import numpy as np
from utils.logger import log
from models.trainer import evaluate_ranker


@torch.no_grad()
def normalize_query(
    text: str,
    cfg: dict,
    text_encoder,
    image_encoder,
    ontology: dict,
    ranker=None,
    image_path: str = None,
    topk: int = 5
) -> dict:
    """
    Real-time multimodal ADR normalization.
    Combines text + (optional) image → top-k predicted MedDRA concepts.

    Args:
        text: Query text string.
        cfg: Config dictionary.
        text_encoder: Loaded TextEncoder.
        image_encoder: Loaded ImageEncoder.
        ontology: Dictionary from load_meddra(cfg)
        ranker: Trained ranker model (defaults to LateFusion).
        image_path: Optional local image path.
        topk: Number of predictions to return.

    Returns:
        dict: {
            "query_text": str,
            "image_used": bool,
            "pred": {concept_id, label, score},
            "topk": [ ... top-k predictions ... ]
        }
    """

    # Prepare encoders
    t_emb = text_encoder.encode([text])
    v_emb = image_encoder.encode([image_path]) if image_path else image_encoder.encode([None])
    concept_emb = ontology["embeddings"]
    concept_ids = ontology["ids"]
    concept_names = ontology["names"]
    concept_texts = ontology["texts"]

    # Choose model
    model = ranker or _get_default_ranker(text_encoder, image_encoder, concept_emb, cfg)

    # Compute logits
    logits = model(t_emb, v_emb, concept_emb)
    if cfg["training"].get("temperature_scale"):
        logits = logits / cfg["training"]["temperature_scale"]

    # Convert to probabilities
    probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
    order = np.argsort(-probs)
    sel = order[:topk]

    preds = [
        {
            "concept_id": concept_ids[i],
            "label": concept_names[i],
            "score": float(probs[i]),
        }
        for i in sel
    ]

    log(f"Normalized text → top concept: {preds[0]['label']} ({preds[0]['score']:.3f})")

    return {
        "query_text": text,
        "image_used": bool(image_path and os.path.exists(str(image_path))),
        "pred": preds[0] if preds else None,
        "topk": preds,
    }


def _get_default_ranker(text_encoder, image_encoder, concept_emb, cfg):
    """Build a lightweight LateFusion model for zero-shot normalization."""
    from models.rankers import LateFusionRanker
    model = LateFusionRanker(
        d_text=text_encoder.output_dim,
        d_img=image_encoder.output_dim
    ).to(torch.device(cfg["device"] if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def batch_normalize(
    texts: list[str],
    cfg: dict,
    text_encoder,
    image_encoder,
    ontology: dict,
    ranker=None,
    topk: int = 5
) -> list[dict]:
    """
    Batch version of normalize_query() for multiple texts.
    Returns a list of prediction dictionaries.
    """
    results = []
    for t in texts:
        try:
            results.append(
                normalize_query(
                    t,
                    cfg,
                    text_encoder,
                    image_encoder,
                    ontology,
                    ranker=ranker,
                    topk=topk,
                )
            )
        except Exception as e:
            log(f"[WARN] Failed to normalize '{t}': {e}", level="warning")
    return results
