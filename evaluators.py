import torch
import pandas as pd
from utils.logger import log
from models.trainer import evaluate_ranker


def evaluate_all(models, test_loader, concept_emb, cfg):
    """
    Evaluate all rankers on the test set and return a DataFrame.
    """
    results = {}
    names = ["LateFusion", "EarlyFusion", "GatedFusion"]

    for name, model in zip(names, models):
        metrics = evaluate_ranker(model, test_loader, concept_emb)
        results[name] = metrics
        log(f"{name} → Acc@1={metrics['Acc@1']:.3f} | MRR={metrics['MRR']:.3f}")

    return pd.DataFrame(results).T
