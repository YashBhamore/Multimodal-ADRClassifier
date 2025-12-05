import torch
import torch.nn.functional as F
from utils.logger import log


def train_ranker(model, train_loader, val_loader, concept_emb, cfg):
    """Train any fusion ranker on labeled text–image pairs."""
    if train_loader is None:
        log("⚠️ No training data; skipping training.", level="warning")
        return model

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 3

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = n = 0
        for t, v, y, conf in train_loader:
            s = model(t, v, concept_emb, conf)
            loss = F.cross_entropy(s, y, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        log(f"[Train] Epoch {ep}/{epochs} | Loss: {total_loss / max(1, n):.4f}")

        # Optional validation
        if val_loader:
            model.eval()
            val_acc = evaluate_ranker(model, val_loader, concept_emb)["Acc@1"]
            log(f"[Val] Epoch {ep} | Acc@1: {val_acc:.3f}")

    return model


def evaluate_ranker(model, loader, concept_emb, ks=[1, 3, 5]):
    """Compute top-k accuracy and MRR."""
    if loader is None:
        return {"Acc@1": 0.0, "MRR": 0.0}
    model.eval()
    all_scores, all_gold = [], []
    for t, v, y, conf in loader:
        s = model(t, v, concept_emb, conf)
        all_scores.append(s)
        all_gold.extend(y.cpu().tolist())

    S = torch.cat(all_scores)
    order = torch.argsort(S, dim=-1, descending=True).cpu().numpy()
    g = torch.tensor(all_gold).numpy()

    acc1 = (order[:, 0] == g).mean()
    ranks = []
    for i in range(len(g)):
        try:
            rank = list(order[i]).index(g[i]) + 1
        except ValueError:
            rank = float("inf")
        ranks.append(rank)
    mrr = sum(1.0 / r for r in ranks if r != float("inf")) / len(ranks)
    return {"Acc@1": float(acc1), "MRR": float(mrr)}
