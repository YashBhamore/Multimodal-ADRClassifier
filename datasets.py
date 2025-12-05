import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.logger import log


class PairDataset(Dataset):
    """Dataset holding text mentions, image paths, labels, and confidence."""
    def __init__(self, df, name2idx):
        self.df = df.reset_index(drop=True)
        self.name2idx = name2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = self.name2idx.get(str(r.get("label", "")).lower(), -1)
        conf = float(r.get("alignment_confidence", 0.0))
        img_path = r.get("image_path", None)
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            img_path = None
        return str(r["text"]), img_path, y, conf


def make_dataloaders(df, cfg, text_enc, img_enc):
    """Split labeled pairs into DataLoaders for training and testing."""
    from sklearn.model_selection import train_test_split
    # df = df.dropna(subset=["mention"])
    # if "label" not in df.columns:
    #     df["label"] = ""

    # name2idx = {lbl.lower(): i for i, lbl in enumerate(df["label"].unique()) if lbl}
    name2idx = {
                    str(lbl): i
                    for i, lbl in enumerate(df["label"].unique())
                    if str(lbl).strip()  # filters out empty or whitespace-only strings
                }
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    def collate(batch):
        m, p, y, c = zip(*batch)
        return (
            text_enc.encode(list(m)),
            img_enc.encode(list(p)),
            torch.tensor(y, dtype=torch.long, device=text_enc.device),
            torch.tensor(c, dtype=torch.float32, device=text_enc.device).unsqueeze(-1)
        )

    train_loader = DataLoader(PairDataset(train_df, name2idx),
                              batch_size= 16,
                              shuffle=True, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_df, name2idx),
                             batch_size= 16,
                             shuffle=False, collate_fn=collate)
    log(f"Built dataloaders → Train: {len(train_df)} / Test: {len(test_df)}")
    return train_loader, None, test_loader
