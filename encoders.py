import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms, models
from PIL import Image
from utils.logger import log


class TextEncoder:
    """Wrapper for transformer-based text embedding model (e.g., SapBERT)."""
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.output_dim = self.model.config.hidden_size
        log(f"Loaded text model: {model_name}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        texts = [re.sub(r"\s+", " ", str(t)).strip() for t in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.model.config.max_position_embeddings, return_tensors="pt").to(self.device)
        outputs = self.model(**batch)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(pooled, dim=-1)


class ImageEncoder:
    """Wrapper for image encoder (ViT via timm or MobileNet fallback)."""
    def __init__(self, model_name: str, pretrained: bool, device: torch.device):
        self.device = device
        # self.output_dim = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        try:
            import timm
            log(f"Building timm encoder: {model_name} (pretrained={pretrained})")
            m = timm.create_model(model_name, pretrained=pretrained)
            if hasattr(m, "reset_classifier"):
                m.reset_classifier(0)
            self.model = m.to(device).eval()
        except Exception as e:
            log(f"[WARN] timm unavailable ({e}); falling back to MobileNetV3.")
            m = models.mobilenet_v3_small(pretrained=pretrained)
            m.classifier = torch.nn.Identity()
            self.model = m.to(device).eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            out = self.model(dummy)
            if out.ndim > 2:
                out = F.adaptive_avg_pool2d(out, 1).flatten(1)
            self.output_dim = out.shape[-1]
            log(f"Image encoder output dimension: {self.output_dim}")

    @torch.no_grad()
    def encode(self, image_paths: list[str]) -> torch.Tensor:
        batch = []
        for p in image_paths:
            try:
                im = Image.open(p).convert("RGB")
                batch.append(self.transform(im))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        X = torch.stack(batch).to(self.device)
        z = self.model(X)
        if z.ndim > 2:
            z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return F.normalize(z, dim=-1)
