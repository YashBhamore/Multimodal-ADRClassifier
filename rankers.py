import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionRanker(nn.Module):
    """Combine text & image scores at the end (equal weight)."""
    def __init__(self, d_text: int, d_img: int, d_concept: int):
        super().__init__()
        # project concepts into image space only if needed
        self.P = nn.Linear(d_concept, d_img, bias=False)
        # if d_concept != d_img:
        #     self.P = nn.Linear(d_concept, d_img, bias=False)
        # else:
        #     self.P = nn.Identity()

    def forward(self, t, v, ce, conf=None):
        s_txt = t @ ce.T
        s_img = v @ F.normalize(self.P(ce), dim=-1).T
        return 0.5 * s_txt + 0.5 * s_img


# class EarlyFusionRanker(nn.Module):
#     """Fuse text & image features before concept comparison."""
#     def __init__(self, d_text: int, d_img: int, d_concept: int):
#         super().__init__()
#         self.proj_t = nn.Linear(d_text, 256)
#         self.proj_v = nn.Linear(d_img, 256)
#         self.proj_c = nn.Linear(d_concept, 256)

#     def forward(self, t, v, ce, conf=None):
#         t = F.normalize(self.proj_t(t), dim=-1)
#         v = F.normalize(self.proj_v(v), dim=-1)
#         z = F.normalize(0.5 * t + 0.5 * v, dim=-1)
#         c = F.normalize(self.proj_c(ce), dim=-1)
#         return z @ c.T

class EarlyFusionRanker(nn.Module):
    """Fuse text & image features before concept comparison."""
    def __init__(self, d_text: int, d_img: int, d_concept: int):
        super().__init__()
        self.proj_t = nn.Linear(d_text, 256)
        self.proj_v = nn.Linear(d_img, 256)
        self.proj_c = nn.Linear(d_concept, 256)

    def forward(self, t, v, ce, conf=None):
        t = F.normalize(self.proj_t(t), dim=-1)
        v = F.normalize(self.proj_v(v), dim=-1)
        z = F.normalize(0.5 * t + 0.5 * v, dim=-1)
        c = F.normalize(self.proj_c(ce), dim=-1)
        return z @ c.T

# class GatedFusionRanker(nn.Module):
#     """Learn to weight text vs. image dynamically based on confidence."""
#     def __init__(self, d_text: int, d_img: int, beta: float = 2.0):
#         super().__init__()
#         self.w_t = nn.Linear(d_text, 1)
#         self.w_v = nn.Linear(d_img, 1)
#         self.P = nn.Linear(d_text, d_img, bias=False)
#         self.beta = beta

#     def forward(self, t, v, ce, conf=None):
#         s_txt = t @ ce.T
#         s_img = v @ F.normalize(self.P(ce), dim=-1).T
#         g_logit = self.w_t(t) - self.w_v(v)
#         if conf is not None:
#             g_logit = g_logit - self.beta * conf
#         g = torch.sigmoid(g_logit)
#         return g * s_txt + (1 - g) * s_img


class GatedFusionRanker(nn.Module):
    """Fuse text & image using learned gating."""
    def __init__(self, d_text: int, d_img: int, d_concept: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_text + d_img, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.proj_c = nn.Linear(d_concept, 256)
        self.proj_t = nn.Linear(d_text, 256)
        self.proj_v = nn.Linear(d_img, 256)

    def forward(self, t, v, ce, conf=None):
        g = self.gate(torch.cat([t, v], dim=-1))
        t_proj = F.normalize(self.proj_t(t), dim=-1)
        v_proj = F.normalize(self.proj_v(v), dim=-1)
        fused = F.normalize(g * t_proj + (1 - g) * v_proj, dim=-1)
        c = F.normalize(self.proj_c(ce), dim=-1)
        return fused @ c.T
