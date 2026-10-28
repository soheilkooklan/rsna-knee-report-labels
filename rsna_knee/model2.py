"""v2 study model: 2.5D windows -> encoder -> slice attention -> per-finding slot attention.

Input per study is one cache row, uint8 [6 slots, S slices, H, W]: S = 12 and 288 px for
cache v1, S = 16 and 256 px for cache v3. The slice count is read from the row at run time.
An instance is a window of three consecutive cached slices used as three channels.
Training draws ``train_windows`` random windows per slot; inference uses all S - 2.

Pooling is two-level. Within a slot, gated attention (Ilse et al. 2018) picks the
slices that matter. Across slots, each finding has its own query, because the
evidence for a cruciate tear sits on sagittal images and the evidence for a
collateral ligament tear on coronal ones. Absent slots are masked inside the
softmax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .report_labels import TARGETS

N_SLOT = 6
GROUP = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GatedAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [N, K, D] -> [N, D]
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))
        a = torch.softmax(a.float(), dim=1).to(x.dtype)
        return (a * x).sum(1)


# Where a radiologist looks first for each finding. Slot order in both caches:
# sagittal fluid, sagittal other, coronal fluid, coronal other, axial fluid, axial other.
# Used to initialise a learnable bias on the finding-to-slot attention.
SLOT_PRIOR = {
    "ACL":              [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
    "MCL":              [0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
    "Medial Meniscus":  [1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
    "Lateral Meniscus": [1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
    "Medial OA":        [0.5, 0.5, 1.0, 1.0, 0.0, 0.0],
    "Lateral OA":       [0.5, 0.5, 1.0, 1.0, 0.0, 0.0],
    "PF OA":            [0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
    "Effusion":         [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Synovitis":        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Baker's":          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Contusion":        [1.0, 0.0, 1.0, 0.0, 0.5, 0.0],
    "Fracture":         [1.0, 0.5, 1.0, 0.5, 0.0, 0.0],
}


class FindingSlotAttention(nn.Module):
    """One query per finding over the slot vectors of one study."""

    def __init__(self, dim: int, n_target: int = len(TARGETS), hidden: int = 512,
                 dropout: float = 0.2, slot_prior: bool = False):
        super().__init__()
        if slot_prior:
            self.prior = nn.Parameter(torch.tensor([SLOT_PRIOR[t] for t in TARGETS]))
        else:
            self.prior = None
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_id = nn.Parameter(torch.randn(N_SLOT, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_target, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out_w = nn.Parameter(torch.randn(n_target, hidden) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(n_target))
        self.hidden = hidden

    def forward(self, slots: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # slots [B, 6, D], present [B, 6]
        h = self.proj(slots) + self.slot_id.unsqueeze(0)
        logit = torch.einsum("bsd,od->bos", h.float(), self.query.float()) / self.hidden ** 0.5
        if self.prior is not None:
            logit = logit + self.prior.float().unsqueeze(0)
        logit = logit.masked_fill(present.unsqueeze(1) < 0.5, float("-inf"))
        a = torch.nan_to_num(torch.softmax(logit, dim=-1), nan=0.0)
        ctx = torch.einsum("bos,bsd->bod", a, h.float())
        ctx = self.drop(ctx)
        return (ctx * self.out_w.float()).sum(-1) + self.out_b.float()


class RadResNet50(nn.Module):
    """RadImageNet ResNet50 trunk, key layout of the public ResNet50.pt checkpoint.

    The checkpoint (Kaggle dataset marwanmath/resnet-50-radimagenet-marwan) was saved
    from ``backbone = nn.Sequential(*list(resnet50().children())[:-2])``, so the keys
    read backbone.0.weight and so on. It expects input scaled to [-1, 1].
    """

    def __init__(self):
        super().__init__()
        import torchvision
        self.backbone = nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])
        self.num_features = 2048

    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))


def build_encoder(name: str, weights_path: Optional[str] = None, pretrained: bool = True, log=print):
    """Return (module, feature_dim, normalisation) for an encoder name.

    ``rad_resnet50`` loads the RadImageNet file strictly. Any other name goes to timm;
    with ``pretrained`` it downloads (training kernels may run with internet on),
    otherwise it loads ``weights_path`` when given. At inference the full model
    checkpoint overwrites everything, so pretrained=False is fine there.
    """
    if name == "rad_resnet50":
        enc = RadResNet50()
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            enc.load_state_dict(sd, strict=True)
            from .targets import file_digest
            log(f"RadImageNet ResNet50 loaded strict from {weights_path}, sha256 {file_digest(weights_path)}")
        return enc, 2048, "rad"
    import timm
    if weights_path:
        enc = timm.create_model(name, pretrained=False, num_classes=0)
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        log(f"{name}: loaded {weights_path}, missing {len(missing)}, unexpected {len(unexpected)}")
    else:
        enc = timm.create_model(name, pretrained=pretrained, num_classes=0)
        log(f"{name}: timm pretrained={pretrained}")
    return enc, enc.num_features, "imagenet"


class StudyNetV2(nn.Module):
    def __init__(self, encoder_name: str, weights_path: Optional[str] = None,
                 pretrained: bool = True, chunk: int = 64, slot_prior: bool = False, log=print):
        super().__init__()
        self.encoder, dim, self.norm_kind = build_encoder(encoder_name, weights_path, pretrained, log)
        self.slice_pool = GatedAttention(dim)
        self.head = FindingSlotAttention(dim, slot_prior=slot_prior)
        self.chunk = chunk
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.norm_kind == "rad":
            return x / 127.5 - 1.0
        return (x / 255.0 - self.mean) / self.std

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        # images [B, 6, K, 3, H, W] uint8, present [B, 6]
        b, s, k = images.shape[:3]
        flat = images.reshape(b * s * k, *images.shape[3:])
        feats = []
        for i in range(0, flat.shape[0], self.chunk):
            feats.append(self.encoder(self.normalise(flat[i:i + self.chunk])))
        f = torch.cat(feats).reshape(b * s, k, -1)
        f = self.slice_pool(f).reshape(b, s, -1)
        return self.head(f, present)


# ---------------------------------------------------------------------------
# instance sampling from a cache row
# ---------------------------------------------------------------------------

def windows_from_row(row: np.ndarray, starts: np.ndarray, group: int = GROUP) -> np.ndarray:
    """row [6, S, H, W] uint8 -> [6, K, 3, H, W].

    group 3: three consecutive cached slices as channels.
    group 1: one slice repeated into three channels.
    """
    idx = starts[:, None] + np.arange(group)[None, :]
    uniq, inv = np.unique(idx, return_inverse=True)
    block = np.asarray(row[:, uniq])          # one read of the needed slices
    x = block[:, inv.reshape(idx.shape)]
    if group == 1:
        x = np.repeat(x, 3, axis=2)
    return x


def sample_starts(train: bool, k: int, rng: np.random.Generator,
                  n_slices: int, group: int = GROUP) -> np.ndarray:
    n = n_slices - group + 1
    if not train:
        if group == 1 and n > k:
            return np.linspace(0, n - 1, k).round().astype(int)
        return np.arange(n)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def crop(images: np.ndarray, size: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    h, w = images.shape[-2:]
    if size >= h:
        return images
    if train:
        oy, ox = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    else:
        oy, ox = (h - size) // 2, (w - size) // 2
    return images[..., oy:oy + size, ox:ox + size]


def jitter(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-study intensity gain and offset. No flips: laterality is baked into the cache."""
    gain = float(rng.uniform(0.9, 1.1))
    offset = float(rng.uniform(-8, 8))
    out = images.astype(np.float32) * gain + offset
    return np.clip(out, 0, 255).astype(np.uint8)
