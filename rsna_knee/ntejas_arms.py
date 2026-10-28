"""Inference for the public NTejas-1 checkpoints (Kaggle dataset tejuuss/rsna-knee-arm-ckpts).

Model code copied from train/train_armA.py of github.com/NTejas-1/RSNA-Knee-Abnormality-Detection
(MIT). Arm A is RadImageNet ResNet50 with [-1, 1] input, arm B the same trunk with
ImageNet weights and mean/std input. Both read the v3 cache: 8 evenly spaced of 16
slices, centre crop 224.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

IMG = 224
SLICES = 8
N_LAB = 12
CKPT_RE = re.compile(r"^(armA|armB)_fold\d+\.pt$")


class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        self.backbone = nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])

    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))


class Attn(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) < 0.5, float("-inf"))
        a = torch.nan_to_num(torch.softmax(a, dim=1), nan=0.0)
        return (a * x).sum(1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = RadEncoder()
        d = 2048
        self.slice_pool = Attn(d)
        self.slot_pool = Attn(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, N_LAB))

    def forward(self, x, pres):
        B, SL, S, H, W = x.shape
        x = x.reshape(B * SL * S, 1, H, W).expand(-1, 3, -1, -1)
        f = self.enc(x).reshape(B, SL, S, -1)
        f = self.slice_pool(f.reshape(B * SL, S, -1)).reshape(B, SL, -1)
        f = self.slot_pool(f, mask=pres)
        return self.head(f)


def find_arm_checkpoints(walk) -> Dict[str, List[Path]]:
    seen = {}
    for p in walk(suffixes=(".pt",)):
        if CKPT_RE.match(p.name) and p.name not in seen:
            seen[p.name] = p
    arms: Dict[str, List[Path]] = {}
    for name in sorted(seen):
        arms.setdefault(name[:4], []).append(seen[name])
    return arms


def _batch(vol: np.ndarray, arm: str) -> torch.Tensor:
    sel = np.linspace(0, vol.shape[1] - 1, SLICES).round().astype(int)
    o = (vol.shape[-1] - IMG) // 2
    v = vol[:, sel][:, :, o:o + IMG, o:o + IMG]
    x = torch.from_numpy(np.ascontiguousarray(v)).float()
    if arm == "armA":
        return x.div_(127.5).sub_(1.0)
    return x.div_(255.0).sub_(0.449).div_(0.226)


def predict_arm(paths: List[Path], arm: str, array, mask, batch: int = 4, log=print) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in paths:
        sd = torch.load(str(p), map_location="cpu", weights_only=False)
        m = Net()
        m.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
        models.append(m.to(dev).eval())
    n = len(array)
    out = np.zeros((n, N_LAB), np.float64)
    with torch.no_grad():
        for i in range(0, n, batch):
            rows = range(i, min(n, i + batch))
            x = torch.stack([_batch(np.asarray(array[r]), arm) for r in rows]).to(dev)
            pr = torch.from_numpy(mask[list(rows)]).to(dev)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                acc = sum(torch.sigmoid(m(x, pr).float()) for m in models)
            out[list(rows)] = (acc / len(models)).cpu().numpy()
    log(f"  {arm}: {len(models)} checkpoints over {n} studies")
    return out
