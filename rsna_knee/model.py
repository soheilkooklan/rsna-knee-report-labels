"""Encoder plus a per-finding attention head over the series of one study.

A study is a bag. Each instance in it is three adjacent slices of one series stacked
as three channels. That is close enough to a natural image for a pretrained encoder
to accept, and thick enough that a structure crossing two slices is not cut in half.

The label is attached to the *study*, so nothing in the supervision says which series
or which slice carries a finding. That fact sets the size of the head. A richer
aggregator, such as slice attention or a transformer over the stack, has no signal
to learn its extra parameters from and spends them fitting noise. What the head does
get is one query per finding, because the protocol is not accidental: cruciates are
read sagittally, the collateral ligaments and the meniscal body coronally, patellar
cartilage axially. Pooling all series identically dilutes the one carrying the
evidence with the five that do not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .report_labels import TARGETS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TargetAttentionHead(nn.Module):
    r"""One attention query per finding, over the instances of one study.

    With instance embeddings :math:`x_i` and a presence mask :math:`m_i`,

    .. math::
        h_i = \phi(x_i) + e_{s(i)}, \qquad
        \alpha_{o,i} = \frac{\exp(\langle h_i, q_o\rangle/\sqrt{H})\, m_i}
                            {\sum_j \exp(\langle h_j, q_o\rangle/\sqrt{H})\, m_j},

    .. math::
        c_o = \sum_i \alpha_{o,i} h_i, \qquad \ell_o = \langle c_o, w_o\rangle + b_o,

    where :math:`e_{s(i)}` is a learned identity for the slot instance *i* came from.
    Masking inside the softmax renormalises over what the study actually contains, so
    a missing axial series moves a finding's attention onto the sequences that are
    present rather than feeding it a zero vector.
    """

    def __init__(self, in_dim: int, n_slot: int, n_target: int = len(TARGETS),
                 hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
        )
        self.slot_identity = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_target, hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_target)
        self.hidden = hidden

    def forward(self, features: torch.Tensor, slot_index: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # features   [B, I, D]   instance embeddings
        # slot_index [B, I]      which slot each instance came from
        # mask       [B, I]      1 where the instance exists
        h = self.project(features) + self.slot_identity[slot_index]
        logits = torch.einsum("bid,od->boi", h, self.query) / (self.hidden ** 0.5)
        logits = logits.masked_fill(~mask.bool().unsqueeze(1), float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        # A study with no instances at all would make every logit -inf and the
        # softmax NaN. It cannot happen with a valid mount, and if it does the run
        # should return the prior rather than poison the batch.
        attention = torch.nan_to_num(attention, nan=0.0)
        context = torch.einsum("boi,bid->bod", attention, h)
        context = self.dropout(context)
        # Each finding reads its own row of the classifier weight, so the twelve
        # outputs do not share a linear map over a shared pooled vector.
        return (context * self.classifier.weight).sum(-1) + self.classifier.bias


class StudyModel(nn.Module):
    """Encoder over instances, attention head over the study."""

    def __init__(self, backbone: nn.Module, feature_dim: int, n_slot: int,
                 n_target: int = len(TARGETS), hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.head = TargetAttentionHead(feature_dim, n_slot, n_target, hidden, dropout)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def encode(self, images: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        """Encode ``[N, 3, H, W]`` uint8-scaled images in chunks.

        Chunking is not an optimisation. A study contributes tens of instances and a
        batch several studies, so the flattened instance axis is what decides peak
        memory, rather than the nominal batch size. Fixing the chunk makes memory a
        function of the chunk instead of the study, which keeps a study with six
        well-filled series from ending the run.
        """
        images = images.float().div_(255.0)
        images = (images - self.mean) / self.std
        outputs = []
        for start in range(0, images.shape[0], chunk):
            outputs.append(self.backbone(images[start:start + chunk]))
        return torch.cat(outputs, dim=0)

    def forward(self, images: torch.Tensor, slot_index: torch.Tensor,
                mask: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        # images [B, I, 3, H, W]
        b, i = images.shape[:2]
        flat = images.reshape(b * i, *images.shape[2:])
        features = self.encode(flat, chunk=chunk).reshape(b, i, -1)
        return self.head(features, slot_index, mask)


ARCH_PREFIXES = (
    "tf_efficientnetv2_", "tf_efficientnet_", "efficientnetv2_", "efficientnet_",
    "convnext_", "resnet", "regnety_", "seresnext",
)


def arch_from_checkpoint(checkpoint_path: str, default: str) -> str:
    """Guess the timm architecture from a weight file name.

    timm publishes weights named after the exact architecture, and the tf_ ports are
    a different model from the native ones even though the parameter names and
    shapes agree: tf_efficientnet_b3 builds Conv2dSame layers and uses different
    batch-norm epsilon. Loading tf_ weights into efficientnet_b3 succeeds silently
    and puts every filter against slightly shifted input, which is the kind of error
    that shows up only as a disappointing score.

    Falls back to ``default`` when the name says nothing useful.
    """
    stem = Path(checkpoint_path).stem.lower()
    for prefix in ARCH_PREFIXES:
        if stem.startswith(prefix):
            # Trim the trailing weight tag: tf_efficientnet_b3_aa-84b4657e -> the
            # architecture is everything up to the tag, and timm names never carry
            # a hyphen, so the hyphen is a reliable cut point.
            name = stem.split("-")[0]
            for tag in ("_aa", "_ap", "_ns", "_in1k", "_in21k", "_ra2", "_ra", "_bt"):
                if name.endswith(tag):
                    name = name[: -len(tag)]
            return name
    return default


def build_backbone(name: str = "efficientnet_b3", pretrained: bool = True,
                   checkpoint_path: Optional[str] = None, log=None):
    """Return ``(module, feature_dim)`` for a timm encoder with its classifier removed.

    On Kaggle the network is off, so ``pretrained=True`` only works when timm can
    find the weights in a mounted dataset. Pass ``checkpoint_path`` for that case;
    training from scratch on 4,407 studies is not a workable fallback and the
    function says so rather than quietly producing a random encoder.

    How many tensors actually loaded is reported through ``log``. A checkpoint that
    matches on names but not on architecture loads a fraction of its tensors and
    raises no exception, so the count is the only thing that distinguishes a working
    encoder from a mostly random one.
    """
    import timm

    say = log or (lambda _msg: None)

    if not checkpoint_path:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                  global_pool="avg")
        return model, model.num_features

    resolved = arch_from_checkpoint(checkpoint_path, name)
    if resolved != name:
        say(f"checkpoint name says architecture {resolved!r}, not {name!r}; using it")
    try:
        model = timm.create_model(resolved, pretrained=False, num_classes=0,
                                  global_pool="avg")
    except RuntimeError:
        say(f"timm has no model {resolved!r}; falling back to {name!r}")
        resolved = name
        model = timm.create_model(resolved, pretrained=False, num_classes=0,
                                  global_pool="avg")

    state = torch.load(checkpoint_path, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}

    wanted = model.state_dict()
    usable = {k: v for k, v in state.items()
              if k in wanted and tuple(v.shape) == tuple(wanted[k].shape)}
    missing, unexpected = model.load_state_dict(usable, strict=False)

    body = [k for k in wanted if "classifier" not in k and "fc." not in k]
    loaded = [k for k in usable if k in body]
    say(f"encoder {resolved}: loaded {len(loaded)}/{len(body)} tensors from "
        f"{Path(checkpoint_path).name}")

    if len(loaded) < 0.9 * len(body):
        raise RuntimeError(
            f"{checkpoint_path} filled only {len(loaded)} of {len(body)} encoder "
            f"tensors in {resolved}. It is weights for a different architecture. "
            f"Attach a checkpoint whose name starts with {resolved}."
        )
    return model, model.num_features


def freeze_all_but_last(backbone: nn.Module, n_blocks: int = 2) -> None:
    """Open only the final stages of the encoder to training.

    Early blocks are generic edge and texture filters. There may not be enough
    supervision here to improve them and there is certainly enough to damage them.
    Late blocks carry the semantics that a proton-density knee needs moved.
    """
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    stages = getattr(backbone, "blocks", None)
    if stages is None:
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        return
    for stage in list(stages)[-n_blocks:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True
    for attr in ("conv_head", "bn2", "norm", "norm_head"):
        module = getattr(backbone, attr, None)
        if isinstance(module, nn.Module):
            for parameter in module.parameters():
                parameter.requires_grad = True


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy against graded targets, weighted per finding.

    The weight is what keeps a report's silence from training as a confident
    negative. A study whose report never mentions synovitis still contributes to the
    synovitis head, but at roughly a third of the pull of one that names it, and the
    twelve findings are weighted independently within the same study.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
