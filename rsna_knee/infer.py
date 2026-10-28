"""Inference and submission writing.

Two rules follow from the metric and are worth stating because both are easy to get
backwards.

**Combine ranks, not probabilities.** The score is a mean of per-target ROC AUCs, and
an AUC is invariant under any strictly increasing map of that target's scores.
Calibration is therefore worth nothing, and averaging raw probabilities across models
lets whichever member happens to be most confident dominate the blend. Averaging
percentile ranks combines the only information the metric reads.

**Write a valid file before doing anything else.** A submission that never appears
scores nothing at all, which is strictly worse than scoring badly. A run killed for
memory dies by SIGKILL and never reaches an exception handler, so the benchmark file
is written in the first second and overwritten only once real predictions exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .paths import INPUT
from .report_labels import TARGETS


def find_root(candidates: Optional[Iterable[Path]] = None) -> Path:
    """Locate the competition mount, wherever Kaggle attached it."""
    defaults = [
        INPUT / "rsna-knee-abnormality-detection",
        INPUT / "competitions/rsna-knee-abnormality-detection",
        Path("data"), Path("."),
    ]
    for candidate in list(candidates or []) + defaults:
        if (candidate / "test.csv").is_file():
            return candidate
    base = INPUT
    if base.is_dir():
        for first in sorted(p for p in base.iterdir() if p.is_dir()):
            for candidate in [first] + sorted(p for p in first.iterdir() if p.is_dir()):
                if (candidate / "test.csv").is_file():
                    return candidate
    raise FileNotFoundError("competition mount not found: no directory holds test.csv")


# Directory names that hold the DICOM tree. The training mount alone is 24,371
# series of roughly a hundred slices, so a plain rglob over /kaggle/input walks on
# the order of two million files on a network mount to find a handful of .npy and
# .pth files. Measured cost of not pruning: about 400 seconds per lookup.
PRUNE_DIR_SUFFIXES = ("_series", "_images")


def walk_inputs(root: Path = INPUT, max_depth: int = 8,
                suffixes: Optional[Sequence[str]] = None):
    """Yield files under the Kaggle mounts, skipping the image trees.

    ``max_depth`` is counted from ``root``. Weight files and caches sit within a
    few levels of a mount root; nothing worth finding is buried deeper, and the
    depth cap is what stops an unexpected deep tree from costing minutes again.
    """
    import os

    root = Path(root)
    if not root.is_dir():
        return
    wanted = {s.lower() for s in suffixes} if suffixes else None
    base_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if len(here.parts) - base_depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [
                d for d in dirnames
                if not d.lower().endswith(PRUNE_DIR_SUFFIXES)
            ]
        for filename in filenames:
            if wanted is not None and Path(filename).suffix.lower() not in wanted:
                continue
            yield here / filename


WEIGHT_SUFFIXES = (".pth", ".pt", ".bin", ".safetensors", ".ckpt")


def find_weights(parts: Sequence[str] = ("efficientnet", "b3")) -> List[str]:
    """Weight files whose path mentions the architecture, best match first.

    Matching on the file name alone misses the common layout where the file is
    generic (``pytorch_model.bin``) and the architecture names the directory above
    it. Every candidate is returned rather than only the winner, because the usual
    failure is a mount holding a different variant, and seeing the list is what
    makes that obvious.
    """
    joined = "_".join(parts).lower()
    hits = []
    for candidate in walk_inputs(suffixes=WEIGHT_SUFFIXES):
        blob = str(candidate).lower()
        if not all(part.lower() in blob for part in parts):
            continue
        score = (joined in candidate.name.lower()) * 2 + (joined in blob)
        hits.append((score, candidate.stat().st_size, str(candidate)))
    hits.sort(reverse=True)
    return [h[2] for h in hits]


def write_benchmark(test: pd.DataFrame, path: Path = Path("submission.csv")) -> None:
    """Write the constant-0.5 file so a valid submission exists from the first second."""
    frame = test[["StudyInstanceUID"]].copy()
    for target in TARGETS:
        frame[target] = 0.5
    frame.to_csv(path, index=False)


def to_ranks(predictions: np.ndarray) -> np.ndarray:
    """Percentile rank each column independently."""
    return pd.DataFrame(predictions).rank(pct=True).to_numpy(dtype=np.float64)


def rank_blend(members: Sequence[np.ndarray],
               weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Weighted mean of per-column percentile ranks."""
    if not members:
        raise ValueError("rank_blend needs at least one member")
    weights = list(weights or [1.0] * len(members))
    if len(weights) != len(members):
        raise ValueError("weights and members disagree in length")
    total = float(sum(weights))
    stacked = sum(w * to_ranks(m) for m, w in zip(members, weights))
    return stacked / total


def write_submission(predictions: np.ndarray, studies: Sequence[str],
                     test: pd.DataFrame, path: Path = Path("submission.csv")) -> pd.DataFrame:
    """Write one submission, rank-transformed and aligned to the test index.

    Any study the pipeline failed to read falls back to 0.5 for every target rather
    than dropping out of the file, because a missing row invalidates the whole
    submission while a neutral row costs only that study.
    """
    frame = pd.DataFrame(to_ranks(predictions), columns=list(TARGETS))
    frame.insert(0, "StudyInstanceUID", list(studies))
    out = test[["StudyInstanceUID"]].merge(frame, on="StudyInstanceUID", how="left")
    out[list(TARGETS)] = out[list(TARGETS)].fillna(0.5)
    out.to_csv(path, index=False)
    return out


def validate_submission(path: Path, test: pd.DataFrame) -> None:
    """Fail loudly on a malformed file, before it is submitted rather than after."""
    frame = pd.read_csv(path)
    expected = ["StudyInstanceUID"] + list(TARGETS)
    if list(frame.columns) != expected:
        raise ValueError(f"columns are {list(frame.columns)}, expected {expected}")
    if len(frame) != len(test):
        raise ValueError(f"{len(frame)} rows, expected {len(test)}")
    if set(frame.StudyInstanceUID) != set(test.StudyInstanceUID):
        raise ValueError("StudyInstanceUID set does not match test.csv")
    values = frame[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("submission contains NaN or infinite values")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("submission values fall outside [0, 1]")


def predict_studies(checkpoints: Sequence[Path], bags, config, batch_size: int = 4,
                    log=print) -> np.ndarray:
    """Average the fold models over every view, in probability space within a fold.

    Members of one cross-validation differ only by which studies they saw, so they
    are calibrated alike and a probability mean is well defined. Different
    architectures are not, which is why the cross-model combination in
    :func:`rank_blend` is done on ranks instead.
    """
    import torch
    from torch.utils.data import DataLoader

    from .data import torch_dataset
    from .dicom_io import N_SLOT
    from .model import StudyModel, build_backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(torch_dataset(bags, None, None, seed=0),
                        batch_size=batch_size, shuffle=False, num_workers=2)

    accumulated = None
    for checkpoint_path in checkpoints:
        state = torch.load(str(checkpoint_path), map_location="cpu")
        backbone, feature_dim = build_backbone(config.backbone, pretrained=False)
        model = StudyModel(backbone, feature_dim, N_SLOT)
        model.load_state_dict(state["model"])
        model.to(device).eval()

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device, non_blocking=True)
                slot_index = batch["slot_index"].to(device)
                mask = batch["mask"].to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images, slot_index, mask, chunk=config.encoder_chunk)
                outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
        member = np.concatenate(outputs, axis=0)
        accumulated = member if accumulated is None else accumulated + member
        log(f"{checkpoint_path.name}: predicted {member.shape[0]} studies")
        del model
        torch.cuda.empty_cache()

    return accumulated / max(len(checkpoints), 1)
