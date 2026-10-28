"""Grouped cross-validation training, with a runtime guard and an out-of-fold audit.

Two references are measured and they do not mean the same thing.

The **holdout** is the fold's share of the report-derived targets. It covers hundreds
of studies, so it can separate a real difference from noise, and it selects the epoch
within a run and the recipe between runs. What it measures is agreement with the
report reader, which is not what the competition scores.

The **gold audit** is the expert annotation, which is what the competition scores,
but only the annotated studies falling in the held-out fold can be used and there are
about a dozen. The Hanley-McNeil standard error of an AUC near 0.85 with that many
positives is around 0.08, so a 95% interval spans roughly plus or minus 0.16 and
competitions are decided by differences an order of magnitude smaller. It is reported
and it is never allowed to arbitrate between two recipes on its own.

The gold studies stay in training, at elevated weight, because they are the only
labels read from the images rather than from text. That is exactly why the audit is
restricted to the held-out fold.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .report_labels import TARGETS


@dataclass
class TrainConfig:
    folds: int = 5
    epochs_frozen: int = 1
    epochs_finetune: int = 3
    batch_size: int = 4
    lr_head: float = 6e-4
    lr_backbone: float = 6e-5
    weight_decay: float = 1e-4
    unfreeze_last: int = 2
    encoder_chunk: int = 24
    gold_weight: float = 4.0
    seed: int = 2026
    workers: int = 2
    time_budget_s: float = 8.0 * 3600
    backbone: str = "efficientnet_b3"


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray,
              threshold: float = 0.5) -> Dict[str, float]:
    """Per-target AUC plus the macro mean, skipping single-class columns.

    The targets this scores are graded, not binary: silence reads 0.05, a trace or
    mild finding 0.45, a plainly stated one 0.80, a severe one 1.0. ROC AUC is
    defined against a binary label, so the grade has to be cut somewhere, and 0.5
    is the cut that matches how the challenge reference standard was built. Two MSK
    radiologists graded against explicit severity thresholds and called borderline
    studies negative, so a "trace effusion" belongs on the negative side here too.

    The cut applies only to the label. Predictions stay continuous, which is what
    the metric ranks, so nothing about the model's output is discretised.
    """
    from sklearn.metrics import roc_auc_score

    scores: Dict[str, float] = {}
    for i, target in enumerate(TARGETS):
        y = y_true[:, i]
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue
        labels = (y[finite] >= threshold).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        scores[target] = float(roc_auc_score(labels, y_pred[finite, i]))
    scores["macro"] = float(np.mean([v for k, v in scores.items() if k != "macro"])) \
        if scores else float("nan")
    return scores


def build_targets(train: pd.DataFrame, config: TrainConfig):
    """Assemble the target and weight matrices the loss reads.

    Where an expert label exists it replaces the report-derived one outright and its
    weight is raised: it is the only supervision drawn from the images, and the whole
    point of the report reader is to approximate it.
    """
    from .report_labels import extract_frame

    derived = extract_frame(train["Report"].fillna(""), index=train.index)
    targets = derived[list(TARGETS)].to_numpy(dtype=np.float32)
    weights = derived[[t + "__w" for t in TARGETS]].to_numpy(dtype=np.float32)

    gold = train[list(TARGETS)].to_numpy(dtype=np.float32)
    has_gold = np.isfinite(gold)
    targets = np.where(has_gold, np.nan_to_num(gold), targets)
    weights = np.where(has_gold, config.gold_weight, weights)
    return targets, weights, has_gold.any(axis=1)


def train_fold(
    fold: int,
    bags_train,
    bags_valid,
    targets: np.ndarray,
    weights: np.ndarray,
    train_rows: Sequence[int],
    valid_rows: Sequence[int],
    config: TrainConfig,
    backbone_checkpoint: Optional[str] = None,
    out_dir: Path = Path("."),
    deadline: Optional[float] = None,
    log=print,
):
    """Fit one fold and return its validation predictions."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from .data import torch_dataset
    from .model import StudyModel, build_backbone, freeze_all_but_last, weighted_bce

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed + fold)

    backbone, feature_dim = build_backbone(
        config.backbone, pretrained=backbone_checkpoint is None,
        checkpoint_path=backbone_checkpoint, log=log,
    )
    from .dicom_io import N_SLOT

    model = StudyModel(backbone, feature_dim, N_SLOT).to(device)

    train_set = Subset(torch_dataset(bags_train, targets, weights, seed=config.seed),
                       list(train_rows))
    valid_set = Subset(torch_dataset(bags_valid, targets, weights, seed=0),
                       list(valid_rows))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.workers, pin_memory=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = {"macro": -1.0}
    best_path = out_dir / f"fold{fold}.pt"
    predictions = np.full((len(valid_rows), len(TARGETS)), 0.5, dtype=np.float32)

    total_epochs = config.epochs_frozen + config.epochs_finetune
    for epoch in range(total_epochs):
        finetuning = epoch >= config.epochs_frozen
        if epoch == 0:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        if epoch == config.epochs_frozen:
            freeze_all_but_last(model.backbone, config.unfreeze_last)

        # The head starts random and has everything to learn; the encoder starts
        # from a good solution and needs only to be moved off it. One learning rate
        # for both either leaves the head untrained or destroys the encoder inside
        # the first few hundred steps.
        groups = [{"params": [p for p in model.head.parameters() if p.requires_grad],
                   "lr": config.lr_head}]
        encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": config.lr_backbone})
        optimiser = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=[g["lr"] for g in groups],
            total_steps=max(1, len(train_loader)), pct_start=0.15,
        )

        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if deadline is not None and time.time() > deadline:
                log(f"fold {fold}: runtime budget reached during epoch {epoch}")
                break
            images = batch["images"].to(device, non_blocking=True)
            slot_index = batch["slot_index"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, slot_index, mask, chunk=config.encoder_chunk)
                loss = weighted_bce(logits, target, weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()

            running += loss.detach().item() * images.shape[0]
            seen += images.shape[0]
            if step % 50 == 0:
                log(f"fold {fold} epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {running / max(seen, 1):.4f}"
                    f"{' (finetuning)' if finetuning else ''}")

        fold_predictions = predict(model, valid_loader, device, config, log=log)
        scores = macro_auc(targets[list(valid_rows)], fold_predictions)
        log(f"fold {fold} epoch {epoch}: holdout macro {scores.get('macro', float('nan')):.4f}")

        if scores.get("macro", -1.0) > best["macro"]:
            best = scores
            predictions = fold_predictions
            torch.save({"model": model.state_dict(),
                        "config": asdict(config),
                        "scores": scores}, best_path)

        if deadline is not None and time.time() > deadline:
            break

    return predictions, best, best_path


def predict(model, loader, device, config: TrainConfig, log=None) -> np.ndarray:
    """Average the logits over every view of every slot, then map to probabilities.

    Slower than a training pass despite needing no gradients, because training draws
    one view per slot and this reads all of them. That is four times the bytes per
    study, off a network mount, so a validation pass over a fold takes minutes.
    It reports progress for that reason: a silent ten-minute gap in the log is
    indistinguishable from a hang.
    """
    import torch

    say = log or (lambda _msg: None)
    model.eval()
    outputs: List[np.ndarray] = []
    total = len(loader)
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step % 50 == 0:
                say(f"  validating {step}/{total}")
            images = batch["images"].to(device, non_blocking=True)
            slot_index = batch["slot_index"].to(device)
            mask = batch["mask"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, slot_index, mask, chunk=config.encoder_chunk)
            outputs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def audit_gold(train: pd.DataFrame, oof: np.ndarray, log=print) -> pd.DataFrame:
    """Per-target AUC of the out-of-fold predictions against the expert labels.

    Reported with a bootstrap interval so that the interval, rather than the point
    estimate, is what gets read. On this many studies the interval is wide enough
    that most recipe comparisons land inside it.
    """
    from sklearn.metrics import roc_auc_score

    gold_mask = train[list(TARGETS)].notna().any(axis=1).to_numpy()
    gold = train.loc[gold_mask, list(TARGETS)].to_numpy(dtype=float)
    predicted = oof[gold_mask]

    rng = np.random.default_rng(0)
    rows = []
    for i, target in enumerate(TARGETS):
        y, p = gold[:, i], predicted[:, i]
        keep = np.isfinite(y)
        if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
            rows.append((target, np.nan, np.nan, np.nan, int(np.nansum(y))))
            continue
        y, p = y[keep].astype(int), p[keep]
        point = roc_auc_score(y, p)
        samples = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) > 1:
                samples.append(roc_auc_score(y[idx], p[idx]))
        rows.append((target, point, float(np.percentile(samples, 2.5)),
                     float(np.percentile(samples, 97.5)), int(y.sum())))

    frame = pd.DataFrame(rows, columns=["target", "auc", "lo", "hi", "n_pos"])
    log(frame.round(3).to_string(index=False))
    log(f"gold macro AUC: {frame.auc.mean():.4f}  (n = {int(gold_mask.sum())} studies)")
    return frame
