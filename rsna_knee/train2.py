"""v2 training: full fine-tune, one cosine schedule, weight averaging of the last epochs.

Note: runs before the release of this file drew windows, crops and jitter from an unseeded
generator; the per-(seed, epoch, study) streams below were added for the paper release.

What changed from v1 and why:
  v1 trained 1 frozen + 3 partly unfrozen epochs and never called its augmentation.
  v2 opens the whole encoder from the first step (head at 10x the encoder rate),
  runs one warm-up + cosine schedule over all epochs, samples random windows and a
  random crop per study (224 px of the 256 px v3 cache in the reported run, 256 px of
  the 288 px v1 cache before it), and averages the weights of the last epochs.

Two numbers are logged per epoch on the held-out fold:
  explicit  AUC against cells the report stated outright (p / n), teacher-free
  gold      AUC against the expert labels of the gold studies in this fold
            (about a dozen studies, so read it as a sanity check only)
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .report_labels import TARGETS


@dataclass
class TrainConfigV2:
    # defaults are those of the reported run (family r50v3)
    encoder: str = "rad_resnet50"      # or a timm name, e.g. convnext_tiny.fb_in22k_ft_in1k
    encoder_weights: Optional[str] = None
    epochs: int = 8
    batch_size: int = 2                # studies per step (split over GPUs)
    accum: int = 4
    lr_encoder: float = 2e-4
    lr_head: float = 2e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.08
    train_windows: int = 4             # windows per slot per step
    img: int = 224                     # crop: 224 of the 256 px v3 cache, 256 of the 288 px v1 cache
    swa_last: int = 3                  # average weights of the last N epochs
    workers: int = 3
    seed: int = 2026
    chunk: int = 64
    time_budget_s: float = 11.0 * 3600
    target_scheme: str = "flat"
    tag: str = "r50v3"
    cache: str = "v3"                  # v1: 288 px, 12 slices, 130 mm   v3: 256 px, 16 slices, 140 mm
    group: int = 3                     # 3: 2.5D windows, 1: single slices
    eval_windows: int = 16             # single-slice mode: slices used at inference
    slot_prior: bool = True            # radiological finding-to-slot prior on the head


def auc_table(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, t in enumerate(TARGETS):
        keep = np.isfinite(y[:, j])
        yy = y[keep, j]
        if keep.sum() < 4 or len(np.unique(yy)) < 2:
            continue
        out[t] = float(roc_auc_score(yy.astype(int), p[keep, j]))
    out["macro"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def make_dataset(array, mask, rows, targets, weights, cfg: TrainConfigV2, train: bool):
    import torch
    from torch.utils.data import Dataset
    from .model2 import crop, jitter, sample_starts, windows_from_row

    rows = np.asarray(rows)

    class _DS(Dataset):
        epoch = 0                      # set by the training loop before each epoch

        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = int(rows[i])
            # one stream per (seed, epoch, study): windows, crop and jitter are reproducible
            rng = np.random.default_rng((cfg.seed, self.epoch, r)) if train else np.random.default_rng(0)
            row = array[r]  # memmap view; only the sampled slices are read below
            starts = sample_starts(train, cfg.train_windows if train else cfg.eval_windows,
                                   rng, row.shape[1], cfg.group)
            x = crop(windows_from_row(row, starts, cfg.group), cfg.img, train, rng)
            if train:
                x = jitter(x, rng)
            item = {
                "x": torch.from_numpy(np.ascontiguousarray(x)),
                "present": torch.from_numpy(mask[r].astype(np.float32)),
                "row": r,
            }
            if targets is not None:
                item["y"] = torch.from_numpy(targets[r].astype(np.float32))
                item["w"] = torch.from_numpy(weights[r].astype(np.float32))
            return item

    return _DS()


def build_model(cfg: TrainConfigV2, pretrained: bool = True, log=print):
    from .model2 import StudyNetV2
    return StudyNetV2(cfg.encoder, cfg.encoder_weights, pretrained=pretrained,
                      chunk=cfg.chunk, slot_prior=cfg.slot_prior, log=log)


def predict(model, loader, device) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(b["x"].to(device, non_blocking=True), b["present"].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def train_one(
    cfg: TrainConfigV2,
    array, mask,
    train_rows: Sequence[int],
    valid_rows: Sequence[int],
    targets: np.ndarray, weights: np.ndarray,
    explicit: Optional[np.ndarray],
    gold: Optional[np.ndarray],
    out_path: Path,
    deadline: Optional[float] = None,
    log=print,
):
    """Fit one model. ``valid_rows`` may be empty (full-data run)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import random
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, pretrained=cfg.encoder_weights is None, log=log).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": cfg.lr_encoder},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    if torch.cuda.device_count() > 1:
        log(f"DataParallel over {torch.cuda.device_count()} GPUs")

    tr = make_dataset(array, mask, train_rows, targets, weights, cfg, train=True)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    ltr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                     num_workers=cfg.workers, pin_memory=True, generator=gen)
    lva = None
    if len(valid_rows):
        va = make_dataset(array, mask, valid_rows, None, None, cfg, train=False)
        lva = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.workers, pin_memory=True)

    steps = max(1, len(ltr) // cfg.accum) * cfg.epochs
    warm = max(1, int(steps * cfg.warmup_frac))

    def lr_mult(s):
        if s < warm:
            return (s + 1) / warm
        t = min(1.0, (s - warm) / max(1, steps - warm))
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    swa_state, swa_n = None, 0
    history: List[Dict] = []
    preds = None
    stopped = False
    for ep in range(cfg.epochs):
        tr.epoch = ep
        net.train()
        opt.zero_grad(set_to_none=True)
        run, seen, t0 = 0.0, 0, time.time()
        t_data, t_mark = 0.0, time.time()
        for bi, b in enumerate(ltr):
            t_data += time.time() - t_mark
            if deadline is not None and time.time() > deadline:
                log(f"time budget reached in epoch {ep}")
                stopped = True
                break
            x = b["x"].to(device, non_blocking=True)
            pr = b["present"].to(device)
            y = b["y"].to(device)
            w = b["w"].to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = net(x, pr)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            loss = (loss * w).sum() / w.sum().clamp_min(1.0)
            scaler.scale(loss / cfg.accum).backward()
            if (bi + 1) % cfg.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += float(loss.detach()) * len(x)
            seen += len(x)
            if bi % 200 == 0:
                log(f"  ep {ep} step {bi}/{len(ltr)} loss {run / max(seen, 1):.4f} "
                    f"{(time.time() - t0) / max(bi, 1):.2f}s/it "
                    f"(waiting for data {t_data / max(bi, 1):.2f}s/it)")
            t_mark = time.time()
        if stopped:
            break

        # running average of the last swa_last epochs
        if ep >= cfg.epochs - cfg.swa_last:
            sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = sd, 1
            else:
                swa_n += 1
                for k in swa_state:
                    if swa_state[k].is_floating_point():
                        swa_state[k] += (sd[k] - swa_state[k]) / swa_n
                    else:
                        swa_state[k] = sd[k]

        rec = {"epoch": ep, "loss": run / max(seen, 1), "min": (time.time() - t0) / 60}
        if lva is not None:
            p = predict(model, lva, device)
            vr = np.asarray(valid_rows)
            if explicit is not None:
                rec["explicit"] = auc_table(explicit[vr], p)["macro"]
            if gold is not None and np.isfinite(gold[vr]).any():
                rec["gold"] = auc_table(gold[vr], p)["macro"]
            preds = p
        history.append(rec)
        # keep the latest weights on disk so a killed session still leaves a model
        torch.save({"model": {k: v.half() if v.is_floating_point() else v
                              for k, v in model.state_dict().items()},
                    "config": asdict(cfg), "targets": list(TARGETS), "history": history}, out_path)
        log(f"epoch {ep}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                          for k, v in rec.items()}))

    if swa_state is not None and swa_n > 1:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in swa_state.items()})
        log(f"loaded average of last {swa_n} epochs")
        if lva is not None:
            preds = predict(model, lva, device)
            vr = np.asarray(valid_rows)
            msg = {}
            if explicit is not None:
                msg["explicit"] = auc_table(explicit[vr], preds)["macro"]
            if gold is not None and np.isfinite(gold[vr]).any():
                msg["gold"] = auc_table(gold[vr], preds)["macro"]
            log(f"averaged weights: {msg}")

    torch.save({"model": {k: v.half() if v.is_floating_point() else v
                          for k, v in model.state_dict().items()},
                "config": asdict(cfg), "targets": list(TARGETS), "history": history}, out_path)
    log(f"saved {out_path}")
    del net, model, opt, ltr, lva, swa_state
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, history


def write_oof(train: pd.DataFrame, oof: np.ndarray, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(oof, columns=list(TARGETS))
    df.insert(0, "StudyInstanceUID", train["StudyInstanceUID"].values)
    df = df[np.isfinite(oof).all(axis=1)]
    df.to_csv(path, index=False)
    return df
