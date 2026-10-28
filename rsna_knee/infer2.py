"""v2 inference: every checkpoint family, folds averaged in probability, families on ranks."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .paths import TEMP
from .report_labels import TARGETS
from .targets import file_digest

CKPT_RE = re.compile(r"^v2_(?P<tag>.+?)_(?P<part>fold\d+|full\d*)\.pt$")


def rezip_dir(d: Path, out_dir: Path = TEMP / "rezipped") -> Path:
    """A .pt file unpacked by an archive tool becomes a folder; pack it back.

    The archive is always rebuilt, to a temporary name first, so an output left by an
    earlier run is never trusted.
    """
    import os
    import zipfile
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (d.name + ".pt")
    tmp = out_dir / (d.name + ".pt.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, arcname=f"{d.name}/{f.relative_to(d).as_posix()}")
    os.replace(tmp, target)
    return target


def content_digest(p: Path) -> str:
    """SHA-256 over the members of a checkpoint, packed (.pt zip) or unpacked (folder).

    The first path component (the archive's own folder name) is dropped and members are
    taken in sorted order, so a checkpoint and the folder an archive tool made from it
    give the same digest, while container details such as zip timestamps do not count.
    A .pt file that is not a zip archive is hashed as raw bytes.
    """
    import hashlib
    import zipfile
    h = hashlib.sha256()
    p = Path(p)
    if p.is_dir():
        members = sorted((f.relative_to(p).as_posix(), f) for f in p.rglob("*") if f.is_file())
        for name, f in members:
            h.update(name.encode() + b"\0")
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            h.update(b"\0")
        return "dir:" + h.hexdigest()
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            infos = sorted((i.filename.split("/", 1)[-1], i) for i in z.infolist() if not i.is_dir())
            for name, info in infos:
                h.update(name.encode() + b"\0")
                with z.open(info) as fh:
                    for block in iter(lambda: fh.read(1 << 20), b""):
                        h.update(block)
                h.update(b"\0")
        return "dir:" + h.hexdigest()
    return "raw:" + file_digest(p)


def find_checkpoints(walk, rezip_to: Optional[Path] = None) -> Dict[str, List[Path]]:
    """Group mounted v2 checkpoints by family tag, one checkpoint per file name.

    A checkpoint may be mounted as a .pt file or as the folder an archive tool unpacked
    it into (the folder holds data.pkl). Every copy of one name is collected first and
    compared by content, whatever order the walk returns them in: identical copies are
    fine, different contents under one name raise.
    """
    copies: Dict[str, List[Path]] = defaultdict(list)
    for p in walk(suffixes=(".pt", ".pkl")):
        if p.name == "data.pkl":
            name, src = p.parent.name + ".pt", p.parent
        else:
            name, src = p.name, p
        if CKPT_RE.match(name):
            copies[name].append(src)
    chosen = {}
    for name, srcs in sorted(copies.items()):
        digests = {content_digest(s): s for s in srcs}
        if len(digests) > 1:
            raise ValueError(f"different checkpoints named {name}: {[str(s) for s in srcs]}")
        files = [s for s in srcs if s.is_file()]
        chosen[name] = files[0] if files else rezip_dir(srcs[0], *([rezip_to] if rezip_to else []))
    fams: Dict[str, List[Path]] = defaultdict(list)
    for name in sorted(chosen):
        fams[CKPT_RE.match(name).group("tag")].append(chosen[name])
    return dict(fams)


def check_folds(fams: Dict[str, List[Path]], n_folds: int) -> None:
    """Every family must hold fold0 ... fold<n_folds-1>; a missing fold raises."""
    for tag, paths in fams.items():
        have = sorted(int(m.group("part")[4:]) for m in (CKPT_RE.match(p.name) for p in paths)
                      if m.group("part").startswith("fold"))
        if have != list(range(n_folds)):
            raise ValueError(f"family {tag} has folds {have}, expected 0 to {n_folds - 1}")


def predict_family(paths: Sequence[Path], array, mask, batch_size: int = 2,
                   workers: int = 2, log=print) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from .train2 import TrainConfigV2, build_model, make_dataset, predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = np.arange(len(array))
    acc = None
    for p in paths:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        cfg = TrainConfigV2(**{k: v for k, v in state["config"].items()
                               if k in TrainConfigV2.__dataclass_fields__})
        cfg.encoder_weights = None
        model = build_model(cfg, pretrained=False, log=lambda *_: None)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in state["model"].items()})
        model.to(device)
        net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        ds = make_dataset(array, mask, rows, None, None, cfg, train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True)
        pred = predict(net, loader, device)
        acc = pred if acc is None else acc + pred
        log(f"  {p.name}: {pred.shape[0]} studies")
        del model, net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return acc / max(1, len(paths))


def rank_blend(members: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    import pandas as pd
    weights = weights or {}
    total, acc = 0.0, None
    for name, m in members.items():
        w = float(weights.get(name, 1.0))
        r = pd.DataFrame(m).rank(pct=True).to_numpy()
        acc = w * r if acc is None else acc + w * r
        total += w
    return acc / total


# fields every fold of one family must share; seed and the fold itself may differ
FAMILY_KEYS = ("encoder", "cache", "img", "group", "slot_prior", "target_scheme", "eval_windows", "tag")


def family_cache(paths: Sequence[Path]) -> str:
    """Cache version of a family, after checking that its folds belong together.

    Every fold must share encoder, cache, crop, window mode, slot prior, target scheme,
    inference windows, family tag and the output order of the findings.
    """
    import torch
    if not paths:
        raise ValueError("empty checkpoint family")
    ref = ref_targets = None
    for p in paths:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
        cfg = state["config"]
        sig = tuple(cfg.get(k) for k in FAMILY_KEYS)
        targets = tuple(state.get("targets", TARGETS))
        if targets != tuple(TARGETS):
            raise ValueError(f"{p.name} was trained with finding order {targets}")
        if ref is None:
            ref, ref_targets = sig, targets
        elif sig != ref:
            raise ValueError(f"{p.name} differs from its family in {dict(zip(FAMILY_KEYS, sig))} "
                             f"vs {dict(zip(FAMILY_KEYS, ref))}")
    return ref[1] or "v1"
