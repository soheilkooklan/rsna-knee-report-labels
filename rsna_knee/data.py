"""Cache construction, fold assignment and the bag Dataset.

Reading dominates the cost of this pipeline. A study holds several series and each
series tens of slices, so decoding a study touches on the order of a hundred and
fifty files. That is affordable once and not affordable once per epoch, so the slots
are decoded a single time into a ``uint8`` memmap and every epoch reads from that.

    bytes = n_study x n_slot x n_slice x px^2

The exponent on ``px`` is what makes this a constraint rather than a detail: the
cache grows with the *square* of resolution and only linearly with slice count, so
coverage is the cheap axis and resolution the expensive one.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .dicom_io import N_SLOT, SLOTS
from .report_labels import TARGETS


@dataclass
class CacheConfig:
    px: int = 288
    slices_per_slot: int = 12       # kept per slot; read as groups of 3 channels
    group: int = 3                  # adjacent slices stacked as channels
    crop_mm: float = 130.0          # below the smallest acquired field of view here
    band: Tuple[float, float] = (0.12, 0.88)

    @property
    def views(self) -> int:
        return self.slices_per_slot // self.group

    def tag(self) -> str:
        """A name that includes everything deciding the pixels, not only their shape.

        Two configurations can agree on resolution, slice count and crop and still
        produce different arrays of identical shape. A tag built from the dimensions
        alone lets the second attach to the first one's file and train against pixels
        it never asked for, with nothing anywhere reporting a mismatch.
        """
        payload = json.dumps(
            {"px": self.px, "slices": self.slices_per_slot, "group": self.group,
             "crop_mm": self.crop_mm, "band": list(self.band)},
            sort_keys=True,
        )
        return f"{self.px}px_{self.slices_per_slot}sl_{int(self.crop_mm)}mm_" \
               f"{hashlib.md5(payload.encode()).hexdigest()[:6]}"


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------

def report_group(report: str) -> str:
    """Group key that keeps byte-identical reports together.

    Some reports repeat verbatim across studies, being a template read for an
    unremarkable knee, so every study in such a group receives the same derived
    target vector. Splitting the group across the fold boundary then scores the
    model on a target whose source it trained on.
    """
    text = (report or "").strip().lower()
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:16]


def site_group(series_rows: pd.DataFrame) -> pd.Series:
    """A site fingerprint per study, from acquisition metadata.

    Studies from one site share a scanner, a protocol and a population, and a network
    learns to recognise the site long before it learns to recognise a meniscal tear.
    Splitting a site across the fold boundary therefore measures site recall and
    reads as skill. Measured on this corpus by others, random K-fold runs roughly
    0.05 AUC above grouped, and the gap widens once the model is good enough to use
    the shortcut.
    """
    key = (
        series_rows.groupby("StudyInstanceUID")
        .apply(lambda g: "|".join(sorted(set(
            f"{r.Anatomical_Plane}:{int(bool(r.Fluid_Sensitive))}{int(bool(r.Fat_Suppression))}"
            for r in g.itertuples()
        ))), include_groups=False)
    )
    return key.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])


def assign_folds(train: pd.DataFrame, n_folds: int = 5, seed: int = 2026) -> pd.Series:
    """Fold index per study, grouped by report and balanced on the gold subset.

    The gold studies are spread across folds deliberately. They are the only labels
    read from the images rather than from text, so a fold holding none of them cannot
    be checked against the thing the competition actually scores.
    """
    rng = np.random.default_rng(seed)
    groups = train["Report"].fillna("").map(report_group)
    is_gold = train[list(TARGETS)].notna().any(axis=1).values

    order = np.argsort(~is_gold, kind="stable")  # gold first, then the rest
    assignment: Dict[str, int] = {}
    counts = np.zeros(n_folds, dtype=int)
    fold = np.empty(len(train), dtype=int)

    for i in order:
        key = groups.iloc[i]
        if key not in assignment:
            smallest = np.flatnonzero(counts == counts.min())
            assignment[key] = int(rng.choice(smallest))
        fold[i] = assignment[key]
        counts[assignment[key]] += 1

    return pd.Series(fold, index=train.index, name="fold")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def shard_bounds(n_studies: int, shard: int, n_shards: int) -> Tuple[int, int]:
    """Half-open study range for one shard, splitting as evenly as possible."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} outside 0..{n_shards - 1}")
    base, extra = divmod(n_studies, n_shards)
    start = shard * base + min(shard, extra)
    return start, start + base + (1 if shard < extra else 0)


def build_cache(
    root: Path,
    series_csv: pd.DataFrame,
    studies: Sequence[str],
    split: str,
    config: CacheConfig,
    out_dir: Path,
    log=print,
    shard: int = 0,
    n_shards: int = 1,
) -> Tuple[np.memmap, np.ndarray, List[str]]:
    """Decode every (study, slot) once into a ``uint8`` memmap on disk.

    Returns the array, a ``[n_study, n_slot]`` presence mask and the study order.
    Decode failures are counted and reported: unreported, a failure is
    indistinguishable from a black knee.

    Kaggle caps ``/kaggle/working`` at 20 GiB, because anything written there can be
    saved permanently with the notebook version. The full training cache at 288 px
    is 24.5 GiB, so it has to be built in shards: one notebook run per shard, each
    saved as its own Dataset, and stage 2 mounts all of them. ``shard`` selects a
    contiguous range of studies and appears in the file name, so two shards cannot
    silently overwrite each other.
    """
    from .dicom_io import laterality, order_slices, pick_slots, read_slot, scan_series

    out_dir.mkdir(parents=True, exist_ok=True)
    studies = list(studies)
    if n_shards > 1:
        start, stop = shard_bounds(len(studies), shard, n_shards)
        studies = studies[start:stop]
        log(f"shard {shard} of {n_shards}: studies {start}..{stop - 1}")
    suffix = "" if n_shards == 1 else f"_s{shard}of{n_shards}"

    shape = (len(studies), N_SLOT, config.slices_per_slot, config.px, config.px)
    gib = np.prod(shape) / 1024 ** 3
    if gib > 19.0:
        log(f"WARNING: this shard is {gib:.1f} GiB and /kaggle/working holds 20 GiB. "
            f"Raise n_shards.")
    path = out_dir / f"{split}_{config.tag()}{suffix}.npy"
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    mask = np.zeros((len(studies), N_SLOT), dtype=np.float32)

    by_study = {uid: g for uid, g in series_csv.groupby("StudyInstanceUID")}
    plane_of = dict(zip(series_csv.SeriesInstanceUID, series_csv.Anatomical_Plane))
    fatsat_of = dict(zip(series_csv.SeriesInstanceUID,
                         series_csv.Fat_Suppression.astype(bool)))
    failures = 0

    for row, uid in enumerate(studies):
        group = by_study.get(uid)
        if group is None:
            continue
        rows = []
        for record in group.itertuples():
            series_dir = root / f"{split}_series" / uid / record.SeriesInstanceUID
            try:
                n = len(list(series_dir.iterdir())) if series_dir.is_dir() else 0
            except OSError:
                n = 0
            rows.append({"SeriesInstanceUID": record.SeriesInstanceUID, "n_slices": n})

        chosen = pick_slots(rows, plane_of, fatsat_of)
        for slot_i, (name, plane, _fs) in enumerate(SLOTS):
            series_uid = chosen.get(name)
            if series_uid is None:
                continue
            series_dir = root / f"{split}_series" / uid / series_uid
            # One unreadable slot costs one slot. Letting it propagate costs the
            # whole run, and a failed Kaggle version saves no usable output, so the
            # hour already spent decoding is lost with it.
            try:
                headers = scan_series(series_dir)
                if not headers:
                    failures += 1
                    continue
                ordered = order_slices(headers)
                volume = read_slot(
                    [h["path"] for h in ordered], config.slices_per_slot,
                    config.px, config.crop_mm, plane, laterality(headers),
                )
            except Exception as exc:
                failures += 1
                if failures <= 20:
                    log(f"  slot {name} of {uid} failed: {type(exc).__name__}: {exc}")
                continue
            if volume is None:
                failures += 1
                continue
            array[row, slot_i] = volume
            mask[row, slot_i] = 1.0

        if row % 200 == 0:
            log(f"cache {split}: {row}/{len(studies)} studies, {failures} slot failures")

    array.flush()
    log(f"cache {split}: done, {failures} slot failures, "
        f"{array.nbytes / 1024 ** 3:.1f} GB at {path}")
    np.save(out_dir / f"{split}_{config.tag()}{suffix}_mask.npy", mask)
    (out_dir / f"{split}_{config.tag()}{suffix}_studies.json").write_text(
        json.dumps(studies))
    return array, mask, studies


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class StudyBags:
    """Turn the cache into per-study bags of ``(3, px, px)`` instances.

    Each slot contributes ``views`` instances, one per group of three adjacent
    slices. In training a single view per slot is drawn, which doubles as
    augmentation along the stack; at inference every view is used and the logits are
    averaged, which is a test-time ensemble that costs nothing to train.
    """

    def __init__(self, array, mask, config: CacheConfig, train: bool = True):
        self.array = array
        self.mask = mask
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return self.array.shape[0]

    def bag(self, row: int, rng: Optional[np.random.Generator] = None):
        cfg = self.config
        views = range(cfg.views) if not self.train else [
            int((rng or np.random.default_rng()).integers(cfg.views))
        ]
        images, slots, present = [], [], []
        for slot_i in range(N_SLOT):
            for view in views:
                lo = view * cfg.group
                images.append(self.array[row, slot_i, lo:lo + cfg.group])
                slots.append(slot_i)
                present.append(self.mask[row, slot_i])
        return (
            np.stack(images).astype(np.uint8),
            np.asarray(slots, dtype=np.int64),
            np.asarray(present, dtype=np.float32),
        )


def torch_dataset(bags: "StudyBags", targets: Optional[np.ndarray],
                  weights: Optional[np.ndarray], seed: int = 0):
    """Wrap :class:`StudyBags` as a ``torch.utils.data.Dataset``."""
    import torch
    from torch.utils.data import Dataset

    class _Dataset(Dataset):
        def __len__(self):
            return len(bags)

        def __getitem__(self, index):
            rng = np.random.default_rng(seed + index)
            images, slots, mask = bags.bag(index, rng)
            item = {
                "images": torch.from_numpy(images),
                "slot_index": torch.from_numpy(slots),
                "mask": torch.from_numpy(mask),
                "index": index,
            }
            if targets is not None:
                item["target"] = torch.from_numpy(targets[index].astype(np.float32))
                item["weight"] = torch.from_numpy(weights[index].astype(np.float32))
            return item

    return _Dataset()


def augment(images: "np.ndarray", rng: np.random.Generator) -> "np.ndarray":
    """Small rigid jitter and an intensity scale.

    Neither flip is available, for different reasons. A horizontal flip would
    reintroduce the axis that the laterality normalisation removed. It would undo,
    once per batch, what the header pass was run to establish. A vertical flip is not
    a nuisance axis at all: a knee is acquired in a canonical orientation, and where
    a finding sits in the frame is information rather than noise, since a Baker cyst
    is identified by lying in the popliteal fossa. What remains is jitter that no
    label depends on.
    """
    import cv2

    angle = float(rng.uniform(-8, 8))
    scale = float(rng.uniform(0.94, 1.06))
    tx, ty = (float(rng.uniform(-0.04, 0.04)) for _ in range(2))
    gain = float(rng.uniform(0.9, 1.1))

    out = np.empty_like(images)
    h, w = images.shape[-2:]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += tx * w
    matrix[1, 2] += ty * h
    for i in range(images.shape[0]):
        for c in range(images.shape[1]):
            out[i, c] = cv2.warpAffine(images[i, c], matrix, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


class ShardedArray:
    """Present several shard memmaps as one array indexed by global study row.

    Stage 1 writes the cache in shards because Kaggle caps ``/kaggle/working`` at
    20 GiB. Stage 2 needs the original study order back. Shards are contiguous and
    non-overlapping by construction (see :func:`shard_bounds`), so a global row maps
    to exactly one shard and a local offset.

    Only the indexing that :class:`StudyBags` performs is supported:
    ``array[row, slot]`` and ``array[row, slot, a:b]``.
    """

    def __init__(self, shards: Sequence[np.ndarray]):
        if not shards:
            raise ValueError("ShardedArray needs at least one shard")
        tail = shards[0].shape[1:]
        for i, part in enumerate(shards):
            if part.shape[1:] != tail:
                raise ValueError(
                    f"shard {i} has trailing shape {part.shape[1:]}, expected {tail}. "
                    "The shards were built with different CacheConfig settings."
                )
        self.shards = list(shards)
        self.offsets = np.cumsum([0] + [len(p) for p in self.shards])
        self.shape = (int(self.offsets[-1]),) + tuple(tail)

    def _locate(self, row: int):
        if not 0 <= row < self.shape[0]:
            raise IndexError(f"row {row} outside 0..{self.shape[0] - 1}")
        which = int(np.searchsorted(self.offsets, row, side="right") - 1)
        return self.shards[which], row - int(self.offsets[which])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            part, local = self._locate(int(key))
            return part[local]
        part, local = self._locate(int(key[0]))
        return part[(local,) + key[1:]]

    @property
    def nbytes(self) -> int:
        return sum(int(p.nbytes) for p in self.shards)


def load_cache(cache_dirs: Sequence[Path], split: str, config: CacheConfig):
    """Load every shard under ``cache_dirs`` and return (array, mask, studies).

    Shards are ordered by their shard index, which is how stage 1 partitioned the
    study list, so concatenating them restores the original order that
    ``train_folds.csv`` was written in. Stage 2 asserts that alignment rather than
    assuming it.

    A missing shard raises. Silently training on part of the corpus while the fold
    assignment still refers to all of it would misalign every label.
    """
    found = {}
    for directory in cache_dirs:
        for npy in sorted(Path(directory).rglob(f"{split}_{config.tag()}*.npy")):
            if npy.name.endswith("_mask.npy"):
                continue
            found[npy.name[:-len(".npy")]] = npy

    if not found:
        raise FileNotFoundError(
            f"no cache shard matching {split}_{config.tag()}*.npy under "
            f"{[str(d) for d in cache_dirs]}. Check that the CacheConfig here matches "
            f"the one stage 1 used; the tag encodes every setting that decides the "
            f"pixels."
        )

    def shard_index(stem: str) -> int:
        m = re.search(r"_s(\d+)of(\d+)$", stem)
        return int(m.group(1)) if m else 0

    expected = None
    for stem in found:
        m = re.search(r"_s(\d+)of(\d+)$", stem)
        if m:
            expected = int(m.group(2))
    if expected is not None and len(found) != expected:
        raise FileNotFoundError(
            f"found {len(found)} shard(s) but the file names say there are {expected}: "
            f"{sorted(found)}. Attach every shard dataset."
        )

    arrays, masks, studies = [], [], []
    for stem in sorted(found, key=shard_index):
        npy = found[stem]
        arr = np.load(npy, mmap_mode="r")
        masks.append(np.load(npy.with_name(f"{stem}_mask.npy")))
        listing = npy.with_name(f"{stem}_studies.json")
        uids = json.loads(listing.read_text()) if listing.is_file() else [None] * len(arr)
        if len(uids) != len(arr):
            raise ValueError(f"{npy.name}: {len(arr)} rows but {len(uids)} study ids")
        arrays.append(arr)
        studies.extend(uids)

    return ShardedArray(arrays), np.concatenate(masks, axis=0), studies
