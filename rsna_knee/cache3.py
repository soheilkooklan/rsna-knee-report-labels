"""Cache v3: the image format of the public NTejas-1 arms (MIT licence).

256 px, 16 slices per slot over the 6-94% band, 140 mm physical crop (row pixel spacing),
slices ordered along the patient axis with the largest position spread, slots by
plane x Fluid_Sensitive, deepest series per slot, laterality from the DICOM tag or
the median x of ImagePositionPatient when it is at least 20 mm from 0.
The decode functions are copied from cache/build_cache.py of
github.com/NTejas-1/RSNA-Knee-Abnormality-Detection so that their public
checkpoints can be run on the same pixels as our own v3 models.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

from .paths import TEMP
from .targets import file_digest

import numpy as np
import pandas as pd

CROP_MM = 140.0
SIZE = 256
N_SLICE = 16
BAND = (0.06, 0.94)
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1),
         ("Coronal", 0), ("Axial", 1), ("Axial", 0)]
N_SLOT = len(SLOTS)
THREADS = 8
ORDER = "axis"      # "axis": largest-spread patient axis (reported runs); "normal": stack normal
QA = {"slices_read": 0, "slices_failed": 0, "slots_built": 0}

try:
    import cv2

    def resize(a, n):
        return cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
except Exception:
    from PIL import Image

    def resize(a, n):
        return np.asarray(Image.fromarray(a).resize((n, n), Image.BILINEAR))


def read_header(path):
    import pydicom
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(name)))


def order_slices(series_dir):
    names = [e.name for e in os.scandir(series_dir) if e.name.endswith(".dcm")]
    if not names:
        return [], None, None
    names.sort(key=natural_key)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        hdrs = list(pool.map(lambda n: read_header(os.path.join(series_dir, n)), names))

    pos, inst, px, lat, iop = [], [], None, None, None
    for h in hdrs:
        if h is None:
            pos.append(None)
            inst.append(None)
            continue
        ipp = getattr(h, "ImagePositionPatient", None)
        pos.append([float(v) for v in ipp] if ipp is not None and len(ipp) >= 3 else None)
        if iop is None:
            o = getattr(h, "ImageOrientationPatient", None)
            if o is not None and len(o) >= 6:
                iop = np.array([float(v) for v in o[:6]])
        iv = getattr(h, "InstanceNumber", None)
        inst.append(float(iv) if iv is not None else None)
        if px is None:
            ps = getattr(h, "PixelSpacing", None)
            if ps is not None and len(ps) >= 1:
                try:
                    px = float(ps[0])
                except Exception:
                    px = None
        if lat is None:
            lv = getattr(h, "Laterality", None) or getattr(h, "ImageLaterality", None)
            if lv:
                s = str(lv).strip().upper()
                if s and s[0] in ("L", "R"):
                    lat = s[0]

    have = [p for p in pos if p is not None]
    idx = list(range(len(names)))
    if len(have) >= max(3, len(names) // 2):
        arr = np.array([p if p is not None else [np.nan] * 3 for p in pos], float)
        if ORDER == "normal" and iop is not None:
            # projection on the stack normal, oriented to increase along the dominant axis
            n = np.cross(iop[:3], iop[3:])
            n = n * np.sign(n[int(np.argmax(np.abs(n)))])
            key = arr @ n
        else:
            # used for every reported run: the patient axis with the largest spread.
            # For parallel slices IPP_k = IPP_0 + k d n, so this key is monotone in k
            # whenever n has a component on that axis (tests/test_units.py).
            spread = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
            axis = int(np.nanargmax(spread))
            key = arr[:, axis]
        key = np.where(np.isfinite(key), key, np.inf)
        idx = list(np.argsort(key, kind="stable"))
    elif any(v is not None for v in inst):
        key = np.array([v if v is not None else np.inf for v in inst], float)
        idx = list(np.argsort(key, kind="stable"))

    if lat is None and len(have) >= 3:
        mx = float(np.median([p[0] for p in have]))
        if abs(mx) >= 20.0:
            lat = "R" if mx < 0 else "L"
    return [names[i] for i in idx], px, lat


def sample_positions(n, k, band):
    if n <= 0:
        return []
    lo = int(round(band[0] * (n - 1)))
    hi = int(round(band[1] * (n - 1)))
    if hi <= lo:
        return [min(max(lo, 0), n - 1)] * k
    return [int(min(max(round(v), 0), n - 1)) for v in np.linspace(lo, hi, k)]


def load_slice(path, px, crop_mm, size):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    ds = pydicom.dcmread(path, force=True)
    a = ds.pixel_array
    try:
        a = apply_modality_lut(a, ds)
    except Exception:
        pass
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] <= 4 else a[a.shape[0] // 2]
    if px and px > 0:
        half = int(round(crop_mm / px / 2.0))
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        y0, y1 = max(0, cy - half), min(a.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(a.shape[1], cx + half)
        if y1 - y0 >= 8 and x1 - x0 >= 8:
            a = a[y0:y1, x0:x1]
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros((size, size), np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return resize((a * 255.0).astype(np.uint8), size)


def build_slot(series_dir, plane):
    names, px, lat = order_slices(series_dir)
    if not names:
        return None
    pos = sample_positions(len(names), N_SLICE, BAND)

    def one(i):
        try:
            return i, load_slice(os.path.join(series_dir, names[i]), px, CROP_MM, SIZE)
        except Exception:
            return i, None

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        got = dict(pool.map(one, sorted(set(pos))))
    QA["slices_read"] += len(got)
    QA["slices_failed"] += sum(v is None for v in got.values())
    fill = next((v for v in got.values() if v is not None), None)
    if fill is None:
        return None
    QA["slots_built"] += 1
    vol = np.stack([got.get(i) if got.get(i) is not None else fill for i in pos])
    if lat == "R":
        vol = vol[:, :, ::-1] if plane in ("Coronal", "Axial") else vol[::-1]
    return np.ascontiguousarray(vol)


def build_study(root, study, rows):
    vol = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    pres = np.zeros(N_SLOT, np.float32)
    depth = {}
    for sid in rows["SeriesInstanceUID"]:
        d = os.path.join(root, study, str(sid))
        try:
            depth[sid] = sum(1 for e in os.scandir(d) if e.name.endswith(".dcm"))
        except Exception:
            depth[sid] = 0
    for si, (plane, fluid) in enumerate(SLOTS):
        cand = rows[(rows["Anatomical_Plane"] == plane) & (rows["Fluid_Sensitive"] == fluid)]
        if not len(cand):
            continue
        sid = max(cand["SeriesInstanceUID"], key=lambda s: depth.get(s, 0))
        if depth.get(sid, 0) == 0:
            continue
        try:
            v = build_slot(os.path.join(root, study, str(sid)), plane)
        except Exception:
            v = None
        if v is not None:
            vol[si] = v
            pres[si] = 1.0
    return vol, pres


def build_split(comp_root: Path, split: str, studies: Sequence[str], out_dir: Path,
                shard: int = 0, n_shards: int = 1, log=print):
    """Decode studies[shard block] into c3_<split>_s<k>of<n>.npy plus mask and id list."""
    import time
    series = pd.read_csv(Path(comp_root) / f"{split}_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    if "Fluid_Sensitive" not in series.columns:
        series["Fluid_Sensitive"] = series["Fat_Suppression"]
    by_study = {st: g for st, g in series.groupby("StudyInstanceUID")}
    studies = list(map(str, studies))
    a = len(studies) * shard // n_shards
    b = len(studies) * (shard + 1) // n_shards
    mine = studies[a:b]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"c3_{split}_s{shard}of{n_shards}"
    arr = np.lib.format.open_memmap(out_dir / f"{stem}.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(mine), N_SLOT, N_SLICE, SIZE, SIZE))
    mask = np.zeros((len(mine), N_SLOT), np.float32)
    root = os.path.join(comp_root, f"{split}_series")
    t0 = time.time()
    for k, st in enumerate(mine):
        g = by_study.get(st)
        if g is not None:
            try:
                arr[k], mask[k] = build_study(root, st, g)
            except Exception as exc:
                log(f"study {st[-12:]} failed: {exc}")
        if (k + 1) % 100 == 0:
            rate = (time.time() - t0) / (k + 1)
            log(f"{split} shard {shard}: {k + 1}/{len(mine)}  {rate:.2f}s/study  "
                f"eta {rate * (len(mine) - k - 1) / 60:.0f} min  fill {mask[:k + 1].mean():.3f}")
    arr.flush()
    np.save(out_dir / f"{stem}_mask.npy", mask)
    (out_dir / f"{stem}_studies.json").write_text(json.dumps(mine))
    log(f"{stem}: {len(mine)} studies, fill {mask.mean():.3f}, "
        f"empty {(mask.sum(1) == 0).sum()}, slices read {QA['slices_read']}, "
        f"failed and replaced {QA['slices_failed']}")
    return arr, mask, mine


class IndexedCache:
    """Rows of several shard files, presented in the order of a given study list."""

    def __init__(self, parts, index):
        self.parts = parts
        self.index = index  # list of (part, row) per requested study
        self.shape = (len(index),) + tuple(parts[0].shape[1:])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        p, r = self.index[int(i)]
        return self.parts[p][r]


def copy_local(dirs: Sequence[Path], dst: Path = TEMP / "c3_local", log=print) -> List[Path]:
    """Copy the cache shards to the local disk when there is room.

    Training reads small windows at random positions; on a network-mounted dataset that can
    be thirty times slower than on the local disk.
    """
    import shutil
    import time
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for d in dirs for p in Path(d).iterdir() if p.is_file() and p.name.startswith("c3_")]
    need = sum(p.stat().st_size for p in files)
    # bytes already on the local disk from an earlier copy count as free for this check
    have = sum(q.stat().st_size for q in dst.rglob("c3_*") if q.is_file())
    free = shutil.disk_usage(dst).free
    log(f"cache {need / 1e9:.1f} GB, local free {free / 1e9:.1f} GB, already local {have / 1e9:.1f} GB")
    if free + have < need + 3e9:
        log("not enough local disk: reading the cache from the input mount")
        return list(dirs)
    t, out = time.time(), []
    for i, d in enumerate(dirs):
        o = dst / f"shard{i}"
        o.mkdir(exist_ok=True)
        for p in sorted(Path(d).iterdir()):
            if p.is_file() and p.name.startswith("c3_"):
                _copy_checked(p, o / p.name, log)
        out.append(o)
    log(f"cache copied to local disk in {time.time() - t:.0f} s")
    return out


def _copy_checked(src: Path, dst: Path, log=print) -> str:
    """Copy ``src`` to ``dst`` unless a verified copy of this very source is there.

    A sidecar ``<name>.src.json`` records the source size, modification time and the
    SHA-256 computed while copying. An existing copy is reused only if the source still
    has that size and time and the local file still has that hash; anything else is
    copied again, to a temporary name first, so a killed copy never looks complete.
    """
    import hashlib
    import shutil
    st = src.stat()
    side = dst.with_name(dst.name + ".src.json")
    if dst.exists() and side.exists():
        rec = json.loads(side.read_text())
        if (rec.get("size"), rec.get("mtime")) == (st.st_size, st.st_mtime) and file_digest(dst) == rec.get("sha256"):
            log(f"  {src.name}: local copy verified ({rec['sha256'][:12]})")
            return rec["sha256"]
    tmp = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for block in iter(lambda: fi.read(1 << 24), b""):
            h.update(block)
            fo.write(block)
    digest = h.hexdigest()
    if file_digest(tmp) != digest:
        raise IOError(f"local copy of {src} does not match the bytes read")
    os.replace(tmp, dst)
    side.write_text(json.dumps({"source": str(src), "size": st.st_size, "mtime": st.st_mtime,
                                "sha256": digest}))
    log(f"  {src.name}: copied, sha256 {digest[:12]}")
    return digest


def read_speed(array, n: int = 16, log=print) -> float:
    """Seconds per study for reading whole studies at random."""
    import time
    rng = np.random.default_rng(0)
    t = time.time()
    for i in rng.choice(len(array), min(n, len(array)), replace=False):
        np.asarray(array[i]).sum()
    s = (time.time() - t) / min(n, len(array))
    log(f"read speed: {s:.2f} s per study (local disk: well under 0.5)")
    return s


def load_cache3(dirs: Sequence[Path], split: str, studies: Sequence[str]):
    """Return (array, mask, missing) aligned to ``studies``. Missing studies get mask 0."""
    parts, masks, where = [], [], {}
    seen = set()
    for d in dirs:
        for npy in sorted(Path(d).rglob(f"c3_{split}_s*of*.npy")):
            if npy.name.endswith("_mask.npy") or npy.name in seen:
                continue
            seen.add(npy.name)
            stem = npy.name[:-4]
            ids = json.loads((npy.parent / f"{stem}_studies.json").read_text())
            pi = len(parts)
            parts.append(np.load(npy, mmap_mode="r"))
            masks.append(np.load(npy.parent / f"{stem}_mask.npy"))
            for k, u in enumerate(ids):
                where[str(u)] = (pi, k)
    if not parts:
        raise FileNotFoundError(f"no c3_{split}_* shards found")
    index, mask, missing = [], np.zeros((len(studies), N_SLOT), np.float32), []
    for i, u in enumerate(map(str, studies)):
        if u in where:
            p, k = where[u]
            index.append((p, k))
            mask[i] = masks[p][k]
        else:
            index.append((0, 0))
            missing.append(u)
    return IndexedCache(parts, index), mask, missing
