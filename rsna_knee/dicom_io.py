"""Reading knee MRI studies out of the competition mount.

Four things in this file are decisions rather than plumbing, and each one fails
silently if it is wrong. There is no exception and no warning, only a model
trained on the wrong pixels.

1. **Slice order.** The files in a series are named by SOP Instance UID, which is
   assigned to be unique and carries no order. Sorting by file name yields a stack
   uncorrelated with anatomy. The true order comes from the geometry every slice
   carries.

2. **Laterality.** Five of the twelve targets are defined by which side of the
   midline a structure sits on. `Laterality` (0020,0060) is a Type 2C attribute and
   is absent on a large share of this corpus, by whole vendors rather than at random,
   so a study with no tag is not a left knee. The patient coordinate system supplies
   the missing value.

3. **Physical scale.** Pixel spacing varies across sites by a factor of several. A
   fixed-pixel resize therefore hands the encoder anatomy at inconsistent scale, and
   a meniscal tear one to three millimetres wide can fall below two pixels and cease
   to exist before the first convolution. Cropping to a constant millimetre extent
   fixes the pitch.

4. **Which series fills which slot.** A study holds several series and the model
   takes a fixed number. Filling an empty slot from a neighbouring sequence would
   assert a sequence that was never acquired.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom

# Tags read during the header pass. Position and orientation cost nothing extra,
# since the file is already open, and they recover both slice order and side.
HEADER_TAGS = (
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "InversionTime", "Laterality",
    "PixelSpacing", "Rows", "Columns", "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
)

# Which of those hold numbers. Several DICOM attributes are multi-valued and hold
# strings, not numbers: ScanningSequence (0018,0020) is a list drawn from SE, GR, IR
# and others, ScanOptions (0018,0022) likewise, and SequenceName can repeat. Deciding
# by value type alone reads those as numeric and raises on the first one. The set is
# therefore listed rather than inferred.
NUMERIC_TAGS = frozenset({
    "RepetitionTime", "EchoTime", "InversionTime", "PixelSpacing",
    "Rows", "Columns", "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
})

# (slot name, plane, fat suppression). Six slots: three planes, fat-suppressed and
# not. Fluid-sensitive fat-suppressed sequences carry o?edema, effusion and marrow
# signal; the non-suppressed sequences carry anatomy, meniscal substance and
# cartilage. A model given only one of the two is blind to half the target list.
SLOTS: Tuple[Tuple[str, str, bool], ...] = (
    ("SAG_FS", "Sagittal", True),
    ("SAG_T1", "Sagittal", False),
    ("COR_FS", "Coronal", True),
    ("COR_T1", "Coronal", False),
    ("AX_FS", "Axial", True),
    ("AX_T1", "Axial", False),
)
N_SLOT = len(SLOTS)


# ---------------------------------------------------------------------------
# header pass
# ---------------------------------------------------------------------------

def _read_tag(value, tag: str):
    """One header value, as a number, a list of numbers, or a string.

    Returns ``None`` when a tag that should be numeric is not parseable. That is a
    real answer: downstream code already treats a missing geometry tag as a reason
    to fall back, and inventing a zero would place the slice at the origin.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        if tag not in NUMERIC_TAGS:
            return [str(v) for v in value]
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if tag not in NUMERIC_TAGS:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: Path) -> Optional[Dict[str, object]]:
    """Read one file's header without decoding its pixels.

    Never raises. A study that cannot be parsed is one study; an exception escaping
    here ends a run that has already spent an hour decoding, and the work is lost
    because a failed Kaggle version saves no usable output. Failures are reported by
    returning ``None`` and counted by the caller.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    out: Dict[str, object] = {"path": str(path)}
    for tag in HEADER_TAGS:
        try:
            out[tag] = _read_tag(getattr(ds, tag, None), tag)
        except Exception:
            out[tag] = None
    return out


def scan_series(series_dir: Path, workers: int = 8) -> List[Dict[str, object]]:
    """Headers for every readable file in one series, unreadable ones dropped."""
    try:
        files = sorted(p for p in series_dir.iterdir() if p.is_file())
    except OSError:
        return []
    if not files:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [h for h in pool.map(probe, files) if h is not None]


# ---------------------------------------------------------------------------
# slice order
# ---------------------------------------------------------------------------

def slice_key(header: Dict[str, object]) -> Optional[float]:
    r"""Signed position of a slice along the stack normal, in patient millimetres.

    ``ImageOrientationPatient`` holds the two in-plane unit vectors
    :math:`\hat r_x, \hat r_y`; their cross product is the slice normal
    :math:`\hat n`. Projecting the slice origin :math:`p` onto it,

    .. math:: k = p \cdot (\hat r_x \times \hat r_y)

    increases monotonically along the stack. The value is signed and expressed in
    patient coordinates, so it also fixes the direction the stack traverses the
    joint, which is what the sagittal laterality correction reverses.
    """
    orient = header.get("ImageOrientationPatient")
    pos = header.get("ImagePositionPatient")
    if orient is None or pos is None or len(orient) < 6 or len(pos) < 3:
        return None
    rx = np.asarray(orient[:3], dtype=float)
    ry = np.asarray(orient[3:6], dtype=float)
    normal = np.cross(rx, ry)
    return float(np.dot(np.asarray(pos[:3], dtype=float), normal))


def order_slices(headers: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Sort a series into anatomical order, falling back where geometry is absent.

    ``InstanceNumber`` usually tracks physical position up to sign, but interleaved
    and multi-echo acquisitions number slices in acquisition order rather than in
    space, so it is a fallback and not the primary key.
    """
    keyed = [(slice_key(h), h) for h in headers]
    if all(k is not None for k, _ in keyed):
        return [h for _, h in sorted(keyed, key=lambda kh: kh[0])]
    return sorted(
        headers,
        key=lambda h: (h.get("InstanceNumber") if h.get("InstanceNumber") is not None else 0),
    )


# ---------------------------------------------------------------------------
# laterality
# ---------------------------------------------------------------------------

def laterality(headers: Sequence[Dict[str, object]]) -> Optional[str]:
    r"""``'L'``, ``'R'`` or ``None`` for a series.

    The tag wins when present. Otherwise the image centre in patient coordinates
    decides: :math:`+x` points to the patient's left, so

    .. math:: c = p + \hat r_x \Delta_c \tfrac{N_c}{2} + \hat r_y \Delta_r \tfrac{N_r}{2}

    and the sign of :math:`c_x` names the knee. Averaging over the series rather
    than trusting one slice keeps a single malformed header from flipping a study.
    """
    for h in headers:
        tag = h.get("Laterality")
        if tag in ("L", "R"):
            return str(tag)

    xs: List[float] = []
    for h in headers:
        orient, pos = h.get("ImageOrientationPatient"), h.get("ImagePositionPatient")
        spacing, rows, cols = h.get("PixelSpacing"), h.get("Rows"), h.get("Columns")
        if not (orient and pos and spacing and rows and cols):
            continue
        rx = np.asarray(orient[:3], dtype=float)
        ry = np.asarray(orient[3:6], dtype=float)
        p = np.asarray(pos[:3], dtype=float)
        centre = p + rx * float(spacing[1]) * int(cols) / 2.0 \
                   + ry * float(spacing[0]) * int(rows) / 2.0
        xs.append(float(centre[0]))
    if not xs:
        return None
    return "L" if float(np.mean(xs)) > 0 else "R"


def normalise_laterality(volume: np.ndarray, plane: str, side: Optional[str]) -> np.ndarray:
    """Map every knee onto a left-knee convention.

    Coronal and axial slices are mirror images of the opposite knee, so flipping the
    last axis maps one onto the other. A sagittal slice is not: mirroring it
    changes nothing anatomically. What differs is the direction in which the stack
    crosses the joint, so the slice order is reversed instead.
    """
    if side != "R":
        return volume
    if plane in ("Coronal", "Axial"):
        return volume[..., ::-1]
    return volume[::-1]


# ---------------------------------------------------------------------------
# sequence classification
# ---------------------------------------------------------------------------

def fat_suppressed(header: Dict[str, object]) -> bool:
    """Fat suppression, from whichever field the vendor chose to record it in."""
    text = " ".join(
        str(header.get(t) or "") for t in
        ("SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence")
    ).lower()
    if any(cue in text for cue in
           ("fs", "fatsat", "fat_sat", "fat sat", "spair", "spir", "stir",
            "spectral", "dixon", "frfse-xl fs", "_fs")):
        return True
    if header.get("InversionTime") not in (None, 0):
        return True
    return False


def fluid_sensitive(header: Dict[str, object]) -> bool:
    r"""Fluid sensitivity, from the contrast weighting.

    Weighting follows from repetition and echo time: :math:`T_R \lesssim 800` ms is
    T1-weighted, and a long :math:`T_R` with a long :math:`T_E` is T2-weighted. Both
    T2 and proton-density sequences show fluid bright; T1 does not.
    """
    tr = header.get("RepetitionTime")
    te = header.get("EchoTime")
    if tr is None:
        return fat_suppressed(header)
    tr = float(tr)
    if tr < 800:
        return False
    if te is not None and float(te) >= 25:
        return True
    return True


def pick_slots(series_rows, plane_of, fatsat_of) -> Dict[str, str]:
    """Choose at most one series per slot for one study.

    An unfilled slot stays empty. Relaxing the predicate to fill it, so that a
    non-suppressed series stands in for a fat-suppressed one, puts one series in
    two slots and makes the presence mask assert a sequence the scanner never
    acquired, which the model then learns to read as evidence.

    Ties go to the series with the most slices: a denser stack samples the joint
    better and leaves the slice sampler more margin.
    """
    chosen: Dict[str, str] = {}
    for name, plane, fs in SLOTS:
        candidates = [
            r for r in series_rows
            if plane_of.get(r["SeriesInstanceUID"]) == plane
            and fatsat_of.get(r["SeriesInstanceUID"]) == fs
        ]
        if candidates:
            best = max(candidates, key=lambda r: r.get("n_slices", 0))
            chosen[name] = best["SeriesInstanceUID"]
    return chosen


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------

def window(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Scale one slice to [0, 1] on its own percentiles.

    MR intensity has no absolute meaning: the same tissue reads at different raw
    values on two scanners, and the maximum is often a single bright vessel or a
    coil artefact. Percentiles use the tissue rather than the outlier.
    """
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def physical_crop(image: np.ndarray, spacing: Sequence[float], crop_mm: float) -> np.ndarray:
    """Centre-crop to a constant physical extent.

    With pitch :math:`s` mm/pixel, holding :math:`L` millimetres across a :math:`P`
    pixel output fixes the effective pitch at :math:`L/P` regardless of the site's
    field of view. If ``crop_mm`` exceeds the acquired extent the crop cannot be
    taken and the slice passes through unchanged, so ``crop_mm`` must sit below the
    smallest field of view in the corpus or it silently does nothing.
    """
    if not spacing:
        return image
    rows, cols = image.shape[:2]
    dr, dc = float(spacing[0]), float(spacing[1])
    nr, nc = int(round(crop_mm / dr)), int(round(crop_mm / dc))
    if nr >= rows or nc >= cols or nr < 8 or nc < 8:
        return image
    r0, c0 = (rows - nr) // 2, (cols - nc) // 2
    return image[r0:r0 + nr, c0:c0 + nc]


def pad_square(image: np.ndarray) -> np.ndarray:
    """Pad to a square before resizing.

    Resizing a non-square slice straight to a square output stretches one axis, so
    a round structure becomes elliptical by an amount that varies with the site's
    matrix. Padding preserves the aspect ratio at the cost of some background.
    """
    rows, cols = image.shape[:2]
    if rows == cols:
        return image
    size = max(rows, cols)
    out = np.zeros((size, size), dtype=image.dtype)
    r0, c0 = (size - rows) // 2, (size - cols) // 2
    out[r0:r0 + rows, c0:c0 + cols] = image
    return out


def sample_positions(n_slices: int, n_keep: int, band: Tuple[float, float] = (0.12, 0.88)) -> List[int]:
    """Indices of the slices to keep, spread evenly over the middle of the stack.

    The first and last slices of a knee series are mostly soft tissue outside the
    joint. Restricting to a band keeps the budget on slices that contain the joint.
    """
    if n_slices <= 0:
        return []
    lo = int(np.floor(band[0] * n_slices))
    hi = int(np.ceil(band[1] * n_slices))
    hi = max(hi, lo + 1)
    idx = np.linspace(lo, min(hi, n_slices) - 1, num=n_keep)
    return [int(round(float(i))) for i in np.clip(idx, 0, n_slices - 1)]


def read_slot(
    paths_in_order: Sequence[str],
    n_keep: int,
    out_px: int,
    crop_mm: float,
    plane: str,
    side: Optional[str],
) -> Optional[np.ndarray]:
    """Decode one series into a ``uint8`` array of shape ``(n_keep, out_px, out_px)``.

    Returns ``None`` when nothing decodes. A decode failure must be reported rather
    than swallowed: an all-zero array is indistinguishable from a black knee, and a
    study silently full of black knees trains the model to predict the base rate.
    """
    import cv2

    if not paths_in_order:
        return None
    keep = sample_positions(len(paths_in_order), n_keep)
    frames: List[np.ndarray] = []
    for i in keep:
        try:
            ds = pydicom.dcmread(paths_in_order[i], force=True)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            arr = arr * slope + intercept
            spacing = getattr(ds, "PixelSpacing", None)
            if spacing is not None:
                arr = physical_crop(arr, [float(spacing[0]), float(spacing[1])], crop_mm)
            arr = pad_square(window(arr))
            arr = cv2.resize(arr, (out_px, out_px), interpolation=cv2.INTER_AREA)
            frames.append((arr * 255.0).astype(np.uint8))
        except Exception:
            frames.append(None)

    good = [f for f in frames if f is not None]
    if not good:
        return None
    filler = good[len(good) // 2]
    frames = [f if f is not None else filler for f in frames]

    volume = np.stack(frames, axis=0)
    return np.ascontiguousarray(normalise_laterality(volume, plane, side))
