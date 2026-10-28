"""Training targets for v2: report labels blended with image-teacher predictions.

Two public sources are bundled in data/ (NTejas-1/RSNA-Knee-Abnormality-Detection,
MIT licence):

  labels_v1_opus45.csv    Claude Opus 4.5 reading of all 4,407 reports, a soft
                          probability and a state (p / n / u) per finding.
  teacher_oof_armA.csv    out-of-fold predictions of an image model (RadImageNet
                          ResNet50, 224 px) trained on those labels.

Measured on the 58 gold studies (tools/reproduce_tables.py):

  report labels alone            0.892 macro AUC
  image-teacher OOF alone        0.896
  0.5 * labels + 0.5 * OOF       0.927   (+0.035, bootstrap 95% CI 0.019 to 0.052)
  OOF only where report silent   0.886

The report is better on findings a radiologist states outright (ACL, MCL, PF OA);
the image teacher is better on findings that are often left unwritten or graded
differently on the images (effusion, Baker's cyst, contusion, fracture). Mixing on
every cell, not only on silent cells, is what helps.

Round 2 adds our own OOF predictions to the teacher pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .paths import INPUT
from .report_labels import TARGETS

ID = "StudyInstanceUID"

# Per-finding share of the image teacher in the target.
# "flat" is the default and was not tuned. "grouped" follows the split above
# (report-explicit findings lean on the report, image-visible ones on the teacher);
# it was chosen after looking at gold-58, so its gold score is optimistic.
TEACHER_SHARE = {
    "flat": {t: 0.5 for t in TARGETS},
    "grouped": {
        "ACL": 0.25, "MCL": 0.25, "PF OA": 0.25,
        "Medial Meniscus": 0.5, "Lateral Meniscus": 0.5,
        "Medial OA": 0.5, "Lateral OA": 0.5, "Synovitis": 0.5,
        "Effusion": 0.75, "Baker's": 0.75, "Contusion": 0.75, "Fracture": 0.75,
    },
    "labels_only": {t: 0.0 for t in TARGETS},
}


def file_digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_file(name: str, roots: Sequence[Path] = (INPUT, Path("."))) -> Optional[Path]:
    """The file called ``name`` under the given roots, skipping DICOM trees.

    Copies with the same content are fine; two different files with this name raise,
    so a stale copy in another mounted dataset is never picked up silently.
    """
    import os
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.lower().endswith(("_series", "_images"))]
            if name in filenames:
                found.append(Path(dirpath) / name)
    if not found:
        return None
    digests = {file_digest(f) for f in found}
    if len(digests) > 1:
        raise ValueError(f"{len(found)} different files named {name}: {[str(f) for f in found]}; "
                         "detach the stale dataset or pass the path explicitly")
    return found[0]


def read_prob_table(path: Path) -> pd.DataFrame:
    """A table with a study id column and the twelve finding columns, indexed by id."""
    df = pd.read_csv(path)
    for cand in (ID, "STUDY_UID", "study_id", "StudyUID"):
        if cand in df.columns:
            df = df.rename(columns={cand: ID})
            break
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}")
    return df.drop_duplicates(ID).set_index(ID)


def load_report_labels(path: Path) -> pd.DataFrame:
    df = read_prob_table(path)
    keep = list(TARGETS) + [t + "_state" for t in TARGETS if t + "_state" in df.columns]
    return df[keep]


def teacher_mean(tables: Sequence[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Average several teachers in rank space, then map back to [0, 1].

    Teachers trained with different recipes are not calibrated alike, so their
    probabilities are converted to per-column percentiles first. A study missing
    from one teacher is averaged over the teachers that have it.
    """
    stacks = []
    for tab in tables:
        t = tab.reindex(index)[list(TARGETS)].astype(float)
        stacks.append(t.rank(pct=True).to_numpy())
    arr = np.stack(stacks)  # [k, n, 12]
    # a cell no teacher covers stays missing; counted explicitly so no empty-slice warning
    have = np.isfinite(arr)
    n = have.sum(axis=0)
    mean = np.where(n > 0, np.where(have, arr, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)
    return pd.DataFrame(mean, index=index, columns=list(TARGETS))


def calibrate_to(reference: pd.Series, ranks: pd.Series) -> pd.Series:
    """Give rank scores the marginal distribution of ``reference`` (quantile map).

    Blending a percentile with a probability would shift every target upwards.
    Mapping the teacher's ranks onto the empirical distribution of the teacher's own
    first member keeps the blend on a probability-like scale.
    """
    ref = np.sort(reference.dropna().to_numpy())
    if len(ref) == 0:
        return ranks
    q = np.clip(ranks.to_numpy(), 0, 1)
    out = np.interp(q, np.linspace(0, 1, len(ref)), ref)
    return pd.Series(np.where(np.isfinite(q), out, np.nan), index=ranks.index)


def build_targets_v2(
    train: pd.DataFrame,
    labels: pd.DataFrame,
    teachers: Sequence[pd.DataFrame],
    scheme: str = "flat",
    gold_weight: float = 2.0,
    use_gold: bool = True,
    prior: float = 0.1,
):
    """Return targets [n,12], weights [n,12], explicit-state matrix [n,12] and gold mask.

    The explicit-state matrix holds 1 / 0 where the report stated the finding
    present / absent and NaN where it was silent. It is the holdout reference: a
    model's agreement with what the radiologist wrote down, free of the teacher.
    """
    idx = pd.Index(train[ID])
    lab = labels.reindex(idx)
    share = TEACHER_SHARE[scheme]

    target = lab[list(TARGETS)].astype(float).copy()
    if teachers:
        tm = teacher_mean(teachers, idx)
        ref0 = teachers[0].reindex(idx)
        for t in TARGETS:
            teach = calibrate_to(ref0[t].astype(float), tm[t]) if len(teachers) > 1 else ref0[t].astype(float)
            w = share[t]
            report_ok, teacher_ok = target[t].notna(), teach.notna()
            mixed = (1 - w) * target[t] + w * teach
            col = target[t].where(~(report_ok & teacher_ok), mixed)
            target[t] = col.where(report_ok, teach)      # no report value: teacher alone

    # neither source: a weak prior
    target = target.fillna(prior)

    explicit = np.full((len(idx), len(TARGETS)), np.nan, dtype=np.float32)
    for j, t in enumerate(TARGETS):
        col = t + "_state"
        if col in lab.columns:
            s = lab[col].astype(str).str.lower().to_numpy()
            explicit[s == "p", j] = 1.0
            explicit[s == "n", j] = 0.0

    targets = target.to_numpy(dtype=np.float32)
    weights = np.ones_like(targets)
    gold = train[list(TARGETS)].to_numpy(dtype=np.float32)
    has_gold = np.isfinite(gold)
    if use_gold:
        targets = np.where(has_gold, np.nan_to_num(gold), targets)
        weights = np.where(has_gold, gold_weight, weights)
    return targets, weights.astype(np.float32), explicit, has_gold.any(axis=1)


def gold_auc_table(train: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Per-finding AUC of ``scores`` against the expert labels."""
    from sklearn.metrics import roc_auc_score
    gold = train[list(TARGETS)].to_numpy(dtype=float)
    rows = []
    for j, t in enumerate(TARGETS):
        keep = np.isfinite(gold[:, j]) & np.isfinite(scores[:, j])
        y = gold[keep, j].astype(int)
        if keep.sum() < 4 or len(np.unique(y)) < 2:
            rows.append((t, np.nan, int(y.sum())))
            continue
        rows.append((t, roc_auc_score(y, scores[keep, j]), int(y.sum())))
    df = pd.DataFrame(rows, columns=["target", "auc", "n_pos"])
    return df


def merge_label_sources(labels: pd.DataFrame, extra: Sequence[pd.DataFrame], log=print) -> pd.DataFrame:
    """Average the bundled report labels with other public report-label tables.

    Only the twelve probability columns are averaged; the p / n / u states stay
    those of the bundled file. Values are clipped to [0, 1]; a table on a 0-100
    scale is rescaled.
    """
    out = labels.copy()
    stack = [labels[list(TARGETS)].astype(float)]
    for tab in extra:
        t = tab.reindex(labels.index)[list(TARGETS)].astype(float)
        if np.nanmax(t.to_numpy()) > 1.5:
            t = t / 100.0
        stack.append(t.clip(0, 1))
        log(f"extra label source covers {t.notna().all(axis=1).mean():.2%} of studies")
    out[list(TARGETS)] = pd.concat(stack).groupby(level=0).mean().reindex(labels.index)
    return out


def load_default_sources(extra_teachers: Sequence[str] = (), extra_labels: Sequence[str] = ()):
    """Locate the bundled label and teacher files on Kaggle or locally."""
    roots = (INPUT, Path(__file__).resolve().parent.parent / "data", Path("data"))
    lab_path = find_file("labels_v1_opus45.csv", roots)
    oof_path = find_file("teacher_oof_armA.csv", roots)
    if lab_path is None:
        raise FileNotFoundError("labels_v1_opus45.csv not found; it ships in data/ of the pipeline dataset")
    labels = load_report_labels(lab_path)
    if extra_labels:
        tabs = []
        for name in extra_labels:
            p = find_file(name, roots)
            if p is None:
                raise FileNotFoundError(f"label file {name} not found")
            tabs.append(read_prob_table(p))
        labels = merge_label_sources(labels, tabs)
    teachers: List[pd.DataFrame] = []
    if oof_path is not None:
        teachers.append(read_prob_table(oof_path))
    for name in extra_teachers:
        p = find_file(name, roots)
        if p is None:
            raise FileNotFoundError(f"teacher file {name} not found")
        teachers.append(read_prob_table(p))
    return labels, teachers, lab_path, oof_path
