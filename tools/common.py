"""Loaders, a fast AUC and the paired bootstrap shared by the analysis scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rsna_knee.report_labels import TARGETS  # noqa: E402

T = list(TARGETS)
GOLD = {t: "GOLD_" + t.upper().replace(" ", "_").replace("'", "") for t in T}
N_BOOT = 1000
SEED = 0


def load():
    """Report labels, teacher OOF and our OOF, aligned on the 4,407 training studies."""
    lab = pd.read_csv(ROOT / "data/labels_v1_opus45.csv").set_index("STUDY_UID")
    arm = pd.read_csv(ROOT / "data/teacher_oof_armA.csv").set_index("STUDY_UID")
    ours = pd.read_csv(ROOT / "data/oof_v2_r50v3_all.csv").set_index("StudyInstanceUID")
    ix = lab.index.intersection(arm.index).intersection(ours.index)
    assert len(ix) == len(lab) == 4407
    return lab.loc[ix], arm.loc[ix, T], ours.loc[ix, T]


def gold_matrix(lab):
    g = lab[lab.IS_GOLD.astype(bool)]
    Y = np.stack([g[GOLD[t]].astype(float).to_numpy() for t in T], 1).astype(int)
    return g.index, Y


def states(lab):
    return np.stack([lab[t + "_state"].astype(str).str.lower().to_numpy() for t in T], 1)


def auc(y, s):
    """ROC AUC with average ranks for ties; nan when one class is missing."""
    y = np.asarray(y).astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_finding(Y, P, rows=None):
    rows = np.arange(len(Y)) if rows is None else rows
    return np.array([auc(Y[rows, j], P[rows, j]) for j in range(Y.shape[1])])


def explicit_per_finding(S, P):
    out = []
    for j in range(len(T)):
        m = np.isin(S[:, j], ["p", "n"])
        out.append(auc(S[m, j] == "p", P[m, j]))
    return np.array(out)


def boot_indices(n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def _ranks(P):
    return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])


def paired_diff(Y, A, B, boots, n_perm=N_BOOT, seed=SEED):
    """Observed macro AUC(A) - AUC(B) on the gold studies, a 95% percentile bootstrap
    interval over studies, and a two-sided paired permutation p value.

    The permutation test swaps the two score sets within a study (all twelve findings
    together) with probability 1/2. Scores are first turned into within-finding ranks, so
    the two sets are on one scale; AUC depends only on these ranks. Under the null that
    the two score sets are exchangeable within studies, the swapped differences give the
    reference distribution; p = (1 + #{|d*| >= |d|}) / (1 + n_perm).
    """
    obs = float(np.nanmean(per_finding(Y, A) - per_finding(Y, B)))
    d = np.array([np.nanmean(per_finding(Y, A, b) - per_finding(Y, B, b)) for b in boots])
    Ra, Rb = _ranks(np.asarray(A, float)), _ranks(np.asarray(B, float))
    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sw = rng.random(len(Ra)) < 0.5
        Pa = np.where(sw[:, None], Rb, Ra)
        Pb = np.where(sw[:, None], Ra, Rb)
        null[i] = np.nanmean(per_finding(Y, Pa) - per_finding(Y, Pb))
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def _placements(y, x):
    """DeLong placement values of one finding: V10 for positives, V01 for negatives."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(x)
    v10 = (r[y] - rankdata(x[y])) / n0
    v01 = 1.0 - (r[~y] - rankdata(x[~y])) / n1
    return y, n1, n0, v10, v01


def delong_macro(Y, A, B):
    """Paired DeLong-type test for the macro AUC difference A - B on the same studies.

    For each finding, every study gets the DeLong placement value of its class, centred
    on that finding's AUC and scaled by 1/sqrt(n(n-1)) of its class; the study's
    contribution to the macro difference is the mean over findings of A minus B. The
    variance is the sum of squared contributions over studies, so the dependence between
    findings scored on the same studies is kept. For one finding this is the DeLong
    variance of a paired difference. Returns (difference, standard error, two-sided p).
    """
    from scipy.stats import norm
    Y = np.asarray(Y)
    A, B = np.asarray(A, float), np.asarray(B, float)
    contrib = np.zeros((len(Y), Y.shape[1]))
    diffs, use = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(bool)
        if y.all() or (~y).all():
            continue
        z = np.zeros(len(Y))
        d = 0.0
        for sign, X in ((1.0, A[:, j]), (-1.0, B[:, j])):
            yy, n1, n0, v10, v01 = _placements(y, X)
            th = v10.mean()
            d += sign * th
            z[yy] += sign * (v10 - th) / np.sqrt(n1 * (n1 - 1))
            z[~yy] += sign * (v01 - th) / np.sqrt(n0 * (n0 - 1))
        contrib[:, j] = z
        diffs.append(d)
        use.append(j)
    diff = float(np.mean(diffs))
    zi = contrib[:, use].mean(axis=1)
    se = float(np.sqrt(np.sum(zi ** 2)))
    p = float(2 * norm.sf(abs(diff) / se)) if se > 0 else float("nan")
    return diff, se, p


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def holm(pvals):
    """Holm step-down adjusted p values, in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj
