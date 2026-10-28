"""Post-hoc corrections in rank space, shared by the analysis scripts and the Kaggle cell.

All functions take a matrix of scores [n_studies, 12] in TARGETS order.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .report_labels import TARGETS

# findings left unchanged by the explaining-away correction: they co-occur by biology
EA_KEEP = ("Effusion", "Synovitis")


def pct_rank(scores: np.ndarray) -> np.ndarray:
    """Percentile rank of every column, ties averaged."""
    return pd.DataFrame(np.asarray(scores, dtype=float)).rank(method="average", pct=True).to_numpy()


def cooccurrence_weights(labels: pd.DataFrame) -> np.ndarray:
    """Positive part of the label correlation, zero diagonal, rows summing to 1."""
    c = labels[list(TARGETS)].astype(float).corr().to_numpy().copy()
    np.fill_diagonal(c, 0.0)
    c = np.clip(c, 0.0, None)
    return c / np.maximum(c.sum(1, keepdims=True), 1e-12)


def explain_away(ranks: np.ndarray, W: np.ndarray, lam: float,
                 keep: Sequence[str] = EA_KEEP) -> np.ndarray:
    """r'_j = r_j + lam * sum_k W[j,k] r_k for every finding not in ``keep``.

    lam < 0 subtracts the usual partners of a finding (explaining away);
    lam > 0 lets them support it.
    """
    r = np.asarray(ranks, dtype=float)
    out = r + lam * r @ np.asarray(W, dtype=float).T
    for name in keep:
        j = TARGETS.index(name)
        out[:, j] = r[:, j]
    return out
