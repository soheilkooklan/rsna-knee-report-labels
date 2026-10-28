# Last cell of a public RSNA knee notebook. Explaining-away correction.
#
# Correlated findings leak into each other's scores: a knee with OA also gets a
# higher meniscus score, a knee with a contusion a higher ACL score. AUC is read
# per finding, so this leakage can cost ranking. Each finding except effusion and
# synovitis gets a small subtraction of the ranks of its usual partners:
#     r_j' = r_j + LAM * sum_k W[j, k] * r_k
# W is the positive part of the report-label correlation over the 4,349 non-gold
# training studies, rows summing to 1 (rsna_knee/rank_tools.py, rounded here).
# Out of fold it changed gold-58 macro AUC by +0.003 to +0.006 (exploratory; see the
# paper). Runtime is milliseconds. On any error the original submission.csv is kept.

LAM = -0.1

W = [
    [0.000, 0.137, 0.091, 0.139, 0.004, 0.043, 0.000, 0.161, 0.075, 0.024, 0.217, 0.108],
    [0.184, 0.000, 0.058, 0.045, 0.034, 0.014, 0.000, 0.154, 0.101, 0.038, 0.216, 0.154],
    [0.068, 0.033, 0.000, 0.056, 0.226, 0.082, 0.089, 0.116, 0.144, 0.116, 0.050, 0.021],
    [0.121, 0.029, 0.065, 0.000, 0.069, 0.216, 0.067, 0.112, 0.137, 0.079, 0.057, 0.047],
    [0.002, 0.015, 0.175, 0.046, 0.000, 0.135, 0.156, 0.114, 0.181, 0.114, 0.036, 0.026],
    [0.030, 0.007, 0.075, 0.170, 0.160, 0.000, 0.157, 0.096, 0.161, 0.098, 0.024, 0.021],
    [0.000, 0.000, 0.099, 0.064, 0.225, 0.191, 0.000, 0.101, 0.206, 0.110, 0.004, 0.000],
    [0.083, 0.059, 0.080, 0.066, 0.102, 0.072, 0.062, 0.000, 0.196, 0.107, 0.104, 0.070],
    [0.036, 0.036, 0.092, 0.075, 0.150, 0.112, 0.118, 0.182, 0.000, 0.106, 0.057, 0.035],
    [0.019, 0.022, 0.121, 0.071, 0.155, 0.112, 0.102, 0.162, 0.173, 0.000, 0.045, 0.019],
    [0.173, 0.129, 0.053, 0.052, 0.050, 0.028, 0.004, 0.161, 0.095, 0.046, 0.000, 0.209],
    [0.123, 0.130, 0.032, 0.062, 0.052, 0.035, 0.000, 0.155, 0.084, 0.028, 0.298, 0.000],
]
ORDER = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEEP = ("Effusion", "Synovitis")


def _explain_away(lam=LAM):
    import shutil, traceback
    import numpy as np
    import pandas as pd

    path = "/kaggle/working/submission.csv"
    backup = "/kaggle/working/submission_before_ea.csv"
    try:
        sub = pd.read_csv(path)
        shutil.copy(path, backup)
        missing = [c for c in ORDER if c not in sub.columns]
        if missing:
            raise ValueError(f"columns not found: {missing}")
        r = sub[ORDER].astype(float).rank(pct=True).to_numpy()
        out = r + lam * r @ np.asarray(W).T
        for j, c in enumerate(ORDER):
            if c not in KEEP:
                sub[c] = out[:, j]
        if not np.isfinite(sub[ORDER].to_numpy(dtype=float)).all():
            raise ValueError("non-finite values")
        sub.to_csv(path, index=False)
        print(f"explaining-away applied to {len(sub)} studies, lambda {lam}")
    except Exception:
        traceback.print_exc()
        try:
            shutil.copy(backup, path)
        except Exception:
            pass
        print("explaining-away skipped; submission.csv unchanged")


_explain_away()
