# ---------------------------------------------------------------------------
# CELL A. Put this right after the last cell of the Versia part.
# Keeps the Versia predictions before the Raptor part overwrites submission.csv.
# ---------------------------------------------------------------------------
import shutil, gc
shutil.copy("/kaggle/working/submission.csv", "/kaggle/working/sub_versia.csv")
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass
print("saved sub_versia.csv")


# ---------------------------------------------------------------------------
# CELL B. Put this as the very last cell, after the Raptor part.
# Rank average of the two submissions. If anything is wrong, Versia alone is kept.
# ---------------------------------------------------------------------------
RAPTOR_WEIGHT = 0.2


def _two_blend(w=RAPTOR_WEIGHT):
    import shutil, traceback
    import numpy as np
    import pandas as pd

    a_path = "/kaggle/working/sub_versia.csv"
    b_path = "/kaggle/working/submission.csv"
    try:
        a = pd.read_csv(a_path)
        b = pd.read_csv(b_path)
        cols = [c for c in a.columns if c != "StudyInstanceUID"]
        if set(cols) - set(b.columns):
            raise ValueError("column mismatch")
        b = a[["StudyInstanceUID"]].merge(b, on="StudyInstanceUID", how="left")
        ra = a[cols].astype(float).rank(pct=True).to_numpy()
        rb = b[cols].astype(float).rank(pct=True).to_numpy()
        if not np.isfinite(rb).all():
            raise ValueError("raptor part has missing rows")
        out = a[["StudyInstanceUID"]].copy()
        out[cols] = (1 - w) * ra + w * rb
        out.to_csv(b_path, index=False)
        print(f"blended {len(out)} studies, raptor weight {w}")
    except Exception:
        traceback.print_exc()
        shutil.copy(a_path, b_path)
        print("blend failed; Versia submission kept")


_two_blend()
