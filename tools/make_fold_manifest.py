"""Fold manifest from the five per-fold OOF files of the r50v3 run.

Each file holds the held-out studies of one fold. The fold index is recovered by
matching the gold macro AUC of the file with the value printed in the training log
of that fold (OOF gold macro: 0.9283, 0.8643, 0.9254, 0.8803, 0.8196).

    python tools/make_fold_manifest.py <oof_f0-like files ...>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from rsna_knee.report_labels import TARGETS

T = list(TARGETS)
GOLD = {t: "GOLD_" + t.upper().replace(" ", "_").replace("'", "") for t in T}
LOG = [0.9283, 0.8643, 0.9254, 0.8803, 0.8196]

lab = pd.read_csv("data/labels_v1_opus45.csv").set_index("STUDY_UID")
rows = []
for f in sys.argv[1:]:
    df = pd.read_csv(f)
    df = df[df.StudyInstanceUID.isin(lab.index)].drop_duplicates("StudyInstanceUID")
    g = lab.loc[df.StudyInstanceUID]
    g = g[g.IS_GOLD]
    p = df.set_index("StudyInstanceUID").loc[g.index]
    aucs = []
    for t in T:
        y = g[GOLD[t]].astype(float)
        if y.nunique() == 2:
            aucs.append(roc_auc_score(y, p[t]))
    m = float(np.mean(aucs))
    k = int(np.argmin([abs(m - v) for v in LOG]))
    print(f"{f}: {len(df)} studies, gold {len(g)}, gold macro {m:.4f} -> fold {k}")
    rows += [{"StudyInstanceUID": u, "fold": k} for u in df.StudyInstanceUID]
man = pd.DataFrame(rows)
man["is_gold"] = lab.loc[man.StudyInstanceUID, "IS_GOLD"].to_numpy()
assert man.StudyInstanceUID.is_unique and man.fold.nunique() == 5
man.sort_values(["fold", "StudyInstanceUID"]).to_csv("data/fold_manifest.csv", index=False)
print(man.groupby("fold").agg(studies=("StudyInstanceUID", "size"), gold=("is_gold", "sum")).to_string())
