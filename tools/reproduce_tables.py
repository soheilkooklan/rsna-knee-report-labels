"""Recompute every number of the paper that comes from the released CSV files.

Writes results/numbers.json (read by tools/make_figures.py and pasted into the paper)
and results/tables.md (the same numbers as Markdown tables). CPU only, a few minutes.

    python tools/reproduce_tables.py
"""

import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn

from common import (ROOT, T, N_BOOT, SEED, load, gold_matrix, states, per_finding,
                    explicit_per_finding, boot_indices, paired_diff, delong_macro, holm, auc, fmt_p)
from rsna_knee.rank_tools import pct_rank, cooccurrence_weights, explain_away

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
R = {"n_boot": N_BOOT, "seed": SEED,
     "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
md = []

lab, arm, ours = load()
gid, Y = gold_matrix(lab)
gpos = np.array([lab.index.get_loc(u) for u in gid])
S = states(lab)
boots = boot_indices(len(gid))

# ------------------------------------------------------------------ data
stated = np.isin(S, ["p", "n"])
R["data"] = {
    "studies": len(lab), "gold": len(gid),
    "cells": int(S.size), "stated": int(stated.sum()), "stated_present": int((S == "p").sum()),
    "per_finding": {t: {"present": int((S[:, j] == "p").sum()), "absent": int((S[:, j] == "n").sum()),
                        "silent": int((~stated[:, j]).sum()),
                        "gold_pos": int(Y[:, j].sum()), "gold_neg": int(len(Y) - Y[:, j].sum())}
                    for j, t in enumerate(T)},
}

# ------------------------------------------------------------------ targets (Table 1)
L = lab.loc[gid, T].to_numpy(float)
A = arm.loc[gid, T].to_numpy(float)
Sg = S[gpos]
silent = ~np.isin(Sg, ["p", "n"])
grouped_share = {t: 0.5 for t in T}
grouped_share.update({"ACL": .25, "MCL": .25, "PF OA": .25, "Effusion": .75, "Baker's": .75,
                      "Contusion": .75, "Fracture": .75})
gw = np.array([grouped_share[t] for t in T])
cand = {"labels": L, "teacher": A, "flat": 0.5 * L + 0.5 * A,
        "silent_only": np.where(silent, A, L), "grouped": (1 - gw) * L + gw * A}
R["targets"] = {k: {"per_finding": dict(zip(T, per_finding(Y, v).round(4))),
                    "macro": float(np.nanmean(per_finding(Y, v)))} for k, v in cand.items()}
R["targets_contrasts"] = {}
for a, b in (("teacher", "labels"), ("flat", "labels"), ("flat", "teacher"),
             ("silent_only", "labels"), ("grouped", "labels")):
    m, lo, hi, p = paired_diff(Y, cand[a], cand[b], boots)
    R["targets_contrasts"][f"{a}-{b}"] = {"diff": m, "lo": lo, "hi": hi, "p": p}
grid = {}
for ws in (0.3, 0.4, 0.5, 0.6):
    for wu in (0.5, 0.6, 0.7, 0.8, 0.9):
        w = np.where(silent, wu, ws)
        grid[f"{ws}/{wu}"] = float(np.nanmean(per_finding(Y, (1 - w) * L + w * A)))
R["state_grid"] = grid

# ------------------------------------------------------------------ our model (Table 3)
O = ours.loc[gid, T].to_numpy(float)
ra_all, ro_all = pct_rank(arm.to_numpy(float)), pct_rank(ours.to_numpy(float))
blend_all = pct_rank(0.5 * ra_all + 0.5 * ro_all)
B = blend_all[gpos]
R["oof"] = {
    "gold": {"ours": dict(zip(T, per_finding(Y, O).round(4))), "teacher": dict(zip(T, per_finding(Y, A).round(4))),
             "macro_ours": float(np.nanmean(per_finding(Y, O))), "macro_teacher": float(np.nanmean(per_finding(Y, A))),
             "macro_blend": float(np.nanmean(per_finding(Y, B)))},
    "explicit": {"ours": dict(zip(T, explicit_per_finding(S, ours.to_numpy(float)).round(4))),
                 "teacher": dict(zip(T, explicit_per_finding(S, arm.to_numpy(float)).round(4)))},
    "rank_corr": float(np.mean([np.corrcoef(ra_all[:, j], ro_all[:, j])[0, 1] for j in range(12)])),
}
R["oof"]["explicit"]["macro_ours"] = float(np.mean(list(R["oof"]["explicit"]["ours"].values())))
R["oof"]["explicit"]["macro_teacher"] = float(np.mean(list(R["oof"]["explicit"]["teacher"].values())))
for name, (a, b) in {"ours-teacher": (O, A), "blend-teacher": (B, A)}.items():
    m, lo, hi, p = paired_diff(Y, a, b, boots)
    R["oof"][name] = {"diff": m, "lo": lo, "hi": hi, "p": p}
r1 = 0.5 * L + 0.5 * A
r2 = 0.5 * L + 0.5 * (ra_all[gpos] + ro_all[gpos]) / 2
m, lo, hi, p = paired_diff(Y, r2, r1, boots)
R["round2"] = {"r1": float(np.nanmean(per_finding(Y, r1))), "r2": float(np.nanmean(per_finding(Y, r2))),
               "contrast": {"diff": m, "lo": lo, "hi": hi, "p": p}}


# ------------------------------------------------------------------ secondary metrics
def ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = np.asarray(y)[o].astype(float)
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def ece(y, p, bins=10):
    p = np.clip(np.asarray(p, float), 0, 1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (p >= lo_) & ((p < hi_) | (hi_ == 1))
        if m.any():
            e += m.mean() * abs(np.mean(y[m]) - p[m].mean())
    return float(e)


R["secondary"] = {}
for name, P_g, P_all in (("ours", O, ours.to_numpy(float)), ("teacher", A, arm.to_numpy(float))):
    aps = [ap(Y[:, j], P_g[:, j]) for j in range(12)]
    eces, sens, spec = [], [], []
    for j in range(12):
        msk = np.isin(S[:, j], ["p", "n"])
        y = (S[msk, j] == "p").astype(float)
        eces.append(ece(y, P_all[msk, j]))
    for j in range(12):
        thr = np.quantile(P_g[:, j], 1 - Y[:, j].mean())          # threshold at the gold prevalence
        pred = P_g[:, j] >= thr
        sens.append(float((pred & (Y[:, j] == 1)).sum() / max(1, Y[:, j].sum())))
        spec.append(float((~pred & (Y[:, j] == 0)).sum() / max(1, (Y[:, j] == 0).sum())))
    R["secondary"][name] = {"ap_gold": dict(zip(T, np.round(aps, 4))), "ap_gold_macro": float(np.nanmean(aps)),
                            "ece_explicit": dict(zip(T, np.round(eces, 4))), "ece_explicit_macro": float(np.mean(eces)),
                            "sens_gold": dict(zip(T, np.round(sens, 3))), "spec_gold": dict(zip(T, np.round(spec, 3)))}
R["secondary"]["gold_prevalence"] = dict(zip(T, Y.mean(0).round(3)))
R["secondary"]["stated_cells"] = {t: int(np.isin(S[:, j], ["p", "n"]).sum()) for j, t in enumerate(T)}

# reliability on the stated report cells, ten equal-width bins
R["reliability"] = {}
edges = np.linspace(0, 1, 11)
for name, P_all in (("ours", ours.to_numpy(float)), ("teacher", arm.to_numpy(float))):
    R["reliability"][name] = {}
    for j, t in enumerate(T):
        msk = np.isin(S[:, j], ["p", "n"])
        y, q = (S[msk, j] == "p").astype(float), np.clip(P_all[msk, j], 0, 1)
        k = np.clip(np.digitize(q, edges) - 1, 0, 9)
        R["reliability"][name][t] = [[float(q[k == b].mean()), float(y[k == b].mean()), int((k == b).sum())]
                                     for b in range(10) if (k == b).any()]

# ------------------------------------------------------------------ explaining away (Table 4, Figure)
W = cooccurrence_weights(lab.loc[~lab.IS_GOLD.astype(bool), T])
R["W_top"] = {t: [T[int(np.argmax(W[j]))], float(W[j].max().round(3))] for j, t in enumerate(T)}
models = {"teacher": ra_all, "ours": ro_all, "blend": blend_all}
LAMS = [-0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
R["ea"] = {}
tests = []
for name, Rk in models.items():
    base_e = explicit_per_finding(S, Rk)
    base_g = per_finding(Y, Rk[gpos])
    rows = {}
    for lam in LAMS:
        P = explain_away(Rk, W, lam)
        e, g = explicit_per_finding(S, P), per_finding(Y, P[gpos])
        row = {"explicit": float(e.mean()), "gold": float(np.nanmean(g)),
               "d_explicit": float(e.mean() - base_e.mean()), "d_gold": float(np.nanmean(g) - np.nanmean(base_g))}
        if lam < 0:
            m, lo, hi, p = paired_diff(Y, P[gpos], Rk[gpos], boots)
            row.update(boot=m, lo=lo, hi=hi, p=p, p_delong=delong_macro(Y, P[gpos], Rk[gpos])[2])
            tests.append((name, lam, p))
        if name == "blend" and lam == -0.1:
            row["per_finding_explicit"] = dict(zip(T, (e - base_e).round(4)))
        rows[str(lam)] = row
    R["ea"][name] = rows
adj = holm([t[2] for t in tests])
adj_d = holm([R["ea"][name][str(lam)]["p_delong"] for name, lam, _ in tests])
for (name, lam, _), a, ad in zip(tests, adj, adj_d):
    R["ea"][name][str(lam)]["p_holm"] = float(a)
    R["ea"][name][str(lam)]["p_delong_holm"] = float(ad)

# ------------------------------------------------------------------ header fields (Table 5, Figure)
meta = pd.read_csv(ROOT / "data/meta_train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
meta = meta.reindex(lab.index)
R["headers"] = {"age_known": float(meta.PatientAge.notna().mean()),
                "sex_known": float(meta.PatientSex.notna().mean()),
                "sex_counts": meta.PatientSex.value_counts().to_dict(),
                "manufacturer_strings": int(meta.Manufacturer.nunique()),
                "models": int(meta.ManufacturerModelName.nunique()),
                "field_counts": meta.MagneticFieldStrength.round(2).value_counts().head(2).to_dict()}
rng = np.random.default_rng(SEED)
half = rng.random(len(lab)) < 0.5
male = (meta.PatientSex.astype(str).str.upper().str[0] == "M").astype(float).to_numpy()
Lall = lab[T].astype(float).to_numpy()
GRIDW = np.round(np.arange(-0.4, 0.41, 0.05), 2)


def score_pair(P):
    return float(explicit_per_finding(S, P).mean()), float(np.nanmean(per_finding(Y, P[gpos])))


corr = {}
for name, Rk in models.items():
    e0, g0 = score_pair(Rk)
    res = {"none": (e0, g0)}
    # sex: per-finding weight chosen on one half by explicit AUC, applied to the other half
    P = Rk.copy()
    for j in range(12):
        for fit, ev in ((half, ~half), (~half, half)):
            fr = np.flatnonzero(fit)
            m = np.isin(S[fr, j], ["p", "n"])
            best = max(GRIDW, key=lambda w: auc(S[fr, j][m] == "p", (Rk[fr, j] + w * male[fr])[m]))
            P[ev, j] = Rk[ev, j] + best * male[ev]
    res["sex"] = score_pair(P)
    for key, short in (("ManufacturerModelName", "model"), ("Manufacturer", "manufacturer"),
                       ("MagneticFieldStrength", "field")):
        site = meta[key].fillna("NA").astype(str).to_numpy()
        dev = pd.DataFrame(Rk).groupby(site).transform("mean").to_numpy() - 0.5
        for c in (0.25, 1.0):
            res[f"{short}_centring_{c}"] = score_pair(Rk - c * dev)
        if short != "field":
            P = Rk.copy()
            for fit, ev in ((half, ~half), (~half, half)):
                df = pd.DataFrame(Lall[fit])
                pri = df.groupby(site[fit]).mean()
                cnt = pd.Series(site[fit]).value_counts()
                glob = Lall[fit].mean(0)
                prior = np.array([(pri.loc[s].to_numpy() * cnt[s] + glob * 20) / (cnt[s] + 20)
                                  if s in pri.index else glob for s in site[ev]])
                P[ev] = Rk[ev] + 0.1 * pct_rank(prior)
            res[f"{short}_prior_0.1"] = score_pair(P)
    for lam in (0.1, 0.2):
        res[f"boost_{lam}"] = score_pair(explain_away(Rk, W, lam, keep=()))
    for lam in (-0.1, -0.2):
        res[f"ea_{lam}"] = score_pair(explain_away(Rk, W, lam))
    corr[name] = {k: {"explicit": v[0], "gold": v[1], "d_explicit": v[0] - e0, "d_gold": v[1] - g0}
                  for k, v in res.items()}
R["corrections"] = corr

# ------------------------------------------------------------------ labels-only student (same recipe, no teacher)
runs = ROOT / "data/runs"
if all((runs / f"oof_v2_r50v3lab_f{k}.csv").exists() for k in range(5)):
    lo = pd.concat([pd.read_csv(runs / f"oof_v2_r50v3lab_f{k}.csv").set_index("StudyInstanceUID") for k in range(5)])
    lo = lo[~lo.index.duplicated()].reindex(lab.index)[T]
    assert lo.notna().all().all()
    Lo = lo.to_numpy(float)
    m, lo_, hi_, p = paired_diff(Y, O, Lo[gpos], boots)
    ex_l, ex_o = explicit_per_finding(S, Lo), explicit_per_finding(S, ours.to_numpy(float))
    rng_e = np.random.default_rng(SEED)
    de = []
    for _ in range(N_BOOT):
        b = rng_e.integers(0, len(S), len(S))
        de.append(np.mean(explicit_per_finding(S[b], ours.to_numpy(float)[b])) - np.mean(explicit_per_finding(S[b], Lo[b])))
    R["labels_only"] = {
        "gold": dict(zip(T, per_finding(Y, Lo[gpos]).round(4))), "macro_gold": float(np.nanmean(per_finding(Y, Lo[gpos]))),
        "explicit": dict(zip(T, ex_l.round(4))), "macro_explicit": float(ex_l.mean()),
        "mixed_minus_labels_gold": {"diff": m, "lo": lo_, "hi": hi_, "p": p},
        "mixed_minus_labels_explicit": {"diff": float(ex_o.mean() - ex_l.mean()),
                                        "lo": float(np.percentile(de, 2.5)), "hi": float(np.percentile(de, 97.5))},
    }

# ------------------------------------------------------------------ fold-0 ablations
fold0_runs = {"v1_noprior_unseeded": "oof_v2_radr50_f0.csv", "v3_noprior_seeded": "oof_v2_r50v3noprior_f0.csv",
              "v3_prior_seeded": "oof_v2_r50v3s2026a_f0.csv", "v3_prior_seeded_repeat": "oof_v2_r50v3s2026b_f0.csv",
              "v3_prior_seed2027": "oof_v2_r50v3seed2027_f0.csv", "v3_prior_seed2028": "oof_v2_r50v3seed2028_f0.csv",
              "v3_prior_labels_only_seeded": "oof_v2_r50v3lab_f0.csv"}
if all((runs / f).exists() for f in fold0_runs.values()):
    man = pd.read_csv(ROOT / "data/fold_manifest.csv").set_index("StudyInstanceUID").reindex(lab.index)
    f0 = (man.fold == 0).to_numpy()
    ids0 = lab.index[f0]
    g0 = np.array([man.loc[u, "fold"] == 0 for u in gid])
    S0 = S[f0]
    tables = {k: pd.read_csv(runs / f).set_index("StudyInstanceUID") for k, f in fold0_runs.items()}
    tables["v3_prior_unseeded"] = ours
    tables["teacher"] = arm
    ref = tables["v3_prior_seeded"].loc[ids0, T].to_numpy(float)
    rng_a = np.random.default_rng(SEED)
    bs = [rng_a.integers(0, len(ids0), len(ids0)) for _ in range(N_BOOT)]
    e_ref = explicit_per_finding(S0, ref).mean()
    R["ablation_fold0"] = {"studies": int(f0.sum()), "gold_studies": int(g0.sum())}
    for k, df in tables.items():
        P0 = df.loc[ids0, T].to_numpy(float)
        e = explicit_per_finding(S0, P0).mean()
        g = np.nanmean(per_finding(Y[g0], df.loc[gid[g0], T].to_numpy(float)))
        d = [explicit_per_finding(S0[b], P0[b]).mean() - explicit_per_finding(S0[b], ref[b]).mean() for b in bs]
        R["ablation_fold0"][k] = {"explicit": float(e), "gold": float(g), "d_explicit": float(e - e_ref),
                                  "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


# ------------------------------------------------------------------ DeLong-type check of the main contrasts
R["delong"] = {}
pairs = {"teacher-labels": (A, L), "flat-labels": (cand["flat"], L), "flat-teacher": (cand["flat"], A),
         "ours-teacher": (O, A), "blend-teacher": (B, A)}
if "labels_only" in R:
    pairs["mixed-labels_only_student"] = (O, Lo[gpos])
for k, (a_, b_) in pairs.items():
    d, se, p = delong_macro(Y, a_, b_)
    R["delong"][k] = {"diff": d, "se": se, "p": p}

# ------------------------------------------------------------------ what the reports say on the gold studies
R["report_states_gold"] = {}
for j, t in enumerate(T):
    pos, neg = Y[:, j] == 1, Y[:, j] == 0
    cnt = lambda m, st: int((Sg[m, j] == st).sum()) if st != "u" else int((~np.isin(Sg[m, j], ["p", "n"])).sum())
    R["report_states_gold"][t] = {"pos_present": cnt(pos, "p"), "pos_absent": cnt(pos, "n"), "pos_silent": cnt(pos, "u"),
                                  "neg_present": cnt(neg, "p"), "neg_absent": cnt(neg, "n"), "neg_silent": cnt(neg, "u")}
    # expert positives whose report does not state the finding present: how often does each source
    # still rank them above the expert negatives (share of pairs, ties count one half)
    miss = pos & (Sg[:, j] != "p")
    if miss.any():
        def pairs_above(P):
            u, v = P[miss, j][:, None], P[neg, j][None, :]
            return float(((u > v) + 0.5 * (u == v)).mean())
        R["report_states_gold"][t].update(missed_above_neg_labels=pairs_above(L), missed_above_neg_teacher=pairs_above(A),
                                          missed_label_median=float(np.median(L[miss, j])),
                                          neg_label_median=float(np.median(L[neg, j])))

# ------------------------------------------------------------------ omission rate against the teacher's advantage
from scipy.stats import spearmanr
adv = per_finding(Y, A) - per_finding(Y, L)
om_all = (~stated).mean(0)
om_gold = silent.mean(0)
rng_o = np.random.default_rng(SEED)
rho = spearmanr(om_all, adv).statistic
null = np.array([spearmanr(rng_o.permutation(om_all), adv).statistic for _ in range(9999)])
rho_g = spearmanr(om_gold, adv).statistic
null_g = np.array([spearmanr(rng_o.permutation(om_gold), adv).statistic for _ in range(9999)])
R["omission"] = {"per_finding": {t: {"omitted_all": float(om_all[j]), "omitted_gold": float(om_gold[j]),
                                     "teacher_minus_labels": float(adv[j])} for j, t in enumerate(T)},
                 "spearman_all": float(rho), "p_all": float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / 10000),
                 "spearman_gold": float(rho_g), "p_gold": float((1 + np.sum(np.abs(null_g) >= abs(rho_g) - 1e-12)) / 10000)}

# ------------------------------------------------------------------ side (left/right) audit: gold sensitivity
aud = pd.read_csv(ROOT / "data/slice_audit.csv", dtype={"StudyInstanceUID": str})


def series_side(r):
    # the rule of cache3.order_slices: tag first, else median x at least 20 mm from 0
    t = r.laterality_tag
    if isinstance(t, str) and t.strip()[:1].upper() in ("L", "R"):
        return t.strip()[:1].upper()
    if r.n_pos >= 3 and pd.notna(r.median_x) and abs(r.median_x) >= 20:
        return "R" if r.median_x < 0 else "L"
    return None


aud["side"] = aud.apply(series_side, axis=1)
by = aud.groupby("StudyInstanceUID")
conflict = by.side.apply(lambda v: v.dropna().nunique() > 1)
unresolved = by.side.apply(lambda v: v.isna().any())
R["side_audit"] = {"series": len(aud), "series_no_side": int(aud.side.isna().sum()),
                   "studies": int(len(conflict)), "studies_conflict": int(conflict.sum()),
                   "studies_any_unresolved": int(unresolved.sum()),
                   "gold_conflict": int(conflict.reindex(gid).fillna(False).sum()),
                   "gold_any_unresolved": int(unresolved.reindex(gid).fillna(False).sum()),
                   "gold_conflict_or_unresolved": int((conflict | unresolved).reindex(gid).fillna(False).sum())}
R["side_sensitivity"] = {}
subsets = {"all": np.ones(len(gid), bool),
           "no_conflict": ~conflict.reindex(gid).fillna(False).to_numpy(bool),
           "no_conflict_no_unresolved": ~(conflict | unresolved).reindex(gid).fillna(False).to_numpy(bool)}
for k, keep in subsets.items():
    Yk = Y[keep]
    bk = boot_indices(int(keep.sum()))
    row = {"n": int(keep.sum()), "min_pos": int(Yk.sum(0).min()),
           "macro": {c: float(np.nanmean(per_finding(Yk, cand[c][keep]))) for c in ("labels", "teacher", "flat")}}
    for a_, b_ in (("teacher", "labels"), ("flat", "labels"), ("flat", "teacher")):
        m, lo, hi, p = paired_diff(Yk, cand[a_][keep], cand[b_][keep], bk)
        row[f"{a_}-{b_}"] = {"diff": m, "lo": lo, "hi": hi, "p": p}
    R["side_sensitivity"][k] = row

json.dump(R, open(OUT / "numbers.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ markdown
md.append("# Tables recomputed from the released files\n")
md.append(f"Differences are observed values on the {len(gid)} gold studies; intervals: {N_BOOT} bootstrap resamples of the studies (seed {SEED}); p: paired permutation test with {N_BOOT} within-study swaps.\n")
md.append("## Targets against gold-58 (target agreement AUC)\n")
md.append("| Finding | Labels | Teacher | Flat | Silent only |\n|---|---|---|---|---|")
for t in T:
    md.append(f"| {t} | " + " | ".join(f"{R['targets'][k]['per_finding'][t]:.3f}"
                                         for k in ("labels", "teacher", "flat", "silent_only")) + " |")
md.append("| Macro | " + " | ".join(f"{R['targets'][k]['macro']:.3f}"
                                     for k in ("labels", "teacher", "flat", "silent_only")) + " |\n")
for k, v in R["targets_contrasts"].items():
    md.append(f"- {k}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}], p {fmt_p(v['p'])}")
md.append("\n## Out-of-fold, ours and teacher\n")
md.append(f"- gold macro ours {R['oof']['gold']['macro_ours']:.4f}, teacher {R['oof']['gold']['macro_teacher']:.4f}, "
          f"blend {R['oof']['gold']['macro_blend']:.4f}")
for k in ("ours-teacher", "blend-teacher"):
    v = R["oof"][k]
    md.append(f"- {k}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}], p {fmt_p(v['p'])}")
md.append(f"- round 2 target {R['round2']['r2']:.4f} vs round 1 {R['round2']['r1']:.4f}: "
          f"{R['round2']['contrast']['diff']:+.4f} [{R['round2']['contrast']['lo']:+.4f}, {R['round2']['contrast']['hi']:+.4f}]")
if "labels_only" in R:
    v, w = R["labels_only"]["mixed_minus_labels_gold"], R["labels_only"]["mixed_minus_labels_explicit"]
    md.append(f"- labels-only student: gold {R['labels_only']['macro_gold']:.4f}, explicit {R['labels_only']['macro_explicit']:.4f}")
    md.append(f"- mixed - labels-only, gold: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}], p {fmt_p(v['p'])}")
    md.append(f"- mixed - labels-only, explicit: {w['diff']:+.4f} [{w['lo']:+.4f}, {w['hi']:+.4f}]")
if "ablation_fold0" in R:
    md.append("- fold-0 runs (explicit, gold, explicit minus the seeded v3 prior run with 95% interval):")
    for k, v in R["ablation_fold0"].items():
        if isinstance(v, dict):
            md.append(f"  - {k}: {v['explicit']:.4f}, {v['gold']:.4f}, {v['d_explicit']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
md.append("\n## Explaining away (observed gold difference, bootstrap interval, permutation p, Holm over the 15 negative settings)\n")
md.append("| Scores | lambda | explicit | gold | gold diff [95% CI] | p | p Holm | p DeLong | DeLong Holm |\n|---|---|---|---|---|---|---|---|---|")
for name in models:
    for lam in LAMS:
        v = R["ea"][name][str(lam)]
        ci = f"{v['boot']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" if "boot" in v else ""
        p = fmt_p(v['p']) if "p" in v else ""
        ph = fmt_p(v['p_holm']) if "p_holm" in v else ""
        pd_ = fmt_p(v['p_delong']) if "p_delong" in v else ""
        pdh = fmt_p(v['p_delong_holm']) if "p_delong_holm" in v else ""
        md.append(f"| {name} | {lam} | {v['explicit']:.4f} | {v['gold']:.4f} | {ci} | {p} | {ph} | {pd_} | {pdh} |")
md.append("\n## Corrections: change in explicit and gold macro AUC\n")
md.append("| Correction | " + " | ".join(f"{m} d_explicit | {m} d_gold" for m in models) + " |")
md.append("|---|" + "---|" * (2 * len(models)))
for k in corr["blend"]:
    md.append(f"| {k} | " + " | ".join(f"{corr[m][k]['d_explicit']:+.4f} | {corr[m][k]['d_gold']:+.4f}"
                                       for m in models) + " |")
md.append("\n## DeLong-type check (difference, standard error, two-sided p)\n")
for k, v in R["delong"].items():
    md.append(f"- {k}: {v['diff']:+.4f}, se {v['se']:.4f}, p {fmt_p(v['p'])}")
md.append("\n## Report state on the gold studies (expert positive: present/absent/silent | expert negative: present/absent/silent)\n")
for t, v in R["report_states_gold"].items():
    md.append(f"- {t}: {v['pos_present']}/{v['pos_absent']}/{v['pos_silent']} | {v['neg_present']}/{v['neg_absent']}/{v['neg_silent']}")
o = R["omission"]
md.append(f"\n## Omission rate and teacher advantage: Spearman {o['spearman_all']:.3f} (p {fmt_p(o['p_all'])}) over all reports, "
          f"{o['spearman_gold']:.3f} (p {fmt_p(o['p_gold'])}) over the gold reports\n")
sa = R["side_audit"]
md.append(f"## Side audit: {sa['series_no_side']} of {sa['series']} series without a side; {sa['studies_conflict']} of {sa['studies']} studies "
          f"with conflicting sides; gold: {sa['gold_conflict']} conflicting, {sa['gold_any_unresolved']} with an unresolved series, "
          f"{sa['gold_conflict_or_unresolved']} either\n")
md.append("| Gold subset | n | labels | teacher | flat | flat-labels [CI] | flat-teacher [CI] | teacher-labels [CI] |\n|---|---|---|---|---|---|---|---|")
for k, v in R["side_sensitivity"].items():
    f = lambda c: f"{v[c]['diff']:+.4f} [{v[c]['lo']:+.4f}, {v[c]['hi']:+.4f}]"
    md.append(f"| {k} | {v['n']} | {v['macro']['labels']:.4f} | {v['macro']['teacher']:.4f} | {v['macro']['flat']:.4f} | "
              f"{f('flat-labels')} | {f('flat-teacher')} | {f('teacher-labels')} |")
(OUT / "tables.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
