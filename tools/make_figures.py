"""Figures of the paper, from results/numbers.json and data/submissions.csv.

    python tools/make_figures.py  ->  results/fig_*.pdf
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, T

OUT = ROOT / "results"
R = json.load(open(OUT / "numbers.json"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
SHORT = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "Medial meniscus", "Lateral Meniscus": "Lateral meniscus",
         "Medial OA": "Medial OA", "Lateral OA": "Lateral OA", "PF OA": "PF OA", "Effusion": "Effusion",
         "Synovitis": "Synovitis", "Baker's": "Baker's cyst", "Contusion": "Contusion", "Fracture": "Fracture"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# 1. what the reports state, per finding
pf = R["data"]["per_finding"]
order = sorted(T, key=lambda t: pf[t]["silent"])
n = R["data"]["studies"]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
y = np.arange(len(order))
pres = np.array([pf[t]["present"] for t in order]) / n
absn = np.array([pf[t]["absent"] for t in order]) / n
sil = np.array([pf[t]["silent"] for t in order]) / n
ax.barh(y, pres, color=BLUE, height=0.7, label="Stated present", edgecolor="white", linewidth=1)
ax.barh(y, absn, left=pres, color=ORANGE, height=0.7, label="Stated absent", edgecolor="white", linewidth=1)
ax.barh(y, sil, left=pres + absn, color=GRID, height=0.7, label="Not mentioned", edgecolor="white", linewidth=1)
ax.set_yticks(y, [SHORT[t] for t in order])
ax.set_xlim(0, 1)
ax.set_xlabel("Share of the 4,407 training studies")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=3, handlelength=1, columnspacing=0.8)
save(fig, "fig_coverage")

# 2. report labels and teacher fail on different findings
tg = R["targets"]
order = sorted(T, key=lambda t: tg["labels"]["per_finding"][t] - tg["teacher"]["per_finding"][t])
fig, ax = plt.subplots(figsize=(3.4, 2.7))
y = np.arange(len(order))
lab_v = np.array([tg["labels"]["per_finding"][t] for t in order])
tea_v = np.array([tg["teacher"]["per_finding"][t] for t in order])
fl_v = np.array([tg["flat"]["per_finding"][t] for t in order])
for i in range(len(order)):
    ax.plot([lab_v[i], tea_v[i]], [y[i], y[i]], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
ax.scatter(lab_v, y, s=26, color=BLUE, zorder=3, label="Report labels", edgecolor="white", linewidth=0.8)
ax.scatter(tea_v, y, s=26, color=ORANGE, zorder=3, label="Image teacher", edgecolor="white", linewidth=0.8)
ax.scatter(fl_v, y, s=30, color=AQUA, marker="D", zorder=4, label="0.5 / 0.5 mixture", edgecolor="white",
           linewidth=0.8)
ax.set_yticks(y, [SHORT[t] for t in order])
ax.set_xlabel("Target agreement AUC with the expert labels (58 studies)")
ax.set_xlim(0.77, 1.012)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.13), ncol=3, handlelength=1, columnspacing=0.8)
save(fig, "fig_complementarity")

# 3. explaining away: lambda sweep
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), sharex=True)
for name, col, lab in (("teacher", ORANGE, "Teacher"), ("ours", BLUE, "Our model"), ("blend", AQUA, "Blend")):
    lams = sorted(map(float, R["ea"][name]))
    de = [R["ea"][name][str(l)]["d_explicit"] for l in lams]
    dg = [R["ea"][name][str(l)]["d_gold"] for l in lams]
    axes[0].plot(lams, de, color=col, lw=2, marker="o", ms=4, label=lab, mec="white", mew=0.6)
    axes[1].plot(lams, dg, color=col, lw=2, marker="o", ms=4, label=lab, mec="white", mew=0.6)
b = R["ea"]["blend"]
neg = sorted(float(l) for l in b if float(l) < 0)
axes[1].fill_between(neg, [b[str(l)]["lo"] for l in neg], [b[str(l)]["hi"] for l in neg], color=AQUA, alpha=0.15,
                     lw=0, label="Blend, 95% interval")
for ax, ttl in zip(axes, ("Report statements (39,248 cells)", "Expert labels (58 studies)")):
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_title(ttl, color=INK, loc="left")
    ax.set_xlabel(r"$\lambda$")
axes[0].set_ylabel("Change in macro AUC")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, handlelength=1.4)
save(fig, "fig_ea")

# 4. the two references disagree for a scanner prior
c = R["corrections"]["blend"]
groups = {
    "Explaining away": [("ea_-0.1", r"EA $-0.1$"), ("ea_-0.2", r"EA $-0.2$")],
    "Co-occurrence support": [("boost_0.1", r"support $+0.1$"), ("boost_0.2", r"support $+0.2$")],
    "Header fields": [("sex", "sex"), ("model_centring_0.25", "model centring"),
                      ("manufacturer_centring_0.25", "maker centring"), ("field_centring_0.25", "field centring"),
                      ("manufacturer_prior_0.1", "manufacturer prior"), ("model_prior_0.1", "model prior")],
}
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for (g, items), col in zip(groups.items(), (AQUA, ORANGE, BLUE)):
    xs = [c[k]["d_explicit"] for k, _ in items]
    ys = [c[k]["d_gold"] for k, _ in items]
    ax.scatter(xs, ys, s=30, color=col, label=g, edgecolor="white", linewidth=0.8, zorder=3)
    for (k, lab), x, yv in zip(items, xs, ys):
        off = {"manufacturer_prior_0.1": (-62, 3), "model_prior_0.1": (-50, -10), "ea_-0.1": (-30, 3),
               "ea_-0.2": (-30, 3), "boost_0.2": (3, 4), "sex": (4, -9)}.get(k)
        if off:
            ax.annotate(lab, (x, yv), xytext=off, textcoords="offset points", fontsize=6.8, color=INK2)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Change in report-statement AUC")
ax.set_ylabel("Change in expert-label AUC")
ax.legend(loc="upper left", handlelength=1)
save(fig, "fig_references")

# 5. leaderboard: public and private score of every scored submission
sub = pd.read_csv(ROOT / "data/submissions.csv")
sub = sub[sub.row > 1].iloc[::-1].reset_index(drop=True)
cols = {"own": BLUE, "public": ORANGE, "public+own": AQUA}
labs = {"own": "Our model alone", "public": "Public ensemble (fork)", "public+own": "Public ensemble with our part"}
fig, ax = plt.subplots(figsize=(3.4, 3.0))
y = np.arange(len(sub))
for i, r in sub.iterrows():
    ax.plot([r.private, r.public], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter(r.public, i, s=22, facecolor="white", edgecolor=cols[r.kind], linewidth=1.2, zorder=3)
    ax.scatter(r.private, i, s=24, color=cols[r.kind], edgecolor="white", linewidth=0.8, zorder=4)
for k in ("own", "public", "public+own"):
    ax.scatter([], [], s=24, color=cols[k], label=labs[k])
ax.scatter([], [], s=22, facecolor="white", edgecolor=MUTED, linewidth=1.2, label="public score (hollow)")
ax.set_yticks(y, [f"{r.row}" for _, r in sub.iterrows()])
ax.set_ylabel("Submission (row)")
ax.set_xlabel("Leaderboard macro AUC (filled: private)")
ax.set_xlim(0.86, 0.962)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handlelength=1, columnspacing=0.8)
save(fig, "fig_leaderboard")

# 6. learning curves (appendix)
# per-epoch validation AUC, transcribed from the Kaggle training logs of family r50v3
lc = pd.read_csv(ROOT / "data/learning_curves.csv").query("family == 'r50v3'")
EP = {k: (g.sort_values("epoch").explicit_auc.to_numpy(), g.sort_values("epoch").gold_auc.to_numpy())
      for k, g in lc.groupby("fold")}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), sharex=True)
for k, (e, g) in EP.items():
    axes[0].plot(range(8), e, color=BLUE, lw=1.2, alpha=0.45)
    axes[1].plot(range(8), g, color=ORANGE, lw=1.2, alpha=0.45)
axes[0].plot(range(8), np.mean([v[0] for v in EP.values()], 0), color=BLUE, lw=2.2)
axes[1].plot(range(8), np.mean([v[1] for v in EP.values()], 0), color=ORANGE, lw=2.2)
axes[0].set_title("Report statements of the held-out fold", loc="left", color=INK)
axes[1].set_title("Expert labels of the held-out fold (10 to 12 studies)", loc="left", color=INK)
for ax in axes:
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Macro AUC")
save(fig, "fig_epochs")

# 7. reliability on the stated report cells (appendix)
fig, axes = plt.subplots(3, 4, figsize=(6.9, 5.0), sharex=True, sharey=True)
for ax, t in zip(axes.ravel(), T):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    for name, col, lab in (("teacher", ORANGE, "Teacher"), ("ours", BLUE, "Student")):
        b = np.array(R["reliability"][name][t])
        ax.plot(b[:, 0], b[:, 1], color=col, lw=1.6, marker="o", ms=3, mec="white", mew=0.5, label=lab)
    ax.set_title(f"{SHORT[t]} (n={R['secondary']['stated_cells'][t]:,})", loc="left", color=INK, fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
for ax in axes[-1]:
    ax.set_xlabel("Predicted probability")
for ax in axes[:, 0]:
    ax.set_ylabel("Share stated present")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, handlelength=1.4)
fig.tight_layout()
save(fig, "fig_reliability")

# 8. private leaderboard: teams per score, our submissions marked
ps = pd.read_csv(ROOT / "data/private_scores.csv")
ps = ps[(ps.private_score >= 0.930) & (ps.private_score <= 0.960)]
fig, ax = plt.subplots(figsize=(6.9, 2.3))
cols = [AQUA if abs(v - 0.943) < 1e-9 else (ORANGE if abs(v - 0.942) < 1e-9 else MUTED) for v in ps.private_score]
ax.bar(ps.private_score, ps.teams, width=0.0008, color=cols, edgecolor="white", linewidth=0.5)
ax.set_yscale("log")
ax.set_xlim(0.9295, 0.9605)
ax.set_xlabel("Private leaderboard score (three decimals)")
ax.set_ylabel("Teams")
ax.grid(axis="x", visible=False)
ax.annotate("1,297 teams at 0.942,\nthe score of the public ensembles", (0.9416, 900), xytext=(0.9302, 700),
            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("ours, 0.943\n(rank 582 of 6,299)", (0.943, 287), xytext=(0.9475, 400), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.annotate("best, 0.960", (0.960, 2), xytext=(0.9545, 12), fontsize=7, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
save(fig, "fig_lb_distribution")
print("figures written to", OUT)

# 9. how often the reports leave a finding out, against the teacher's advantage on the gold studies
om = R["omission"]["per_finding"]
x = np.array([om[t]["omitted_all"] for t in T])
yv = np.array([om[t]["teacher_minus_labels"] for t in T])
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
col = [ORANGE if v > 0 else BLUE for v in yv]
ax.scatter(x, yv, s=30, c=col, zorder=3, edgecolor="white", linewidth=0.8)
off = {"ACL": (6, 0), "MCL": (6, 0), "Medial Meniscus": (-5, -12), "Lateral Meniscus": (6, 1),
       "Medial OA": (-6, 0), "Lateral OA": (6, 1), "PF OA": (6, 0), "Effusion": (6, 0),
       "Synovitis": (0, -11), "Baker's": (6, 0), "Contusion": (6, 0), "Fracture": (6, 0)}
ha = {"Medial OA": "right", "Synovitis": "center"}
for t, xi, yi in zip(T, x, yv):
    dx, dy = off[t]
    ax.annotate(SHORT[t], (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                color=INK2, ha=ha.get(t, "left"), va="center")
ax.set_xlim(0, 0.95)
ax.set_ylim(-0.17, 0.17)
ax.set_xlabel("Share of the 4,407 reports that do not mention the finding")
ax.set_ylabel("Teacher minus report labels\n(target agreement AUC, 58 studies)")
ax.text(0.94, 0.155, "teacher better", ha="right", va="top", fontsize=7, color=ORANGE)
ax.text(0.94, -0.155, "report labels better", ha="right", va="bottom", fontsize=7, color=BLUE)
save(fig, "fig_omission")
