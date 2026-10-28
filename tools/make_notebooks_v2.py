"""Write the Kaggle notebooks 04 (train), 05 (submit), 06 (cache v3), 07 (slice audit) and 08 (slot attention)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

BOOT = r'''# Find the pipeline package without walking the DICOM tree.
import sys, os, time, gc, json
from pathlib import Path

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

def walk(root="/kaggle/input", max_depth=8, suffixes=None):
    base = len(Path(root).parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if len(here.parts) - base >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if not d.lower().endswith(("_series", "_images"))]
        for f in filenames:
            if suffixes is None or Path(f).suffix.lower() in suffixes:
                yield here / f

PKG = next((p.parent.parent for p in walk(suffixes=(".py",))
            if p.name == "targets.py" and p.parent.name == "rsna_knee"), None)
if PKG is None:
    raise FileNotFoundError("attach the rsna-knee-pipeline dataset (v2, it must contain rsna_knee/targets.py)")
sys.path.insert(0, str(PKG))

import numpy as np
import pandas as pd
from rsna_knee.report_labels import TARGETS
log(f"pipeline: {PKG}")
'''

TRAIN_CFG = r'''# ---- settings for this run (defaults: the reported family r50v3) ---------------
ENCODER = "rad_resnet50"          # or a timm name, e.g. "convnext_tiny.fb_in22k_ft_in1k"
TAG = "r50v3"                     # family name; checkpoints are v2_<TAG>_fold<k>.pt
SEED = 2026
FOLDS = [0]                       # one fold per notebook run
FULL_DATA = False                 # True: one extra model on all studies (v2_<TAG>_full.pt)
EPOCHS = 8
CACHE = "v3"                      # "v1": old 288 px cache   "v3": cache from notebook 06
GROUP = 3                         # 3: 2.5D windows   1: single slices (8 per slot)
SCHEME = "flat"                   # teacher share: "flat" (0.5), "grouped", "labels_only"
EXTRA_LABELS = []                 # other public report-label csv files, e.g. ["llm_labels_v4_blend.csv"]
SLOT_PRIOR = True                 # radiological finding-to-slot prior
CROP = True                       # random crop in training (v3: 224 of 256 px); False: whole slice
LOCAL_COPY = True                 # copy the v3 cache to the local disk before training (much faster reads)
EXTRA_TEACHERS = []               # round 2: e.g. ["oof_v2_radr50_f0.csv", "oof_v2_radr50_f1234.csv"]
LR_ENCODER = 2e-4 if ENCODER == "rad_resnet50" else 1e-4
# ---------------------------------------------------------------------------

from rsna_knee.data import CacheConfig, load_cache
from rsna_knee.targets import build_targets_v2, gold_auc_table, load_default_sources, TEACHER_SHARE
from rsna_knee.train2 import TrainConfigV2, train_one, write_oof, auc_table

WEIGHTS = None
if ENCODER == "rad_resnet50":
    WEIGHTS = next((str(p) for p in walk(suffixes=(".pt", ".pth")) if p.name == "ResNet50.pt"), None)
    if WEIGHTS is None:
        raise FileNotFoundError("attach Kaggle dataset marwanmath/resnet-50-radimagenet-marwan (ResNet50.pt)")
    log(f"RadImageNet weights: {WEIGHTS}")
else:
    # timm downloads the weights: turn Internet ON for this training notebook,
    # or attach a .pth whose file name starts with the encoder name.
    WEIGHTS = next((str(p) for p in walk(suffixes=(".pth", ".pt", ".safetensors"))
                    if p.name.startswith(ENCODER.split(".")[0])), None)
    log(f"timm weights file: {WEIGHTS} (None means download)")

FOLDS_CSV = next(p for p in walk(suffixes=(".csv",)) if p.name == "train_folds.csv")
train = pd.read_csv(FOLDS_CSV)
if CACHE == "v1":
    CFG = CacheConfig(px=288, slices_per_slot=12, group=3, crop_mm=130.0)
    CACHE_DIRS = sorted({p.parent for p in walk(suffixes=(".npy",)) if p.name.startswith("train_")})
    array, mask, studies = load_cache(CACHE_DIRS, "train", CFG)
    assert studies == train.StudyInstanceUID.tolist(), "cache order does not match train_folds.csv"
    IMG = 256 if CROP else 288
else:
    from rsna_knee.cache3 import load_cache3, copy_local, read_speed
    CACHE_DIRS = sorted({p.parent for p in walk(suffixes=(".npy",)) if p.name.startswith("c3_train_")})
    if LOCAL_COPY:
        CACHE_DIRS = copy_local(CACHE_DIRS, log=log)
    array, mask, missing = load_cache3(CACHE_DIRS, "train", train.StudyInstanceUID.tolist())
    read_speed(array, log=log)
    if missing:
        raise FileNotFoundError(f"{len(missing)} studies missing from the v3 cache: attach every shard")
    IMG = 224 if CROP else 256
EMPTY = mask.sum(1) == 0
log(f"cache {array.shape}, slots filled {mask.mean():.3f}, empty studies {EMPTY.sum()}")
'''

TRAIN_TARGETS = r'''labels, teachers, lab_path, oof_path = load_default_sources(EXTRA_TEACHERS, EXTRA_LABELS)
log(f"report labels: {lab_path}\nteachers: {[str(oof_path)] + EXTRA_TEACHERS}")

# Quality of the training targets themselves on gold-58, before gold overrides them.
t_nogold, _, _, _ = build_targets_v2(train, labels, teachers, scheme=SCHEME, use_gold=False)
lab_only, _, _, _ = build_targets_v2(train, labels, [], scheme="labels_only", use_gold=False)
cmp = gold_auc_table(train, lab_only).rename(columns={"auc": "labels"})
cmp["targets"] = gold_auc_table(train, t_nogold)["auc"]
log("\n" + cmp.round(3).to_string(index=False))
log(f"gold macro  labels {cmp.labels.mean():.4f}  ->  targets {cmp.targets.mean():.4f}")

targets, weights, explicit, is_gold = build_targets_v2(train, labels, teachers, scheme=SCHEME)
gold = train[list(TARGETS)].to_numpy(dtype=np.float32)
gold[~is_gold] = np.nan
'''

TRAIN_LOOP = r'''OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
tcfg = TrainConfigV2(encoder=ENCODER, encoder_weights=WEIGHTS, epochs=EPOCHS,
                     lr_encoder=LR_ENCODER, target_scheme=SCHEME, tag=TAG,
                     cache=CACHE, group=GROUP, img=IMG,
                     train_windows=4 if GROUP == 3 else 8, eval_windows=16 if GROUP == 3 else 8,
                     slot_prior=SLOT_PRIOR, seed=SEED)
deadline = T0 + tcfg.time_budget_s
oof = np.full((len(train), len(TARGETS)), np.nan, dtype=np.float32)
hist = {}
for fold in FOLDS:
    if time.time() > deadline:
        log(f"no time left for fold {fold}"); break
    tr = np.flatnonzero((train.fold.values != fold) & ~EMPTY)
    va = np.flatnonzero(train.fold.values == fold)
    log(f"==== fold {fold}: {len(tr)} train / {len(va)} valid")
    p, h = train_one(tcfg, array, mask, tr, va, targets, weights, explicit, gold,
                     OUT / f"v2_{TAG}_fold{fold}.pt", deadline=deadline, log=log)
    if p is not None:
        oof[va] = p
    hist[f"fold{fold}"] = h
    gc.collect()

if FULL_DATA and time.time() < deadline:
    log("==== full-data model")
    _, h = train_one(tcfg, array, mask, np.flatnonzero(~EMPTY), [], targets, weights, None, None,
                     OUT / f"v2_{TAG}_full.pt", deadline=deadline, log=log)
    hist["full"] = h

RUN = f"{TAG}_f{''.join(map(str, FOLDS))}"
json.dump(hist, open(OUT / f"history_{RUN}.json", "w"), indent=1)

# provenance: settings, package versions and a hash of every checkpoint written by this run
import hashlib, platform, torch, timm
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
manifest = {"config": {k: v for k, v in vars(tcfg).items() if k != "encoder_weights"},
            "folds": FOLDS, "scheme": SCHEME, "crop": CROP, "cache": CACHE,
            "labels": str(lab_path), "teachers": [str(oof_path)] + EXTRA_TEACHERS,
            "inputs_sha256": {str(q): sha256(q) for q in [lab_path, oof_path] + ([WEIGHTS] if WEIGHTS else [])},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "checkpoints": {p.name: sha256(p) for p in sorted(OUT.glob(f"v2_{TAG}_*.pt"))}}
json.dump(manifest, open(OUT / f"run_{RUN}.json", "w"), indent=1, default=str)
log(f"manifest: {manifest['checkpoints']}")
'''

TRAIN_REPORT = r'''done = np.isfinite(oof).all(axis=1)
if done.any():
    write_oof(train, oof, OUT / f"oof_v2_{TAG}_f{''.join(map(str, FOLDS))}.csv")
    ex = auc_table(explicit[done], oof[done])
    log(f"OOF vs explicit report statements ({done.sum()} studies): macro {ex['macro']:.4f}")
    g = gold_auc_table(train[done].reset_index(drop=True), oof[done])
    log("\n" + g.round(3).to_string(index=False))
    log(f"OOF gold macro: {g.auc.mean():.4f}   (labels alone were {cmp.labels.mean():.4f})")
log("done. Save this version's output as a Dataset, e.g. rsna-knee-v2-" + TAG)
'''

SUB_1 = r'''from rsna_knee.infer import find_root, write_benchmark, write_submission, validate_submission
ROOT = find_root()
test = pd.read_csv(ROOT / "test.csv")
SUB = Path("/kaggle/working/submission.csv")

# family weights for the rank blend; families not listed get 1.0
FAMILY_WEIGHTS = {}
# False: no constant file is written first, and any inference failure stops the notebook,
# so a timeout or a crash leaves no submission.csv instead of a benchmark that looks valid
ALLOW_BENCHMARK = False
if ALLOW_BENCHMARK:
    write_benchmark(test, SUB)
    log(f"benchmark written for {len(test)} studies")
# every mounted family must hold fold0 ... fold4; set None to allow a single-fold family
REQUIRE_FOLDS = 5
'''

SUB_2 = r'''from rsna_knee.data import CacheConfig, build_cache
from rsna_knee.infer2 import find_checkpoints, check_folds, predict_family, rank_blend, family_cache
from rsna_knee.ntejas_arms import find_arm_checkpoints, predict_arm

FAMILIES = find_checkpoints(walk)
if REQUIRE_FOLDS:
    check_folds(FAMILIES, REQUIRE_FOLDS)
ARMS = find_arm_checkpoints(walk)
NEED = {}
for k, v in FAMILIES.items():
    NEED[k] = family_cache(v)
    log(f"family {k} ({NEED[k]}): {[p.name for p in v]}")
for k, v in ARMS.items():
    log(f"public arm {k}: {[p.name for p in v]}")
if not FAMILIES and not ARMS:
    raise FileNotFoundError("no checkpoints mounted")
test_series = pd.read_csv(ROOT / "test_series.csv")
'''

SUB_3 = r'''try:
    members = {}
    if "v1" in NEED.values():
        CFG = CacheConfig(px=288, slices_per_slot=12, group=3, crop_mm=130.0)
        C1 = Path("/kaggle/temp/test_cache"); C1.mkdir(parents=True, exist_ok=True)
        a1, m1, s1 = build_cache(ROOT, test_series, test.StudyInstanceUID.tolist(), "test", CFG, C1, log=log)
        assert s1 == test.StudyInstanceUID.tolist()
        log(f"v1 test cache done at {(time.time() - T0) / 60:.1f} min")
    if "v3" in NEED.values() or ARMS:
        from rsna_knee.cache3 import build_split
        a3, m3, s3 = build_split(ROOT, "test", test.StudyInstanceUID.tolist(), Path("/kaggle/temp/c3"), log=log)
        log(f"v3 test cache done at {(time.time() - T0) / 60:.1f} min")
    for tag, paths in FAMILIES.items():
        arr, msk = (a1, m1) if NEED[tag] == "v1" else (a3, m3)
        members[tag] = predict_family(paths, arr, msk, log=log)
        log(f"family {tag} done at {(time.time() - T0) / 60:.1f} min")
    for arm, paths in ARMS.items():
        members["ntejas_" + arm] = predict_arm(paths, arm, a3, m3, log=log)
        log(f"arm {arm} done at {(time.time() - T0) / 60:.1f} min")
    weights = {k: FAMILY_WEIGHTS.get(k, 0.5 if k.startswith("ntejas_") else 1.0) for k in members}
    log(f"blend weights: {weights}")
    pred = rank_blend(members, weights)
    out = write_submission(pred, test.StudyInstanceUID.tolist(), test, SUB)
    validate_submission(SUB, test)
    log(f"submission written and validated: {out.shape}")
except Exception:
    import traceback
    traceback.print_exc()
    if not ALLOW_BENCHMARK:
        raise
    log("inference failed; the benchmark file stays in place (ALLOW_BENCHMARK = True)")
assert pd.read_csv(SUB)[list(TARGETS)].nunique().min() > 1 or ALLOW_BENCHMARK, "submission is constant"
log(f"total {(time.time() - T0) / 60:.1f} min")
'''

CACHE3 = r'''# CPU notebook (Accelerator: None). Run once per shard: SHARD = 0, 1, 2.
SHARD, N_SHARDS = 0, 3

from rsna_knee.infer import find_root
from rsna_knee.data import assign_folds
from rsna_knee.cache3 import build_split

ROOT = find_root()
train = pd.read_csv(ROOT / "train.csv")
train["fold"] = assign_folds(train, n_folds=5, seed=2026)
OUT = Path("/kaggle/working")
train.to_csv(OUT / "train_folds.csv", index=False)
build_split(ROOT, "train", train.StudyInstanceUID.tolist(), OUT, shard=SHARD, n_shards=N_SHARDS, log=log)
log("done. Save the output as a Dataset: rsna-knee-cache-v3-" + str(SHARD))
'''


AUDIT = r'''# CPU notebook (Accelerator: None). Reads only DICOM headers of every training series and
# writes slice_audit.csv: how the largest-spread axis order compares with the stack-normal order,
# the slice spacing, pixel anisotropy and the physical span of one 3-slice cache window.
from concurrent.futures import ThreadPoolExecutor
import pydicom
from rsna_knee.infer import find_root
from rsna_knee.cache3 import natural_key

ROOT = find_root()
series = pd.read_csv(ROOT / "train_series.csv", dtype=str)
PLANE_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

def audit(row):
    d = ROOT / "train_series" / row.StudyInstanceUID / row.SeriesInstanceUID
    out = {"StudyInstanceUID": row.StudyInstanceUID, "SeriesInstanceUID": row.SeriesInstanceUID,
           "plane": row.Anatomical_Plane}
    try:
        names = sorted((e.name for e in os.scandir(d) if e.name.endswith(".dcm")), key=natural_key)
    except Exception:
        return out
    pos, iop, ps, th, sb, lat, pid = [], None, None, None, None, None, None
    for n in names:
        try:
            h = pydicom.dcmread(d / n, stop_before_pixels=True, force=True)
        except Exception:
            continue
        p = getattr(h, "ImagePositionPatient", None)
        if p is not None and len(p) >= 3:
            pos.append([float(v) for v in p[:3]])
        if iop is None and getattr(h, "ImageOrientationPatient", None) is not None:
            iop = np.array([float(v) for v in h.ImageOrientationPatient[:6]])
        if ps is None and getattr(h, "PixelSpacing", None) is not None:
            ps = [float(v) for v in h.PixelSpacing[:2]]
        th = th or (float(h.SliceThickness) if getattr(h, "SliceThickness", None) else None)
        sb = sb or (float(h.SpacingBetweenSlices) if getattr(h, "SpacingBetweenSlices", None) else None)
        lat = lat or str(getattr(h, "Laterality", "") or getattr(h, "ImageLaterality", "") or "") or None
        pid = pid or str(getattr(h, "PatientID", "") or "") or None
    out.update(n_files=len(names), n_pos=len(pos), has_iop=iop is not None, thickness=th, spacing_tag=sb,
               laterality_tag=lat, patient_id=pid)
    if ps:
        out["px_row"], out["px_col"] = ps
    if len(pos) >= 3 and iop is not None:
        a = np.array(pos)
        axis = int(np.argmax(a.max(0) - a.min(0)))
        nrm = np.cross(iop[:3], iop[3:])
        k_axis, k_nrm = np.argsort(a[:, axis], kind="stable"), np.argsort(a @ nrm, kind="stable")
        out["axis"] = axis
        out["axis_is_plane_axis"] = axis == PLANE_AXIS.get(row.Anatomical_Plane, -1)
        out["same_order"] = bool((k_axis == k_nrm).all())
        out["reversed_order"] = bool((k_axis == k_nrm[::-1]).all())
        proj = np.sort(a @ nrm)
        gaps = np.diff(proj)
        out["spacing_ipp"] = float(np.median(np.abs(gaps))) if len(gaps) else None
        out["max_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.abs(nrm).max() / np.linalg.norm(nrm), 0, 1))))
        n = len(pos)
        lo, hi = round(0.06 * (n - 1)), round(0.94 * (n - 1))
        out["window_span_mm"] = float(2 * (hi - lo) / 15 * out["spacing_ipp"])
        out["window_acquired_slices"] = float(2 * (hi - lo) / 15 + 1)
        out["iop_plane"] = ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(nrm)))]
        out["median_x"] = float(np.median(a[:, 0]))
    return out

with ThreadPoolExecutor(32) as ex:
    rows = list(ex.map(audit, series.itertuples(index=False)))
aud = pd.DataFrame(rows)
aud.to_csv("/kaggle/working/slice_audit.csv", index=False)
ok = aud.dropna(subset=["same_order"])
log(f"series {len(aud)}, with geometry {len(ok)}")
log(f"axis order equals stack-normal order: {(ok.same_order | ok.reversed_order).mean():.4f} "
    f"(same direction {ok.same_order.mean():.4f}, reversed {ok.reversed_order.mean():.4f})")
log(f"largest-spread axis is the plane's own axis: {ok.axis_is_plane_axis.mean():.4f}")
log(f"pixel spacing anisotropic (>1%): {((aud.px_row - aud.px_col).abs() / aud.px_row > 0.01).mean():.4f}")
log("window span (mm) by plane:\n" + ok.groupby("plane").window_span_mm.describe().round(2).to_string())
log("tilt from the nearest axis (deg) by plane:\n" + ok.groupby("plane").max_tilt_deg.describe().round(1).to_string())
log(f"plane from the slice normal agrees with the series label: {(ok.iop_plane == ok.plane).mean():.4f}")
log(f"series without enough positions or orientation: {1 - len(ok) / len(aud):.4f}")
per_study = aud.dropna(subset=["patient_id"]).groupby("StudyInstanceUID").patient_id.first()
shared = per_study.map(per_study.value_counts()) > 1
log(f"PatientID present for {len(per_study)} of {aud.StudyInstanceUID.nunique()} studies; "
    f"{per_study.nunique()} distinct values; studies sharing a PatientID with another study: {int(shared.sum())}")
FOLDS_CSV = next((p for p in walk(suffixes=(".csv",)) if p.name == "train_folds.csv"), None)
if FOLDS_CSV is not None:
    folds = pd.read_csv(FOLDS_CSV).set_index("StudyInstanceUID").fold
    cross = pd.DataFrame({"pid": per_study, "fold": folds.reindex(per_study.index)}).groupby("pid").fold.nunique()
    log(f"PatientIDs whose studies fall into more than one fold: {int((cross > 1).sum())}")
lt = aud.laterality_tag.notna()
log(f"laterality tag present: {lt.mean():.4f}; median x at least 20 mm from 0 where absent: "
    f"{(aud.loc[~lt, 'median_x'].abs() >= 20).mean():.4f}")
'''

ATTN = r'''# GPU notebook. Inputs: rsna-knee-pipeline, the three cache v3 shards and the five fold notebooks
# knee r50v3 fold0 ... fold4. Writes slot_attention.csv: for every fold, the learned slot prior and the
# mean finding-to-slot attention on that fold's held-out studies.
import torch
from torch.utils.data import DataLoader
from rsna_knee.cache3 import load_cache3, copy_local
from rsna_knee.train2 import TrainConfigV2, build_model, make_dataset
from rsna_knee.infer2 import find_checkpoints
from rsna_knee.targets import file_digest
TAG = "r50v3"

FOLDS_CSV = next(p for p in walk(suffixes=(".csv",)) if p.name == "train_folds.csv")
train = pd.read_csv(FOLDS_CSV)
CACHE_DIRS = copy_local(sorted({p.parent for p in walk(suffixes=(".npy",)) if p.name.startswith("c3_train_")}), log=log)
array, mask, _ = load_cache3(CACHE_DIRS, "train", train.StudyInstanceUID.tolist())
ckpts = {p.name: p for p in find_checkpoints(walk).get(TAG, [])}
pd.DataFrame([{"checkpoint": n, "sha256": file_digest(p), "bytes": p.stat().st_size} for n, p in ckpts.items()]
             ).to_csv("/kaggle/working/checkpoint_manifest.csv", index=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for k in range(5):
    p = ckpts.get(f"v2_{TAG}_fold{k}.pt")
    if p is None:
        log(f"fold {k}: checkpoint not mounted"); continue
    st = torch.load(p, map_location="cpu", weights_only=False)
    cfg = TrainConfigV2(**{a: b for a, b in st["config"].items() if a in TrainConfigV2.__dataclass_fields__})
    cfg.encoder_weights = None
    m = build_model(cfg, pretrained=False, log=lambda *_: None)
    m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in st["model"].items()})
    m.to(dev).eval()
    caught = []
    head = m.head
    def hook(mod, inp, out):
        slots, present = inp
        h = mod.proj(slots) + mod.slot_id.unsqueeze(0)
        lg = torch.einsum("bsd,od->bos", h.float(), mod.query.float()) / mod.hidden ** 0.5
        if mod.prior is not None:
            lg = lg + mod.prior.float().unsqueeze(0)
        lg = lg.masked_fill(present.unsqueeze(1) < 0.5, float("-inf"))
        caught.append(torch.nan_to_num(torch.softmax(lg, -1), nan=0.0).cpu().numpy())
    hk = head.register_forward_hook(hook)
    va = np.flatnonzero(train.fold.values == k)
    dl = DataLoader(make_dataset(array, mask, va, None, None, cfg, train=False), batch_size=2, num_workers=2)
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                m(b["x"].to(dev), b["present"].to(dev))
    hk.remove()
    att = np.concatenate(caught).mean(0)
    prior = head.prior.detach().float().cpu().numpy() if head.prior is not None else np.zeros_like(att)
    for j, t in enumerate(TARGETS):
        for s in range(6):
            rows.append({"fold": k, "finding": t, "slot": s, "attention": float(att[j, s]),
                         "prior": float(prior[j, s])})
    log(f"fold {k}: {len(va)} studies")
pd.DataFrame(rows).to_csv("/kaggle/working/slot_attention.csv", index=False)
log("done")
'''


def nb(cells):
    return {
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = HERE / "notebooks"
    out.mkdir(exist_ok=True)
    json.dump(nb([BOOT, TRAIN_CFG, TRAIN_TARGETS, TRAIN_LOOP, TRAIN_REPORT]),
              open(out / "04_train_v2.ipynb", "w"), indent=1)
    json.dump(nb([BOOT, SUB_1, SUB_2, SUB_3]), open(out / "05_submit_v2.ipynb", "w"), indent=1)
    json.dump(nb([BOOT, CACHE3]), open(out / "06_build_cache_v3.ipynb", "w"), indent=1)
    json.dump(nb([BOOT, AUDIT]), open(out / "07_slice_audit.ipynb", "w"), indent=1)
    json.dump(nb([BOOT, ATTN]), open(out / "08_slot_attention.ipynb", "w"), indent=1)
    print("written", sorted(p.name for p in out.iterdir()))
