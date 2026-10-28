"""Unit tests that need neither DICOM files nor a GPU.

    python -m pytest tests/test_units.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsna_knee.report_labels import TARGETS  # noqa: E402
from rsna_knee.targets import build_targets_v2  # noqa: E402
from rsna_knee.rank_tools import cooccurrence_weights, explain_away, pct_rank, EA_KEEP  # noqa: E402

T = list(TARGETS)


# ---------------------------------------------------------------- slice order

def _stack(normal, n=20, gap=3.0, rng=None):
    rng = rng or np.random.default_rng(0)
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    origin = rng.normal(0, 30, 3)
    return np.array([origin + k * gap * normal for k in range(n)])


def _axis_order(pos):
    spread = pos.max(0) - pos.min(0)
    return np.argsort(pos[:, int(np.argmax(spread))], kind="stable")


@pytest.mark.parametrize("seed", range(200))
def test_axis_order_matches_normal_order_for_parallel_stacks(seed):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=3)                       # any oblique orientation
    pos = _stack(normal, rng=rng)
    shuffled = rng.permutation(len(pos))
    p = pos[shuffled]
    by_axis = shuffled[_axis_order(p)]
    by_normal = shuffled[np.argsort(p @ normal, kind="stable")]
    assert (by_axis == by_normal).all() or (by_axis == by_normal[::-1]).all()


def test_axis_order_on_standard_planes():
    for normal, axis in (([1, 0, 0], 0), ([0, 1, 0], 1), ([0, 0, 1], 2), ([0.97, 0.24, 0], 0)):
        pos = _stack(normal)
        assert int(np.argmax(pos.max(0) - pos.min(0))) == axis
        assert (_axis_order(pos[::-1]) == np.arange(len(pos))[::-1]).all() or \
               (_axis_order(pos[::-1]) == np.arange(len(pos))).all()


# ---------------------------------------------------------------- targets

def _frames(report, teacher):
    ids = [f"s{i}" for i in range(len(report))]
    train = pd.DataFrame({"StudyInstanceUID": ids})
    for t in T:
        train[t] = np.nan                              # no gold labels
    lab = pd.DataFrame({t: report for t in T}, index=pd.Index(ids, name="StudyInstanceUID"))
    for t in T:
        lab[t + "_state"] = "u"
    tea = pd.DataFrame({t: teacher for t in T}, index=pd.Index(ids, name="StudyInstanceUID"))
    return train, lab, tea


def test_target_fallbacks():
    # report and teacher present / report only / teacher only / neither
    report = [0.8, 0.6, np.nan, np.nan]
    teacher = [0.2, np.nan, 0.4, np.nan]
    train, lab, tea = _frames(report, teacher)
    y, w, explicit, gold = build_targets_v2(train, lab, [tea], scheme="flat", prior=0.1)
    assert np.allclose(y[:, 0], [0.5, 0.6, 0.4, 0.1])
    assert np.isnan(explicit).all() and not gold.any()


def test_labels_only_ignores_teacher():
    train, lab, tea = _frames([0.8, 0.3], [0.0, 1.0])
    y, *_ = build_targets_v2(train, lab, [tea], scheme="labels_only")
    assert np.allclose(y[:, 0], [0.8, 0.3])


def test_gold_overrides_target_with_weight():
    train, lab, tea = _frames([0.8, 0.3], [0.2, 0.7])
    train.loc[0, T] = 1.0
    y, w, _, gold = build_targets_v2(train, lab, [tea], scheme="flat", gold_weight=2.0)
    assert np.allclose(y[0], 1.0) and np.allclose(w[0], 2.0) and gold[0] and not gold[1]


# ---------------------------------------------------------------- explaining away

def test_explain_away_keeps_fluid_findings_and_lambda_zero():
    rng = np.random.default_rng(1)
    labels = pd.DataFrame(rng.random((200, 12)) > 0.6, columns=T).astype(float)
    labels["Contusion"] = np.maximum(labels["Contusion"], labels["Fracture"])
    W = cooccurrence_weights(labels)
    assert np.allclose(np.diag(W), 0) and (W >= 0).all()
    assert np.allclose(W.sum(1)[W.sum(1) > 0], 1)
    r = pct_rank(rng.random((50, 12)))
    assert np.allclose(explain_away(r, W, 0.0), r)
    out = explain_away(r, W, -0.1)
    for name in EA_KEEP:
        j = T.index(name)
        assert np.allclose(out[:, j], r[:, j])


def test_target_order_matches_submission_columns():
    assert T == ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                 "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def test_released_files_align():
    lab = pd.read_csv(ROOT / "data/labels_v1_opus45.csv")
    tea = pd.read_csv(ROOT / "data/teacher_oof_armA.csv")
    oof = pd.read_csv(ROOT / "data/oof_v2_r50v3_all.csv")
    man = pd.read_csv(ROOT / "data/fold_manifest.csv")
    for df, col in ((lab, "STUDY_UID"), (tea, "STUDY_UID"), (oof, "StudyInstanceUID"), (man, "StudyInstanceUID")):
        assert df[col].is_unique and len(df) == 4407
    assert set(lab.STUDY_UID) == set(tea.STUDY_UID) == set(oof.StudyInstanceUID) == set(man.StudyInstanceUID)
    for df in (lab, tea, oof):
        assert all(t in df.columns for t in T)
    assert man.fold.between(0, 4).all() and man.groupby("fold").is_gold.sum().min() >= 10


def test_kaggle_cell_weights_match_rank_tools():
    import re
    src = (ROOT / "tools/kaggle_cells/explaining_away_cell.py").read_text()
    block = src[src.index("W = ["):src.index("]\nORDER")]
    W_cell = np.array([[float(v) for v in re.findall(r"[0-9.]+", line)]
                       for line in block.splitlines()[1:] if "[" in line])
    lab = pd.read_csv(ROOT / "data/labels_v1_opus45.csv")
    W = cooccurrence_weights(lab[~lab.IS_GOLD.astype(bool)])
    assert W_cell.shape == (12, 12) and np.abs(W_cell - W).max() < 6e-4


# ---------------------------------------------------------------- file and checkpoint discovery

def _walker(root):
    def walk(suffixes=None):
        for f in sorted(Path(root).rglob("*")):
            if f.is_file() and (suffixes is None or f.suffix in suffixes):
                yield f
    return walk


def test_checkpoints_same_name_same_size_different_content_raise(tmp_path):
    from rsna_knee.infer2 import find_checkpoints
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a/v2_r50v3_fold0.pt").write_bytes(b"abcd")
    (tmp_path / "b/v2_r50v3_fold0.pt").write_bytes(b"wxyz")
    with pytest.raises(ValueError):
        find_checkpoints(_walker(tmp_path))


def test_checkpoints_identical_copies_and_fold_check(tmp_path):
    from rsna_knee.infer2 import find_checkpoints, check_folds
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        for k in range(5):
            (tmp_path / d / f"v2_r50v3_fold{k}.pt").write_bytes(bytes([k]) * 8)
    fams = find_checkpoints(_walker(tmp_path))
    assert len(fams["r50v3"]) == 5
    check_folds(fams, 5)
    fams["r50v3"] = fams["r50v3"][:4]
    with pytest.raises(ValueError):
        check_folds(fams, 5)


def _ckpt(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.full((4,), float(value))}, "config": {"cache": "v3"}}, path)
    return path


def _unpack(pt, into):
    import zipfile
    with zipfile.ZipFile(pt) as z:
        z.extractall(into)
    return into / pt.stem


def _rev_walker(root):
    def walk(suffixes=None):
        yield from reversed(list(_walker(root)(suffixes)))
    return walk


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_checkpoint_packed_and_unpacked_identical(tmp_path, order):
    from rsna_knee.infer2 import find_checkpoints
    pt = _ckpt(tmp_path / "a/v2_r50v3_fold0.pt", 1)
    _unpack(pt, tmp_path / "b")
    walker = (_walker if order == "forward" else _rev_walker)(tmp_path)
    fams = find_checkpoints(walker, rezip_to=tmp_path / "rz")
    assert fams["r50v3"] == [pt]


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_checkpoint_packed_and_unpacked_conflicting_raise(tmp_path, order):
    from rsna_knee.infer2 import find_checkpoints
    pt = _ckpt(tmp_path / "a/v2_r50v3_fold0.pt", 1)
    other = _ckpt(tmp_path / "x/v2_r50v3_fold0.pt", 2)
    _unpack(other, tmp_path / "b")
    other.unlink()
    walker = (_walker if order == "forward" else _rev_walker)(tmp_path)
    with pytest.raises(ValueError):
        find_checkpoints(walker, rezip_to=tmp_path / "rz")


def test_checkpoint_two_unpacked_folders(tmp_path):
    from rsna_knee.infer2 import find_checkpoints, content_digest
    import torch
    pt = _ckpt(tmp_path / "x/v2_r50v3_fold0.pt", 3)
    _unpack(pt, tmp_path / "a"); _unpack(pt, tmp_path / "b")
    digest = content_digest(pt)
    pt.unlink()
    fams = find_checkpoints(_walker(tmp_path), rezip_to=tmp_path / "rz")
    (got,) = fams["r50v3"]
    assert content_digest(got) == digest
    assert torch.load(got, weights_only=False)["model"]["w"].tolist() == [3.0] * 4
    # a different second folder under the same name raises
    other = _ckpt(tmp_path / "y/v2_r50v3_fold0.pt", 4)
    import shutil
    shutil.rmtree(tmp_path / "b")
    _unpack(other, tmp_path / "b")
    other.unlink()
    with pytest.raises(ValueError):
        find_checkpoints(_walker(tmp_path), rezip_to=tmp_path / "rz")


def test_rezip_overwrites_stale_output(tmp_path):
    from rsna_knee.infer2 import rezip_dir, content_digest
    pt = _ckpt(tmp_path / "x/v2_r50v3_fold0.pt", 5)
    d = _unpack(pt, tmp_path / "a")
    (tmp_path / "rz").mkdir()
    (tmp_path / "rz/v2_r50v3_fold0.pt").write_bytes(b"stale")
    out = rezip_dir(d, tmp_path / "rz")
    assert content_digest(out) == content_digest(pt)


def test_family_check_rejects_mixed_target_scheme(tmp_path):
    import torch
    from rsna_knee.infer2 import family_cache
    base = {"encoder": "rad_resnet50", "cache": "v3", "img": 224, "group": 3, "slot_prior": True,
            "eval_windows": 16, "tag": "r50v3"}
    a = tmp_path / "v2_r50v3_fold0.pt"; b = tmp_path / "v2_r50v3_fold1.pt"
    torch.save({"config": {**base, "target_scheme": "flat", "seed": 1}}, a)
    torch.save({"config": {**base, "target_scheme": "flat", "seed": 2}}, b)
    assert family_cache([a, b]) == "v3"
    torch.save({"config": {**base, "target_scheme": "labels_only", "seed": 2}}, b)
    with pytest.raises(ValueError):
        family_cache([a, b])
    with pytest.raises(ValueError):
        family_cache([])


def test_copy_local_replaces_stale_copy(tmp_path):
    from rsna_knee.cache3 import copy_local
    src = tmp_path / "src"; src.mkdir()
    (src / "c3_train_s0of1.npy").write_bytes(b"new content")
    dst = tmp_path / "dst"
    (dst / "shard0").mkdir(parents=True)
    (dst / "shard0/c3_train_s0of1.npy").write_bytes(b"old content")  # same size, no record
    msgs = []
    out = copy_local([src], dst, log=msgs.append)
    assert (out[0] / "c3_train_s0of1.npy").read_bytes() == b"new content"
    copy_local([src], dst, log=msgs.append)
    assert any("verified" in m for m in msgs)
    (src / "c3_train_s0of1.npy").write_bytes(b"newer conten")
    import os, time
    os.utime(src / "c3_train_s0of1.npy", (time.time() + 5, time.time() + 5))
    out = copy_local([src], dst, log=msgs.append)
    assert (out[0] / "c3_train_s0of1.npy").read_bytes() == b"newer conten"


@pytest.mark.parametrize("bad", ["/", "/kaggle", "/kaggle/input", "/kaggle/working", "/kaggle/input/x", "/tmp"])
def test_scratch_root_refuses_mounts(bad):
    from rsna_knee.paths import check_scratch_root
    with pytest.raises(ValueError):
        check_scratch_root(Path(bad))


def test_smoke_test_never_targets_real_kaggle(tmp_path):
    import importlib.util
    from rsna_knee.paths import check_scratch_root
    assert check_scratch_root(tmp_path / "fake-kaggle") == (tmp_path / "fake-kaggle").resolve()
    src = (ROOT / "tests/smoke_kaggle.py").read_text()
    assert 'Path("/kaggle")' not in src and "rmtree(K)" not in src


def test_teacher_mean_all_missing_row_no_warning():
    import warnings
    from rsna_knee.targets import teacher_mean
    idx = pd.Index(["a", "b"])
    t1 = pd.DataFrame({t: [0.2, np.nan] for t in TARGETS}, index=idx)
    t2 = pd.DataFrame({t: [0.6, np.nan] for t in TARGETS}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = teacher_mean([t1, t2], idx)
    assert m.loc["b"].isna().all() and m.loc["a"].notna().all()


def test_find_file_rejects_different_copies(tmp_path):
    from rsna_knee.targets import find_file
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a/labels.csv").write_text("x\n1\n")
    (tmp_path / "b/labels.csv").write_text("x\n1\n")
    assert find_file("labels.csv", [tmp_path]).name == "labels.csv"
    (tmp_path / "b/labels.csv").write_text("x\n2\n")
    with pytest.raises(ValueError):
        find_file("labels.csv", [tmp_path])
    assert find_file("missing.csv", [tmp_path]) is None


# ---------------------------------------------------------------- statistics

def _common():
    sys.path.insert(0, str(ROOT / "tools"))
    import common
    return common


def test_delong_macro_one_finding_matches_fast_delong():
    from scipy.stats import rankdata, norm
    c = _common()
    rng = np.random.default_rng(1)
    y = (rng.random(60) < 0.4).astype(int)
    a = y + rng.normal(0, 1, 60)
    b = y + rng.normal(0, 1.3, 60)
    m, n = y.sum(), (1 - y).sum()

    def comps(x):
        tz = rankdata(np.r_[x[y == 1], x[y == 0]])
        v01 = (tz[:m] - rankdata(x[y == 1])) / n
        v10 = 1 - (tz[m:] - rankdata(x[y == 0])) / m
        return tz[:m].sum() / m / n - (m + 1) / 2 / n, v01, v10
    a1, pa, na = comps(a)
    a2, pb, nb = comps(b)
    S = np.cov(np.vstack([pa, pb])) / m + np.cov(np.vstack([na, nb])) / n
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    d, se2, p = c.delong_macro(y[:, None], a[:, None], b[:, None])
    assert abs(d - (a1 - a2)) < 1e-12 and abs(se2 - se) < 1e-12
    assert abs(p - 2 * norm.sf(abs(a1 - a2) / se)) < 1e-12


def test_delong_macro_identical_scores_give_zero_difference():
    c = _common()
    rng = np.random.default_rng(2)
    Y = (rng.random((40, 3)) < 0.5).astype(int)
    P = rng.random((40, 3))
    d, se, p = c.delong_macro(Y, P, P)
    assert d == 0 and se == 0
