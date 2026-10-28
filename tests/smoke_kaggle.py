"""CPU end-to-end run of both v2 notebooks against a fake Kaggle tree.

Builds, inside a new temporary folder: a pipeline dataset, a 2-shard train cache,
train_folds.csv, a random RadImageNet-layout ResNet50.pt, and a tiny competition mount
with synthetic DICOM series for the test split. Then executes the notebook cells in
order, with every /kaggle path in them pointed at the temporary folder, and checks the
outputs. The real /kaggle is never read, written or deleted.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_TRAIN = 30
K = None  # the fake Kaggle root, set in main()


def fresh(root):
    """Lay out an empty fake Kaggle tree under ``root``, a folder this script created."""
    sys.path.insert(0, str(ROOT))
    from rsna_knee.paths import check_scratch_root
    root = check_scratch_root(root)
    for d in ("input/pipe", "input/cache0", "input/cache1", "input/rad", "input/comp", "working", "temp"):
        (root / d).mkdir(parents=True)
    shutil.copytree(ROOT / "rsna_knee", root / "input/pipe/rsna_knee")
    shutil.copytree(ROOT / "data", root / "input/pipe/data")
    return root


def fake_train():
    sys.path.insert(0, str(ROOT))
    from rsna_knee.data import CacheConfig, shard_bounds
    from rsna_knee.report_labels import TARGETS

    lab = pd.read_csv(ROOT / "data/labels_v1_opus45.csv")
    gold = lab[lab.IS_GOLD].head(10)
    rest = lab[~lab.IS_GOLD].head(N_TRAIN - 10)
    sel = pd.concat([gold, rest])
    gmap = dict(zip([f"GOLD_{t.upper().replace(' ', '_').replace(chr(39), '')}" for t in TARGETS], TARGETS))
    gmap["GOLD_BAKERS"] = "Baker's"
    train = pd.DataFrame({"StudyInstanceUID": sel.STUDY_UID.values, "Report": "x"})
    for gc, t in gmap.items():
        train[t] = sel[gc].map(lambda v: np.nan if pd.isna(v) else float(v in (True, "True", 1, 1.0))).values
    train["fold"] = np.arange(len(train)) % 3
    cfg = CacheConfig(px=288, slices_per_slot=12, group=3, crop_mm=130.0)
    rng = np.random.default_rng(0)
    uids = train.StudyInstanceUID.tolist()
    for s in range(2):
        a, b = shard_bounds(len(uids), s, 2)
        d = K / f"input/cache{s}"
        stem = f"train_{cfg.tag()}_s{s}of2"
        arr = np.lib.format.open_memmap(d / f"{stem}.npy", mode="w+", dtype=np.uint8,
                                        shape=(b - a, 6, 12, 288, 288))
        arr[:] = rng.integers(0, 255, arr.shape, dtype=np.uint8)
        arr.flush()
        m = (rng.random((b - a, 6)) > 0.2).astype(np.float32)
        m[:, 0] = 1
        np.save(d / f"{stem}_mask.npy", m)
        (d / f"{stem}_studies.json").write_text(json.dumps(uids[a:b]))
    train.to_csv(K / "input/cache0/train_folds.csv", index=False)

    import torch
    from rsna_knee.model2 import RadResNet50
    torch.save(RadResNet50().state_dict(), K / "input/rad/ResNet50.pt")


def fake_c3_train():
    train = pd.read_csv(K / "input/cache0/train_folds.csv")
    uids = train.StudyInstanceUID.tolist()
    rng = np.random.default_rng(1)
    for s, (a, b) in enumerate([(0, 15), (15, len(uids))]):
        d = K / f"input/c3_{s}"
        d.mkdir()
        stem = f"c3_train_s{s}of2"
        arr = np.lib.format.open_memmap(d / f"{stem}.npy", mode="w+", dtype=np.uint8, shape=(b - a, 6, 16, 256, 256))
        arr[:] = rng.integers(0, 255, arr.shape, dtype=np.uint8)
        arr.flush()
        np.save(d / f"{stem}_mask.npy", np.ones((b - a, 6), np.float32))
        (d / f"{stem}_studies.json").write_text(json.dumps(uids[a:b]))


def fake_dicom_mount():
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    comp = K / "input/comp"
    rows_csv, test_rows = [], []
    planes = {"Sagittal": ([0, 1, 0, 0, 0, -1], 0), "Coronal": ([1, 0, 0, 0, 0, -1], 1),
              "Axial": ([1, 0, 0, 0, 1, 0], 2)}
    for s in range(3):
        study = generate_uid()
        test_rows.append(study)
        for plane, (iop, ax) in planes.items():
            for fs in (0, 1):
                series = generate_uid()
                sd = comp / "test_series" / study / series
                sd.mkdir(parents=True)
                rows_csv.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                 "Anatomical_Plane": plane, "Fat_Suppression": fs,
                                 "Fluid_Sensitive": fs, "SeriesDescription": f"{plane} {'FS' if fs else 'T1'}"})
                normal = np.cross(iop[:3], iop[3:])
                for i in range(20):
                    meta = FileMetaDataset()
                    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
                    ds.Rows = ds.Columns = 320
                    ds.PixelSpacing = [0.5, 0.5]
                    ds.ImageOrientationPatient = iop
                    ds.ImagePositionPatient = list((normal * i * 3.0 + np.array([-80.0, -80, 0])).astype(float))
                    ds.InstanceNumber = i + 1
                    ds.RepetitionTime, ds.EchoTime = 3000, 30
                    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
                    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
                    ds.PixelData = np.random.randint(0, 2000, (320, 320), dtype=np.uint16).tobytes()
                    ds.save_as(sd / f"{meta.MediaStorageSOPInstanceUID}.dcm", enforce_file_format=True)
    pd.DataFrame(rows_csv).to_csv(comp / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": test_rows}).to_csv(comp / "test.csv", index=False)


def run_notebook(path, patches):
    cells = ["".join(c["source"]) for c in json.load(open(path))["cells"]]
    g = {"__name__": "__main__"}
    for i, src in enumerate(cells):
        for a, b in patches:
            src = src.replace(a, b)
        src = src.replace('"/kaggle', f'"{K}')
        assert '"/kaggle' not in src
        print(f"\n######## {path.name} cell {i}", flush=True)
        exec(compile(src, f"{path.name}:{i}", "exec"), g)
    return g


def check_outputs():
    """The submission and the checkpoints the notebooks wrote are complete and consistent."""
    import torch
    from rsna_knee.report_labels import TARGETS
    sub = pd.read_csv(K / "working/submission.csv")
    test = pd.read_csv(K / "input/comp/test.csv")
    assert set(sub.StudyInstanceUID) == set(test.StudyInstanceUID) and len(sub) == len(test)
    vals = sub[list(TARGETS)].to_numpy(float)
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
    for p in sorted((K / "input/models").glob("v2_*.pt")):
        state = torch.load(p, map_location="cpu", weights_only=False)
        assert state["config"]["target_scheme"] == "flat", p.name
        assert state.get("targets") == list(TARGETS), p.name
    print("outputs checked: submission rows, value range, checkpoint provenance")


def main():
    global K
    with tempfile.TemporaryDirectory(prefix="fake-kaggle-") as tmp:
        K = Path(tmp) / "kaggle"
        os.environ["RSNA_KNEE_KAGGLE_ROOT"] = str(K)  # read by rsna_knee.paths at import
        K = fresh(K)
        fake_train()
        fake_dicom_mount()
        here = os.getcwd()
        os.chdir(K / "working")
        try:
            run_all()
            check_outputs()
        finally:
            os.chdir(here)


def run_all():
    patches = [('ENCODER = "convnext_tiny.fb_in22k_ft_in1k"', 'ENCODER = "rad_resnet50"'),
               ('CACHE = "v3" ', 'CACHE = "v1" '), ('TAG = "r50v3"', 'TAG = "radr50"'),
               ('FOLDS = [0, 1, 2, 3, 4]', 'FOLDS = [0]'), ("EPOCHS = 8", "EPOCHS = 2"),
               ("FULL_DATA = False", "FULL_DATA = True"),
               ('seed=SEED)', 'seed=SEED, batch_size=1, accum=2, workers=0); tcfg.img = 96')]
    run_notebook(ROOT / "notebooks/04_train_v2.ipynb", patches)
    # move outputs into an input dataset as the guide says
    (K / "input/models").mkdir()
    for p in (K / "working").glob("v2_*.pt"):
        shutil.copy(p, K / "input/models" / p.name)
    print(sorted(p.name for p in (K / "working").iterdir()))
    # round 2 teacher: our own OOF as extra teacher, under a name the release does not use
    shutil.copy(K / "working/oof_v2_radr50_f0.csv", K / "input/models/oof_smoke_teacher.csv")
    run_notebook(ROOT / "notebooks/04_train_v2.ipynb",
                 patches + [('EXTRA_TEACHERS = []', 'EXTRA_TEACHERS = ["oof_smoke_teacher.csv"]'),
                            ('TAG = "radr50"', 'TAG = "r2"'), ("FULL_DATA = True", "FULL_DATA = False")])
    for p in (K / "working").glob("v2_r2*.pt"):
        shutil.copy(p, K / "input/models" / p.name)
    # v3 cache, single-slice mode
    fake_c3_train()
    run_notebook(ROOT / "notebooks/04_train_v2.ipynb",
                 patches + [('CACHE = "v1"', 'CACHE = "v3"'), ('GROUP = 3 ', 'GROUP = 1 '),
                            ('TAG = "radr50"', 'TAG = "r3"'), ("FULL_DATA = True", "FULL_DATA = False")])
    for p in (K / "working").glob("v2_r3*.pt"):
        shutil.copy(p, K / "input/models" / p.name)
    import torch
    from rsna_knee.ntejas_arms import Net
    for arm in ("armA", "armB"):
        torch.save({"model": Net().state_dict()}, K / "input/models" / f"{arm}_fold0.pt")
    run_notebook(ROOT / "notebooks/05_submit_v2.ipynb", [("REQUIRE_FOLDS = 5", "REQUIRE_FOLDS = None")])
    print(pd.read_csv(K / "working/submission.csv").round(3).to_string())


if __name__ == "__main__":
    main()
