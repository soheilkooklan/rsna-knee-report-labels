# Read one DICOM header per study (no pixels) for train and test, write the
# patient and scanner fields to /kaggle/working/meta_train.csv and meta_test.csv.
# CPU only, a few minutes.

import os, glob, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pydicom

ROOT = next(p for p in glob.glob("/kaggle/input/**/test.csv", recursive=True)
            if "rsna-knee" in p).rsplit("/", 1)[0]
FIELDS = ["PatientAge", "PatientSex", "PatientWeight", "PatientSize", "Manufacturer",
          "ManufacturerModelName", "MagneticFieldStrength", "InstitutionName", "StationName",
          "BodyPartExamined", "Laterality", "StudyDate", "PatientBirthDate"]


def first_file(study_dir):
    for s in sorted(os.listdir(study_dir)):
        d = os.path.join(study_dir, s)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".dcm"):
                    return os.path.join(d, f)
    return None


def probe(args):
    uid, base = args
    row = {"StudyInstanceUID": uid}
    f = first_file(os.path.join(base, uid)) if os.path.isdir(os.path.join(base, uid)) else None
    if f is None:
        return row
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        for k in FIELDS:
            v = getattr(ds, k, None)
            row[k] = None if v in (None, "") else str(v)
    except Exception as e:
        row["error"] = repr(e)[:80]
    return row


for split in ("train", "test"):
    t0 = time.time()
    ids = pd.read_csv(f"{ROOT}/{split}.csv")["StudyInstanceUID"].astype(str).tolist()
    base = next(d for d in (f"{ROOT}/{split}_series", f"{ROOT}/{split}_images", f"{ROOT}/{split}")
                if os.path.isdir(d))
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(probe, [(u, base) for u in ids]))
    df = pd.DataFrame(rows)
    df.to_csv(f"/kaggle/working/meta_{split}.csv", index=False)
    print(f"{split}: {len(df)} studies in {time.time() - t0:.0f}s")
    print(df.notna().mean().round(3).to_string())
    if "PatientAge" in df:
        print("PatientAge examples:", df["PatientAge"].dropna().head(8).tolist())
    if "PatientSex" in df:
        print("PatientSex counts:", df["PatientSex"].value_counts().to_dict())
    print()
