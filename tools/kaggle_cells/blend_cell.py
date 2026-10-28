# Last cell of a forked public ensemble notebook.
# Needs two more inputs: rsna-knee-pipeline and rsna-knee-v2-models.
# Reads the submission.csv the notebook already wrote, adds our models on ranks,
# and writes submission.csv again. On any error the public file stays as it was.

OUR_WEIGHT = 0.3


def _our_blend(our_weight=OUR_WEIGHT):
    import os, sys, time, shutil, traceback
    from pathlib import Path
    import numpy as np
    import pandas as pd

    t0 = time.time()

    def say(msg):
        print(f"[ours {time.time() - t0:7.1f}s] {msg}", flush=True)

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

    sub_path = Path("/kaggle/working/submission.csv")
    if not sub_path.is_file():
        say("no submission.csv from the public part; nothing to blend into")
        return
    backup = Path("/kaggle/working/submission_public.csv")
    shutil.copy(sub_path, backup)
    try:
        pkg = next((p.parent.parent for p in walk(suffixes=(".py",))
                    if p.name == "targets.py" and p.parent.name == "rsna_knee"), None)
        if pkg is None:
            raise FileNotFoundError("rsna-knee-pipeline is not attached")
        sys.path.insert(0, str(pkg))
        from rsna_knee.report_labels import TARGETS
        from rsna_knee.infer import find_root
        from rsna_knee.data import CacheConfig, build_cache
        from rsna_knee.infer2 import find_checkpoints, predict_family, rank_blend, family_cache

        root = find_root()
        test = pd.read_csv(root / "test.csv")
        base = pd.read_csv(sub_path)
        base = test[["StudyInstanceUID"]].merge(base, on="StudyInstanceUID", how="left")
        fams = find_checkpoints(walk)
        if not fams:
            raise FileNotFoundError("rsna-knee-v2-models is not attached")
        for k, v in fams.items():
            say(f"family {k}: {[p.name for p in v]}")
        need = {k: family_cache(v) for k, v in fams.items()}

        series = pd.read_csv(root / "test_series.csv")
        arrays = {}
        if "v1" in need.values():
            cfg = CacheConfig(px=288, slices_per_slot=12, group=3, crop_mm=130.0)
            d = Path("/kaggle/temp/our_v1"); d.mkdir(parents=True, exist_ok=True)
            a, m, s = build_cache(root, series, test.StudyInstanceUID.tolist(), "test", cfg, d, log=say)
            arrays["v1"] = (a, m)
        if "v3" in need.values():
            from rsna_knee.cache3 import build_split
            a, m, s = build_split(root, "test", test.StudyInstanceUID.tolist(), Path("/kaggle/temp/our_v3"), log=say)
            arrays["v3"] = (a, m)

        members = {}
        for tag, paths in fams.items():
            a, m = arrays[need[tag]]
            members[tag] = predict_family(paths, a, m, log=say)
        ours = rank_blend(members)

        pub = base[list(TARGETS)].to_numpy(dtype=float)
        pub_r = pd.DataFrame(pub).rank(pct=True).to_numpy()
        mix = (1 - our_weight) * pub_r + our_weight * ours
        if not np.isfinite(mix).all():
            raise ValueError("blend is not finite")
        out = base[["StudyInstanceUID"]].copy()
        out[list(TARGETS)] = mix
        out.to_csv(sub_path, index=False)
        say(f"blended {len(out)} studies, our weight {our_weight}")
    except Exception:
        traceback.print_exc()
        shutil.copy(backup, sub_path)
        say("our part failed; the public submission.csv is kept unchanged")


_our_blend()
