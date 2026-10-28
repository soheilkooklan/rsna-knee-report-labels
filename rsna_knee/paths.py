"""Root of the Kaggle file tree.

On Kaggle this is /kaggle. The CPU smoke test sets RSNA_KNEE_KAGGLE_ROOT to a
temporary folder before importing the package, so nothing under the real /kaggle
is read or written by a test.
"""

import os
from pathlib import Path

KAGGLE = Path(os.environ.get("RSNA_KNEE_KAGGLE_ROOT", "/kaggle"))
INPUT = KAGGLE / "input"
TEMP = KAGGLE / "temp"
WORKING = KAGGLE / "working"

# folders a test fixture may never delete
PROTECTED = tuple(Path(p) for p in ("/", "/kaggle", "/kaggle/input", "/kaggle/working",
                                    "/kaggle/temp", "/home", "/root", "/tmp"))


def check_scratch_root(root: Path) -> Path:
    """Return ``root`` resolved, or raise if it is a mount or system folder.

    Used before any recursive deletion of a test tree.
    """
    r = Path(root).resolve()
    if r in PROTECTED or Path("/kaggle") in (r, *r.parents):
        raise ValueError(f"refusing to use {r} as a scratch root")
    if len(r.parts) < 3:
        raise ValueError(f"refusing to use {r} as a scratch root: too close to /")
    return r
