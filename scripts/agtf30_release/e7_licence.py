#!/usr/bin/env python3
"""E7: apply the owner's licence decision (2026-11-07) to the frozen v1.0 release folder.

    python e7_licence.py                       # LICENSE.txt + NOTICE.txt into the release, licence fields in MANIFEST.json
    python e6_release_check.py --stage         # copy the updated card into index/, re-hash small files (h5 hashes kept)
    python e6_release_check.py --lock          # read-only again
    python e1_freeze.py --verify               # re-hash everything against SHA256SUMS.txt

Licence: CC BY 4.0 (LICENSE.txt = the official legal code, creativecommons.org/licenses/by/4.0/legalcode.txt).
Sources checked by the owner: nasa/AGTF30 and nasa/T-MATS are Apache-2.0, N-CMAPSS is CC0 1.0.
The read-only .h5 files are NOT touched: their 'licence' attribute ("not yet decided by the owner") is superseded by
LICENSE.txt; MANIFEST.json and the data card are authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

REPO = pathlib.Path(__file__).resolve().parents[2]
DST = pathlib.Path(f"{WORK}/agtf30_v1")
SRC = {"LICENSE.txt": REPO / "docs/agtf30_v1/LICENSE.txt", "NOTICE.txt": REPO / "docs/agtf30_v1/NOTICE.txt"}
LICENCE = {
    "licence": "CC-BY-4.0",
    "licence_file": "LICENSE.txt",
    "notice_file": "NOTICE.txt",
    "licence_decision": "owner, 2026-11-07",
    "licence_sources": {"AGTF30 (github.com/nasa/AGTF30)": "Apache-2.0",
                        "T-MATS (github.com/nasa/T-MATS)": "Apache-2.0",
                        "N-CMAPSS flight conditions (Arias Chao et al., Data 2021, 6(1), 5)": "CC0-1.0"},
    "licence_note": ("the 'licence' attribute inside the .h5 files ('not yet decided by the owner') predates the "
                     "decision and is superseded by LICENSE.txt; the read-only .h5 files were not changed"),
    "second_copy": ("none yet (one physical disk on this PC); location to be decided by the owner (external disk, "
                    "lab Mac or AI Panther); Zenodo at paper time"),
}


def apply(dst: pathlib.Path):
    for name, src in SRC.items():
        t = dst / name
        if t.exists():
            os.chmod(t, 0o666)
        shutil.copyfile(src, t)
    m = dst / "MANIFEST.json"
    os.chmod(m, 0o666)
    out = {}
    for k, v in json.loads(m.read_text()).items():
        if k in LICENCE:
            continue
        out[k] = v
        if k == "split_rule":                       # licence fields right after the provenance rules
            out.update(LICENCE)
    assert "licence" in out
    m.write_text(json.dumps(out, indent=1))
    print("LICENCE_APPLIED", list(LICENCE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(DST))
    apply(pathlib.Path(ap.parse_args().dst))
