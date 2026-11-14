#!/usr/bin/env python3
"""Compare every unit of a mixer run's h5 files with the same unit of a released subset, bit for bit.

    python mixer_verify.py <runs root>/<name> [--release <folder with the released AGTF30_*.h5>]

For each unit (= engine slot + 1) it compares the raw bytes of W, X_s, X_s_clean, X_v, X_a, T, A, Y (the unit's
rows), its rows of the `cycles` table and its row of the `units` table (string fields by value). Prints one JSON line per unit and
MIXER_VERIFY_PASS / MIXER_VERIFY_FAIL. Reads only the unit's rows from the release (located through `cycles`).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np

WORK = os.environ.get("AGTF30_WORK", "work")     # work root; the v1.0 files are in <work>/agtf30_v1
GROUPS = ["W", "X_s", "X_s_clean", "X_v", "X_a", "T", "A", "Y"]


def unit_rows(h, unit: int):
    """(split, first row, n rows, cycles rows) of a unit; rows are ordered by split, unit, cycle."""
    cyc = h["cycles"][:]
    sel = np.flatnonzero(cyc["unit"] == unit)
    if not len(sel):
        return None
    sp = cyc["split"][sel[0]].decode()
    same = np.flatnonzero(cyc["split"] == sp.encode())
    before = same[same < sel[0]]
    first = int(cyc["seconds"][before].sum())
    n = int(cyc["seconds"][sel].sum())
    return sp, first, n, cyc[sel]


def units_equal(ra, rb) -> bool:
    """Field by field, bit for bit. String fields compare by value, because the family field was widened from
    S4 (v1.0 files) to S8 (v1.1 names); the width is storage, not data."""
    if len(ra) != 1 or len(rb) != 1 or set(ra.dtype.names) != set(rb.dtype.names):
        return False
    for k in rb.dtype.names:
        x, y = ra[k], rb[k]
        if x.dtype.kind == "S":
            if x[0] != y[0]:
                return False
        elif x.dtype != y.dtype or x.tobytes() != y.tobytes():
            return False
    return True


def compare_unit(run_h5: pathlib.Path, rel_h5: pathlib.Path, unit: int) -> dict:
    out = {"file": run_h5.name, "unit": unit, "equal": {}}
    with h5py.File(run_h5, "r") as a, h5py.File(rel_h5, "r") as b:
        ra, rb = unit_rows(a, unit), unit_rows(b, unit)
        if ra is None or rb is None:
            return dict(out, ok=False, error=f"unit {unit} missing in {'run' if ra is None else 'release'}")
        out.update(split=[ra[0], rb[0]], rows=[ra[2], rb[2]])
        for g in GROUPS:
            xa = a[f"{g}_{ra[0]}"][ra[1]:ra[1] + ra[2]]
            xb = b[f"{g}_{rb[0]}"][rb[1]:rb[1] + rb[2]]
            out["equal"][g] = xa.shape == xb.shape and xa.dtype == xb.dtype and xa.tobytes() == xb.tobytes()
        out["equal"]["A_unit_column"] = bool(np.all(a[f"A_{ra[0]}"][ra[1]:ra[1] + ra[2], 0] == unit))
        out["equal"]["cycles"] = ra[3].tobytes() == rb[3].tobytes()
        ua, ub = a["units"][:], b["units"][:]
        out["equal"]["units"] = units_equal(ua[ua["unit"] == unit], ub[ub["unit"] == unit])
        for g in ("W_var", "X_s_var", "X_v_var", "X_a_var", "T_var", "A_var"):
            out["equal"][g] = a[g][:].tobytes() == b[g][:].tobytes()
    out["ok"] = all(out["equal"].values())
    if not out["ok"]:
        out["differs"] = [k for k, v in out["equal"].items() if not v]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--release", default=f"{WORK}/agtf30_v1", help="folder with the released AGTF30_<FAM>.h5")
    ap.add_argument("--h5-dir", default="h5")
    a = ap.parse_args(argv)
    res = []
    for f in sorted((pathlib.Path(a.run) / a.h5_dir).glob("AGTF30_*.h5")):
        rel = pathlib.Path(a.release) / f.name
        if not rel.exists():
            print(json.dumps({"file": f.name, "ok": False, "error": f"no {rel}"}))
            res.append(False)
            continue
        with h5py.File(f, "r") as h:
            units = [int(u) for u in h["units"]["unit"]]
        for u in units:
            r = compare_unit(f, rel, u)
            print(json.dumps(r))
            res.append(r["ok"])
    ok = bool(res) and all(res)
    print("MIXER_VERIFY_PASS" if ok else "MIXER_VERIFY_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
