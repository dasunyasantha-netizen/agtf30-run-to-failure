#!/usr/bin/env python3
"""C2 (part 1): N-CMAPSS source-flight library for the AGTF30 generator.

    python gen_library.py --out $AGTF30_WORK/agtf30_c/library [--workers 6]

One .npz per N-CMAPSS unit (dev and test, 9 files; DS08d is damaged and skipped) with the
1 Hz flight inputs the generator needs, plus index.json (unit list with class and cycle
offsets). Per row: alt, Mach, TRA, T2 (W, copied into the AGTF30 files), Nf (N-CMAPSS fan
speed), dT (ambient minus ISA, degR), and the B4 throttle match: AGTF30 PLA and the clip flag
(0 matched, -1 below the PLA-40 fan speed, +1 above the PLA-80 fan speed).

Throttle rule (B_RECIPE section 5, unchanged): N1_target = Nf / 2284.5 * N1_REF_AGTF30, with
N1_REF_AGTF30 = N1 at PLA 80 at the N-CMAPSS peak condition (2214.6 rpm). The lever is found
by inverting AGTF30's own schedule N1(PLA) = Nc1(TF, alt, MN) * sqrt(T2/518.67),
TF = (PLA-40)/40. Nc1 is linear in TF between the 11 table nodes for a fixed (alt, MN), so the
inverse is exact per segment (B4 used a 0.05-degree PLA grid; same rule, finer).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys

import h5py
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_recipe"))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from b4_throttle import (NCMAPSS_DIR, NF_REF_NCMAPSS, PLA_IDLE, PLA_MAX, REF_ALT, REF_MN, REF_T2,  # noqa: E402
                         T_REF, PowerManagement, isa_ts)

FILES = ["N-CMAPSS_DS01-005", "N-CMAPSS_DS02-006", "N-CMAPSS_DS03-012", "N-CMAPSS_DS04", "N-CMAPSS_DS05",
         "N-CMAPSS_DS06", "N-CMAPSS_DS07", "N-CMAPSS_DS08a-009", "N-CMAPSS_DS08c-008"]   # DS08d skipped (damaged)
BIN = pathlib.Path(f"{WORK}/agtf30_cbuild/bin_b")


def pm_tables(bin_dir=BIN):
    return PowerManagement(json.loads((pathlib.Path(bin_dir) / "ic_tables.json").read_text()))


def pla_match(pm: PowerManagement, nf, alt, mn, t2):
    """Exact per-segment inverse of AGTF30's PLA -> N1 schedule. Returns pla, flag, n1_target."""
    n1_ref = pm.n1(PLA_MAX, REF_ALT, REF_MN, REF_T2).item()
    tgt = np.asarray(nf, float) / NF_REF_NCMAPSS * n1_ref
    tf_nodes = pm.f.grid[0]
    n = len(tgt)
    N = np.empty((n, len(tf_nodes)))
    for k, tf in enumerate(tf_nodes):
        N[:, k] = pm.nc1(PLA_IDLE + tf * (PLA_MAX - PLA_IDLE), alt, mn) * np.sqrt(np.asarray(t2, float) / T_REF)
    assert np.all(np.diff(N, axis=1) > 0), "N1(PLA) not strictly increasing"
    flag = np.where(tgt < N[:, 0], -1, np.where(tgt > N[:, -1], 1, 0)).astype(np.int8)
    tc = np.clip(tgt, N[:, 0], N[:, -1])
    k = np.minimum((tc[:, None] >= N[:, 1:]).sum(axis=1), len(tf_nodes) - 2)   # segment index (N increasing)
    r = np.arange(n)
    f = (tc - N[r, k]) / (N[r, k + 1] - N[r, k])
    tf = tf_nodes[k] + f * (tf_nodes[k + 1] - tf_nodes[k])
    return PLA_IDLE + tf * (PLA_MAX - PLA_IDLE), flag, tgt


def build_unit(job):
    fname, part, unit, out = job
    with h5py.File(NCMAPSS_DIR / f"{fname}.h5", "r") as f:
        A = f[f"A_{part}"][:]
        m = np.flatnonzero(A[:, 0] == unit)
        s = slice(m[0], m[-1] + 1)
        assert len(m) == m[-1] - m[0] + 1, "unit rows not contiguous"
        W = f[f"W_{part}"][s]
        xs = [x.decode() for x in np.array(f["X_s_var"]).ravel()]
        nf = f[f"X_s_{part}"][s, xs.index("Nf")]
        A = A[s]
    alt, mn, tra, t2 = W.T
    dT = t2 / (1 + 0.2 * mn ** 2) - isa_ts(alt)
    pm = pm_tables()
    pla = np.empty(len(nf)); flag = np.empty(len(nf), np.int8)
    for a in range(0, len(nf), 200000):
        b = slice(a, a + 200000)
        pla[b], flag[b], _ = pla_match(pm, nf[b], alt[b], mn[b], t2[b])
    cyc = A[:, 1].astype(int)
    starts = np.flatnonzero(np.r_[True, np.diff(cyc) != 0])
    assert np.all(np.diff(cyc[starts]) == 1) and cyc[starts[0]] == 1, "cycles not 1..n in order"
    key = f"{fname.replace('N-CMAPSS_', '')}_{part}_u{unit}"
    np.savez(out / f"{key}.npz", alt=alt, mn=mn, tra=tra, t2=t2, nf=nf, dT=dT, pla=pla, flag=flag,
             cycle=cyc, hs=A[:, 3].astype(np.int8), starts=starts)
    lens = np.diff(np.r_[starts, len(cyc)])
    return {"key": key, "file": fname, "part": part, "unit": int(unit), "Fc": int(A[0, 2]),
            "n_cycles": int(len(starts)), "seconds": lens.tolist(), "hours": float(lens.sum() / 3600),
            "max_flight_s": int(lens.max()), "flag_below": int((flag == -1).sum()), "flag_above": int((flag == 1).sum()),
            "abs_dT_max": float(np.abs(dT).max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/agtf30_c/library")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fname in FILES:
        with h5py.File(NCMAPSS_DIR / f"{fname}.h5", "r") as f:
            for part in ("dev", "test"):
                for u in np.unique(f[f"A_{part}"][:, 0]).astype(int):
                    jobs.append((fname, part, int(u), out))
    with cf.ProcessPoolExecutor(a.workers) as ex:
        units = list(ex.map(build_unit, jobs))
    units.sort(key=lambda u: u["key"])
    idx = {"units": units, "n_units": len(units),
           "by_class": {c: {"units": sum(u["Fc"] == c for u in units),
                            "flights": sum(u["n_cycles"] for u in units if u["Fc"] == c),
                            "hours": round(sum(u["hours"] for u in units if u["Fc"] == c), 1)} for c in (1, 2, 3)},
           "max_flight_s": max(u["max_flight_s"] for u in units),
           "clip_below_s": sum(u["flag_below"] for u in units), "clip_above_s": sum(u["flag_above"] for u in units),
           "seconds_total": sum(sum(u["seconds"]) for u in units), "abs_dT_max": max(u["abs_dT_max"] for u in units)}
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    print(json.dumps({k: v for k, v in idx.items() if k != "units"}, indent=1))


if __name__ == "__main__":
    main()
