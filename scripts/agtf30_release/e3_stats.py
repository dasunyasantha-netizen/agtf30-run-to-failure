#!/usr/bin/env python3
"""E3 (pass 1): per-unit statistics of AGTF30 v1.0 and N-CMAPSS, read with the same loader (agtf30_data).

    python e3_stats.py --out $AGTF30_WORK/agtf30_v1_work/e3 [--only N-CMAPSS_DS01-005] [--max-units 2]

One unit in memory at a time (memory about 1.5 GB). Per file a pickle <out>/<file stem>.pkl with one record per
unit (split, class, life, flights, hours, onset, T at the first and last flight, per-flight features from
agtf30_data.flight_features, X_s min/max, noise estimates) plus a row subsample of W and X_s for the
coverage and sensor-range figures. e3_report.py turns these into tables and figures.

Noise estimate (the same for both datasets): per flight and channel, sigma = 1.4826 * MAD(second difference) /
sqrt(6), relative to the flight's rms; unit value = median over its flights. A smooth 1 Hz signal has small
second differences, so this measures white noise plus fast dynamics; on AGTF30 X_s_clean it gives the
estimator's floor. AGTF30 also gets the exact value: rms(X_s - X_s_clean) / rms(X_s_clean).
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import h5py
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agtf30_gen"))
from gen_run import set_priority  # noqa: E402  (Idle priority class, checked)

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_learn"))
from agtf30_data import Normaliser, flight_features, iter_units  # noqa: E402

V1 = pathlib.Path(f"{WORK}/agtf30_v1")
NC = pathlib.Path(NCMAPSS)
AGTF30 = [V1 / f"AGTF30_{f}.h5" for f in ("HPT", "HPC", "LPT", "FAN", "MIX")]
NCMAPSS = [NC / f"N-CMAPSS_{k}.h5" for k in ("DS01-005", "DS02-006", "DS03-012", "DS04", "DS05", "DS06", "DS07",
                                              "DS08a-009", "DS08c-008")]     # DS08d-010 is truncated on disk
SUB_X, SUB_W = 200, 50          # row subsample steps for sensor ranges and operating-condition coverage




def flight_bounds(cyc):
    s = np.flatnonzero(np.r_[True, np.diff(cyc) != 0])
    return s, np.r_[s[1:], len(cyc)]


def d2_noise(X, cyc):
    """Per-flight robust second-difference sigma relative to the flight rms; median over flights (per channel)."""
    out = []
    for s, e in zip(*flight_bounds(cyc)):
        x = X[s:e]
        if len(x) < 30:
            continue
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        mad = np.median(np.abs(d2 - np.median(d2, 0)), 0)
        out.append(1.4826 * mad / np.sqrt(6) / np.sqrt(np.mean(x ** 2, 0)))
    return np.median(np.array(out), 0)


def exact_noise(X, Xc, cyc):
    out = []
    for s, e in zip(*flight_bounds(cyc)):
        out.append(np.sqrt(np.mean((X[s:e] - Xc[s:e]) ** 2, 0)) / np.sqrt(np.mean(Xc[s:e] ** 2, 0)))
    return np.median(np.array(out), 0)


def unit_record(key, u, norm, is_agtf30):
    A, an = u["A"], u["A_names"]
    cyc = u["cycle"]
    hs = A[:, an.index("hs")]
    ab = np.flatnonzero(hs == 0)
    s, e = flight_bounds(cyc)
    ff = flight_features(u, norm)
    rec = {"key": key, "file": u["file"], "unit": u["unit"], "Fc": int(A[0, an.index("Fc")]),
           "life": int(cyc[-1] + u["rul"][-1]), "flights": int(len(s)), "first_cycle": int(cyc[0]),
           "hours": len(cyc) / 3600.0, "flight_hours": (e - s) / 3600.0,
           "onset": int(cyc[ab[0]]) if len(ab) else -1,
           "T0": u["T"][0].copy(), "T_eol": u["T"][-1].copy(), "T_names": u["T_names"],
           "F": ff["F"].astype(np.float32), "F_cycle": ff["cycle"], "F_rul": ff["rul"],
           "xs_min": u["X_s"].min(0), "xs_max": u["X_s"].max(0), "noise_d2": d2_noise(u["X_s"], cyc),
           "W_sub": u["W"][::SUB_W].astype(np.float32), "X_sub": u["X_s"][::SUB_X].astype(np.float32)}
    if is_agtf30:
        rec["noise_exact"] = exact_noise(u["X_s"], u["X_s_clean"], cyc)
        rec["noise_d2_clean"] = d2_noise(u["X_s_clean"], cyc)
    return rec


def fit_norm(files):
    return Normaliser().fit(iter_units(files, "dev", max_cycle=10))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/agtf30_v1_work/e3")
    ap.add_argument("--only", default=None, help="comma list of file stems")
    ap.add_argument("--max-units", type=int, default=None, help="test: units per split")
    ap.add_argument("--h5", default=str(V1), help="folder of the AGTF30 files to describe (v1.1: $AGTF30_WORK/agtf30_v11)")
    ap.add_argument("--fams", default="HPT,HPC,LPT,FAN,MIX", help="subsets in --h5 (v1.1: HPTLPT,LPCHPC,ALL)")
    a = ap.parse_args(argv)
    set_priority("idle")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    agtf30 = [pathlib.Path(a.h5) / f"AGTF30_{f}.h5" for f in a.fams.split(",")]
    # the AGTF30 normaliser is always fitted on the v1.0 dev engines, so residuals of later subsets are comparable
    for name, files, fit_files in (("AGTF30", agtf30, AGTF30), ("N-CMAPSS", NCMAPSS, NCMAPSS)):
        todo = [f for f in files if a.only is None or f.stem in a.only.split(",")]
        if not todo:
            continue
        npath = out / f"normaliser_{name}.pkl"
        if npath.exists():
            norm = pickle.loads(npath.read_bytes())
        else:
            t0 = time.time()
            norm = fit_norm(fit_files)
            npath.write_bytes(pickle.dumps(norm))
            print(name, "normaliser r2", np.round(norm.r2, 4).tolist(), f"{time.time() - t0:.0f}s", flush=True)
        for f in todo:
            dst = out / f"{f.stem}.pkl"
            if dst.exists() and a.max_units is None:
                print("skip (done)", f.stem, flush=True)
                continue
            t0 = time.time()
            recs = {}
            extra = ("A", "T", "X_s_clean") if name == "AGTF30" else ("A", "T")
            for sp in ("dev", "test"):
                n = 0
                for key, u in iter_units([f], sp, extra=extra):
                    r = unit_record(key, u, norm, name == "AGTF30")
                    r["split"], r["dataset"] = sp, name
                    recs[f"{sp}:{key}"] = r
                    n += 1
                    if a.max_units and n >= a.max_units:
                        break
            meta = {"dataset": name, "file": f.name, "xs_names": u["xs_names"], "w_names": u["w_names"],
                    "feature_names": [f"mean_{c}" for c in u["w_names"] + u["xs_names"]] +
                                     [f"res_{c}" for c in u["xs_names"]]}
            if name == "AGTF30":
                with h5py.File(f, "r") as h:
                    meta["cycles"] = h["cycles"][:]
                    meta["units"] = h["units"][:]
                    meta["attrs"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps({"meta": meta, "units": recs}))
            tmp.replace(dst)
            print(f.stem, len(recs), "units", f"{time.time() - t0:.0f}s", flush=True)
    print("E3_STATS_DONE")


if __name__ == "__main__":
    main()
