#!/usr/bin/env python3
"""B8 supporting checks for the run plan.

1. N-CMAPSS flight pool per length class (flights, flight-hours) from b5/ncmapss_onset.json.
2. Does N-CMAPSS's fan speed at a given flight condition change with its own wear?  Corrected fan speed
   is fitted on the W variables (TRA, alt, Mach polynomial) over each DS01 dev unit's first 10 cycles
   (TRA > 40) and the mean residual over its last 10 cycles is reported (% of mean corrected speed).
   If ~0, any N-CMAPSS cycle can be a source flight for throttle matching (B4 rule).
Writes $AGTF30_WORK/agtf30_b/b8/b8_checks.json
"""
import collections, glob, json, pathlib
import h5py
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README

B = pathlib.Path(f"{WORK}/agtf30_b")


def main():
    on = json.loads((B / "b5/ncmapss_onset.json").read_text())
    pool = collections.defaultdict(lambda: {"units": 0, "flights": 0, "hours": 0.0})
    for u in on:
        p = pool[u["Fc"]]; p["units"] += 1; p["flights"] += u["eol"]; p["hours"] += u["total_hours"]
    for p in pool.values():
        p["hours"] = round(p["hours"], 1); p["h_per_flight"] = round(p["hours"] / p["flights"], 3)
    f = glob.glob(NCMAPSS + "**/N-CMAPSS_DS01-005.h5", recursive=True)[0]
    with h5py.File(f, "r") as h:
        W, X, A = h["W_dev"][::10], h["X_s_dev"][::10], h["A_dev"][::10]
        xv = [s.decode() for s in h["X_s_var"][:]]; wv = [s.decode() for s in h["W_var"][:]]
    alt, mn, tra, t2 = (W[:, wv.index(k)] for k in ("alt", "Mach", "TRA", "T2"))
    nc = X[:, xv.index("Nf")] / np.sqrt(t2 / 518.67)
    a = alt / 1e4
    F = np.column_stack([np.ones_like(tra), tra, tra**2, tra**3, a, a**2, mn, mn**2, tra * mn, tra * a])
    nf = {}
    for unit in np.unique(A[:, 0]):
        m = A[:, 0] == unit; eol = A[m, 1].max()
        e = m & (A[:, 1] <= 10) & (tra > 40); l = m & (A[:, 1] >= eol - 9) & (tra > 40)
        b, *_ = np.linalg.lstsq(F[e], nc[e], rcond=None)
        ref = float(np.mean(nc[e]))
        nf[int(unit)] = {"eol": int(eol), "fit_resid_sd_pct": float(np.std(nc[e] - F[e] @ b) / ref * 100),
                         "late_mean_shift_pct": float(np.mean(nc[l] - F[l] @ b) / ref * 100)}
    out = {"pool_by_class": dict(sorted(pool.items())), "ncmapss_nf_vs_wear_DS01_dev": nf}
    (B / "b8").mkdir(parents=True, exist_ok=True)
    (B / "b8/b8_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
