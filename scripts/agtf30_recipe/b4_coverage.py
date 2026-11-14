#!/usr/bin/env python3
"""B4 coverage: idle-floor / top-of-lever flags of the %-max-fan-speed rule over every 10th sample of all N-CMAPSS files (dev+test).
Output $AGTF30_WORK/agtf30_b/b4/coverage_all_ncmapss.json."""
import sys, json, glob, os, numpy as np, h5py
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from b4_throttle import PowerManagement, NCMAPSS_DIR, NF_REF_NCMAPSS, REF_ALT, REF_MN, REF_T2, isa_ts, BIN
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
pm = PowerManagement(json.loads((BIN / "ic_tables.json").read_text()))
n1_ref = pm.n1(80, REF_ALT, REF_MN, REF_T2).item()
tot = {"rows": 0, "idle": 0, "top": 0}; res = {}
for fn in sorted(glob.glob(str(NCMAPSS_DIR / "*.h5"))):
    b = os.path.basename(fn)
    if "DS08d" in b: continue
    with h5py.File(fn, "r") as f:
        xs = [x.decode() for x in np.array(f["X_s_var"]).ravel()]
        for sp in ("dev", "test"):
            W = f[f"W_{sp}"][::10]; nf = f[f"X_s_{sp}"][::10, xs.index("Nf")]; A = f[f"A_{sp}"][::10, :2]
            alt, mn, tra, t2 = W.T
            dT = t2 / (1 + 0.2 * mn**2) - isa_ts(alt)
            tgt = nf / NF_REF_NCMAPSS * n1_ref
            lo = pm.n1(40.0, alt, mn, t2); hi = pm.n1(80.0, alt, mn, t2)
            r = {"rows": len(nf), "idle": int((tgt < lo).sum()), "top": int((tgt > hi).sum()),
                 "max_over_top_pct": float(100 * np.max(tgt / hi - 1)), "min_margin_idle_pct": float(100 * np.min(tgt / lo - 1)),
                 "dT": [float(dT.min()), float(dT.max())], "mn_max": float(mn.max()), "alt": [float(alt.min()), float(alt.max())],
                 "pct_fan": [float(100 * nf.min() / NF_REF_NCMAPSS), float(100 * nf.max() / NF_REF_NCMAPSS)]}
            res[f"{b}:{sp}"] = r
            for k in ("rows", "idle", "top"): tot[k] += r[k]
            print(b, sp, json.dumps(r), flush=True)
res["total"] = tot; print(tot)
json.dump(res, open(f"{WORK}/agtf30_b/b4/coverage_all_ncmapss.json", "w"), indent=1)
