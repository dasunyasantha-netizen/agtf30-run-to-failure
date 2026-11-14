#!/usr/bin/env python3
"""B4 per-phase table of the three matched flights (reads b4_throttle.py outputs).

    python b4_tabulate.py --dir $AGTF30_WORK/agtf30_b/b4   -> phase_table.md + phase_table.json
Phases from a 61 s smoothed altitude rate (+-5 ft/s band), as in agtf30_flights.py.
"""
import argparse, json, pathlib, sys
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from cb_run import read_out  # noqa: E402
from b4_throttle import NF_REF_NCMAPSS  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=f"{WORK}/agtf30_b/b4"); a = ap.parse_args()
d = pathlib.Path(a.dir); rows = []
for z in sorted(d.glob("match_*.npz")):
    fid = z.stem[6:]; M = np.load(z)
    names, Y = read_out(d / f"run_{fid}" / "dyn.bin"); c = {k: i for i, k in enumerate(names)}
    t = np.round(Y[:, 0]).astype(int); n1 = Y[:, c["eng.Shaft.N_Fan"]]
    rate = np.gradient(np.convolve(M["alt"], np.ones(61) / 61, mode="same"))
    ph = np.where(rate > 5, "climb", np.where(rate < -5, "descent", "level"))
    for p in ("climb", "level", "descent"):
        k = t[(ph[t] == p) & (t >= 60)]
        if not len(k):
            continue
        err = (n1[np.isin(t, k)] - M["n1_target"][k]) / M["n1_target"][k] * 100
        rows.append({"flight": fid, "phase": p, "seconds": int(len(k)),
                     "alt_ft": [float(np.min(M["alt"][k])), float(np.max(M["alt"][k]))],
                     "mach_med": float(np.median(M["mn"][k])),
                     "fan_pct_NC_med": float(np.median(M["nf"][k] / NF_REF_NCMAPSS * 100)),
                     "fan_pct_range": [float(np.min(M["nf"][k] / NF_REF_NCMAPSS * 100)), float(np.max(M["nf"][k] / NF_REF_NCMAPSS * 100))],
                     "tra_med": float(np.median(M["tra"][k])), "pla_med": float(np.median(M["pla"][k])),
                     "pla_range": [float(np.min(M["pla"][k])), float(np.max(M["pla"][k]))],
                     "n1_err_abs_p95_pct": float(np.percentile(np.abs(err), 95)),
                     "flagged_s": int(np.sum(M["flag"][k] != 0))})
L = ["| Flight | Phase | s | Alt ft | Mach (med) | N-CMAPSS fan % med [range] | TRA med | PLA med [range] | abs N1 err p95 % | flagged s |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['flight']} | {r['phase']} | {r['seconds']} | {r['alt_ft'][0]:.0f}-{r['alt_ft'][1]:.0f} | {r['mach_med']:.2f} | "
             f"{r['fan_pct_NC_med']:.1f} [{r['fan_pct_range'][0]:.1f}-{r['fan_pct_range'][1]:.1f}] | {r['tra_med']:.1f} | "
             f"{r['pla_med']:.1f} [{r['pla_range'][0]:.1f}-{r['pla_range'][1]:.1f}] | {r['n1_err_abs_p95_pct']:.2f} | {r['flagged_s']} |")
(d / "phase_table.md").write_text("\n".join(L) + "\n"); (d / "phase_table.json").write_text(json.dumps(rows, indent=1))
print("\n".join(L))
