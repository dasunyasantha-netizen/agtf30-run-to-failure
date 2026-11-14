#!/usr/bin/env python3
"""B5: summarise life_<tag>_<FAM>.json -> $AGTF30_WORK/agtf30_b/b5/summary_<tag>.json and a printed table."""
import json, sys, numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
S = {}
for f in "HPT HPC LPT FAN MIX".split():
    D = json.load(open(f"{WORK}/agtf30_b/b5/life_{tag}_{f}.json"))
    E = D["engines"]
    eol = np.array([e["eol"] for e in E]); on = np.array([e["onset"] for e in E])
    lim = {}
    for e in E: lim[e["limit"]] = lim.get(e["limit"], 0) + 1
    fe = [e["theta_eol"] for e in E]
    keys = sorted(fe[0])
    s = {"params": D["params"], "n": len(E), "eol_min": int(eol.min()), "eol_med": float(np.median(eol)), "eol_max": int(eol.max()),
         "eol_mean": float(eol.mean()), "eol_sd": float(eol.std()), "in_60_100": int(((eol >= 60) & (eol <= 100)).sum()),
         "onset_min": int(on.min()), "onset_med": float(np.median(on)), "onset_max": int(on.max()),
         "eol_by_class": {c: float(np.median([e["eol"] for e in E if e["cls"] == c])) for c in (1, 2, 3)},
         "died": sum(e["died"] for e in E), "solver_fail": sum(e["solver_fail"] for e in E), "limit": lim,
         "fault_eff_loss_eol_pct": [float(np.min([e["g_eol"] for e in E])), float(np.median([e["g_eol"] for e in E])), float(np.max([e["g_eol"] for e in E]))],
         "theta_eol_median_pct": {k: float(np.median([x[k] for x in fe])) for k in keys},
         "hi0": [float(min(e["hi0"] for e in E)), float(max(e["hi0"] for e in E))],
         "T45_rise_eol_degR": float(np.median([e["hist"][-1]["T45"] - 2351.154 for e in E if e["hist"][-1]["ok"]])),
         "min_SM_at_eol": {k: float(min(e["hist"][-1][k] for e in E)) for k in ("SMFan", "SMLPC", "SMHPC")},
         "hi_monotone_violations": int(sum(sum(1 for a, b in zip(e["hist"], e["hist"][1:]) if b["hi"] is not None and a["hi"] is not None and b["hi"] > a["hi"] + 1e-3) for e in E))}
    S[f] = s
    print(f"{f}: life {s['eol_min']}/{s['eol_med']:.0f}/{s['eol_max']} (60-100: {s['in_60_100']}/20), by class {s['eol_by_class']}, onset {s['onset_min']}/{s['onset_med']:.0f}/{s['onset_max']}, "
          f"died {s['died']} fail {s['solver_fail']} lim {lim}, fault eff loss EOL {['%.2f' % x for x in s['fault_eff_loss_eol_pct']]}, HI0 {['%.3f' % x for x in s['hi0']]}, "
          f"minSM {s['min_SM_at_eol']}, HI up-steps {s['hi_monotone_violations']}")
    print("   theta EOL median %:", {k.replace('MWS.', '').replace('.s_', ' '): round(v, 2) for k, v in s["theta_eol_median_pct"].items()})
json.dump(S, open(f"{WORK}/agtf30_b/b5/summary_{tag}.json", "w"), indent=1)
