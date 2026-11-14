#!/usr/bin/env python3
"""G1 (v1.1 feasibility): limits-versus-solver sweep for the three candidate fault subsets.

    python g1_v11_sweep.py --fam HPTLPT --out $AGTF30_WORK/agtf30_g/g1 [--priority idle]

Same procedure as B3 (b3_sweep.py, B_RECIPE section 4), for the N-CMAPSS failure modes v1.0 lacks:
  HPTLPT  (N-CMAPSS DS02/DS03): HPT + LPT at equal size, v1.0 turbine directions
  LPCHPC  (N-CMAPSS DS06):      LPC + HPC at equal size, compressor direction -1.5 for both
  ALL     (N-CMAPSS DS08):      all five modules at equal size, each with its v1.0 direction
  LPC     diagnostic only (LPC alone, -1.5), to see whether the LPC can end a life by itself.
For each subset the wear size s (% efficiency loss of each faulted module) is raised step by step:
  * reference point at constant corrected fan speed (B3 column 1) and at constant healthy take-off
    thrust (B3 column 2, the death check the generator uses): first crossing of each death limit,
    first steady-state solver failure;
  * the five B3 closed-loop flight points (take-off, top of climb, cruise, descent, SS start at climb
    power), up to max(6 %, 1.5 x the constant-thrust death size): first failure, first iteration-cap step.
Output: <out>/<fam>.json (rows + summary). v1.0 data and code paths are only read.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agtf30_gen"))
from b3_sweep import SM_MIN, crossings, dyn_point, ref_sweep, theta_of  # noqa: E402
from b_ss import T45_MAX, ss_point, ss_point_thrust  # noqa: E402

# per 1 % efficiency loss of each faulted module: carrier -> change (fraction x 100)
D_FAN = {"MWS.FAN.s_Eff": -1.0, "MWS.FAN.s_Wc": -0.7}     # v1.0 FAN (N-CMAPSS DS04 ratio)
D_LPC = {"MWS.LPC.s_Eff": -1.0, "MWS.LPC.s_Wc": -1.5}     # new: ProDiMES compressor 1-2, JT9D long-term 1.76
D_HPC = {"MWS.HPC.s_Eff": -1.0, "MWS.HPC.s_Wc": -1.5}     # v1.0 HPC
D_HPT = {"MWS.HPT.s_Eff": -1.0, "MWS.HPT.s_Wc": +0.5}     # v1.0 HPT (ProDiMES turbine)
D_LPT = {"MWS.LPT.s_Eff": -1.0, "MWS.LPT.s_Wc": +0.5}     # v1.0 LPT
FAMILIES = {"HPTLPT": {**D_HPT, **D_LPT}, "LPCHPC": {**D_LPC, **D_HPC},
            "ALL": {**D_FAN, **D_LPC, **D_HPC, **D_HPT, **D_LPT}, "LPC": dict(D_LPC)}
PROFILES = ("takeoff", "toc", "cruise", "descent", "toc_ssstart")


def thrust_sweep(direction: dict, fn: float, smax: float, step: float) -> list[dict]:
    rows = []
    for s in np.arange(0, smax + 1e-9, step):
        r = ss_point_thrust(theta_of(direction, s), fn)
        rows.append({"s": float(s), "ok": bool(r["ok"]),
                     **{k: r.get(k) for k in ("SM.SMFan", "SM.SMLPC", "SM.SMHPC", "S45.Tt", "S4.Tt", "Shaft.N_Fan",
                                              "Shaft.N_HPC", "S36.Ps", "Perf.Fnet")}})
        if not r["ok"] and len(rows) > 2 and not rows[-2]["ok"]:
            break
        if r["ok"] and r["S45.Tt"] > T45_MAX + 60 and min(r["SM.SMFan"], r["SM.SMLPC"], r["SM.SMHPC"]) < SM_MIN:
            break   # well past death on two limits: enough
    return rows


def death(cross: dict) -> float | None:
    x = [v for k, v in cross.items() if v is not None and k in ("SMFan<8", "SMLPC<8", "SMHPC<8", "T45>2414")]
    return min(x) if x else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=list(FAMILIES))
    ap.add_argument("--out", default=f"{WORK}/agtf30_g/g1")
    ap.add_argument("--max", type=float, default=20.0)
    ap.add_argument("--step-ref", type=float, default=0.25)
    ap.add_argument("--step-dyn", type=float, default=0.5)
    ap.add_argument("--priority", default="idle", choices=["idle", "below_normal", "normal"])
    a = ap.parse_args(argv)
    from gen_run import set_priority
    set_priority(a.priority)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = FAMILIES[a.fam]
    t0 = time.time()
    fn = ss_point({})["Perf.Fnet"]                        # healthy take-off thrust at the reference point
    res = {"fam": a.fam, "direction": d, "fn_ref": fn}
    rows_n = ref_sweep(d, a.max, a.step_ref)
    res["ref_const_N1"] = {"cross": crossings(rows_n), "rows": rows_n}
    print(a.fam, "const N1", res["ref_const_N1"]["cross"], f"{time.time()-t0:.0f}s", flush=True)
    rows_f = thrust_sweep(d, fn, a.max, a.step_ref)
    res["ref_const_Fn"] = {"cross": crossings(rows_f), "rows": rows_f}
    print(a.fam, "const Fn", res["ref_const_Fn"]["cross"], f"{time.time()-t0:.0f}s", flush=True)
    s_death = death(res["ref_const_Fn"]["cross"])
    smax = min(a.max, max(6.0, 1.5 * s_death if s_death else a.max))
    res["dyn"] = {}
    for prof in PROFILES:
        rr, bad = [], 0
        for s in np.arange(0, smax + 1e-9, a.step_dyn):
            r = dyn_point(theta_of(d, s), prof, out / f"work_{a.fam}")
            rr.append({"s": float(s), **r})
            bad = bad + 1 if not (r["ss_ok"] and r["complete"]) else 0
            if bad >= 2:
                break
        res["dyn"][prof] = {"swept_to": rr[-1]["s"],
                            "first_fail": next((r["s"] for r in rr if not (r["ss_ok"] and r["complete"])), None),
                            "first_cap": next((r["s"] for r in rr if r.get("cap_steps", 0) > 0), None), "rows": rr}
        print(" ", a.fam, prof, {k: v for k, v in res["dyn"][prof].items() if k != "rows"},
              f"{time.time()-t0:.0f}s", flush=True)
    res["summary"] = {"death_const_Fn": s_death, "death_limit": None if s_death is None else
                      next(k for k, v in res["ref_const_Fn"]["cross"].items() if v == s_death),
                      "dyn_swept_to": smax, "elapsed_s": round(time.time() - t0)}
    (out / f"{a.fam}.json").write_text(json.dumps(res, indent=1))
    print(a.fam, "summary", res["summary"], flush=True)


if __name__ == "__main__":
    main()
