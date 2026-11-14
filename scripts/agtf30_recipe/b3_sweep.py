#!/usr/bin/env python3
"""B3: limits vs solver sweep per fault family (standalone AGTF30 C runners).

    python b3_sweep.py --out $AGTF30_WORK/agtf30_b/b3 [--max 20] [--step-ref 0.25] [--step-dyn 0.5]

For each fault family the wear magnitude s (% efficiency loss of the faulted part;
flow carriers move with the family ratio) is increased step by step:
  * reference check point (steady state, no limiters): SLS, MN 0, ISA+27 degR, PLA 80;
    records stall margins, T45, convergence -> first crossing of each death limit;
  * four hard flight points, closed loop (cb_run: SS start + cb_dyn) on short
    profiles: take-off (idle -> PLA 80 + roll to MN 0.25, hot day), top of climb,
    cruise, low-power descent (PLA chop to idle); records completion, solver steps at
    the iteration cap, limiter use, min margins -> first solver trouble.
Families and single carriers are defined in FAMILIES / SINGLES below.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from b_ss import BIN, T45_MAX, ss_point  # noqa: E402
from cb_run import read_out  # noqa: E402

SM_MIN = 8.0                 # % (Chapman & Litt 2017: LPC/HPC transient minimum)
T45_MN0 = 2241.0             # degR, AGTF30 T45 limit at MN 0 (T45_BET)
CAP = 64                     # MWS.In.Max_IterDyn (solver iteration cap per step)

# per 1 % efficiency loss: carrier -> theta change (fraction). Compressor flow/eff ratio 1.5, fan 1.0 (JT9D)
# (ProDiMES 1-2), turbine flow/eff ratio -0.5 (ProDiMES -1..-0.5; JT9D HPT flow +0.25 %).
FAMILIES = {
    "HPT": {"MWS.HPT.s_Eff": -1.0, "MWS.HPT.s_Wc": +0.5},
    "HPC": {"MWS.HPC.s_Eff": -1.0, "MWS.HPC.s_Wc": -1.5},
    "LPT": {"MWS.LPT.s_Eff": -1.0, "MWS.LPT.s_Wc": +0.5},
    "FAN": {"MWS.FAN.s_Eff": -1.0, "MWS.FAN.s_Wc": -1.0},
    "MIX": {"MWS.HPC.s_Eff": -1.0, "MWS.HPC.s_Wc": -1.5, "MWS.HPT.s_Eff": -1.0, "MWS.HPT.s_Wc": +0.5},
}
SINGLES = {f"{c}.{p}{sg}": {f"MWS.{c}.s_{p}": (-1.0 if sg == "-" else 1.0)}
           for c in ("FAN", "LPC", "HPC", "HPT", "LPT") for p in ("Eff", "Wc") for sg in "-+"
           if not (p == "Eff" and sg == "+")}


def profile(name: str) -> np.ndarray:
    t = np.arange(0, 61, 1.0)
    alt, mn, dT, pla = (np.zeros_like(t) for _ in range(4))
    if name == "takeoff":
        dT[:] = 27
        pla[:] = np.interp(t, [0, 20, 25, 60], [42, 42, 80, 80])
        mn[:] = np.interp(t, [0, 25, 55, 60], [0, 0, 0.25, 0.25])
    elif name == "toc":            # start at part power (SS start), then climb power dynamically
        alt[:] = 35000; mn[:] = 0.8
        pla[:] = np.interp(t, [0, 10, 15, 60], [60, 60, 80, 80])
    elif name == "cruise":
        alt[:] = 35000; mn[:] = 0.78; pla[:] = 65
    elif name == "descent":        # part power, then chop to idle
        alt[:] = 25000; mn[:] = 0.7
        pla[:] = np.interp(t, [0, 10, 12, 60], [60, 60, 40, 40])
    elif name == "toc_ssstart":    # SS start directly at climb power (stress test of the IC solver)
        alt[:] = 35000; mn[:] = 0.8; pla[:] = 80
    return np.c_[t, alt, mn, dT, pla]


def theta_of(direction: dict, s: float) -> dict:
    return {k: v * s / 100.0 for k, v in direction.items()}


def dyn_point(theta: dict, prof: str, work: pathlib.Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    fcsv = work / f"{prof}.csv"
    if not fcsv.exists():
        np.savetxt(fcsv, profile(prof), delimiter=",", header="t,Alt,MN,dT,PLA", comments="")
    run = work / "run"
    if run.exists():
        shutil.rmtree(run)
    cmd = [sys.executable, str(HERE.parent / "agtf30_cbuild" / "cb_run.py"), "--bin", str(BIN),
           "--flight", str(fcsv), "--out", str(run), "--rate", "full"]
    for k, v in theta.items():
        cmd += ["--theta", f"{k}={float(v)!r}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if not (run / "run.json").exists():
        return {"ss_ok": False, "dyn_rc": None, "complete": False, "error": (p.stderr or p.stdout)[-400:]}
    R = json.loads((run / "run.json").read_text())
    out = {"ss_ok": R.get("ss", {}).get("converged") == 1.0, "dyn_rc": R.get("dyn", {}).get("rc"),
           "complete": bool(R.get("dyn", {}).get("complete"))}
    if (run / "dyn.bin").exists():
        n, Y = read_out(run / "dyn.bin")
        c = {k: i for i, k in enumerate(n)}
        it = Y[:, c["loop.Solver.Iterations"]]
        out.update(t_end=float(Y[-1, 0]), cap_steps=int((it >= CAP).sum()), steps=int(len(it)),
                   max_iter=float(it.max()),
                   min_SMFan=float(Y[:, c["eng.SM.SMFan"]].min()), min_SMLPC=float(Y[:, c["eng.SM.SMLPC"]].min()),
                   min_SMHPC=float(Y[:, c["eng.SM.SMHPC"]].min()), max_T45=float(Y[:, c["eng.S45.Tt"]].max()),
                   N1_end=float(Y[-1, c["eng.Shaft.N_Fan"]]), Fn_end=float(Y[-1, c["eng.Perf.Fnet"]]),
                   regs=sorted(set(int(x) for x in Y[:, c["cntrl.RegNum"]])))
    return out


def ref_sweep(direction: dict, smax: float, step: float) -> list[dict]:
    rows = []
    for s in np.arange(0, smax + 1e-9, step):
        r = ss_point(theta_of(direction, s))
        rows.append({"s": float(s), "ok": r["ok"], "iters": r.get("iters"),
                     **{k: r.get(k) for k in ("SM.SMFan", "SM.SMLPC", "SM.SMHPC", "S45.Tt", "S4.Tt",
                                              "Shaft.N_Fan", "Shaft.N_HPC", "S36.Ps", "Perf.Fnet")}})
        if not r["ok"] and len(rows) > 2 and not rows[-2]["ok"]:
            break
    return rows


def crossings(rows: list[dict]) -> dict:
    def first(pred):
        for r in rows:
            if r["ok"] and pred(r):
                return r["s"]
        return None
    fail = next((r["s"] for r in rows if not r["ok"]), None)
    return {"SMFan<8": first(lambda r: r["SM.SMFan"] < SM_MIN), "SMLPC<8": first(lambda r: r["SM.SMLPC"] < SM_MIN),
            "SMHPC<8": first(lambda r: r["SM.SMHPC"] < SM_MIN), "T45>2414": first(lambda r: r["S45.Tt"] > T45_MAX),
            "SMFan<0": first(lambda r: r["SM.SMFan"] < 0), "N3>22500": first(lambda r: r["Shaft.N_HPC"] > 22500),
            "Ps3>790": first(lambda r: r["S36.Ps"] > 790), "ss_fail": fail}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/agtf30_b/b3")
    ap.add_argument("--max", type=float, default=20.0)
    ap.add_argument("--step-ref", type=float, default=0.25)
    ap.add_argument("--step-dyn", type=float, default=0.5)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--no-dyn", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"ref_singles": {}, "ref_families": {}, "dyn": {}}
    for nm, d in SINGLES.items():
        rows = ref_sweep(d, a.max, a.step_ref)
        summary["ref_singles"][nm] = crossings(rows)
        (out / f"ref_single_{nm}.json").write_text(json.dumps(rows))
    for fam in a.families.split(","):
        d = FAMILIES[fam]
        rows = ref_sweep(d, a.max, a.step_ref)
        summary["ref_families"][fam] = crossings(rows)
        (out / f"ref_family_{fam}.json").write_text(json.dumps(rows))
        print(fam, summary["ref_families"][fam], flush=True)
        if a.no_dyn:
            continue
        # closed loop on hard flight points: up to the first death crossing x 1.5 and at least 6 %
        x = [v for k, v in summary["ref_families"][fam].items() if v is not None and k in
             ("SMFan<8", "SMLPC<8", "SMHPC<8", "T45>2414")]
        smax = min(a.max, max(6.0, 1.5 * min(x) if x else a.max))
        summary["dyn"][fam] = {}
        for prof in ("takeoff", "toc", "cruise", "descent", "toc_ssstart"):
            res, bad = [], 0
            for s in np.arange(0, smax + 1e-9, a.step_dyn):
                r = dyn_point(theta_of(d, s), prof, out / "work")
                res.append({"s": float(s), **r})
                bad = bad + 1 if not (r["ss_ok"] and r["complete"]) else 0
                if bad >= 2:
                    break
            first_fail = next((r["s"] for r in res if not (r["ss_ok"] and r["complete"])), None)
            first_cap = next((r["s"] for r in res if r.get("cap_steps", 0) > 0), None)
            summary["dyn"][fam][prof] = {"swept_to": res[-1]["s"], "first_fail": first_fail, "first_cap": first_cap}
            (out / f"dyn_{fam}_{prof}.json").write_text(json.dumps(res))
            print(" ", fam, prof, summary["dyn"][fam][prof], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
