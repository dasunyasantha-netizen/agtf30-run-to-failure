#!/usr/bin/env python3
"""Phase B4: throttle matching by percent of maximum fan speed (Q-024 decision 5).

    python b4_throttle.py --out $AGTF30_WORK/agtf30_b/b4            # table, SS check, 3 flights
    python b4_throttle.py --out $AGTF30_WORK/agtf30_b/b4 --no-dyn   # table + SS check only

Rule.  At every second of a real N-CMAPSS flight, AGTF30's fan runs at the same percent of
its own maximum as N-CMAPSS's fan did:  N1_target = (Nf / NF_REF_NCMAPSS) * N1_REF_AGTF30.
Both 100 % references are "maximum fan speed of a healthy engine at maximum lever" and are
taken at the same flight condition, the one where N-CMAPSS's healthy fan peaks (see REF_*).

Lever lookup.  AGTF30's controller turns PLA into a corrected fan-speed demand
Nc1 = interp3(PM_AltVec, PM_TF, PM_MNVec, PM_NcArray) with TF = (PLA-40)/40
(setup_Controller.m) and holds the physical fan speed N1 = Nc1 * sqrt(T2/518.67) in steady
state (checked below against the steady-state runner and the dynamic runner's N1cReq).
So the healthy-engine table "PLA -> N1 at (alt, Mach, T2)" is the controller's own schedule
and the inverse is a 1-D monotone interpolation per second.  Seconds whose target lies
below the PLA-40 fan speed (idle floor) or above the PLA-80 fan speed (top of lever) are
clipped and flagged; seconds where a limiter (T45, N1, Ps3, Wf/Ps3) holds the fan below the
demand show up in the dynamic check (cntrl.RegNum != 0).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
sys.path.insert(0, str(HERE))
from cb_run import read_out  # noqa: E402
from b_ss import BIN, ss_point  # noqa: E402

NCMAPSS_DIR = pathlib.Path(NCMAPSS)
T_REF = 518.67          # degR, corrected-speed reference
PLA_IDLE, PLA_MAX = 40.0, 80.0
# N-CMAPSS 100 %: maximum healthy fan speed (cycles <= 5, dev, all files except the truncated
# DS08d): 2284.5 rpm at alt 5429 ft, Mach 0.3614, TRA 88.77 (= max TRA), T2 512.35 degR
# (DS05/06/07; $AGTF30_WORK/agtf30_b/b4/ncmapss_nf_stats.json).
NF_REF_NCMAPSS = 2284.5
REF_ALT, REF_MN, REF_T2 = 5429.0, 0.3614, 512.35
# Flights for the check: the Stage 1 selection, one per N-CMAPSS length class.
FLIGHTS = (("N-CMAPSS_DS01-005.h5", 1, 1), ("N-CMAPSS_DS04.h5", 1, 1), ("N-CMAPSS_DS08c-008.h5", 1, 1))


def isa_ts(alt_ft):
    """ISA static temperature, degR (troposphere lapse, isothermal above 36089 ft)."""
    a = np.asarray(alt_ft, float)
    return np.where(a < 36089.0, 518.67 - 0.00356616 * a, 389.97)


class PowerManagement:
    """AGTF30's PLA -> corrected fan-speed schedule (ic_tables.json export of setup_Controller)."""

    def __init__(self, tables: dict):
        TF = np.array(tables["PM_TF"], float).ravel()
        Alt = np.array(tables["PM_AltVec"], float).ravel()
        MN = np.array(tables["PM_MNVec"], float).ravel()
        PM = np.array(tables["PM_NcArray"], float)
        if PM.ndim == 1:
            PM = PM.reshape(tables["PM_NcArray_size"], order="F")
        self.alt_rng, self.mn_rng = (Alt[0], Alt[-1]), (MN[0], MN[-1])
        self.f = RegularGridInterpolator((TF, Alt, MN), PM)

    def nc1(self, pla, alt, mn):
        pla, alt, mn = np.broadcast_arrays(np.asarray(pla, float), np.asarray(alt, float), np.asarray(mn, float))
        tf = (pla - PLA_IDLE) / (PLA_MAX - PLA_IDLE)
        q = np.stack([tf, np.clip(alt, *self.alt_rng), np.clip(mn, *self.mn_rng)], axis=-1)
        return self.f(q)

    def n1(self, pla, alt, mn, t2):
        return self.nc1(pla, alt, mn) * np.sqrt(np.asarray(t2, float) / T_REF)


def pla_for_n1(pm: PowerManagement, n1_target, alt, mn, t2, step=0.05):
    """Invert N1(PLA) per sample. Returns PLA (clipped to [40, 80]) and flags:
    0 = matched, -1 = below the idle-floor fan speed, +1 = above the PLA-80 fan speed."""
    grid = np.arange(PLA_IDLE, PLA_MAX + 1e-9, step)
    n = len(n1_target)
    N = pm.n1(grid[None, :], np.asarray(alt)[:, None], np.asarray(mn)[:, None], np.asarray(t2)[:, None])
    assert np.all(np.diff(N, axis=1) > 0), "N1(PLA) not strictly increasing"
    pla = np.empty(n)
    for i in range(n):
        pla[i] = np.interp(n1_target[i], N[i], grid)
    flag = np.where(n1_target < N[:, 0], -1, np.where(n1_target > N[:, -1], 1, 0))
    return pla, flag, N[:, 0], N[:, -1]


def read_flight(fname: str, unit: int, cycle: int) -> dict:
    with h5py.File(NCMAPSS_DIR / fname, "r") as f:
        A = f["A_dev"][:, :3]
        idx = np.flatnonzero((A[:, 0] == unit) & (A[:, 1] == cycle))
        s = slice(idx[0], idx[-1] + 1)
        assert len(idx) == idx[-1] - idx[0] + 1, "flight rows not contiguous"
        W = f["W_dev"][s]
        xs = [x.decode() for x in np.array(f["X_s_var"]).ravel()]
        nf = f["X_s_dev"][s, xs.index("Nf")]
    alt, mn, tra, t2 = W.T
    ts = t2 / (1 + 0.2 * mn ** 2)
    return {"file": fname, "unit": unit, "cycle": cycle, "Fc": int(A[idx[0], 2]), "alt": alt, "mn": mn,
            "tra": tra, "t2": t2, "dT": ts - isa_ts(alt), "nf": nf}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/agtf30_b/b4")
    ap.add_argument("--no-dyn", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = json.loads((BIN / "ic_tables.json").read_text())
    pm = PowerManagement(T)
    S = {}

    # ---- AGTF30 100 % reference: PLA 80 at the N-CMAPSS peak condition
    n1_ref = pm.n1(PLA_MAX, REF_ALT, REF_MN, REF_T2).item()
    S["reference"] = {"NF_REF_NCMAPSS_rpm": NF_REF_NCMAPSS, "N1_REF_AGTF30_rpm": n1_ref,
                      "condition": {"alt_ft": REF_ALT, "MN": REF_MN, "T2_degR": REF_T2}}

    # ---- lookup table (healthy engine): PLA x alt x MN at ISA, physical N1 and % of reference
    pla_g = np.arange(40, 80.1, 5.0); alt_g = np.array([0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3])
    mn_g = np.round(np.arange(0, 0.81, 0.1), 2)
    rows = []
    for alt in alt_g:
        for mn in mn_g:
            t2 = float(isa_ts(alt)) * (1 + 0.2 * mn ** 2)
            n1 = pm.n1(pla_g, alt, mn, t2)
            rows.append({"alt": float(alt), "MN": float(mn), "T2": t2, "N1": n1.tolist(),
                         "pctN1": (100 * n1 / n1_ref).tolist()})
    S["table"] = {"PLA": pla_g.tolist(), "rows": rows, "note": "ISA day; for other temperatures N1 scales with sqrt(T2)"}

    # ---- steady-state runner check of N1 = Nc1(PLA) * sqrt(T2/518.67), incl. hot/cold days
    chk = []
    for alt, mn, dT, pla in [(0, 0, 27, 80), (0, 0, 0, 40), (0, 0.25, 15, 70), (5000, 0.36, 0, 80),
                             (10000, 0.4, -20, 55), (20000, 0.6, 10, 75), (25000, 0.7, 0, 45),
                             (30000, 0.75, -15, 65), (35000, 0.8, 0, 80), (35000, 0.6, 20, 50)]:
        t2 = (float(isa_ts(alt)) + dT) * (1 + 0.2 * mn ** 2)
        r = ss_point({}, alt, mn, dT, pla)
        pred = pm.n1(pla, alt, mn, t2).item()
        chk.append({"alt": alt, "MN": mn, "dT": dT, "PLA": pla, "ok": r["ok"], "N1_ss": r.get("Shaft.N_Fan"),
                    "N1_pred": pred, "T2_ss": r.get("S2.Tt"), "T2_pred": t2,
                    "rel_err": (r["Shaft.N_Fan"] - pred) / pred if r["ok"] else None})
    S["ss_check"] = chk
    S["ss_check_max_rel_err"] = max(abs(c["rel_err"]) for c in chk if c["rel_err"] is not None)

    # ---- three real flights
    S["flights"] = []
    for fname, unit, cycle in FLIGHTS:
        F = read_flight(fname, unit, cycle)
        tgt = F["nf"] / NF_REF_NCMAPSS * n1_ref
        pla, flag, n1_idle, n1_top = pla_for_n1(pm, tgt, F["alt"], F["mn"], F["t2"])
        n = len(tgt)
        env_dT = np.abs(F["dT"]) > 30
        fid = f"{pathlib.Path(fname).stem.replace('N-CMAPSS_', '')}_u{unit}_c{cycle}"
        Fs = {"flight": fid, "Fc": F["Fc"], "seconds": n, "hours": n / 3600,
              "pct_fan_NC": [float(x) for x in np.percentile(100 * F["nf"] / NF_REF_NCMAPSS, [0, 5, 50, 95, 100])],
              "pla": [float(x) for x in np.percentile(pla, [0, 5, 50, 95, 100])],
              "tra": [float(x) for x in np.percentile(F["tra"], [0, 5, 50, 95, 100])],
              "corr_pla_tra": float(np.corrcoef(pla, F["tra"])[0, 1]),
              "s_idle_floor": int((flag == -1).sum()), "s_above_top": int((flag == 1).sum()),
              "max_shortfall_above_top_pct": float(100 * np.max(np.where(flag == 1, tgt / n1_top - 1, 0))),
              "max_excess_below_idle_pct": float(100 * np.max(np.where(flag == -1, n1_idle / tgt - 1, 0))),
              "dT_range": [float(F["dT"].min()), float(F["dT"].max())], "s_dT_outside_30": int(env_dT.sum())}
        csv = out / f"flight_{fid}.csv"
        np.savetxt(csv, np.c_[np.arange(n, dtype=float), F["alt"], F["mn"], np.clip(F["dT"], -30, 30), pla],
                   delimiter=",", header="time_s,altitude_ft,mach,dT_degF,pla_deg", comments="", fmt="%.10g")
        np.savez_compressed(out / f"match_{fid}.npz", alt=F["alt"], mn=F["mn"], t2=F["t2"], dT=F["dT"], tra=F["tra"],
                            nf=F["nf"], n1_target=tgt, pla=pla, flag=flag, n1_idle=n1_idle, n1_top=n1_top)
        if not a.no_dyn:
            Fs["dyn"] = run_dyn(csv, out / f"run_{fid}", tgt)
        S["flights"].append(Fs)
        print(json.dumps({k: v for k, v in Fs.items() if k != "dyn"}), flush=True)
        if "dyn" in Fs:
            print("  dyn:", json.dumps(Fs["dyn"]), flush=True)
    (out / "b4_summary.json").write_text(json.dumps(S, indent=1))
    print("reference", S["reference"], "ss_check_max_rel_err", S["ss_check_max_rel_err"])


def run_dyn(csv: pathlib.Path, run: pathlib.Path, tgt: np.ndarray) -> dict:
    """Closed-loop healthy flight at 1 Hz output; compare achieved N1 with the target."""
    if run.exists():
        shutil.rmtree(run)
    cmd = [sys.executable, str(HERE.parent / "agtf30_cbuild" / "cb_run.py"), "--bin", str(BIN),
           "--flight", str(csv), "--out", str(run), "--rate", "1hz"]
    subprocess.run(cmd, capture_output=True, text=True)
    R = json.loads((run / "run.json").read_text())
    D = {"ss_converged": R.get("ss", {}).get("converged"), "rc": R.get("dyn", {}).get("rc"),
         "complete": R.get("dyn", {}).get("complete"), "wall_s": R.get("dyn", {}).get("wall_s")}
    if not (run / "dyn.bin").exists():
        return D
    names, Y = read_out(run / "dyn.bin")
    c = {k: i for i, k in enumerate(names)}
    t = Y[:, 0]
    k = np.clip(np.round(t).astype(int), 0, len(tgt) - 1)
    n1 = Y[:, c["eng.Shaft.N_Fan"]]
    err = (n1 - tgt[k]) / tgt[k] * 100
    reg = Y[:, c["cntrl.RegNum"]].astype(int)
    settle = t >= 60          # skip the first minute (start transient)
    regs, counts = np.unique(reg, return_counts=True)
    D.update(t_end=float(t[-1]), samples=int(len(t)),
             n1_err_pct={"p5": float(np.percentile(err[settle], 5)), "median": float(np.median(err[settle])),
                         "p95": float(np.percentile(err[settle], 95)),
                         "abs_le_1pct_share": float(np.mean(np.abs(err[settle]) <= 1.0))},
             n1_err_on_reg0_pct={"p5": float(np.percentile(err[settle & (reg == 0)], 5)),
                                 "p95": float(np.percentile(err[settle & (reg == 0)], 95))}
             if np.any(settle & (reg == 0)) else None,
             regnum_seconds={int(r): int(n) for r, n in zip(regs, counts)},
             min_SMFan=float(Y[:, c["eng.SM.SMFan"]].min()), min_SMLPC=float(Y[:, c["eng.SM.SMLPC"]].min()),
             min_SMHPC=float(Y[:, c["eng.SM.SMHPC"]].min()), max_T45=float(Y[:, c["eng.S45.Tt"]].max()),
             max_iter=float(Y[:, c["loop.Solver.Iterations"]].max()))
    return D


if __name__ == "__main__":
    main()
