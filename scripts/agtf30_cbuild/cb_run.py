#!/usr/bin/env python3
"""Run one AGTF30 flight with the standalone C runners (no MATLAB).

    cb_run.py --bin <dir with cb_ss/cb_dyn + ic_tables.json> --flight flight.csv \
              --out <run dir> [--theta MWS.HPT.s_Eff=-0.02 ...] [--rate 1hz|full]

Steps (same as the AGTF30 package with ICPoint 'auto', setup_InputsSS):
  1. start guess: envelope check (AGTF30.inEnvelope), PLA -> corrected fan speed
     schedule, VAFN/VBV schedules and the SS_Mtrx table lookups at the first
     flight sample (port of the 'otherwise' branch of setup_InputsSS.m);
  2. cb_ss: steady-state solve with the run's health parameters; converged flag,
     solver independents and station values at the last step -> dynamic ICs;
  3. cb_dyn: the whole flight closed loop; channels to out_dyn.bin.
Health parameters are healthy * (1 + theta) (the campaign's carrier rule); the
healthy values are the build-time ones reported by ``cb_dyn -l``.
Writes in --out: params_ss.txt, ss.bin, params_dyn.txt, dyn.bin, run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

EXE = ".exe" if os.name == "nt" else ""


# ---------------------------------------------------------------- MATLAB-like interpolation
def _seg(grid: np.ndarray, q: float) -> tuple[int, float]:
    """Interval index i and fraction f with grid[i] <= q <= grid[i+1] (NaN outside)."""
    if not (grid[0] <= q <= grid[-1]):
        return -1, float("nan")
    i = int(np.searchsorted(grid, q, side="right") - 1)
    i = min(max(i, 0), len(grid) - 2)
    return i, (q - grid[i]) / (grid[i + 1] - grid[i])


def interp_grid(grids: list[np.ndarray], V: np.ndarray, q: list[float]) -> float:
    """Multilinear interpolation on an ndgrid-ordered array V (V.shape == lens of grids)."""
    segs = [_seg(g, x) for g, x in zip(grids, q)]
    if any(i < 0 for i, _ in segs):
        return float("nan")
    out = 0.0
    n = len(grids)
    for corner in range(1 << n):
        w = 1.0
        idx = []
        for d in range(n):
            i, f = segs[d]
            b = (corner >> d) & 1
            w *= f if b else (1.0 - f)
            idx.append(i + b)
        if w != 0.0:
            out += w * V[tuple(idx)]
    return float(out)


def interp1(x, y, q):
    return float(np.interp(q, x, y)) if x[0] <= q <= x[-1] else float("nan")


# ---------------------------------------------------------------- IC start guess
def start_guess(T: dict, alt0: float, mn0: float, dT0: float, pla0: float) -> dict:
    # AGTF30.inEnvelope
    cond = 1
    Alt, MN, dT = alt0, mn0, dT0
    msgs = []
    if dT > 30 or dT < -30:
        cond = 0; dT = max(min(dT, 30), -30); msgs.append("dT out of envelope")
    if MN > 0.8 or MN < 0:
        cond = 0; MN = max(min(MN, 0.8), 0); msgs.append("MN out of envelope")
    if cond == 1:
        hi = interp1([0, 0.2, 0.5, 0.6, 0.7, 0.8], [1e4, 1e4, 2.5e4, 3.5e4, 4e4, 4e4], MN)
        lo = interp1([0, 0.5, 0.6, 0.7, 0.8], [0, 0, 1e3, 2e3, 25e3], MN)
        if Alt > hi or Alt < lo:
            cond = 0; Alt = max(min(Alt, hi), lo); msgs.append("Alt out of envelope")
    G = {"AltIC": Alt, "MNIC": MN, "dTambIC": dT, "envelope_ok": cond == 1, "messages": msgs}
    AltV, NcV, MNV = (np.array(T[k], float) for k in ("Alt", "Nc", "MN"))
    clamp = lambda v, lo, hi, nm: (msgs.append(f"{nm} clamped") or lo) if v < lo else \
        ((msgs.append(f"{nm} clamped") or hi) if v > hi else v)
    Alt = clamp(Alt, AltV.min(), AltV.max(), "Alt")
    MN = clamp(MN, MNV.min(), MNV.max(), "MN")
    man = T["manual"]
    if man["WfManEn"] or man["N1ManEn"]:
        raise SystemExit("cb_run: manual Wf/N1 request modes are not supported (package default is PLA)")
    TF = (pla0 - 40.0) / (80.0 - 40.0)
    # interp3(PM_AltVec, PM_TF, PM_MNVec, PM_NcArray, Alt, TF, MN): meshgrid order, V(TF, Alt, MN)
    PM = np.array(T["PM_NcArray"], float).reshape(T["PM_NcArray_size"], order="F") \
        if np.ndim(T["PM_NcArray"]) == 1 else np.array(T["PM_NcArray"], float)
    Nc = interp_grid([np.array(T["PM_TF"], float).ravel(), np.array(T["PM_AltVec"], float).ravel(),
                      np.array(T["PM_MNVec"], float).ravel()], PM, [TF, Alt, MN])
    Nc = clamp(Nc, NcV.min(), NcV.max(), "Nc")
    # interp2(MNVec, AltVec, SS.Ncmin, MN, Alt): V(Alt, MN)
    Ncmin = np.array(T["Ncmin"], float); Ncmax = np.array(T["Ncmax"], float)
    NcMin = interp_grid([AltV, MNV], Ncmin, [Alt, MN])
    NcMax = interp_grid([AltV, MNV], Ncmax, [Alt, MN])
    Nc = clamp(Nc, NcMin, NcMax, "Nc")
    G["ssNc1"] = Nc
    # interp2(VAFN_MN, VAFN_Nc1, VAFN_sch, MN, Nc): V(Nc1, MN)
    if man["VAFNManEn"] or man["VBVManEn"]:
        raise SystemExit("cb_run: manual VAFN/VBV modes are not supported")
    G["VAFNIC"] = interp_grid([np.array(T["VAFN_Nc1"], float).ravel(), np.array(T["VAFN_MN"], float).ravel()],
                              np.array(T["VAFN_sch"], float), [Nc, MN])
    G["VBVIC"] = interp_grid([np.array(T["VBV_Nc1"], float).ravel(), np.array(T["VBV_MN"], float).ravel()],
                             np.array(T["VBV_sch"], float), [Nc, MN])
    dT_adj = 1 + dT / 1000.0
    ind = [np.array(v, float).reshape(T["Ind_size"], order="F") if np.ndim(v) == 1 else np.array(v, float)
           for v in T["Ind"]]
    L = lambda k: interp_grid([AltV, NcV, MNV], ind[k - 1], [Alt, Nc, MN])
    ICcom = [L(1) / dT_adj, L(2) / (1 + (dT_adj - 1) * MN), L(3) / (1 + (dT_adj - 1) * MN),
             L(4), L(5), L(6), L(7)]
    ICss = [L(8) * dT_adj, L(9) * dT_adj, L(10) * dT_adj, L(11), L(12) + 1]
    G["SSIC"] = ICcom + ICss
    return G


# ---------------------------------------------------------------- runner I/O
def list_params(exe: pathlib.Path) -> dict[str, list[float] | int]:
    out = subprocess.run([str(exe), "-l"], check=True, capture_output=True, text=True).stdout
    P = {}
    for ln in out.splitlines():
        f = ln.split()
        P[f[0]] = [float(x) for x in f[2:]] if len(f) > 2 else int(f[1])
    return P


def write_params(file: pathlib.Path, values: dict[str, list[float]], known: dict) -> None:
    with open(file, "w") as fh:
        for k, v in values.items():
            if k not in known:
                continue          # value not used by this model (e.g. SS-only input)
            v = list(np.atleast_1d(np.asarray(v, float)).ravel())
            fh.write(k + " " + " ".join(repr(float(x)) for x in v) + "\n")


def read_out(file: pathlib.Path) -> tuple[list[str], np.ndarray]:
    with open(file, "rb") as fh:
        if fh.readline() != b"CBOUT1\n":
            raise ValueError(f"{file}: not a CBOUT1 file")
        hdr = fh.readline().decode().split()
        n = int(hdr[0]); names = hdr[1:]
        data = np.frombuffer(fh.read(), dtype="<f8")
    return names, data.reshape(-1, n)


def run(args) -> dict:
    b = pathlib.Path(args.bin); out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ss_exe, dyn_exe = b / ("cb_ss" + EXE), b / ("cb_dyn" + EXE)
    T = json.loads((b / "ic_tables.json").read_text())
    R = {"flight": str(args.flight), "theta": {}, "rate": args.rate, "ok": False}
    t0 = time.perf_counter()
    F = np.loadtxt(args.flight, delimiter=",", skiprows=1, ndmin=2)
    Pdyn = list_params(dyn_exe); Pss = list_params(ss_exe)
    health = {}
    for c in ("FAN", "LPC", "HPC", "HPT", "LPT"):
        for s in ("Nc", "Wc", "PR", "Eff"):
            health[f"MWS.{c}.s_{s}"] = f"CB_{c}_s_{s}"
    vals = {}
    for th in args.theta:
        path, v = th.split("=")
        if path not in health:
            raise SystemExit(f"cb_run: unknown carrier {path}")
        cb = health[path]
        vals[cb] = Pdyn[cb][0] * (1.0 + float(v))
        R["theta"][path] = {"theta": float(v), "healthy": Pdyn[cb][0], "effective": vals[cb]}
    G = start_guess(T, F[0, 1], F[0, 2], F[0, 3], F[0, 4])
    R["start_guess"] = G
    ss_vals = dict(vals, CB_ssNc1=G["ssNc1"], CB_AltIC=G["AltIC"], CB_MNIC=G["MNIC"],
                   CB_dTambIC=G["dTambIC"], CB_VAFNIC=G["VAFNIC"], CB_VBVIC=G["VBVIC"], CB_SSIC=G["SSIC"])
    write_params(out / "params_ss.txt", ss_vals, Pss)
    t1 = time.perf_counter()
    p = subprocess.run([str(ss_exe), "-p", str(out / "params_ss.txt"), "-o", str(out / "ss.bin"), "-r", "full"],
                       capture_output=True, text=True)
    R["ss"] = {"rc": p.returncode, "stderr": p.stderr.strip(), "wall_s": time.perf_counter() - t1}
    # rc 4 = the steady-state model's own Stop block (converged before MWS.In.Max_IterSS), as in Simulink
    if p.returncode not in (0, 4):
        return finish(R, out, t0)
    names, Y = read_out(out / "ss.bin")
    last = dict(zip(names, Y[-1]))
    R["ss"]["converged"] = last["converged"]
    if last["converged"] == 0:
        R["error"] = "steady-state IC generator failed (converged == 0)"
        return finish(R, out, t0)
    ind = [last[f"independents[{k}]"] for k in range(1, 13)]
    N2IC, N3IC, WfIC = ind[7], ind[8], ind[9]
    dyn_vals = dict(vals, CB_ICcom=ind[:7], CB_N2IC=N2IC, CB_N3IC=N3IC, CB_WfIC=WfIC,
                    CB_VAFNIC=G["VAFNIC"], CB_VBVIC=G["VBVIC"],
                    CB_Sensor_ICN1=N2IC / 3.1, CB_Sensor_ICN2=N2IC, CB_Sensor_ICN3=N3IC,
                    CB_Sensor_ICPa=last["Pa"], CB_Sensor_ICP2=last["S2.Pt"], CB_Sensor_ICP21=last["S21.Pt"],
                    CB_Sensor_ICP25=last["S25.Pt"], CB_Sensor_ICPs3=last["S36.Pt"], CB_Sensor_ICP5=last["S5.Pt"],
                    CB_Sensor_ICT2=last["S2.Tt"], CB_Sensor_ICT25=last["S25.Tt"], CB_Sensor_ICT3=last["S36.Tt"],
                    CB_Sensor_ICT45=last["S45.Tt"])
    R["dyn_ic"] = {k: (list(map(float, v)) if isinstance(v, list) else float(v)) for k, v in dyn_vals.items()}
    write_params(out / "params_dyn.txt", dyn_vals, Pdyn)
    t2 = time.perf_counter()
    p = subprocess.run([str(dyn_exe), "-p", str(out / "params_dyn.txt"), "-i", str(args.flight),
                        "-o", str(out / "dyn.bin"), "-r", args.rate], capture_output=True, text=True)
    R["dyn"] = {"rc": p.returncode, "stderr": p.stderr.strip(), "wall_s": time.perf_counter() - t2,
                "bytes": (out / "dyn.bin").stat().st_size if (out / "dyn.bin").exists() else -1}
    try:   # peak resident memory of the largest child (the runners), Linux: kB
        import resource
        R["dyn"]["children_maxrss_kb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except ImportError:
        pass
    # rc 4 here = the dynamic model's non-convergence stop (MWS.In.DynRunNonConv): flight incomplete
    R["dyn"]["complete"] = p.returncode == 0
    R["ok"] = p.returncode == 0
    return finish(R, out, t0)


def finish(R, out, t0):
    R["total_wall_s"] = time.perf_counter() - t0
    (out / "run.json").write_text(json.dumps(R, indent=1))
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta", action="append", default=[])
    ap.add_argument("--rate", default="1hz", choices=["1hz", "full"])
    R = run(ap.parse_args(argv))
    print(json.dumps({k: R.get(k) for k in ("ok", "total_wall_s", "error")}))
    return 0 if R["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
