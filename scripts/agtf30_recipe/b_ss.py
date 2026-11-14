#!/usr/bin/env python3
"""Phase B steady-state point with the standalone AGTF30 C runner (no MATLAB).

    from b_ss import ss_point, health_index
    r = ss_point({"MWS.HPT.s_Eff": -0.02}, alt=0, mn=0, dT=27, pla=80)

Uses the extended steady-state runner built by ``cb_build('pkgs','gen_ss',true)``
(bin dir $AGTF30_WORK/agtf30_cbuild/bin_b): the unchanged out_SS channels plus stall
margins, T4/T45/T48/T50, Ps3, shaft speeds, thrust and map speeds.  The start
guess is the cb_run.py port of setup_InputsSS 'auto' (PLA -> corrected fan speed
schedule, VAFN/VBV schedules).  The engine is solved at the commanded corrected
fan speed with the given health, i.e. the controller's steady state WITHOUT its
limiters: margins are therefore the unconstrained ("EGT-margin style") margins.

Health carriers: value = healthy * (1 + theta), the campaign rule.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from cb_run import list_params, read_out, start_guess, write_params  # noqa: E402

import json  # noqa: E402

BIN = pathlib.Path(os.environ.get("AGTF30_BIN_B", f"{WORK}/agtf30_cbuild/bin_b"))
EXE = ".exe" if os.name == "nt" else ""

# AGTF30 limits used for death (setup_Controller.m; Chapman & Litt 2017)
T45_MAX = 2414.0      # degR, setup_Controller.m T45_MAX
N1_MAX = 2300.0       # rpm fan, setup_Controller.m N1_MAX
N3_MAX = 22500.0      # rpm core, setup_Controller.m N3_MAX
PS3_MAX = 790.0       # psia, setup_Controller.m Ps3_MAX

_cache: dict = {}


def _tables():
    if "T" not in _cache:
        _cache["T"] = json.loads((BIN / "ic_tables.json").read_text())
        _cache["Pss"] = list_params(BIN / ("cb_ss" + EXE))
    return _cache["T"], _cache["Pss"]


def carriers():
    out = {}
    for c in ("FAN", "LPC", "HPC", "HPT", "LPT"):
        for s in ("Nc", "Wc", "PR", "Eff"):
            out[f"MWS.{c}.s_{s}"] = f"CB_{c}_s_{s}"
    return out


def ss_point(theta: dict[str, float], alt=0.0, mn=0.0, dT=27.0, pla=80.0,
             workdir: str | None = None, ssic: list[float] | None = None,
             nc1: float | None = None) -> dict:
    """Solve one steady-state point. Returns the converged channels plus 'ok'."""
    T, Pss = _tables()
    C = carriers()
    vals = {}
    for path, th in theta.items():
        cb = C[path]
        vals[cb] = Pss[cb][0] * (1.0 + float(th))
    G = start_guess(T, alt, mn, dT, pla)
    if nc1 is not None:          # explicit corrected fan speed (e.g. constant-thrust reference)
        G["ssNc1"] = nc1
    ss_vals = dict(vals, CB_ssNc1=G["ssNc1"], CB_AltIC=G["AltIC"], CB_MNIC=G["MNIC"],
                   CB_dTambIC=G["dTambIC"], CB_VAFNIC=G["VAFNIC"], CB_VBVIC=G["VBVIC"],
                   CB_SSIC=ssic if ssic is not None else G["SSIC"])
    tmp = tempfile.mkdtemp(dir=workdir) if workdir else tempfile.mkdtemp()
    pf, of = pathlib.Path(tmp) / "p.txt", pathlib.Path(tmp) / "o.bin"
    write_params(pf, ss_vals, Pss)
    p = subprocess.run([str(BIN / ("cb_ss" + EXE)), "-p", str(pf), "-o", str(of), "-r", "full"],
                       capture_output=True, text=True)
    R = {"rc": p.returncode, "ssNc1": G["ssNc1"], "VAFN": G["VAFNIC"], "VBV": G["VBVIC"],
         "envelope_ok": G["envelope_ok"], "ok": False}
    if p.returncode in (0, 4) and of.exists():
        names, Y = read_out(of)
        last = dict(zip(names, map(float, Y[-1])))
        R.update(last)
        R["iters"] = int(Y.shape[0])
        R["Wf"] = last["independents[10]"]
        R["ok"] = last["converged"] == 1.0
    for f in (pf, of):
        try:
            f.unlink()
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return R


def ss_point_retry(theta, alt=0.0, mn=0.0, dT=27.0, pla=80.0, nc1=None, ssic=None) -> dict:
    """ss_point with a retry ladder for isolated Newton-Raphson non-convergence (iteration
    cap): given start, cold start (package guess), then corrected fan speed nudged by +-0.05 %."""
    tries = [(nc1, ssic), (nc1, None)]
    if nc1 is not None:
        tries += [(nc1 * 1.0005, None), (nc1 * 0.9995, None)]
    k = 0
    while k < len(tries):
        n, ic = tries[k]
        r = ss_point(theta, alt, mn, dT, pla, nc1=n, ssic=ic)
        if r["ok"]:
            r["retries"] = k
            return r
        if nc1 is None and k == 1:   # default request: nudge the package's own corrected fan speed
            tries += [(r["ssNc1"] * 1.0005, None), (r["ssNc1"] * 0.9995, None)]
        k += 1
    r["retries"] = len(tries)
    return r


def ss_point_thrust(theta: dict[str, float], fn_target: float, alt=0.0, mn=0.0, dT=27.0,
                    pla=80.0, tol=1.0, maxit=8) -> dict:
    """Steady state at a fixed net thrust (secant on the corrected fan speed request)."""
    r0 = ss_point_retry(theta, alt, mn, dT, pla)
    if not r0["ok"]:
        return r0
    x0, f0 = r0["ssNc1"], r0["Perf.Fnet"] - fn_target
    x1 = x0 * (1 + 0.5 * (-f0) / fn_target)
    r = r0
    for _ in range(maxit):
        if abs(f0) < tol:
            break
        ic = [r[f"independents[{k}]"] for k in range(1, 13)]   # warm start from the last solution
        r = ss_point_retry(theta, alt, mn, dT, pla, nc1=x1, ssic=ic)
        if not r["ok"]:
            return r
        f1 = r["Perf.Fnet"] - fn_target
        if abs(f1) < tol or f1 == f0:
            break
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
    r["fn_target"] = fn_target
    return r


def margins(r: dict, sm_min: dict[str, float]) -> dict[str, float]:
    """Margins above the death limits, in the limit's own units.
    sm_min: minimum stall margin per compressor in % (e.g. {'Fan': 8, 'LPC': 8, 'HPC': 8})."""
    return {"Fan": r["SM.SMFan"] - sm_min["Fan"],
            "LPC": r["SM.SMLPC"] - sm_min["LPC"],
            "HPC": r["SM.SMHPC"] - sm_min["HPC"],
            "T45": T45_MAX - r["S45.Tt"]}


def health_index(r: dict, r0: dict, sm_min: dict[str, float]) -> tuple[float, str, dict]:
    """N-CMAPSS-style HI (Arias Chao et al. 2021, Eq. in Sec. 3): each margin is
    normalised by its healthy value at the same point, HI = min over limits, clipped to [0, 1]."""
    m, m0 = margins(r, sm_min), margins(r0, sm_min)
    h = {k: m[k] / m0[k] for k in m}
    k = min(h, key=h.get)
    return float(np.clip(h[k], 0.0, 1.0)), k, h
