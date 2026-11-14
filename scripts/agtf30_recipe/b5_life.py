#!/usr/bin/env python3
"""B5: wear model + reference-point-only life simulation (C steady-state runner).

Wear per engine (all values are fractional changes of the AGTF30 map modifiers MWS.<C>.s_Eff / s_Wc):
  theta(t) = u * SHORT                         initial (build + running-in) wear, u ~ U(0,1) per module, x S_INIT
           + r_bg * (t / L_BG) * LAM * LONG     background wear on all parts (JT9D long-term split)
           + g(tau) * DIR_f                     fault growth after onset, tau = flights since onset
  g(tau)   = exp(K_f * a * tau**b) - 1   [% efficiency loss of the fault module], a~U(0.001,0.003), b~U(1.4,1.6)
             (N-CMAPSS Eq. 4 in percent units; K_f = time-scale factor for the shortened life)
  onset    : sum over flights of (ALPHA + hours) > E_s, E_s ~ N(E_MU, E_SD)   (fit to N-CMAPSS onsets)
Death check after each flight: one steady state at SLS, MN 0, ISA+27, constant healthy take-off thrust
(PLA 80 healthy Fnet), no limiters. HI = min normalised margin (T45 to 2414 degR; fan/LPC/HPC stall margin to 8 %).
Outputs $AGTF30_WORK/agtf30_b/b5/life_<tag>.json
"""
import sys, json, pathlib, argparse, time
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from b_ss import ss_point, ss_point_thrust, health_index, T45_MAX  # noqa: E402

OUT = pathlib.Path(f"{WORK}/agtf30_b/b5")
SM_MIN = {"Fan": 8.0, "LPC": 8.0, "HPC": 8.0}
MODS = ("FAN", "LPC", "HPC", "HPT", "LPT")
def vec(d):  # {"FAN": (eff%, wc%)} -> theta dict (fractions)
    t = {}
    for m, (e, w) in d.items():
        if e: t[f"MWS.{m}.s_Eff"] = e / 100.0
        if w: t[f"MWS.{m}.s_Wc"] = w / 100.0
    return t
# Sallee 1978 (NASA CR-135448) Table IX, JT9D-7A short-term module deterioration (points / %), 15-engine average
SHORT = {"FAN": (-0.25, -0.25), "LPC": (-0.5, -0.5), "HPC": (-0.5, -1.25), "HPT": (-0.5, +0.25), "LPT": (-0.5, 0.0)}
# Sallee 1978 Tables XXIII-XXV, bottom-up module deterioration at 3500 cycles, mean of airlines A/B/C
# (HPC flow capacity not reported: Sallee p.122 says it affects only N2) -> 0
LONG = {"FAN": (-0.933, -1.2), "LPC": (-0.967, -1.7), "HPC": (-2.367, 0.0), "HPT": (-1.367, +0.633), "LPT": (-0.567, +0.233)}
# fault directions per 1 % efficiency loss (flow change in %)
DIR = {"HPT": {"HPT": (-1, +0.5)}, "HPC": {"HPC": (-1, -1.5)}, "LPT": {"LPT": (-1, +0.5)},
       "FAN": {"FAN": (-1, -0.7)}, "MIX": {"HPC": (-1, -1.5), "HPT": (-1, +0.5)},
       # v1.1 additions (docs/solo_reports/G_V11_FAULTS.md, owner approval 2026-11-09): N-CMAPSS DS02/03, DS06, DS08
       "HPTLPT": {"HPT": (-1, +0.5), "LPT": (-1, +0.5)},
       "LPCHPC": {"LPC": (-1, -1.5), "HPC": (-1, -1.5)},
       "ALL": {"FAN": (-1, -0.7), "LPC": (-1, -1.5), "HPC": (-1, -1.5), "HPT": (-1, +0.5), "LPT": (-1, +0.5)}}
FAM_SEED = {"HPT": 1, "HPC": 2, "LPT": 3, "FAN": 4, "MIX": 5,   # fixed per family (independent of --fams order)
            "HPTLPT": 6, "LPCHPC": 7, "ALL": 8}                # v1.1
ALPHA, E_MU, E_SD = 1.48, 89.1, 8.0        # onset usage threshold (equivalent hours), N-CMAPSS fit
CLASS_H = {1: (0.8, 1.6), 2: (1.8, 3.2), 3: (3.0, 4.8)}   # per-flight hours U(lo,hi), N-CMAPSS class means 1.2/2.5/3.9 h

def scale(d, k):
    return {m: (e * k, w * k) for m, (e, w) in d.items()}
def add(*ts):
    o = {}
    for t in ts:
        for k, v in t.items(): o[k] = o.get(k, 0.0) + v
    return o

_ref = {}
def ref_point(theta):
    if "h" not in _ref:
        h = ss_point({})
        _ref["h"], _ref["fn"] = h, h["Perf.Fnet"]
    r = ss_point_thrust(theta, _ref["fn"])
    if not r["ok"]:
        return r, None, None, None
    hi, lim, parts = health_index(r, _ref["h"], SM_MIN)
    return r, hi, lim, parts

def engine(fam, idx, P, rng):
    cls = 1 + idx % 3
    u = {m: rng.uniform(0, 1) for m in MODS}
    init = vec({m: (SHORT[m][0] * u[m] * P["s_init"], SHORT[m][1] * u[m] * P["s_init"]) for m in MODS})
    r_bg = rng.uniform(0.8, 1.2)
    a, b = rng.uniform(0.001, 0.003), rng.uniform(1.4, 1.6)
    Es = rng.normal(E_MU, E_SD)
    E, onset, hist = 0.0, None, []
    for t in range(0, P["tmax"] + 1):
        if t > 0:
            E += ALPHA + rng.uniform(*CLASS_H[cls])
            if onset is None and E > Es: onset = t
        tau = (t - onset) if onset is not None else 0
        g = np.expm1(P["K"][fam] * a * tau ** b) if tau > 0 else 0.0
        bg = vec(scale(LONG, r_bg * (t / P["L_bg"]) * P["lam"]))
        th = add(init, bg, vec(scale(DIR[fam], g)))
        r, hi, lim, parts = ref_point(th)
        row = {"t": t, "g": float(g), "ok": bool(r["ok"]), "hi": hi, "lim": lim}
        if r["ok"]:
            row.update(T45=r["S45.Tt"], SMFan=r["SM.SMFan"], SMLPC=r["SM.SMLPC"], SMHPC=r["SM.SMHPC"], N1=r["Shaft.N_Fan"],
                       parts={k: round(v, 4) for k, v in parts.items()})
        hist.append(row)
        if not r["ok"] or hi <= 0.0:
            break
    last = hist[-1]
    return {"fam": fam, "idx": idx, "cls": cls, "u": u, "r_bg": r_bg, "a": a, "b": b, "E_s": Es, "onset": onset,
            "eol": last["t"], "died": bool(last["ok"] and last["hi"] <= 0), "solver_fail": not last["ok"],
            "limit": last["lim"], "g_eol": last["g"], "hi0": hist[0]["hi"],
            "theta_eol": {k: round(v * 100, 3) for k, v in th.items()}, "hist": hist}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sim", choices=["probe", "sim"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--fams", default="HPT,HPC,LPT,FAN,MIX")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--K", default="HPT=1,HPC=1,LPT=1,FAN=1,MIX=1")
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--s_init", type=float, default=1.0)
    ap.add_argument("--L_bg", type=float, default=80.0)
    ap.add_argument("--tmax", type=int, default=160)
    A = ap.parse_args(argv)
    if A.mode == "probe":
        res = {}
        for name, th in [("healthy", {}), ("short_x1", vec(SHORT)), ("long_x1", vec(LONG)), ("long_x0.25", vec(scale(LONG, .25))),
                         ("long_x0.5", vec(scale(LONG, .5)))] + [(f"dir_{f}_1pct", vec(DIR[f])) for f in DIR] + \
                        [(f"long_{m}", vec({m: LONG[m]})) for m in MODS]:
            r, hi, lim, parts = ref_point(th)
            res[name] = {"ok": r["ok"], "hi": hi, "lim": lim, "T45": r.get("S45.Tt"), "parts": parts}
            print(name, res[name], flush=True)
        json.dump(res, open(OUT / "probe.json", "w"), indent=1)
        return
    P = {"K": {k: float(v) for k, v in (x.split("=") for x in A.K.split(","))}, "lam": A.lam, "s_init": A.s_init,
         "L_bg": A.L_bg, "tmax": A.tmax}
    allr = {"params": P, "engines": []}
    t0 = time.time()
    for fi, fam in enumerate(A.fams.split(",")):
        for i in range(A.n):
            rng = np.random.default_rng(1000 * FAM_SEED[fam] + i)
            e = engine(fam, i, P, rng)
            allr["engines"].append(e)
            print(f"{fam} {i:2d} cls{e['cls']} onset {e['onset']} eol {e['eol']} died {e['died']} fail {e['solver_fail']} "
                  f"lim {e['limit']} hi0 {e['hi0']:.3f} g_eol {e['g_eol']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(allr, open(OUT / f"life_{A.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
