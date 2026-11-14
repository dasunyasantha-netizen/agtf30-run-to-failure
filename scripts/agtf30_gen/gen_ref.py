#!/usr/bin/env python3
"""Reference point at exactly the recipe thrust (31,766 lbf), and repair of run folders written before this fix.

b_ss.ss_point_thrust (phase B, unchanged) stalls 34 lbf (+0.11 %) above the target when the exact fan-speed request
does not converge: its retry ladder then converges only at the request nudged by +0.05 %, every secant step lands
on the same nudged point, and the secant stops with the thrust outside its own 1 lbf tolerance. The HI of that
flight is then slightly low (main run: 13 of 24,251 flights, HI 0.003-0.06 low; two of them were end-of-life
flights). ss_point_thrust_exact returns b_ss's result unchanged whenever it meets the tolerance (all other flights
stay bitwise the same); otherwise it solves two fan-speed requests that bracket the target (warm-started from the
stalled solution) and interpolates every channel linearly in thrust to the target (bracket width ~0.2 % thrust).

    python gen_ref.py --out $AGTF30_WORK/agtf30_c/main            # dry: list affected flights and the corrected HI
    python gen_ref.py --out $AGTF30_WORK/agtf30_c/main --apply    # rewrite their c####.json (old ref kept as ref_unfixed);
                                                            # an end-of-life flight whose HI is now > 0 loses its
                                                            # done.json (kept as done_unfixed.json): re-running the
                                                            # same gen_run command then flies that engine on.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "agtf30_recipe"))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))

TOL_LBF = 1.0                                     # = b_ss.ss_point_thrust's tol
STEPS = [-0.0010, -0.0008, -0.0012, -0.0006, -0.0015, -0.0020, 0.0010, 0.0015, 0.0020, -0.0030]


def ss_point_thrust_exact(theta: dict, fn_target: float, **kw) -> dict:
    from b_ss import ss_point_retry, ss_point_thrust
    r = ss_point_thrust(theta, fn_target, **kw)
    if not r["ok"] or abs(r["Perf.Fnet"] - fn_target) <= TOL_LBF:
        return r
    ic = [r[f"independents[{k}]"] for k in range(1, 13)]
    above = r["Perf.Fnet"] > fn_target
    pts = [r]
    for s in STEPS:                               # a converged point on the other side of the target
        q = ss_point_retry(theta, nc1=r["ssNc1"] * (1 + s), ssic=ic)
        if q["ok"] and (q["Perf.Fnet"] > fn_target) != above:
            pts.append(q)
            break
    if len(pts) < 2:
        r["thrust_exact"] = {"ok": False, "note": "no bracketing point; b_ss result kept"}
        return r
    a, b = pts
    w = (fn_target - a["Perf.Fnet"]) / (b["Perf.Fnet"] - a["Perf.Fnet"])
    out = dict(a)
    for k, v in a.items():
        if isinstance(v, float) and isinstance(b.get(k), float):
            out[k] = v + w * (b[k] - v)
    out["Perf.Fnet"] = float(fn_target)
    out["thrust_exact"] = {"ok": True, "method": "linear in thrust between two converged fan-speed requests",
                           "fnet": [a["Perf.Fnet"], b["Perf.Fnet"]], "nc1": [a["ssNc1"], b["ssNc1"]], "w": w,
                           "retries": [a.get("retries"), b.get("retries")]}
    return out


def _j(p):
    return json.loads(pathlib.Path(p).read_text())


def repair(out: pathlib.Path, apply: bool) -> list[dict]:
    import gen_engine as G
    fixed = []
    for p in sorted((out / "runs").glob("*/e*/c*.json")):
        s = _j(p)
        if "ref_unfixed" in s or abs(s["ref"]["Fnet"] - G.FN_REF_RECIPE) <= TOL_LBF + 0.5:
            continue                              # Fnet of the new-engine reference is 31,766.0 +- 0.5
        d = p.parent
        c = int(p.stem[1:])
        new = G.ref_check(s["theta"])
        rec = {"file": str(p), "cycle": c, "Fnet_old": s["ref"]["Fnet"], "HI_old": s["HI"], "HI_new": new.get("HI"),
               "Fnet_new": new.get("Fnet"), "exact": new.get("thrust_exact"), "ok": new["ok"]}
        done = _j(d / "done.json") if (d / "done.json").exists() else None
        rec["was_eol"] = bool(done and done["eol"] == c)
        rec["eol_moves"] = bool(rec["was_eol"] and new["ok"] and new["HI"] > 0.0)
        fixed.append(rec)
        if apply and new["ok"]:
            s["ref_unfixed"] = s["ref"]
            s["ref"], s["HI"] = new, new["HI"]
            tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, p)
            if rec["eol_moves"]:
                os.replace(d / "done.json", d / "done_unfixed.json")
    return fixed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    res = repair(out, a.apply)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "exact"}))
    if a.apply:
        (out / "ref_repair.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"REF_REPAIR {'applied' if a.apply else 'dry'}: {len(res)} flights, eol moves {sum(r['eol_moves'] for r in res)}")


if __name__ == "__main__":
    main()
