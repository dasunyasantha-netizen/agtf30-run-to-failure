#!/usr/bin/env python3
"""C3: per-flight loop of one engine, with a checkpoint after every flight.

    python gen_engine.py --out $AGTF30_WORK/agtf30_c/dry --fam HPT --slot 0 --n-per-fam 10

Per flight c = 1, 2, ...:
  1. maintenance hook: events scheduled before this flight are applied (none in v1);
  2. wear theta(c) (frozen within the flight, gen_recipe.theta);
  3. dynamic C runner (cb_run: steady-state IC at the first sample, then cb_dyn closed loop),
     1 Hz output; input = 30 s pre-roll at the first sample (not saved) + the flight + 1 s pad;
  4. reference-point steady state at constant new-engine take-off thrust (31,766 lbf; gen_ref keeps it exact),
     SLS, MN 0, ISA+27 degR, no limiters -> HI and the 4 margins (b_ss.health_index);
  5. checkpoint c<cycle>.npz (1 Hz channels) + c<cycle>.json (summary) written atomically;
  6. stop after the flight with HI <= 0 (RUL = 0 there): done.json.
Solver failure (agreed decision #7): the flight is retried with the half-step runner
(bin_fine, MWS.In.Ts 0.0075 s instead of 0.015 s); if it still fails, or the reference point
fails after b_ss's own retry ladder, the engine is dropped (dropped.json, logged) and the
job runner replaces it. A stopped job resumes at the first flight without a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agtf30_recipe"))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
import gen_recipe as R  # noqa: E402

PREROLL_S = 30
PAD_S = 1
SM_MIN = {"Fan": 8.0, "LPC": 8.0, "HPC": 8.0}
FN_REF_RECIPE = 31766.0          # lbf, recipe section 3 (checked against the computed value)
CAP_ITER = 64                    # dynamic solver iteration cap (B5 'capped seconds')
# 1 Hz channels kept from cb_dyn (everything the writer needs)
DYN_KEEP = ["eng.S2.Tt", "eng.S24.Tt", "eng.S36.Tt", "eng.S48.Tt", "eng.S5.Tt", "eng.S4.Tt", "eng.S45.Tt",
            "eng.S17.Pt", "eng.S2.Pt", "eng.S21.Pt", "eng.S24.Pt", "eng.S36.Ps", "eng.S36.Pt", "eng.S4.Pt",
            "eng.S45.Pt", "eng.S5.Pt", "eng.Shaft.N_Fan", "eng.Shaft.N_LPC", "eng.Shaft.N_HPC", "eng.S21.W",
            "eng.S24.W", "eng.S25.W", "eng.S48.W", "eng.S5.W", "eng.SM.SMFan", "eng.SM.SMLPC", "eng.SM.SMHPC",
            "eng.Perf.Fnet", "eng.Perf.TSFC", "eng.Data.HPC_Data.Wbleeds", "act.Wfact", "act.VBVact",
            "act.VAFNact", "cntrl.RegNum", "cntrl.N1cReq", "cntrl.T45Lim", "in.PLA", "in.Alt", "in.MN",
            "loop.Solver.Iterations"]


def default_cfg(out, lib=f"{WORK}/agtf30_c/library", n_per_fam=10):
    return {"out": str(out), "lib": str(lib), "n_per_fam": int(n_per_fam),
            "bin": os.environ.get("AGTF30_BIN_B", f"{WORK}/agtf30_cbuild/bin_b"),
            "bin_fine": os.environ.get("AGTF30_BIN_FINE", f"{WORK}/agtf30_cbuild/bin_fine"),
            "tmp": os.environ.get("AGTF30_TMP", "")}


class Library:
    """Lazy access to the N-CMAPSS source flights (one unit kept in memory)."""

    def __init__(self, lib):
        self.lib = pathlib.Path(lib)
        self.index = R.load_index(lib)
        self._key, self._z = None, None

    def flight(self, key: str, cycle: int) -> dict:
        if key != self._key:
            with np.load(self.lib / f"{key}.npz") as z:
                self._z = {k: z[k] for k in z.files}
            self._key = key
        z = self._z
        a = int(z["starts"][cycle - 1])
        b = int(z["starts"][cycle]) if cycle < len(z["starts"]) else len(z["cycle"])
        assert z["cycle"][a] == cycle and z["cycle"][b - 1] == cycle
        return {k: z[k][a:b] for k in ("alt", "mn", "tra", "t2", "nf", "dT", "pla", "flag", "hs")}


_ref: dict = {}


def reference_new():
    """New engine (theta = 0) at the reference point: the HI normaliser and the thrust target."""
    if "r0" not in _ref:
        from b_ss import ss_point
        r0 = ss_point({})
        assert r0["ok"], "new-engine reference point failed"
        assert abs(r0["Perf.Fnet"] - FN_REF_RECIPE) < 1.0, f"reference thrust {r0['Perf.Fnet']} != recipe 31766"
        _ref["r0"] = r0
    return _ref["r0"]


def ref_check(th: dict) -> dict:
    from b_ss import health_index, margins
    from gen_ref import ss_point_thrust_exact
    r0 = reference_new()
    r = ss_point_thrust_exact(th, r0["Perf.Fnet"])
    out = {"ok": bool(r["ok"]), "retries": r.get("retries")}
    if "thrust_exact" in r:                  # b_ss stalled outside its thrust tolerance (gen_ref)
        out["thrust_exact"] = r["thrust_exact"]
    if r["ok"]:
        hi, lim, parts = health_index(r, r0, SM_MIN)
        m = margins(r, SM_MIN)
        out.update(HI=hi, limit=lim, hi_parts=parts, margin_T45=m["T45"], margin_SMFan=m["Fan"],
                   margin_SMLPC=m["LPC"], margin_SMHPC=m["HPC"], T45=r["S45.Tt"], N1=r["Shaft.N_Fan"],
                   N3=r["Shaft.N_HPC"], Fnet=r["Perf.Fnet"], Ps3=r.get("S36.Ps", r.get("Ps3", float("nan"))))
    return out


def write_flight_csv(path: pathlib.Path, F: dict) -> int:
    n = len(F["alt"])
    alt = np.r_[np.full(PREROLL_S, F["alt"][0]), F["alt"], np.full(PAD_S, F["alt"][-1])]
    mn = np.r_[np.full(PREROLL_S, F["mn"][0]), F["mn"], np.full(PAD_S, F["mn"][-1])]
    dT = np.clip(np.r_[np.full(PREROLL_S, F["dT"][0]), F["dT"], np.full(PAD_S, F["dT"][-1])], -30, 30)
    pla = np.r_[np.full(PREROLL_S, F["pla"][0]), F["pla"], np.full(PAD_S, F["pla"][-1])]
    t = np.arange(len(alt), dtype=float)
    np.savetxt(path, np.c_[t, alt, mn, dT, pla], delimiter=",", header="time_s,altitude_ft,mach,dT_degF,pla_deg",
               comments="", fmt="%.17g")
    return n


ENV_RETRIES, ENV_WAIT_S = 4, 60


class EnvFault(RuntimeError):
    """The PC, not the engine model, failed (memory, crash, file). Never a drop: the slot stays resumable."""


def solver_failure(rec: dict) -> bool:
    """Decision #7 applies only to the model's own non-convergence stop (rc 4) or a steady-state start that did
    not converge; any other failure is an environment fault."""
    return (rec["rc"] == 4 and "stop requested by the model" in rec["stderr"]) or rec["ss_converged"] in (0, 0.0, False)


def run_dynamic(cfg, th: dict, csv: pathlib.Path, rundir: pathlib.Path, n: int) -> dict:
    """Closed-loop flight; decision #7 ladder: normal step, then half step. Returns the 1 Hz rows."""
    from cb_run import read_out, run as cb_run
    tries = [("normal", cfg["bin"], 0.015), ("half_step", cfg["bin_fine"], 0.0075)]
    log = []
    for name, b, step in tries:
        for env_try in range(ENV_RETRIES + 1):
            if rundir.exists():
                shutil.rmtree(rundir)
            t0 = time.perf_counter()
            Rr = cb_run(SimpleNamespace(bin=b, flight=str(csv), out=str(rundir), rate="1hz",
                                        theta=[f"{k}={v!r}" for k, v in th.items()]))
            rec = {"attempt": name, "step_s": step, "ok": bool(Rr.get("ok")), "rc": Rr.get("dyn", {}).get("rc"),
                   "ss_converged": Rr.get("ss", {}).get("converged"), "error": Rr.get("error"),
                   "stderr": (Rr.get("dyn", {}).get("stderr") or Rr.get("ss", {}).get("stderr") or "")[-300:],
                   "wall_s": time.perf_counter() - t0}
            log.append(rec)
            if rec["ok"] or solver_failure(rec):
                break
            rec["env_fault"] = True             # not the solver: crash, out of memory, missing file -> same step again
            if env_try == ENV_RETRIES:
                raise EnvFault(f"flight run failed {ENV_RETRIES + 1} times without a solver stop: {rec}")
            time.sleep(ENV_WAIT_S * (env_try + 1))
        if Rr.get("ok"):
            names, Y = read_out(rundir / "dyn.bin")
            if len(Y) < PREROLL_S + n:
                rec["ok"] = False; rec["error"] = f"short output {len(Y)} rows"
                continue
            c = {k: i for i, k in enumerate(names)}
            Yk = Y[PREROLL_S:PREROLL_S + n][:, [c[k] for k in DYN_KEEP]]
            shutil.rmtree(rundir, ignore_errors=True)
            return {"ok": True, "Y": np.ascontiguousarray(Yk), "attempts": log, "step_s": step}
    return {"ok": False, "attempts": log}


def apply_events(th: dict, events: list, cycle: int) -> dict:
    """Maintenance hook (v1: no events). Events scheduled before `cycle` would change theta here."""
    for ev in events:
        if ev["before_cycle"] == cycle:
            raise NotImplementedError("maintenance events are not used in v1")
    return th


def _atomic_json(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def engine_dir(cfg, fam, engine_index) -> pathlib.Path:
    return pathlib.Path(cfg["out"]) / "runs" / fam / f"e{engine_index:04d}"


def run_engine(cfg, fam: str, slot: int, engine_index: int, lib: Library | None = None, max_flights: int = 400,
               stop_after: int | None = None) -> dict:
    """Fly one engine to its end (or `stop_after` flights, for tests). Resumable."""
    lib = lib or Library(cfg["lib"])
    classes = R.class_plan(fam, cfg["n_per_fam"])
    splits = R.split_plan(fam, classes)
    cls = classes[slot]
    d = engine_dir(cfg, fam, engine_index); d.mkdir(parents=True, exist_ok=True)
    if (d / "done.json").exists():
        return json.loads((d / "done.json").read_text())
    if (d / "dropped.json").exists():
        return json.loads((d / "dropped.json").read_text())
    E = R.sample_engine(fam, engine_index, cls)
    seq = R.flight_sequence(fam, engine_index, cls, lib.index)
    hours = R.flight_hours(seq, lib.index)
    onset = R.onset_flight(E, hours)
    E.update(slot=slot, split=splits[slot], onset=onset, recipe_id=R.RECIPE_ID)
    th0, _, _ = R.theta(E, 0, onset)
    if not (d / "engine.json").exists():
        r0 = ref_check(th0)
        _atomic_json(d / "engine.json", dict(E, hi0=r0.get("HI"), ref0=r0, seq_head=seq[:200]))
    events: list = []            # v1: no maintenance
    t_eng = time.perf_counter()
    c = 1
    while (d / f"c{c:04d}.json").exists():
        c += 1
    while c <= max_flights:
        if stop_after is not None and c > stop_after:
            return {"status": "stopped", "family": fam, "slot": slot, "engine_index": engine_index, "cycles": c - 1}
        key, src_cycle = seq[c - 1]
        F = lib.flight(key, src_cycle)
        th, g, tau = R.theta(E, c, onset)
        th = apply_events(th, events, c)
        csv = d / "flight.csv"
        n = write_flight_csv(csv, F)
        t0 = time.perf_counter()
        dyn = run_dynamic(cfg, th, csv, d / "run_tmp", n)
        t_dyn = time.perf_counter() - t0
        if not dyn["ok"]:
            rec = {"status": "dropped", "reason": "dynamic flight failed at normal and half step", "family": fam,
                   "slot": slot, "engine_index": engine_index, "cycle": c, "attempts": dyn["attempts"],
                   "theta": th, "source": [key, src_cycle]}
            _atomic_json(d / "dropped.json", rec)
            return rec
        t1 = time.perf_counter()
        ref = ref_check(th)
        t_ref = time.perf_counter() - t1
        if not ref["ok"]:
            rec = {"status": "dropped", "reason": "reference-point steady state failed after retries",
                   "family": fam, "slot": slot, "engine_index": engine_index, "cycle": c, "theta": th,
                   "source": [key, src_cycle], "ref": ref}
            _atomic_json(d / "dropped.json", rec)
            return rec
        Y = dyn["Y"]
        kreg = DYN_KEEP.index("cntrl.RegNum"); kit = DYN_KEEP.index("loop.Solver.Iterations")
        np.savez(d / f"c{c:04d}.tmp.npz", Y=Y, flag=F["flag"])
        os.replace(d / f"c{c:04d}.tmp.npz", d / f"c{c:04d}.npz")
        summ = {"cycle": c, "source_key": key, "source_cycle": int(src_cycle), "seconds": int(n), "hours": n / 3600,
                "theta": th, "g": g, "tau": tau, "onset": onset, "hs": int(c <= (onset or 10 ** 9)),
                "ref": ref, "HI": ref["HI"], "limiter_s": int((Y[:, kreg] != 0).sum()),
                "regnum_values": sorted({int(x) for x in np.unique(Y[:, kreg])}),
                "clipped_s": int((F["flag"] != 0).sum()), "cap_s": int((Y[:, kit] >= CAP_ITER).sum()),
                "dyn_step_s": dyn["step_s"], "dyn_attempts": dyn["attempts"], "wall_dyn_s": t_dyn, "wall_ref_s": t_ref,
                "nonfinite": int((~np.isfinite(Y)).sum())}
        _atomic_json(d / f"c{c:04d}.json", summ)
        if ref["HI"] <= 0.0:
            rec = {"status": "done", "family": fam, "slot": slot, "engine_index": engine_index, "eol": c,
                   "limit": ref["limit"], "onset": onset, "cls": cls, "split": splits[slot],
                   "wall_s_this_session": time.perf_counter() - t_eng}
            _atomic_json(d / "done.json", rec)
            csv.unlink(missing_ok=True)
            return rec
        c += 1
    rec = {"status": "dropped", "reason": f"no end of life within {max_flights} flights", "family": fam,
           "slot": slot, "engine_index": engine_index, "cycle": c}
    _atomic_json(d / "dropped.json", rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=f"{WORK}/agtf30_c/library")
    ap.add_argument("--fam", required=True)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--engine-index", type=int, default=None)
    ap.add_argument("--n-per-fam", type=int, default=10)
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = default_cfg(a.out, a.lib, a.n_per_fam)
    r = run_engine(cfg, a.fam, a.slot, a.slot if a.engine_index is None else a.engine_index, stop_after=a.stop_after)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
