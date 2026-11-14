#!/usr/bin/env python3
"""Drop analysis of a finished generator run (decision #7 drops).

    python gen_drops.py --out $AGTF30_WORK/agtf30_c/main [--workers 6]

For every dropped engine (runs/<FAM>/e####/dropped.json):
  1. where it failed: steady-state start of the flight (IC not converged) or the dynamic run stopped by the model
     (time in the flight, altitude, Mach, PLA at the stop); wear at the failure (HI of the last good flight and the
     HI the failing flight would have had);
  2. the failing source flight re-flown, with the same decision #7 ladder (normal step, then half step), on
     (a) the same engine with its failure wear (reproducibility), (b) the same engine new (theta at t = 0: build wear
     only), (c) a nominal new engine (theta = 0). Fails on (b) and (c) -> flight-related (the flight itself
     cannot be flown); passes there -> the failure needs this engine's wear state (the --ladder pass then tells a
     narrow numerical event from a wear threshold);
  3. how often other engines of the run flew the same source flight successfully, and at which HI;
  4. the life the dropped engine would have had (b5_life.engine, unchanged, fed with its parameters and its real
     flight hours, as in gen_checks): the dropped engines' would-be lives against the kept lives, per subset;
  5. per-flight drop rate by HI band (flights flown vs drops) and drop rate per subset.
Writes <out>/drops_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys

import numpy as np

HERE = pathlib.Path(os.environ.get("AGTF30_GEN_DIR", pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "agtf30_recipe"))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
import gen_recipe as R  # noqa: E402

FAMS = ["HPT", "HPC", "LPT", "FAN", "MIX"]          # --fams HPTLPT,LPCHPC,ALL for v1.1


def _j(p):
    return json.loads(pathlib.Path(p).read_text())


def failure_point(rec: dict, F: dict, preroll: int) -> dict:
    a = rec["attempts"][0]
    if a["ss_converged"] in (0, 0.0, False):
        return {"kind": "flight_start_ss", "t_s": 0, "alt": float(F["alt"][0]), "mn": float(F["mn"][0]),
                "pla": float(F["pla"][0])}
    m = re.search(r"at t=([0-9.]+)", a["stderr"])
    t = float(m.group(1)) if m else float("nan")
    k = int(min(max(t - preroll, 0), len(F["alt"]) - 1)) if m else 0
    return {"kind": "dynamic_stop", "t_s": round(t - preroll, 1), "flight_s": len(F["alt"]), "alt": float(F["alt"][k]),
            "mn": float(F["mn"][k]), "pla": float(F["pla"][k])}


def refly(job):
    """job = (tag, cfg, key, src_cycle, theta, workdir) -> decision #7 ladder result."""
    import gen_engine as G
    tag, cfg, key, src_cycle, th, wd = job
    wd = pathlib.Path(wd); wd.mkdir(parents=True, exist_ok=True)
    lib = G.Library(cfg["lib"])
    F = lib.flight(key, src_cycle)
    n = G.write_flight_csv(wd / "flight.csv", F)
    r = G.run_dynamic(cfg, th, wd / "flight.csv", wd / "run_tmp", n)
    ref = G.ref_check(th) if th else {"ok": True, "HI": None}
    shutil.rmtree(wd, ignore_errors=True)
    return tag, {"ok": r["ok"], "step_s": r.get("step_s"),
                 "attempts": [{k: x[k] for k in ("attempt", "ok", "rc", "ss_converged")} for x in r["attempts"]],
                 "ref_HI": ref.get("HI")}


def would_be_life(job):
    """b5_life.engine (unchanged) with this engine's parameters and real flight hours, as gen_checks."""
    import b5_life
    from gen_ref import ss_point_thrust_exact
    b5_life.ss_point_thrust = ss_point_thrust_exact     # the same exact-thrust reference as the generator (gen_ref)
    b5_engine = b5_life.engine
    from gen_checks import FedRng
    d, idx_path = job
    E = _j(pathlib.Path(d) / "engine.json")
    index = R.load_index(idx_path)
    seq = R.flight_sequence(E["family"], E["engine_index"], E["cls"], index)
    hours = list(R.flight_hours(seq, index))
    P = {"K": {E["family"]: E["K"]}, "lam": R.LAM, "s_init": R.S_INIT, "L_bg": R.L_BG, "tmax": 400}
    b5 = b5_engine(E["family"], 0, P, FedRng(E, hours))
    return str(d), {"b5_eol": int(b5["eol"]), "died": b5["died"], "solver_fail": b5["solver_fail"],
                    "hi": [round(x["hi"], 5) for x in b5["hist"]]}


def wear_ladder(out: pathlib.Path, cfg: dict, workers: int):
    """Second pass: the failing flight on the same engine at neighbouring and larger wear (theta after t flights,
    t = c-1, c+1, the midpoint to its would-be end of life, and that end of life). Fails only at t = c -> a narrow
    numerical event of this flight; fails from some wear on -> a wear threshold for this flight."""
    res = _j(out / "drops_analysis.json")
    jobs = []
    for i, x in enumerate(res["drops"]):
        E = _j(pathlib.Path(x["dir"]) / "engine.json")
        c, L = x["cycle"], max(x["would_be_life"], x["cycle"] + 2)
        for t in sorted({max(c - 1, 0), c + 1, (c + L) // 2, L}):
            th, _, _ = R.theta(E, t, E["onset"])
            jobs.append(((i, t), cfg, x["source"][0], x["source"][1], th,
                         str(out / "drops_work" / f"d{i:02d}_t{t:03d}")))
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for (i, t), r in pool.imap_unordered(refly, jobs):
            res["drops"][i].setdefault("ladder", {})[str(t)] = {"ok": r["ok"], "step_s": r["step_s"], "ref_HI": r["ref_HI"]}
            print(f"ladder {res['drops'][i]['fam']} e{res['drops'][i]['engine']} c{res['drops'][i]['cycle']} t{t}: "
                  f"ok={r['ok']}", flush=True)
    shutil.rmtree(out / "drops_work", ignore_errors=True)
    for x in res["drops"]:
        lad = {int(t): v["ok"] for t, v in x["ladder"].items()}
        near = [t for t, ok in lad.items() if abs(t - x["cycle"]) <= 1 and not ok]
        far = [t for t, ok in lad.items() if abs(t - x["cycle"]) > 1 and not ok]
        x["ladder_verdict"] = ("also fails at a distant wear (mid-life or end of life)" if far else
                               "local: also fails at a neighbouring flight's wear, passes further away" if near else
                               "narrow: fails only at its own wear")
    (out / "drops_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    for x in res["drops"]:
        print(x["fam"], x["engine"], "c", x["cycle"], {int(t): v["ok"] for t, v in sorted(x["ladder"].items(),
              key=lambda kv: int(kv[0]))}, x["ladder_verdict"])


def main(argv=None):
    import gen_engine as G
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fams", default=",".join(FAMS), help="subsets of the run (v1.1: HPTLPT,LPCHPC,ALL)")
    ap.add_argument("--skip-refly", action="store_true")
    ap.add_argument("--ladder", action="store_true", help="second pass on an existing drops_analysis.json")
    ap.add_argument("--keep-refly", action="store_true",
                    help="recompute everything else; take the re-flight and ladder results from the existing json")
    a = ap.parse_args(argv)
    FAMS[:] = a.fams.split(",")
    out = pathlib.Path(a.out)
    plan = _j(out / "plan.json")
    cfg = plan["cfg"]
    if a.ladder:
        return wear_ladder(out, cfg, a.workers)
    lib = G.Library(cfg["lib"])

    # every flown flight: (source flight) -> successes; per family flights and HI before each flight
    flown, flights_by_fam, hi_before = {}, {f: 0 for f in FAMS}, []
    kept, dropped = [], []
    for fam in FAMS:
        for d in sorted((out / "runs" / fam).glob("e*")):
            E = _j(d / "engine.json")
            hi_prev = E["hi0"]
            for c in range(1, 401):
                p = d / f"c{c:04d}.json"
                if not p.exists():
                    break
                s = _j(p)
                flown.setdefault((s["source_key"], s["source_cycle"]), []).append(
                    {"fam": fam, "engine": E["engine_index"], "cycle": c, "HI_before": hi_prev})
                hi_before.append((fam, hi_prev, 0))
                flights_by_fam[fam] += 1
                hi_prev = s["HI"]
            if (d / "done.json").exists():
                kept.append({"fam": fam, "engine": E["engine_index"], "slot": E["slot"], "cls": E["cls"],
                             "eol": _j(d / "done.json")["eol"], "dir": str(d)})
            elif (d / "dropped.json").exists():
                rec = _j(d / "dropped.json")
                key, sc = rec["source"]
                F = lib.flight(key, sc)
                hi_last = E["hi0"] if rec["cycle"] == 1 else _j(d / f"c{rec['cycle'] - 1:04d}.json")["HI"]
                hi_before.append((fam, hi_last, 1))
                th_new, _, _ = R.theta(E, 0, E["onset"])
                dropped.append({"fam": fam, "engine": E["engine_index"], "slot": E["slot"], "cls": E["cls"],
                                "cycle": rec["cycle"], "source": [key, sc], "reason": rec["reason"],
                                "failure": failure_point(rec, F, G.PREROLL_S), "hi0": E["hi0"],
                                "HI_last_good_flight": hi_last,
                                "wear_used": round((E["hi0"] - hi_last) / E["hi0"], 4),
                                "theta_fail": rec["theta"], "theta_new": th_new, "dir": str(d)})

    res = {"n_engines": len(kept) + len(dropped), "n_kept": len(kept), "n_dropped": len(dropped),
           "flights_flown": sum(flights_by_fam.values()),
           "drop_rate_per_subset": {f: {"dropped": sum(x["fam"] == f for x in dropped),
                                        "engines": sum(x["fam"] == f for x in dropped) + sum(x["fam"] == f for x in kept),
                                        "flights": flights_by_fam[f]} for f in FAMS}}
    for f, v in res["drop_rate_per_subset"].items():
        v["per_engine"] = round(v["dropped"] / v["engines"], 4)
        v["per_1000_flights"] = round(1000 * v["dropped"] / v["flights"], 3)

    for x in dropped:                                     # other engines' successful flights of the same source flight
        ok = [y for y in flown.get(tuple(x["source"]), [])]
        x["same_flight_flown_ok_by_others"] = len(ok)
        x["same_flight_ok_HI_before"] = sorted(round(y["HI_before"], 3) for y in ok)
        same_unit = [k for k in flown if k[0] == x["source"][0]]
        x["same_source_unit_flights_flown_ok"] = sum(len(flown[k]) for k in same_unit)

    ctx = mp.get_context("spawn")
    if a.keep_refly:
        prev = {(y["dir"], y["cycle"]): y for y in _j(out / "drops_analysis.json")["drops"]}
        for x in dropped:
            x.update({k: prev[(x["dir"], x["cycle"])][k] for k in ("refly", "class", "reproduced", "ladder",
                                                                   "ladder_verdict") if k in prev[(x["dir"], x["cycle"])]})
    elif not a.skip_refly:
        jobs = []
        work = out / "drops_work"
        for i, x in enumerate(dropped):
            for tag, th in (("same_wear", x["theta_fail"]), ("new_engine", x["theta_new"]), ("nominal", {})):
                jobs.append(((i, tag), cfg, x["source"][0], x["source"][1], th, str(work / f"d{i:02d}_{tag}")))
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            for (i, tag), r in pool.imap_unordered(refly, jobs):
                dropped[i].setdefault("refly", {})[tag] = r
                print(f"refly {dropped[i]['fam']} e{dropped[i]['engine']} c{dropped[i]['cycle']} {tag}: ok={r['ok']}",
                      flush=True)
        shutil.rmtree(work, ignore_errors=True)
        for x in dropped:
            rf = x["refly"]
            x["class"] = ("flight-related" if not rf["new_engine"]["ok"] and not rf["nominal"]["ok"] else
                          "needs this wear state (new engine flies it)" if rf["new_engine"]["ok"] and rf["nominal"]["ok"]
                          else "mixed")
            x["reproduced"] = not rf["same_wear"]["ok"]

    idx_path = pathlib.Path(cfg["lib"])
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        wb = dict(pool.imap_unordered(would_be_life, [(x["dir"], idx_path) for x in dropped]))
    for x in dropped:
        x["would_be_life"] = wb[x["dir"]]["b5_eol"]
        x["would_be_died"] = wb[x["dir"]]["died"]
        x["hi_at_failure_b5"] = wb[x["dir"]]["hi"][x["cycle"]] if x["cycle"] < len(wb[x["dir"]]["hi"]) else None
        x["life_fraction_at_failure"] = round(x["cycle"] / x["would_be_life"], 3)

    # bias: would-be lives of the dropped engines vs the kept lives (same class mix), and the life mean with/without
    bias = {}
    from scipy.stats import mannwhitneyu
    for f in list(FAMS) + ["all"]:
        k = np.array([y["eol"] for y in kept if f in ("all", y["fam"])], float)
        dl = np.array([y["would_be_life"] for y in dropped if f in ("all", y["fam"])], float)
        b = {"kept_n": len(k), "kept_mean": round(k.mean(), 2), "kept_sd": round(k.std(ddof=1), 2),
             "dropped_n": len(dl), "dropped_would_be": dl.astype(int).tolist()}
        if len(dl):
            b["dropped_mean"] = round(dl.mean(), 2)
            b["mean_if_dropped_had_been_kept"] = round(np.r_[k, dl].mean(), 2)
            b["shift_of_mean"] = round(k.mean() - np.r_[k, dl].mean(), 2)
            b["dropped_life_percentiles_in_kept"] = [round(float((k < v).mean() + 0.5 * (k == v).mean()), 3) for v in dl]
            if len(dl) >= 3:
                b["mannwhitney_p"] = round(float(mannwhitneyu(dl, k).pvalue), 3)
        bias[f] = b
    res["bias"] = bias
    # an engine that flies L flights meets a bad flight with probability 1-(1-p)^L: longer lives are dropped more often
    p = len(dropped) / res["flights_flown"]
    k = np.array([y["eol"] for y in kept], float)
    w = 1.0 / (1 - p) ** k                       # undo the survival weighting
    res["length_bias_model"] = {"p_drop_per_flight": p, "kept_mean": round(k.mean(), 2),
                                "reweighted_mean": round(float((w * k).sum() / w.sum()), 2)}
    hb = np.array([(h, dflag) for _, h, dflag in hi_before], float)
    bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    res["drop_rate_by_HI_band"] = [{"HI": list(bd), "flights": int(((hb[:, 0] >= bd[0]) & (hb[:, 0] < bd[1])).sum()),
                                    "drops": int(((hb[:, 0] >= bd[0]) & (hb[:, 0] < bd[1]) & (hb[:, 1] == 1)).sum())}
                                   for bd in bands]
    for r in res["drop_rate_by_HI_band"]:
        r["per_1000"] = round(1000 * r["drops"] / max(r["flights"], 1), 3)
    from scipy.stats import poisson_means_test
    b = res["drop_rate_by_HI_band"]
    lo = {"drops": sum(r["drops"] for r in b[:3]), "flights": sum(r["flights"] for r in b[:3])}
    res["drop_rate_HI_above_vs_below_0p75"] = {"above": b[3], "below": lo, "poisson_means_p": round(float(
        poisson_means_test(b[3]["drops"], b[3]["flights"], lo["drops"], lo["flights"]).pvalue), 3)}
    for x in dropped:
        del x["theta_fail"], x["theta_new"]
    res["drops"] = dropped
    (out / "drops_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "drops"}, indent=1, default=str))
    for x in dropped:
        print(x["fam"], x["engine"], "c", x["cycle"], x["source"], x["failure"]["kind"], "HI", round(x["HI_last_good_flight"], 3),
              x.get("class"), "others_ok", x["same_flight_flown_ok_by_others"], "would_be", x["would_be_life"])


if __name__ == "__main__":
    main()
