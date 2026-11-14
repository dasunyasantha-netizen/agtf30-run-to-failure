#!/usr/bin/env python3
"""A4 throughput: N simultaneous AGTF30 flights (cb_run.py) on one node.

    cb_throughput.py --bin bin --flight f.csv --n 96 --out dir [--rate 1hz] [--seed 1]

Each run gets its own degraded health setting (HPT efficiency theta uniform in
[-0.04, 0], HPC efficiency and flow theta uniform in [-0.02, 0]; seeded) so the
runs are realistic but different. Records wall time of the whole batch, and per
run the steady-state and flight runner wall times, peak RSS and output bytes.
Writes <out>/throughput.json.
"""
import argparse
import json
import os
import platform
import pathlib
import random
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", default="1hz")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    procs = []
    t0 = time.time()
    for i in range(a.n):
        th = [f"MWS.HPT.s_Eff={rng.uniform(-0.04, 0):.6f}", f"MWS.HPC.s_Eff={rng.uniform(-0.02, 0):.6f}",
              f"MWS.HPC.s_Wc={rng.uniform(-0.02, 0):.6f}"]
        cmd = [sys.executable, str(HERE / "cb_run.py"), "--bin", a.bin, "--flight", a.flight,
               "--out", str(out / f"r{i:03d}"), "--rate", a.rate]
        for t in th:
            cmd += ["--theta", t]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    rcs = [p.wait() for p in procs]
    wall = time.time() - t0
    runs = []
    for i in range(a.n):
        f = out / f"r{i:03d}" / "run.json"
        R = json.loads(f.read_text()) if f.exists() else {}
        runs.append({"ok": R.get("ok"), "total_s": R.get("total_wall_s"), "ss_s": R.get("ss", {}).get("wall_s"),
                     "dyn_s": R.get("dyn", {}).get("wall_s"), "bytes": R.get("dyn", {}).get("bytes"),
                     "maxrss_kb": R.get("dyn", {}).get("children_maxrss_kb"),
                     "stderr": R.get("dyn", {}).get("stderr", "")[-200:]})
    ok = [r for r in runs if r["ok"]]
    nrows = sum(1 for _ in open(a.flight)) - 1
    S = {"n": a.n, "ok": len(ok), "rcs_nonzero": sum(1 for r in rcs if r), "batch_wall_s": wall,
         "flight": a.flight, "flight_seconds": nrows, "rate": a.rate, "host": platform.node(),
         "cpus": os.cpu_count(),
         "per_run_total_s_mean": statistics.mean(r["total_s"] for r in ok) if ok else None,
         "per_run_total_s_max": max(r["total_s"] for r in ok) if ok else None,
         "per_run_dyn_s_mean": statistics.mean(r["dyn_s"] for r in ok) if ok else None,
         "per_run_ss_s_mean": statistics.mean(r["ss_s"] for r in ok) if ok else None,
         "maxrss_mb_max": max((r["maxrss_kb"] or 0) for r in ok) / 1024 if ok else None,
         "bytes_mean": statistics.mean(r["bytes"] for r in ok) if ok else None,
         "flights_per_hour_node": len(ok) / wall * 3600,
         "flight_hours_per_wall_hour": len(ok) * nrows / wall,
         "runs": runs}
    (out / "throughput.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
