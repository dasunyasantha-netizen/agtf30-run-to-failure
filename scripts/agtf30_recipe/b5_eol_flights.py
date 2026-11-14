#!/usr/bin/env python3
"""B5 check: the worst end-of-life wear of each family (largest fault loss in life_<tag>_<FAM>.json) flown
closed loop on the 3 B4 N-CMAPSS flights (cb_run, 1 Hz). Reports completion, solver cap steps, limiter use,
min stall margins, max T45. Output $AGTF30_WORK/agtf30_b/b5/eol_flights_<tag>.json"""
import sys, json, pathlib, subprocess, concurrent.futures as cf
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from b_ss import BIN  # noqa: E402
from cb_run import read_out  # noqa: E402
tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
B4 = pathlib.Path(f"{WORK}/agtf30_b/b4"); W = pathlib.Path(f"{WORK}/agtf30_b/b5/eol_flights_{tag}")
FL = ["DS01-005_u1_c1", "DS04_u1_c1", "DS08c-008_u1_c1"]
jobs = []
for f in "HPT HPC LPT FAN MIX".split():
    E = json.load(open(f"{WORK}/agtf30_b/b5/life_{tag}_{f}.json"))["engines"]
    e = max(E, key=lambda x: x["g_eol"])
    th = {k: v / 100.0 for k, v in e["theta_eol"].items()}
    for fl in FL:
        jobs.append((f, e["idx"], fl, th))
def run(j):
    f, idx, fl, th = j
    out = W / f"{f}_{fl}"
    cmd = [sys.executable, str(HERE.parent / "agtf30_cbuild" / "cb_run.py"), "--bin", str(BIN), "--flight", str(B4 / f"flight_{fl}.csv"),
           "--out", str(out), "--rate", "1hz"] + sum((["--theta", f"{k}={v!r}"] for k, v in th.items()), [])
    subprocess.run(cmd, capture_output=True, text=True)
    R = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    r = {"fam": f, "engine": idx, "flight": fl, "ss_ok": R.get("ss", {}).get("converged") == 1.0,
         "dyn_rc": R.get("dyn", {}).get("rc"), "complete": bool(R.get("dyn", {}).get("complete"))}
    if (out / "dyn.bin").exists():
        n, Y = read_out(out / "dyn.bin"); c = {k: i for i, k in enumerate(n)}
        it = Y[:, c["loop.Solver.Iterations"]]; reg = Y[:, c["cntrl.RegNum"]]
        r.update(seconds=int(len(Y)), cap_seconds=int((it >= 64).sum()), limiter_seconds=int((reg != 0).sum()),
                 regs=sorted(set(int(x) for x in reg)), min_SMFan=float(Y[:, c["eng.SM.SMFan"]].min()),
                 min_SMLPC=float(Y[:, c["eng.SM.SMLPC"]].min()), min_SMHPC=float(Y[:, c["eng.SM.SMHPC"]].min()),
                 max_T45=float(Y[:, c["eng.S45.Tt"]].max()))
    print(r, flush=True)
    return r
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(run, jobs))
json.dump(res, open(f"{WORK}/agtf30_b/b5/eol_flights_{tag}.json", "w"), indent=1)
