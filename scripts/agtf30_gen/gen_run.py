#!/usr/bin/env python3
"""C5: run many engines in parallel (this PC, or one SLURM array task), resumable.

    # this PC, pilot (10 per subset, 12 at once), detached:
    python gen_run.py --out $AGTF30_WORK/agtf30_c/pilot --n-per-fam 10 --workers 12
    # dry run: slots 0-1 of every subset
    python gen_run.py --out $AGTF30_WORK/agtf30_c/dry --n-per-fam 10 --slots 0-1 --workers 10
    # SLURM array (AI Panther): task i of n takes every n-th engine; paths from the environment
    #   AGTF30_BIN_B, AGTF30_BIN_FINE (Linux builds), --lib on scratch
    python gen_run.py --out $SCRATCH/agtf30_main --lib $SCRATCH/library --n-per-fam 60 \
        --workers $SLURM_CPUS_PER_TASK --array-index $SLURM_ARRAY_TASK_ID --array-count 8
    # progress:
    python gen_run.py --out $AGTF30_WORK/agtf30_c/pilot --status

The whole process tree runs at Idle CPU priority by default (--priority idle): it slows down while the
owner works on this PC and runs at full speed when the PC is otherwise idle. Use --priority normal on SLURM if wanted.

Every engine checkpoints per flight (gen_engine), so re-running the same command resumes.
A dropped engine (decision #7) is replaced by engine index slot + k*1000 (k = 1, 2, 3) and the
drop is written to <out>/drops.jsonl and run.log; nothing is dropped silently.
The merge into .h5 files is a separate step (gen_merge.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_recipe as R  # noqa: E402

MAX_REPLACEMENTS = 3


def _log(out: pathlib.Path, msg: str):
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a") as f:
        f.write(line + "\n")


def set_priority(level: str):
    """Lower this process's CPU priority; workers and runner exes inherit it (Windows priority class / POSIX nice)."""
    if level == "normal":
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetPriorityClass.restype = wintypes.BOOL
        cls = {"idle": 0x40, "below_normal": 0x4000}[level]   # IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS
        if not k.SetPriorityClass(k.GetCurrentProcess(), cls):
            raise OSError(f"SetPriorityClass({level}) failed, error {ctypes.get_last_error()}")
    else:
        os.nice({"idle": 19, "below_normal": 10}[level])


def run_slot(task):
    cfg, fam, slot = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import gen_engine as G
    lib = G.Library(cfg["lib"])
    hist = []
    for k in range(MAX_REPLACEMENTS + 1):
        idx = slot + k * R.REPLACE_STRIDE
        try:
            r = G.run_engine(cfg, fam, slot, idx, lib=lib)
        except (G.EnvFault, MemoryError, OSError) as e:   # the PC failed, not the engine: no drop, slot resumable
            hist.append({"status": "env_error", "reason": f"{type(e).__name__}: {e}"[:2000], "family": fam,
                         "slot": slot, "engine_index": idx})
            break
        except Exception as e:  # noqa: BLE001  (logged as a drop, never silent)
            r = {"status": "dropped", "reason": f"exception {type(e).__name__}: {e}", "family": fam, "slot": slot,
                 "engine_index": idx}
            G._atomic_json(G.engine_dir(cfg, fam, idx) / "dropped.json", r)
        hist.append(r)
        if r["status"] == "done":
            break
    return {"family": fam, "slot": slot, "final": hist[-1], "history": hist}


def parse_slots(s: str | None, n: int) -> list[int]:
    if not s:
        return list(range(n))
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def status(out: pathlib.Path):
    runs = out / "runs"
    done = drop = flights = 0
    secs = 0
    engines = 0
    for d in sorted(runs.glob("*/e*")):
        engines += 1
        done += (d / "done.json").exists()
        drop += (d / "dropped.json").exists()
        for j in d.glob("c*.json"):
            flights += 1
            secs += json.loads(j.read_text())["seconds"]
    plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else {}
    ntask = len(plan.get("tasks", []))
    fin = sorted(out.glob("job_*.json"))
    print(json.dumps({"slots_planned": ntask, "engine_dirs": engines, "engines_done": done, "engines_dropped": drop,
                      "flights": flights, "flight_hours": round(secs / 3600, 1), "jobs_finished": len(fin)}))
    env = sum(r["final"]["status"] == "env_error" for f in fin for r in json.loads(f.read_text())["results"])
    if fin and env:
        print(f"GEN_JOB_ENV_ERROR {env} slots stopped by PC faults (not drops): re-run the same command to resume")
    elif fin:
        print("GEN_JOB_DONE")
    elif (out / "pid.txt").exists() and not _alive(int((out / "pid.txt").read_text().split()[0])):
        print("GEN_JOB_DEAD (no job_*.json and the job process is gone: re-run the same command to resume)")


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in r
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=f"{WORK}/agtf30_c/library")
    ap.add_argument("--fams", default="HPT,HPC,LPT,FAN,MIX")
    ap.add_argument("--n-per-fam", type=int, default=10)
    ap.add_argument("--slots", default=None, help="e.g. 0-1 or 0,3,5 (default all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--array-index", type=int, default=0)
    ap.add_argument("--array-count", type=int, default=1)
    ap.add_argument("--priority", default="idle", choices=["idle", "below_normal", "normal"],
                    help="CPU priority of the whole process tree (default idle: yields to the owner's work)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.status:
        return status(out)
    set_priority(a.priority)
    import gen_engine as G
    cfg = G.default_cfg(out, a.lib, a.n_per_fam)
    tasks = []
    for fam in a.fams.split(","):
        cls = R.class_plan(fam, a.n_per_fam)
        for s in parse_slots(a.slots, a.n_per_fam):
            tasks.append((cls[s], fam, s))
    tasks.sort(key=lambda x: (-x[0], x[1], x[2]))          # longest (class 3) first
    tasks = [t for i, t in enumerate(tasks) if i % a.array_count == a.array_index]
    (out / "plan.json").write_text(json.dumps({"cfg": cfg, "fams": a.fams, "n_per_fam": a.n_per_fam,
                                               "slots": a.slots, "workers": a.workers,
                                               "array": [a.array_index, a.array_count],
                                               "tasks": [[f, s, c] for c, f, s in tasks]}, indent=1))
    (out / "pid.txt").write_text(f"{os.getpid()}\n")
    _log(out, f"start {len(tasks)} slots, {a.workers} workers, priority {a.priority}, array {a.array_index}/{a.array_count}, bin {cfg['bin']}")
    t0 = time.time()
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_slot, [(cfg, f, s) for _, f, s in tasks]):
            res.append(r)
            for h in r["history"]:
                if h["status"] == "dropped":
                    _log(out, f"DROP {h['family']} slot {h['slot']} engine {h['engine_index']}: {h.get('reason')}")
                    with open(out / "drops.jsonl", "a") as f:
                        f.write(json.dumps(h) + "\n")
            for h in r["history"]:
                if h["status"] == "env_error":
                    _log(out, f"ENV_ERROR {h['family']} slot {h['slot']} engine {h['engine_index']} (not a drop; "
                              f"re-run the same command to resume): {h['reason'][:300]}")
            fin = r["final"]
            _log(out, f"slot {r['family']}/{r['slot']} -> {fin['status']} engine {fin['engine_index']} "
                      f"eol {fin.get('eol')} limit {fin.get('limit')} ({len(res)}/{len(tasks)}, {time.time()-t0:.0f}s)")
    summ = {"wall_s": time.time() - t0, "slots": len(tasks), "done": sum(r["final"]["status"] == "done" for r in res),
            "drops": sum(h["status"] == "dropped" for r in res for h in r["history"]),
            "env_errors": sum(r["final"]["status"] == "env_error" for r in res), "results": res}
    jp = out / f"job_{a.array_index:03d}.json"
    if jp.exists():                          # a resumed job keeps the earlier record (its wall time is the run's)
        os.replace(jp, jp.with_name(f"{jp.stem}_{int(jp.stat().st_mtime)}.json"))
    jp.write_text(json.dumps(summ, indent=1))
    _log(out, f"GEN_JOB_DONE wall {summ['wall_s']:.0f}s done {summ['done']}/{len(tasks)} drops {summ['drops']} env_errors {summ['env_errors']}")


if __name__ == "__main__":
    main()
