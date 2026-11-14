#!/usr/bin/env python3
"""After a generator run: merge -> C6 checks -> judge -> learnability check, one step after the other, as one
detached job at Idle priority (inherited by every step). Each step's output goes to <out>/post_<step>.log.

    python gen_post.py --out $AGTF30_WORK/agtf30_c/main --rerun HPT:30:3 --loader-units 3 --reused-flight-h 9995.7

Ends with POST_DONE in <out>/post.log, or POST_FAILED <step> (later steps are not run).
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_run import set_priority  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerun", default="HPT:30:3")
    ap.add_argument("--loader-units", type=int, default=3)
    ap.add_argument("--reused-flight-h", type=float, default=0.0)
    ap.add_argument("--steps", default="merge,checks,judge,learn")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    set_priority("idle")
    py = sys.executable
    cmds = {"merge": [py, HERE / "gen_merge.py", "--out", out, "--filter", "gzip1"],
            "checks": [py, HERE / "gen_checks.py", "--out", out, "--ref", "DS01-005", "--rerun", a.rerun,
                       "--loader-units", str(a.loader_units)],
            "judge": [py, HERE / "gen_judge.py", "--out", out, "--reused-flight-h", str(a.reused_flight_h)],
            "learn": [py, HERE.parent / "agtf30_learn" / "learn_check.py", "--h5", out / "h5", "--out", out / "learn"]}
    log = out / "post.log"
    for step in a.steps.split(","):
        with open(log, "a") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} start {step}\n")
        with open(out / f"post_{step}.log", "w") as f:
            rc = subprocess.run([str(c) for c in cmds[step]], stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode
        with open(log, "a") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} end {step} rc {rc}\n")
            if rc != 0:
                f.write(f"POST_FAILED {step}\n")
                return rc
    with open(log, "a") as f:
        f.write("POST_DONE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
