#!/usr/bin/env python3
"""G2 (v1.1 feasibility): reference-point-only life simulation for the new fault subsets, to calibrate K.

    python g2_v11_life.py --fams HPTLPT --K HPTLPT=1.5 --n 20 --tag k1 [--priority idle]

Runs b5_life.py unchanged (same wear model, onset rule and death check as v1.0, recipe settings
lam = 0.12 and s_init = 0.2) with the three new fault directions added in memory only. v1.0 code is not
modified. Family seeds 6, 7, 8 (v1.0 uses 1-5), so no v1.0 engine is reproduced or changed.
Output $AGTF30_WORK/agtf30_g/g2/life_<tag>.json.
"""
import pathlib
import sys
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agtf30_gen"))
import b5_life  # noqa: E402

# per 1 % efficiency loss of each faulted module: (efficiency %, flow %); same values as g1_v11_sweep.py
NEW_DIR = {"HPTLPT": {"HPT": (-1, +0.5), "LPT": (-1, +0.5)},
           "LPCHPC": {"LPC": (-1, -1.5), "HPC": (-1, -1.5)},
           "ALL": {"FAN": (-1, -0.7), "LPC": (-1, -1.5), "HPC": (-1, -1.5), "HPT": (-1, +0.5), "LPT": (-1, +0.5)}}
NEW_SEED = {"HPTLPT": 6, "LPCHPC": 7, "ALL": 8}

if __name__ == "__main__":
    args = sys.argv[1:]
    prio = "idle"
    if "--priority" in args:
        i = args.index("--priority"); prio = args[i + 1]; del args[i:i + 2]
    from gen_run import set_priority
    set_priority(prio)
    b5_life.DIR.update(NEW_DIR)
    b5_life.FAM_SEED.update(NEW_SEED)
    b5_life.OUT = pathlib.Path(f"{WORK}/agtf30_g/g2")
    b5_life.OUT.mkdir(parents=True, exist_ok=True)
    defaults = ["--lam", "0.12", "--s_init", "0.2", "--L_bg", "80", "--tmax", "160"]
    b5_life.main(defaults + args)
