#!/usr/bin/env python3
"""E4: baseline table for the data card from learn_check.py run on the frozen v1.0 files.

    python e4_baseline.py --learn $AGTF30_WORK/agtf30_v1_work/e4/learn.json --ref $AGTF30_WORK/agtf30_v1/provenance/learn__learn.json \
        --out docs/agtf30_v1/baseline.json

learn_check.py is the D2 check unchanged (train on all 210 training engines, test on the 90 test engines, per-flight
windows of 10, 'res' features, hidden 32, 60 epochs, seeds 0-2). This script reports test RMSE and N-CMAPSS score,
overall and per subset, as mean +- SD over the 3 seeds, and compares every number with the main-run learn.json
(same code and data before the freeze) as a reproduction check.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

FAMS = ["HPT", "HPC", "LPT", "FAN", "MIX"]
ARCH = [("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")]


def table(L, fs="res"):
    S = L["sets"][fs]
    B = S["baseline_train_mean"]
    rows = [{"model": "Mean guess (training-mean RUL %.1f)" % B["value"], "rmse": [B["rmse"], 0.0],
             "score": [B["score"], 0.0],
             "per_subset": {f: {"rmse": [B["per_file"][f"AGTF30_{f}"]["rmse"], 0.0],
                                "score": [B["per_file"][f"AGTF30_{f}"]["score"], 0.0]} for f in FAMS}}]
    for k, name in ARCH:
        R = S["models"][k]["runs"]
        ms = lambda xs: [float(np.mean(xs)), float(np.std(xs))]   # noqa: E731  (population SD, as learn_check)
        rows.append({"model": name, "rmse": ms([r["rmse"] for r in R]), "score": ms([r["score"] for r in R]),
                     "per_subset": {f: {"rmse": ms([r["per_file"][f"AGTF30_{f}"]["rmse"] for r in R]),
                                        "score": ms([r["per_file"][f"AGTF30_{f}"]["score"] for r in R])} for f in FAMS},
                     "seeds": [r["seed"] for r in R]})
    return rows


def markdown(rows):
    h = "| Model | RMSE (flights) | N-CMAPSS score | " + " | ".join(f"{f} RMSE / score" for f in FAMS) + " |"
    L = [h, "|---|---|---|" + "---|" * len(FAMS)]
    for r in rows:
        pm = lambda v, d=2: f"{v[0]:.{d}f}" + (f" ± {v[1]:.{d}f}" if v[1] else "")   # noqa: E731
        L.append(f"| {r['model']} | {pm(r['rmse'])} | {pm(r['score'], 0)} | " + " | ".join(
            f"{pm(r['per_subset'][f]['rmse'], 1)} / {pm(r['per_subset'][f]['score'], 0)}" for f in FAMS) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", default=f"{WORK}/agtf30_v1_work/e4/learn.json")
    ap.add_argument("--ref", default=f"{WORK}/agtf30_v1/provenance/learn__learn.json")
    ap.add_argument("--out", default="docs/agtf30_v1/baseline.json")
    ap.add_argument("--fams", default=",".join(FAMS), help="AGTF30 subsets (v1.1: HPTLPT,LPCHPC,ALL)")
    a = ap.parse_args(argv)
    FAMS[:] = a.fams.split(",")
    L = json.loads(pathlib.Path(a.learn).read_text())
    rows = table(L)
    if a.ref.lower() == "none":          # no earlier run of the same code and data to compare with (v1.1)
        diff = None
    else:
        ref = table(json.loads(pathlib.Path(a.ref).read_text()))
        diff = max(abs(x["rmse"][0] - y["rmse"][0]) for x, y in zip(rows, ref))
    out = {"source": a.learn, "n_train_units": L["n_train_units"], "n_test_units": L["n_test_units"],
           "n_test_windows": L["sets"]["res"]["baseline_train_mean"]["n"], "window": L["window"], "epochs": L["epochs"],
           "seeds": L["seeds"], "features": "res (W flight means + X_s operating-condition residual means)",
           "rows": rows, "max_abs_rmse_diff_vs_main_run": diff, "markdown": markdown(rows)}
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(out["markdown"]); print("max |RMSE difference| vs main-run learn.json:", diff)


if __name__ == "__main__":
    main()
