#!/usr/bin/env python3
"""Age-only baselines for the learnability check (no sensor data, no test labels used for fitting).

For a test flight at age c (flights flown, c = 1 .. end of life), the prediction is the mean remaining life
eol - c over the development engines that are still flying at age c (eol >= c):
  age        : all development engines of all files;
  age_class  : only development engines of the same flight-length class (Fc).
If no development engine reached age c, the oldest age that has one is used.
Also reports the training-mean guess (mean RUL over all development flights), as in learn_check.py.

Only the per-engine 'units' table of each file is read (eol, split, Fc). RMSE is over all test flights,
one per flight, the same 11,424 targets as learn_check.py; --pred checks this against its pred_res.npz.

  python age_baseline.py --h5 <folder with AGTF30_*.h5> --out age_baseline.json [--pred pred_res.npz]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


def load(files):
    rows = []
    for p in files:
        with h5py.File(p, "r") as h:
            u = h["units"][:]
        for r in u:
            rows.append((p.stem, int(r["unit"]), r["split"].decode(), int(r["Fc"]), int(r["eol"])))
    return rows


def mean_rul_by_age(lives, max_age):
    """m[c] = mean(eol - c | eol >= c) for c = 1..max_age (carried forward where no engine is left)."""
    lives = np.asarray(lives)
    m = np.full(max_age + 1, np.nan)
    for c in range(1, max_age + 1):
        alive = lives[lives >= c]
        m[c] = (alive - c).mean() if len(alive) else m[c - 1]
    return m


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred", help="learn_check.py pred_res.npz, to confirm the same test targets")
    a = ap.parse_args(argv)
    files = sorted(pathlib.Path(a.h5).glob("AGTF30_*.h5"))
    rows = load(files)
    dev = [r for r in rows if r[2] == "dev"]
    test = [r for r in rows if r[2] == "test"]
    max_age = max(r[4] for r in rows)
    m_all = mean_rul_by_age([r[4] for r in dev], max_age)
    m_cls = {fc: mean_rul_by_age([r[4] for r in dev if r[3] == fc], max_age) for fc in sorted({r[3] for r in dev})}
    train_mean = float(np.mean([l - c for *_, l in dev for c in range(1, l + 1)]))

    y, p_mean, p_age, p_cls, subset, key = [], [], [], [], [], []
    for stem, unit, _, fc, eol in test:
        for c in range(1, eol + 1):
            y.append(eol - c); p_mean.append(train_mean); p_age.append(m_all[c]); p_cls.append(m_cls[fc][c])
            subset.append(stem.replace("AGTF30_", "")); key.append((f"{stem}:{unit}", c))
    y, subset = np.array(y, float), np.array(subset)
    preds = {"train_mean": np.array(p_mean), "age": np.array(p_age), "age_class": np.array(p_cls)}

    R = {"files": [f.name for f in files], "n_dev_engines": len(dev), "n_test_engines": len(test),
         "n_test_flights": int(len(y)), "train_mean_rul": train_mean, "rmse": {}, "rmse_per_subset": {}}
    for k, p in preds.items():
        R["rmse"][k] = rmse(y, p)
        R["rmse_per_subset"][k] = {s: rmse(y[subset == s], p[subset == s]) for s in sorted(set(subset))}
    if a.pred:
        d = np.load(a.pred)
        theirs = {(str(u), int(c)): float(t) for u, c, t in zip(d["unit"], d["cycle"], d["y"])}
        mine = {k: float(t) for k, t in zip(key, y)}
        R["same_targets_as_learn_check"] = bool(theirs == mine)
    pathlib.Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("n_dev_engines", "n_test_engines", "n_test_flights", "train_mean_rul", "rmse")}
                     | ({"same_targets": R["same_targets_as_learn_check"]} if a.pred else {}), indent=1))


if __name__ == "__main__":
    main()
