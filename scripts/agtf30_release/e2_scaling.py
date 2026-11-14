#!/usr/bin/env python3
"""E2: nested scaling subsets of the 210 training engines (B_RECIPE section 9, agreed #8).

    python e2_scaling.py --h5 $AGTF30_WORK/agtf30_v1 --out configs/agtf30_v1

Index files only (no data copies): scaling_subsets.json and scaling_subsets.csv. Engines are named as the
shared loader names units ('AGTF30_<FAM>:<unit>', agtf30_data.iter_units), so a list can be passed to
iter_units(..., units=...) directly. The test set is the same 90 engines (18 per subset) for every size.

Design (one draw = one seed): the 210 training engines are 15 strata (5 subsets x 3 length classes) of 14.
One random order of all 210 is built from 14 rounds; each round takes one engine from every stratum, in 3 blocks
of 5 positions: position j of block k is subset fo[j] with class cp[(j + k + r) % 3] (fo, cp random permutations,
r a random offset, new per round), so every block has each subset once and every round each stratum once.
The engine taken from a stratum is the next one of that stratum's random permutation. The size-n list is the
first n engines of the order, so the lists are nested, and any prefix of whole blocks has equal subset counts
and class counts that differ by at most 1 (which class gets the extra one is random):
  10: 2 per subset, classes 4/3/3; 25: 5 per subset, 9/8/8; 50: 10, 17/17/16; 100: 20, 34/33/33; 200: 40, 67/67/66.
Draw 0 is the default list; draws 1 and 2 are independent repeats for the spread across draws.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter

import h5py
import numpy as np
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

FAMS = ["HPT", "HPC", "LPT", "FAN", "MIX"]
SIZES = [10, 25, 50, 100, 200]
SEEDS = [202611060, 202611061, 202611062]     # draw 0 (default), 1, 2


def engines(h5: pathlib.Path):
    rows = []
    for f in FAMS:
        with h5py.File(h5 / f"AGTF30_{f}.h5", "r") as h:
            for u in h["units"][:]:
                rows.append({"key": f"AGTF30_{f}:{int(u['unit'])}", "family": f, "unit": int(u["unit"]),
                             "cls": int(u["Fc"]), "split": u["split"].decode(), "eol": int(u["eol"])})
    return rows


def draw_order(train: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    strata = {}
    for e in sorted(train, key=lambda e: (e["family"], e["unit"])):
        strata.setdefault((e["family"], e["cls"]), []).append(e)
    sizes = {len(v) for v in strata.values()}
    assert len(strata) == 15 and sizes == {14}, (len(strata), sizes)
    perm = {k: [v[i] for i in rng.permutation(len(v))] for k, v in strata.items()}
    taken = Counter()
    order = []
    for _ in range(14):
        fo = [FAMS[i] for i in rng.permutation(5)]
        cp = [1 + int(i) for i in rng.permutation(3)]
        r = int(rng.integers(3))
        for k in range(3):
            for j in range(5):
                s = (fo[j], cp[(j + k + r) % 3])
                order.append(perm[s][taken[s]])
                taken[s] += 1
    return order


def check(order, train, test):
    keys = [e["key"] for e in order]
    assert len(keys) == len(set(keys)) == len(train) == 210
    assert set(keys) == {e["key"] for e in train}
    assert not set(keys) & {e["key"] for e in test}
    out = {}
    for n in SIZES + [210]:
        sub = order[:n]
        fam = Counter(e["family"] for e in sub)
        cls = Counter(e["cls"] for e in sub)
        fc = Counter((e["family"], e["cls"]) for e in sub)
        out[n] = {"per_subset": {f: fam[f] for f in FAMS}, "per_class": {c: cls[c] for c in (1, 2, 3)},
                  "per_subset_class_min_max": [min(fc[(f, c)] for f in FAMS for c in (1, 2, 3)),
                                               max(fc.values())],
                  "life_mean": round(float(np.mean([e["eol"] for e in sub])), 2)}
        assert max(fam.values()) - min(fam.values()) == 0, (n, fam)
        assert max(cls.values()) - min(cls.values()) <= 1, (n, cls)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=f"{WORK}/agtf30_v1")
    ap.add_argument("--out", default="configs/agtf30_v1")
    a = ap.parse_args(argv)
    E = engines(pathlib.Path(a.h5))
    train = [e for e in E if e["split"] == "dev"]
    test = sorted((e for e in E if e["split"] == "test"), key=lambda e: (FAMS.index(e["family"]), e["unit"]))
    J = {"dataset": "AGTF30 v1.0", "design": __doc__.split("Design")[1].strip(), "sizes": SIZES,
         "seeds": SEEDS, "default_draw": 0, "unit_key": "AGTF30_<FAM>:<unit> (agtf30_data.iter_units key)",
         "test": [e["key"] for e in test], "train_all": sorted(e["key"] for e in train), "draws": [], "balance": []}
    rows = []
    for d, seed in enumerate(SEEDS):
        order = draw_order(train, seed)
        J["balance"].append(check(order, train, test))
        J["draws"].append({"draw": d, "seed": seed, "order": [e["key"] for e in order],
                           "subsets": {str(n): [e["key"] for e in order[:n]] for n in SIZES}})
        for rank, e in enumerate(order):
            rows.append({"draw": d, "seed": seed, "rank": rank + 1, "key": e["key"], "family": e["family"],
                         "unit": e["unit"], "cls": e["cls"], "eol": e["eol"],
                         **{f"in_{n}": int(rank < n) for n in SIZES}})
    for e in test:
        rows.append({"draw": "test", "seed": "", "rank": "", "key": e["key"], "family": e["family"], "unit": e["unit"],
                     "cls": e["cls"], "eol": e["eol"], **{f"in_{n}": 1 for n in SIZES}})
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "scaling_subsets.json").write_text(json.dumps(J, indent=1))
    with open(out / "scaling_subsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # overlap between draws (how independent the draws are at each size)
    ov = {n: [len(set(J["draws"][i]["subsets"][str(n)]) & set(J["draws"][j]["subsets"][str(n)]))
              for i, j in ((0, 1), (0, 2), (1, 2))] for n in SIZES}
    print(json.dumps({"balance_draw0": J["balance"][0], "overlap_pairs": ov}, indent=1))


if __name__ == "__main__":
    main()
