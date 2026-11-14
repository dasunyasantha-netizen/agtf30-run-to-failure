#!/usr/bin/env python3
"""E1: freeze the AGTF30 dataset v1.0 (phase E order).

    python e1_freeze.py --move     # main/h5/*.h5 -> $AGTF30_WORK/agtf30_v1 (same disk: a rename, no copy)
    python e1_freeze.py --attrs    # add the v1.0 release attributes (attributes only; data untouched, verified)
    python e1_freeze.py --manifest # MANIFEST.json + provenance/ (copies of the main run's check files)
    python e1_freeze.py --hash     # SHA256SUMS.txt for every file in the release folder (Idle priority)
    python e1_freeze.py --lock     # read-only flag on every file
    python e1_freeze.py --verify   # re-hash and compare with SHA256SUMS.txt

The .h5 files are moved, not copied: D: has 57 GB free and the files are 65.7 GB. The main run's scratch
(main/runs) can re-merge them bit for bit except the 'created' attribute, so it is the backup until the owner
decides on deletion. A pointer file is left in main/h5.

Release attributes (added to every .h5; the merge's own attributes are kept unchanged):
  dataset, dataset_version, release_date, run_commit (code of the simulation run), ref_fix_commit (13 reference
  points repaired), merge_commit (= git_commit), seed_rule, split_rule, n_dev_units, n_test_units, licence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys

import h5py
import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agtf30_gen"))
from gen_run import set_priority  # noqa: E402  (Idle priority class, checked)

HERE = pathlib.Path(__file__).resolve().parent
MAIN = pathlib.Path(f"{WORK}/agtf30_c/main")
DST = pathlib.Path(f"{WORK}/agtf30_v1")
FAMS = ["HPT", "HPC", "LPT", "FAN", "MIX"]
VERSION = "1.0"
RELEASE_DATE = "2026-11-06"
RUN_COMMIT = "a52a30689c7bfc55b4840822a5f4c43fb7424f64"      # main/code/GIT_COMMIT.txt (launch 2026-11-04)
REF_FIX_COMMIT = "69e7016"                                    # gen_ref.py fix + repair of 13 flights
SEED_RULE = ("numpy SeedSequence([recipe_id=20261103, family_id (HPT 1, HPC 2, LPT 3, FAN 4, MIX 5), engine_index]);"
             " child streams (0,) wear parameters, (1,) flight sequence, (2, cycle) sensor noise of that flight;"
             " units.seed = the engine's entropy; replacement engines (decision #7) have engine_index = slot + k*1000")
SPLIT_RULE = ("70/30 by engine within each subset, stratified by length class: 42 dev (14 per class) / 18 test"
              " (6 per class); SeedSequence([20261103, family_id, 999999])")
PROVENANCE = ["checks.json", "drops_analysis.json", "drops.jsonl", "ref_repair.json", "plan.json",
              "judge/judge.json", "learn/learn.json", "h5/merge.json", "code/GIT_COMMIT.txt"]


def files():
    return [DST / f"AGTF30_{f}.h5" for f in FAMS]




def move():
    DST.mkdir(parents=True, exist_ok=True)
    for f in FAMS:
        s, d = MAIN / "h5" / f"AGTF30_{f}.h5", DST / f"AGTF30_{f}.h5"
        if d.exists():
            print("already moved", d.name)
            continue
        os.replace(s, d)                                    # same volume: rename
        print("moved", s, "->", d)
    (MAIN / "h5" / "MOVED_TO_agtf30_v1.txt").write_text(
        f"The five AGTF30_<FAM>.h5 files of this run were moved (renamed, not copied) to {DST} on "
        f"{dt.date.today()} as dataset version {VERSION} (phase E, E1). merge.json stays here.\n")


def fingerprint(path: pathlib.Path) -> dict:
    """Shapes, dtypes and sha256 of the first and last 50,000 rows of every dataset (cheap content check)."""
    out = {}
    with h5py.File(path, "r") as h:
        for k in sorted(h):
            ds = h[k]
            n = ds.shape[0]
            m = hashlib.sha256()
            for sl in (slice(0, min(n, 50_000)), slice(max(0, n - 50_000), n)):
                m.update(np.ascontiguousarray(ds[sl]).tobytes())
            out[k] = [list(ds.shape), str(ds.dtype), m.hexdigest()]
    return out


def attrs():
    for p in files():
        before = fingerprint(p)
        with h5py.File(p, "r") as h:
            u = h["units"][:]
        n_dev = int((u["split"] == b"dev").sum())
        n_test = int((u["split"] == b"test").sum())
        if os.stat(p).st_mode & stat.S_IWRITE == 0:
            sys.exit(f"{p} is read-only: release attributes must be added before --lock")
        with h5py.File(p, "r+") as h:
            h.attrs.update({
                "dataset": "AGTF30 run-to-failure dataset (N-CMAPSS layout)",
                "dataset_version": VERSION, "release_date": RELEASE_DATE,
                "run_commit": RUN_COMMIT, "ref_fix_commit": REF_FIX_COMMIT,
                "merge_commit": h.attrs["git_commit"], "seed_rule": SEED_RULE, "split_rule": SPLIT_RULE,
                "n_dev_units": n_dev, "n_test_units": n_test,
                "data_card": "docs/AGTF30 dataset card.md (repository engine-digital-twin)",
                "licence": "not yet decided by the owner (see the data card)"})
        after = fingerprint(p)
        same = before == after
        print(p.name, "dev", n_dev, "test", n_test, "data unchanged:", same, flush=True)
        if not same:
            sys.exit(f"DATA CHANGED in {p}")


def manifest():
    prov = DST / "provenance"
    prov.mkdir(exist_ok=True)
    for r in PROVENANCE:
        src = MAIN / r
        shutil.copy2(src, prov / r.replace("/", "__"))
    merge = {m["family"]: m for m in json.loads((MAIN / "h5" / "merge.json").read_text())}
    M = {"dataset": "AGTF30 run-to-failure dataset (N-CMAPSS layout)", "version": VERSION,
         "release_date": RELEASE_DATE, "recipe": "docs/solo_reports/B_RECIPE.md (approved 2026-11-03, sections 2-9, 10a)",
         "generator_report": "docs/solo_reports/C_GENERATOR.md section 10", "run_commit": RUN_COMMIT,
         "ref_fix_commit": REF_FIX_COMMIT, "seed_rule": SEED_RULE, "split_rule": SPLIT_RULE,
         "source_run": str(MAIN), "files": {}}
    for p in files():
        with h5py.File(p, "r") as h:
            a = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
            u = h["units"][:]
            cyc = h["cycles"][:]
            rows = {sp: int(h[f"A_{sp}"].shape[0]) for sp in ("dev", "test")}
            cols = {g: [x.decode() for x in h[f"{g}_var"][:]] for g in ("W", "X_s", "X_s_clean", "X_v", "X_a", "T", "A")}
        fam = a["family"]
        M["files"][p.name] = {
            "family": fam, "bytes": p.stat().st_size, "units": len(u),
            "dev_units": sorted(int(x) for x in u["unit"][u["split"] == b"dev"]),
            "test_units": sorted(int(x) for x in u["unit"][u["split"] == b"test"]),
            "flights": int(len(cyc)), "flight_hours": round(float(cyc["hours"].sum()), 2), "rows": rows,
            "units_per_class": {str(c): int((u["Fc"] == c).sum()) for c in (1, 2, 3)},
            "life_min_median_max": [int(u["eol"].min()), float(np.median(u["eol"])), int(u["eol"].max())],
            "merge_wall_s": round(merge[fam]["wall_s"], 1), "columns": cols, "attrs": a}
    M["totals"] = {"files": len(M["files"]), "bytes": sum(f["bytes"] for f in M["files"].values()),
                   "units": sum(f["units"] for f in M["files"].values()),
                   "flights": sum(f["flights"] for f in M["files"].values()),
                   "flight_hours": round(sum(f["flight_hours"] for f in M["files"].values()), 2)}
    (DST / "MANIFEST.json").write_text(json.dumps(M, indent=1))
    print(json.dumps(M["totals"]))


def sha256(p: pathlib.Path) -> str:
    m = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            m.update(b)
    return m.hexdigest()


def release_files():
    return sorted((p for p in DST.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"),
                  key=lambda p: p.relative_to(DST).as_posix())


def hash_all():
    set_priority("idle")
    old = read_sums()
    lines = []
    for p in release_files():
        rel = p.relative_to(DST).as_posix()
        h = old.get(rel) if p.suffix == ".h5" and old.get(rel) else sha256(p)   # h5 hashes are kept once written
        lines.append(f"{h} *{rel}")
        print(lines[-1], flush=True)
        (DST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", newline="\n")   # LF, else sha256sum -c fails
    print("HASH_DONE", len(lines))


def read_sums() -> dict:
    s = DST / "SHA256SUMS.txt"
    if not s.exists():
        return {}
    return {ln.split(" *", 1)[1]: ln.split(" *", 1)[0] for ln in s.read_text().splitlines() if " *" in ln}


def lock():
    for p in release_files() + [DST / "SHA256SUMS.txt"]:
        os.chmod(p, stat.S_IREAD)
        print("read-only", p.relative_to(DST).as_posix())


def verify():
    set_priority("idle")
    sums = read_sums()
    bad = [rel for rel, h in sums.items() if sha256(DST / rel) != h]
    extra = [p.relative_to(DST).as_posix() for p in release_files() if p.relative_to(DST).as_posix() not in sums]
    print(json.dumps({"files": len(sums), "mismatch": bad, "not_listed": extra}))
    print("VERIFY_OK" if not bad and not extra else "VERIFY_FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("move", "attrs", "manifest", "hash", "lock", "verify"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--dst", default=str(DST))
    a = ap.parse_args()
    DST = pathlib.Path(a.dst)
    for k, fn in (("move", move), ("attrs", attrs), ("manifest", manifest), ("hash", hash_all), ("lock", lock),
                  ("verify", verify)):
        if getattr(a, k):
            fn()
