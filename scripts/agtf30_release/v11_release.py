#!/usr/bin/env python3
"""Freeze the AGTF30 dataset v1.1 add-on (three subsets HPTLPT, LPCHPC, ALL) in $AGTF30_WORK/agtf30_v11.

    python v11_release.py --attrs      # release attributes in the three .h5 files (attributes only; data verified unchanged)
    python v11_release.py --manifest   # MANIFEST.json (v1.1 files + the unchanged v1.0 files it extends)
    python v11_release.py --stage      # LICENSE, NOTICE, data card, index files and figures into the release folder
    python v11_release.py --hash       # SHA256SUMS.txt for every file (h5 hashes kept once written; Idle priority)
    python v11_release.py --lock       # read-only flag on every file
    python v11_release.py --verify     # re-hash and compare with SHA256SUMS.txt

v1.1 = v1.0 ($AGTF30_WORK/agtf30_v1, five files, unchanged and not copied) + three new files. The release folder holds only
the new files; MANIFEST.json lists the v1.0 files with their v1.0 SHA-256 so the full dataset is defined in one place.
Same steps and attribute names as e1_freeze.py / e7_licence.py / e6_release_check.py --stage for v1.0.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import stat
import sys

import h5py
import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "agtf30_gen"))
import e1_freeze as E1  # noqa: E402  (fingerprint, sha256)
from gen_run import set_priority  # noqa: E402

DST = pathlib.Path(f"{WORK}/agtf30_v11")
V1 = pathlib.Path(f"{WORK}/agtf30_v1")
RUN = pathlib.Path(f"{WORK}/agtf30_g/v11_main")          # metadata copy of the cluster run folder (no .npz)
FAMS = ["HPTLPT", "LPCHPC", "ALL"]
VERSION = "1.1"
RELEASE_DATE = "2026-11-11"
RUN_COMMIT = "b3f88b8299135ae1dd090e9760191c64f7afbfff"    # cluster package for the engine runs (job 233591)
MERGE_COMMIT = "42154837ef3a9ab7dac32fd0c46d95f3f19a61fe"  # gen_merge.py with the 8-byte family field (job 234192)
SEED_RULE = ("numpy SeedSequence([recipe_id=20261103, family_id (HPT 1, HPC 2, LPT 3, FAN 4, MIX 5, HPTLPT 6, LPCHPC 7,"
             " ALL 8), engine_index]); child streams (0,) wear parameters, (1,) flight sequence, (2, cycle) sensor noise"
             " of that flight; units.seed = the engine's entropy; replacement engines have engine_index = slot + k*1000")
SPLIT_RULE = ("70/30 by engine within each subset, stratified by length class: 42 dev (14 per class) / 18 test"
              " (6 per class); SeedSequence([20261103, family_id, 999999])")
RECIPE = ("docs/solo_reports/B_RECIPE.md (approved 2026-11-03) with the v1.1 addendum docs/solo_reports/G_V11_FAULTS.md"
          " (owner decisions 2026-11-09: fault directions section 2, K section 4)")
LICENCE = {
    "licence": "CC-BY-4.0", "licence_file": "LICENSE.txt", "notice_file": "NOTICE.txt",
    "licence_decision": "owner, 2026-11-07 (v1.0; the same licence applies to v1.1)",
    "licence_sources": {"AGTF30 (github.com/nasa/AGTF30)": "Apache-2.0",
                        "T-MATS (github.com/nasa/T-MATS)": "Apache-2.0",
                        "N-CMAPSS flight conditions (Arias Chao et al., Data 2021, 6(1), 5)": "CC0-1.0"},
}
PROVENANCE = {"checks.json": RUN / "checks.json", "drops.jsonl": RUN / "drops.jsonl", "plan.json": RUN / "plan.json",
              "judge.json": RUN / "judge" / "judge.json", "learn.json": pathlib.Path(f"{WORK}/agtf30_v11_work/e4/learn.json")}
INDEX = [("docs/AGTF30 dataset card.md", "AGTF30 dataset card.md"),
         ("docs/agtf30_v11/stats.json", "stats.json"), ("docs/agtf30_v11/stats_tables.md", "stats_tables.md"),
         ("docs/agtf30_v11/baseline.json", "baseline.json"), ("docs/agtf30_v11/release_check.json", "release_check.json")]


def files():
    return [DST / f"AGTF30_{f}.h5" for f in FAMS]


def attrs():
    for p in files():
        before = E1.fingerprint(p)
        with h5py.File(p, "r") as h:
            u = h["units"][:]
            assert h.attrs["family"] == p.stem.split("_", 1)[1], p
        n_dev, n_test = int((u["split"] == b"dev").sum()), int((u["split"] == b"test").sum())
        with h5py.File(p, "r+") as h:
            h.attrs.update({
                "dataset": "AGTF30 run-to-failure dataset (N-CMAPSS layout)",
                "dataset_version": VERSION, "release_date": RELEASE_DATE, "extends": "dataset version 1.0 (unchanged)",
                "run_commit": RUN_COMMIT, "merge_commit": h.attrs["git_commit"], "seed_rule": SEED_RULE,
                "split_rule": SPLIT_RULE, "n_dev_units": n_dev, "n_test_units": n_test,
                "data_card": "docs/AGTF30 dataset card.md (repository engine-digital-twin)",
                "licence": "CC-BY-4.0 (see LICENSE.txt)"})
        same = before == E1.fingerprint(p)
        print(p.name, "dev", n_dev, "test", n_test, "data unchanged:", same, flush=True)
        if not same:
            sys.exit(f"DATA CHANGED in {p}")


def manifest():
    prov = DST / "provenance"
    prov.mkdir(exist_ok=True)
    for name, src in PROVENANCE.items():
        shutil.copy2(src, prov / name)
    merge = {m["family"]: m for m in json.loads((prov / "merge.json").read_text())}
    cluster_sums = {ln.split()[1]: ln.split()[0] for ln in (DST / "provenance" / "SHA256SUMS_cluster_merge.txt").read_text().splitlines()}
    v1 = json.loads((V1 / "MANIFEST.json").read_text())
    v1_sums = {ln.split(" *", 1)[1]: ln.split(" *", 1)[0]
               for ln in (V1 / "SHA256SUMS.txt").read_text().splitlines() if " *" in ln}
    M = {"dataset": "AGTF30 run-to-failure dataset (N-CMAPSS layout)", "version": VERSION,
         "release_date": RELEASE_DATE, "recipe": RECIPE,
         "generator_report": "docs/solo_reports/G_V11_FAULTS.md sections 6-7",
         "run_commit": RUN_COMMIT, "merge_commit": MERGE_COMMIT, "seed_rule": SEED_RULE, "split_rule": SPLIT_RULE,
         **LICENCE,
         "second_copy": "AI Panther scratch " + os.environ.get("AGTF30_SCRATCH", "$AGTF30_SCRATCH")
                        + "/agtf30_v11/main (merged files without the"
                        " release attributes, and the per-flight run folder); the lab Mac was not reachable (2026-11-10)",
         "source_run": "AI Panther jobs 233591 (engines, 2 x 90 CPUs) and 234192 (merge); metadata copy " + str(RUN),
         "extends": {"version": v1["version"], "folder": str(V1), "release_date": v1["release_date"],
                     "note": "the five v1.0 files are part of v1.1 unchanged; they are not copied into this folder",
                     "files": {n: {"family": f["family"], "units": f["units"], "bytes": f["bytes"],
                                   "sha256": v1_sums[n]} for n, f in v1["files"].items()},
                     "totals": v1["totals"]},
         "files": {}}
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
            "merge_wall_s": round(merge[fam]["wall_s"], 1),
            "sha256_as_merged_on_cluster": cluster_sums[p.name], "columns": cols, "attrs": a}
    tot = lambda F: {"files": len(F), "bytes": sum(f["bytes"] for f in F.values()),   # noqa: E731
                     "units": sum(f["units"] for f in F.values()),
                     "flights": sum(f.get("flights", 0) for f in F.values()),
                     "flight_hours": round(sum(f.get("flight_hours", 0) for f in F.values()), 2)}
    M["totals"] = tot(M["files"])
    t1 = v1["totals"]
    M["totals_with_v1_0"] = {k: (round(M["totals"][k] + t1[k], 2) if k == "flight_hours" else M["totals"][k] + t1[k])
                             for k in ("files", "bytes", "units", "flights", "flight_hours")}
    (DST / "MANIFEST.json").write_text(json.dumps(M, indent=1))
    print(json.dumps({"v1.1 files": M["totals"], "with v1.0": M["totals_with_v1_0"]}))


def stage():
    idx = DST / "index"
    (idx / "figures").mkdir(parents=True, exist_ok=True)
    pairs = [(REPO / "docs/agtf30_v1/LICENSE.txt", DST / "LICENSE.txt"), (REPO / "docs/agtf30_v11/NOTICE.txt", DST / "NOTICE.txt")]
    pairs += [(REPO / s, idx / n) for s, n in INDEX if (REPO / s).exists()]
    pairs += [(f, idx / "figures" / f.name) for f in sorted((REPO / "docs/agtf30_v11/figures").glob("*.png"))]
    # all eight subsets (v1.0 + v1.1) as one dataset: statistics, figures and the 480-engine baseline
    allx = REPO / "docs/agtf30_all"
    pairs += [(allx / n, idx / "all_subsets" / n) for n in ("stats.json", "stats_tables.md", "baseline.json")]
    pairs += [(f, idx / "all_subsets" / "figures" / f.name) for f in sorted((allx / "figures").glob("*.png"))]
    # drop analysis of the v1.1 run (gen_drops.py on AI Panther, job 242769)
    pairs += [(REPO / "docs/agtf30_v11/drops_analysis.json", DST / "provenance" / "drops_analysis.json")]
    (idx / "all_subsets" / "figures").mkdir(parents=True, exist_ok=True)
    for s, t in pairs:
        if t.exists():
            os.chmod(t, 0o666)
        shutil.copyfile(s, t)
    print("STAGE_DONE", len(pairs), "files")


def release_files():
    return sorted((p for p in DST.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"),
                  key=lambda p: p.relative_to(DST).as_posix())


def read_sums() -> dict:
    s = DST / "SHA256SUMS.txt"
    return {} if not s.exists() else {ln.split(" *", 1)[1]: ln.split(" *", 1)[0]
                                      for ln in s.read_text().splitlines() if " *" in ln}


def hash_all(rehash_h5=False):
    set_priority("idle")
    old = read_sums()
    lines = []
    for p in release_files():
        rel = p.relative_to(DST).as_posix()
        h = old.get(rel) if p.suffix == ".h5" and old.get(rel) and not rehash_h5 else E1.sha256(p)
        lines.append(f"{h} *{rel}")
        print(lines[-1], flush=True)
    s = DST / "SHA256SUMS.txt"
    if s.exists():
        os.chmod(s, 0o666)
    s.write_text("\n".join(lines) + "\n", newline="\n")      # LF, else sha256sum -c fails
    print("HASH_DONE", len(lines))


def lock():
    for p in DST.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    print("LOCK_DONE")


def verify():
    set_priority("idle")
    sums = read_sums()
    bad = [rel for rel, h in sums.items() if E1.sha256(DST / rel) != h]
    extra = [p.relative_to(DST).as_posix() for p in release_files() if p.relative_to(DST).as_posix() not in sums]
    print(json.dumps({"files": len(sums), "mismatch": bad, "not_listed": extra}))
    print("VERIFY_OK" if not bad and not extra else "VERIFY_FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("attrs", "manifest", "stage", "hash", "rehash-h5", "lock", "verify"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.attrs:
        attrs()
    if a.manifest:
        manifest()
    if a.stage:
        stage()
    if a.hash or a.rehash_h5:
        hash_all(a.rehash_h5)
    if a.lock:
        lock()
    if a.verify:
        verify()
