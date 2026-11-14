#!/usr/bin/env python3
"""E6: release check of the AGTF30 dataset v1.0 (phase E order).

    python e6_release_check.py --stage            # copy card, index files, figures into <dst>/index; re-hash (h5 kept)
    <fresh venv>/python e6_release_check.py --check --out $AGTF30_WORK/agtf30_v1_work/e6

--check is meant for a fresh virtual environment with only numpy, h5py and torch (the versions of E4). It does not
import the project package: learn_check imports only agtf30_data and rul_metrics from scripts/agtf30_learn. Steps, each
written to release_check.json as soon as it ends:
  1. checksums: re-hash every file listed in SHA256SUMS.txt (the .h5 included) and list files that are not listed;
  2. h5py: every file opens; names, shapes and dtypes equal MANIFEST.json; every dataset is read completely
     (decompression check); non-finite values outside W31/W32; values of RegNum and PLA_clip; rows per engine equal
     the cycles table's seconds;
  3. loader: agtf30_data.iter_units reads every engine of both splits (42 dev / 18 test per file); per engine
     cycles 1..eol, RUL = eol - cycle, RUL 0 on the last flight, rows = cycles-table seconds;
  4. reproduce one E4 number: the LSTM, 'res' features, seed 0 (learn_check's code and settings) -> test RMSE
     compared with docs/agtf30_v1 / learn.json (9.428493933515362);
  5. sizes: the release folder and the scratch folders under $AGTF30_WORK that could be deleted (nothing is deleted).
Runs at Idle CPU priority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import h5py
import numpy as np
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DST = pathlib.Path(f"{WORK}/agtf30_v1")
LEARN_JSON = pathlib.Path(f"{WORK}/agtf30_v1_work/e4/learn.json")
GROUPS = ["W", "X_s", "X_s_clean", "X_v", "X_a", "T", "A"]
INDEX = [("docs/AGTF30 dataset card.md", "AGTF30 dataset card.md"),
         ("configs/agtf30_v1/scaling_subsets.json", "scaling_subsets.json"),
         ("configs/agtf30_v1/scaling_subsets.csv", "scaling_subsets.csv"),
         ("docs/agtf30_v1/stats.json", "stats.json"), ("docs/agtf30_v1/stats_tables.md", "stats_tables.md"),
         ("docs/agtf30_v1/baseline.json", "baseline.json"),
         ("docs/agtf30_v1/release_check.json", "release_check.json")]
SCRATCH = [f"{WORK}/agtf30_c/main/runs", f"{WORK}/agtf30_c/main/h5", f"{WORK}/agtf30_c/main/work",
           f"{WORK}/agtf30_c/main/judge", f"{WORK}/agtf30_c/main/learn", f"{WORK}/agtf30_c/pilot", f"{WORK}/agtf30_c/dry",
           f"{WORK}/agtf30_c/library", f"{WORK}/agtf30_c", f"{WORK}/agtf30_v1_work", f"{WORK}/agtf30_b"]


def set_idle():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetPriorityClass.restype = wintypes.BOOL
        k.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k.GetPriorityClass.restype = wintypes.DWORD
        if not k.SetPriorityClass(k.GetCurrentProcess(), 0x40):
            raise OSError("SetPriorityClass failed")
        return hex(k.GetPriorityClass(k.GetCurrentProcess()))
    os.nice(19)
    return "nice 19"


def sha256(p: pathlib.Path) -> str:
    m = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            m.update(b)
    return m.hexdigest()


def read_sums(dst) -> dict:
    return {ln.split(" *", 1)[1]: ln.split(" *", 1)[0]
            for ln in (dst / "SHA256SUMS.txt").read_text().splitlines() if " *" in ln}


def stage(dst):
    idx = dst / "index"
    (idx / "figures").mkdir(parents=True, exist_ok=True)
    for src, name in INDEX:
        t = idx / name
        if t.exists():
            os.chmod(t, 0o666)
        shutil.copyfile(REPO / src, t)
    for f in sorted((REPO / "docs/agtf30_v1/figures").glob("*.png")):
        t = idx / "figures" / f.name
        if t.exists():
            os.chmod(t, 0o666)
        shutil.copyfile(f, t)
    old = read_sums(dst)
    lines = []
    for p in sorted((p for p in dst.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"),
                    key=lambda p: p.relative_to(dst).as_posix()):
        rel = p.relative_to(dst).as_posix()
        lines.append(f"{old[rel] if p.suffix == '.h5' else sha256(p)} *{rel}")   # h5 hashes kept from E1
    s = dst / "SHA256SUMS.txt"
    if s.exists():
        os.chmod(s, 0o666)
    s.write_text("\n".join(lines) + "\n", newline="\n")   # LF, else sha256sum -c fails
    print("STAGE_DONE", len(lines), "files listed")


def lock(dst):
    for p in dst.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)
    print("LOCK_DONE")


def dir_size(path: pathlib.Path):
    n = b = 0
    for root, _, fs in os.walk(path):
        for f in fs:
            try:
                b += os.path.getsize(os.path.join(root, f)); n += 1
            except OSError:
                pass
    return {"files": n, "bytes": b, "GiB": round(b / 2 ** 30, 2)}


def load_learn_check():
    """Import learn_check from scripts/agtf30_learn (numpy, h5py, torch only)."""
    sys.path.insert(0, str(REPO / "scripts/agtf30_learn"))
    import learn_check  # noqa: E402
    return learn_check


def check(dst, out):
    out.mkdir(parents=True, exist_ok=True)
    R = {"python": sys.version, "executable": sys.executable, "venv": sys.prefix != sys.base_prefix,
         "user_site_enabled": bool(__import__("site").ENABLE_USER_SITE), "priority": set_idle(),
         "numpy": np.__version__, "h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version}

    def save():
        (out / "release_check.json").write_text(json.dumps(R, indent=1))
    save()
    files = sorted(dst.glob("AGTF30_*.h5"))
    M = json.loads((dst / "MANIFEST.json").read_text())

    # 1. checksums
    t0 = time.time()
    sums = read_sums(dst)
    listed = set(sums)
    present = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"}
    bad = [rel for rel in sorted(sums) if sha256(dst / rel) != sums[rel]]
    R["checksums"] = {"listed": len(sums), "mismatch": bad, "not_listed": sorted(present - listed),
                      "missing": sorted(listed - present), "ok": not bad and present == listed,
                      "wall_s": round(time.time() - t0)}
    print("checksums", R["checksums"]["ok"], flush=True); save()

    # 2. h5py full read
    t0 = time.time()
    H = {}
    for f in files:
        mf = M["files"][f.name]
        r = {"ok": True, "problems": []}
        with h5py.File(f, "r") as h:
            r["version"] = str(h.attrs.get("dataset_version"))
            if r["version"] != str(M["version"]):          # MANIFEST.json version (1.0 or 1.1)
                r["problems"].append("dataset_version")
            for g in GROUPS:
                names = [x.decode() for x in h[f"{g}_var"][:]]
                if names != mf["columns"][g]:
                    r["problems"].append(f"{g}_var names")
            for sp in ("dev", "test"):
                nrow = mf["rows"][sp]
                for g in GROUPS + ["Y"]:
                    d = h[f"{g}_{sp}"]
                    want = (nrow, 1 if g == "Y" else len(mf["columns"][g]))
                    if d.shape != want or d.dtype != (np.int64 if g == "Y" else np.float64):
                        r["problems"].append(f"{g}_{sp} shape/dtype {d.shape} {d.dtype}")
                nonfin = {}
                reg, clip, ymin, ymax = set(), 0, None, None
                unit_rows = {}
                for g in GROUPS + ["Y"]:
                    d = h[f"{g}_{sp}"]
                    bad_cols = np.zeros(d.shape[1], np.int64)
                    for s in range(0, d.shape[0], 1 << 20):
                        x = d[s:s + (1 << 20)]
                        bad_cols += (~np.isfinite(x)).sum(0) if x.dtype.kind == "f" else 0
                        if g == "X_a":
                            names = mf["columns"]["X_a"]
                            reg.update(np.unique(x[:, names.index("RegNum")]).tolist())
                            clip += int(x[:, names.index("PLA_clip")].sum())
                        if g == "Y":
                            ymin = int(x.min()) if ymin is None else min(ymin, int(x.min()))
                            ymax = int(x.max()) if ymax is None else max(ymax, int(x.max()))
                        if g == "A":
                            u, c = np.unique(x[:, 0].astype(int), return_counts=True)
                            for a, b in zip(u, c):
                                unit_rows[int(a)] = unit_rows.get(int(a), 0) + int(b)
                    names = ["RUL"] if g == "Y" else mf["columns"][g]
                    for k, v in zip(names, bad_cols):
                        if v:
                            nonfin[f"{g}.{k}"] = int(v)
                cyc = h["cycles"][:]
                cyc = cyc[cyc["split"] == sp.encode()]
                sec = {int(u): int(cyc["seconds"][cyc["unit"] == u].sum()) for u in np.unique(cyc["unit"])}
                allowed = {k for k in nonfin if k in ("X_v.W31", "X_v.W32") and nonfin[k] == nrow}
                if set(nonfin) - allowed:
                    r["problems"].append(f"{sp} non-finite {sorted(set(nonfin) - allowed)}")
                if unit_rows != sec:
                    r["problems"].append(f"{sp} rows per unit differ from cycles.seconds")
                if sorted(unit_rows) != sorted(mf[f"{sp}_units"]):
                    r["problems"].append(f"{sp} unit ids differ from MANIFEST")
                r[sp] = {"rows": nrow, "units": len(unit_rows), "nonfinite": nonfin, "RegNum_values": sorted(reg),
                         "PLA_clip_seconds": clip, "RUL_min_max": [ymin, ymax]}
            r["events_rows"] = int(h["events"].shape[0])
            r["units_rows"] = int(h["units"].shape[0])
        r["ok"] = not r["problems"]
        H[f.name] = r
        print("h5py", f.name, r["ok"], r["problems"], f"{time.time() - t0:.0f}s", flush=True)
    R["h5py"] = {"files": H, "ok": all(v["ok"] for v in H.values()), "wall_s": round(time.time() - t0)}
    save()

    # 3. loader (+ features for 4.)
    t0 = time.time()
    import torch
    torch.set_num_threads(4)
    R["torch"] = torch.__version__
    LC = load_learn_check()
    A = LC  # names used by learn_check: Normaliser, iter_units, flight_features, ...
    eol = {}
    for f in files:
        with h5py.File(f, "r") as h:
            for row in h["units"][:]:
                eol[f"{f.stem}:{int(row['unit'])}"] = int(row["eol"])
            cyc = h["cycles"][:]
            for u in np.unique(cyc["unit"]):
                eol[f"{f.stem}:{int(u)}:rows"] = int(cyc["seconds"][cyc["unit"] == u].sum())
    norm = A.Normaliser().fit(A.iter_units(files, "dev", "X_s", max_cycle=10))
    names = A.feature_names(next(A.iter_units(files, "dev", "X_s", max_cycle=1))[1])
    F, L = {"dev": {}, "test": {}}, {"problems": []}
    for sp in ("dev", "test"):
        per_file = {}
        for k, u in A.iter_units(files, sp, "X_s"):
            c, rul = u["cycle"], u["rul"]
            cu = np.unique(c)
            e = eol[k]
            if not (cu[0] == 1 and cu[-1] == e and len(cu) == e and np.all(rul == e - c) and rul[-1] == 0
                    and len(c) == eol[k + ":rows"]):
                L["problems"].append(k)
            per_file[u["file"]] = per_file.get(u["file"], 0) + 1
            F[sp][k] = A.flight_features(u, norm)
        L[sp] = per_file
        if set(per_file.values()) != {42 if sp == "dev" else 18}:
            L["problems"].append(f"{sp} units per file {per_file}")
    L["ok"] = not L["problems"]
    L["wall_s"] = round(time.time() - t0)
    R["loader"] = L
    print("loader", L["ok"], L["problems"][:5], flush=True); save()

    # 4. reproduce the E4 LSTM seed 0 ('res')
    t0 = time.time()
    cols = A.select(None, names, "res")
    Xtr, ytr, _, _ = A.make_windows(F["dev"], 10, cols)
    Xte, yte, ute, _ = A.make_windows(F["test"], 10, cols)
    sc = A.Scaler().fit(Xtr)
    Xtr, Xte = sc(Xtr), sc(Xte)
    torch.manual_seed(0)
    m = LC.RNN(Xtr.shape[-1], "lstm")
    LC.train(m, Xtr, ytr, 60, 0)
    p = LC.predict(m, Xte)
    got = LC.metrics(yte, p, ute)
    ref = json.loads(LEARN_JSON.read_text())["sets"]["res"]
    ref0 = next(r for r in ref["models"]["lstm"]["runs"] if r["seed"] == 0)
    base = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    R["reproduce"] = {"what": f"LSTM, res features, seed 0, all dev / test engines ({LEARN_JSON}), window 10, 60 epochs",
                      "rmse": got["rmse"], "rmse_e4": ref0["rmse"], "abs_diff": abs(got["rmse"] - ref0["rmse"]),
                      "score": got["score"], "score_e4": ref0["score"], "n_windows": got["n"],
                      "baseline_rmse": base, "baseline_rmse_e4": ref["baseline_train_mean"]["rmse"],
                      "ok": abs(got["rmse"] - ref0["rmse"]) < 0.01, "wall_s": round(time.time() - t0)}
    print("reproduce", R["reproduce"]["rmse"], R["reproduce"]["rmse_e4"], flush=True); save()

    # 5. sizes
    R["sizes"] = {"release": dir_size(dst), "release_h5_bytes": sum(f.stat().st_size for f in files),
                  "scratch": {s: dir_size(pathlib.Path(s)) for s in SCRATCH if pathlib.Path(s).exists()},
                  "disk_free_work_GiB": round(shutil.disk_usage(WORK).free / 2 ** 30, 1)}
    R["ok"] = all(R[k]["ok"] for k in ("checksums", "h5py", "loader", "reproduce"))
    save()
    print("E6_DONE ok" if R["ok"] else "E6_DONE FAILED", flush=True)


def main(argv=None):
    global LEARN_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--out", default=f"{WORK}/agtf30_v1_work/e6")
    ap.add_argument("--learn-json", default=str(LEARN_JSON), help="E4 learn.json to reproduce (v1.1: $AGTF30_WORK/agtf30_v11_work/e4/learn.json)")
    a = ap.parse_args(argv)
    LEARN_JSON = pathlib.Path(a.learn_json)
    dst = pathlib.Path(a.dst)
    if a.stage:
        stage(dst)
    if a.check:
        check(dst, pathlib.Path(a.out))
    if a.lock:
        lock(dst)


if __name__ == "__main__":
    main()
