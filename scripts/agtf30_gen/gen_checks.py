#!/usr/bin/env python3
"""C6 dry-run checks on merged files (and later the pilot).

    python gen_checks.py --out $AGTF30_WORK/agtf30_c/dry [--ref DS01-005] [--rerun HPT:0:5]

1. files open with h5py and load with ml_pipeline.data_loader.ncmapss_loader (unchanged) for dev and
   test; loader columns and dtypes equal those of a real N-CMAPSS file (--ref, test split). The loader belongs
   to the original project and is not part of this code release: when it cannot be imported this step is
   skipped and recorded as skipped (the released files were checked with it, see provenance/checks.json);
   raw names / column order / dtypes of W, X_s, X_v, T, Y equal N-CMAPSS's;
2. no NaN or inf except X_v W31/W32 (which must be all NaN);
3. per unit: HI(1) = the new-engine-with-build-wear value (0.91-0.98 in B5), HI > 0 before the last
   flight, HI <= 0 on the last; HI never rises; RUL = eol - cycle in Y, A and the cycles table;
   life equals the B5 reference-only simulation (b5_life.engine, unchanged) fed with this engine's
   sampled parameters and real flight hours;
4. X_s - X_s_clean: sigma / rms(clean) per channel = 10^(-65/20); X_s_clean equals the runner output;
5. determinism: re-run the first N flights of one engine into a fresh folder -> bitwise identical;
6. speed (flight-hours per wall hour) and compressed size per flight-hour, with pilot/main projections.
Writes <out>/checks.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import warnings

import h5py
import numpy as np
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "agtf30_recipe")); sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
import gen_recipe as R  # noqa: E402
from gen_engine import DYN_KEEP  # noqa: E402
from gen_merge import SIG_REL, X_S  # noqa: E402

NC = pathlib.Path(NCMAPSS)
PILOT_H, MAIN_H = 9000.0, 54000.0          # recipe section 9
BLOCK_ROWS = 2_000_000


class FedRng:
    """Stands in for numpy's Generator inside b5_life.engine: returns this engine's values in b5's draw order."""

    def __init__(self, E, hours):
        self.q = [E["u"][m] for m in ("FAN", "LPC", "HPC", "HPT", "LPT")] + [E["r_bg"], E["a"], E["b"]]
        self.Es, self.h = E["E_s"], list(hours)
        self.k = 0

    def uniform(self, lo, hi):
        if self.q:
            return self.q.pop(0)
        v = self.h[self.k]; self.k += 1
        return v

    def normal(self, mu, sd):
        return self.Es


def check_raw_layout(path, ref):
    out = {}
    with h5py.File(path, "r") as h, h5py.File(ref, "r") as r:
        for g in ("A", "W", "X_s", "X_v", "T"):
            a = [x.decode() for x in h[f"{g}_var"][:]]
            b = [x.decode() for x in r[f"{g}_var"][:]]
            out[f"{g}_names_equal"] = a == b if g != "A" else a[:4] == b
        for g in ("W", "X_s", "X_v", "T", "Y", "A"):
            for sp in ("dev", "test"):
                out[f"{g}_{sp}_dtype_equal"] = h[f"{g}_{sp}"].dtype == r[f"{g}_{sp}"].dtype
                ncol_ok = h[f"{g}_{sp}"].shape[1] == r[f"{g}_{sp}"].shape[1] or g == "A"
                out[f"{g}_{sp}_ncols_equal"] = bool(ncol_ok)
    return out


def excerpt(src, dst, n_units):
    """Copy of `src` holding only the first n_units units of each split (all other datasets and attributes whole):
    the project loader reads a whole split into one DataFrame, which does not fit in memory for a main-run file."""
    with h5py.File(src, "r") as h, h5py.File(dst, "w") as o:
        o.attrs.update(dict(h.attrs))
        ends = {}
        for sp in ("dev", "test"):
            ucol = h[f"A_{sp}"][:, 0]
            st = np.flatnonzero(np.r_[True, np.diff(ucol) != 0]) if len(ucol) else np.zeros(0, int)
            ends[sp] = int(st[n_units]) if len(st) > n_units else len(ucol)
        for k in h:
            sp = k.rsplit("_", 1)[-1]
            if sp in ends:
                o.create_dataset(k, data=h[k][:ends[sp]])
            else:
                h.copy(k, o)


def check_loader(paths, ref_key, n_units=None):
    try:
        from ml_pipeline.data_loader.ncmapss_loader import NCMAPSSLoader
    except ImportError as e:   # optional: the original project's loader is not in this release
        print("loader check skipped:", e, flush=True)
        return {"skipped": f"ml_pipeline loader not installed ({e})"}
    L = NCMAPSSLoader()
    tmp = pathlib.Path(tempfile.mkdtemp(dir=os.environ.get("AGTF30_TMP") or None))
    res = {}
    for p in paths:
        key = "AGTF30-" + p.stem.split("_")[1]
        dst = tmp / f"N-CMAPSS_{key}.h5"
        if n_units:
            excerpt(p, dst, n_units)
        else:
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy(p, dst)
        for sp in ("dev", "test"):
            with h5py.File(p, "r") as h:
                if h[f"A_{sp}"].shape[0] == 0:      # the loader cannot take an empty split (dry run only)
                    res[f"{key}_{sp}"] = {"rows": 0, "skipped": "empty split"}
                    continue
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                df = L.load(tmp, datasets=[key], split=sp)
            res[f"{key}_{sp}"] = {"rows": len(df), "units": int(df["unit_id"].nunique()),
                                  "columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
    ref = L.load(NC, datasets=[ref_key], split="test")
    rc, rt = list(ref.columns), {c: str(t) for c, t in ref.dtypes.items()}
    for k, v in res.items():
        if "skipped" in v:
            continue
        v["columns_equal_ncmapss"] = v["columns"] == rc
        v["dtypes_equal_ncmapss"] = v["dtypes"] == rt
        del v["columns"], v["dtypes"]
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def check_file(path, out):
    import b5_life
    from gen_ref import ss_point_thrust_exact
    b5_life.ss_point_thrust = ss_point_thrust_exact     # the same exact-thrust reference as the generator (gen_ref)
    b5_engine = b5_life.engine
    res = {"file": path.name}
    with h5py.File(path, "r") as h:
        xv = [x.decode() for x in h["X_v_var"][:]]
        bad = {}
        for k in h:
            if not (k.endswith("_dev") or k.endswith("_test")):
                continue
            nbad, w31 = 0, True
            for r0 in range(0, h[k].shape[0], BLOCK_ROWS):      # blocks: a main-run split is ~40 M rows
                X = h[k][r0:r0 + BLOCK_ROWS]
                nf = ~np.isfinite(X)
                if k.startswith("X_v_"):
                    w = [xv.index("W31"), xv.index("W32")]
                    w31 &= bool(np.isnan(X[:, w]).all())
                    nf[:, w] = False
                nbad += int(nf.sum())
            if k.startswith("X_v_"):
                res[f"{k}_W31_W32_all_nan"] = w31
            if nbad:
                bad[k] = nbad
        res["nonfinite_outside_W31_W32"] = bad
        cyc = h["cycles"][:]
        units = h["units"][:]
        noise = {}
        per_unit = []
        slices = {}                                          # split -> unit -> (first row, end row); units are contiguous
        for sp in ("dev", "test"):
            ucol = h[f"A_{sp}"][:, 0]
            st = np.flatnonzero(np.r_[True, np.diff(ucol) != 0]) if len(ucol) else np.zeros(0, int)
            en = np.r_[st[1:], len(ucol)]
            slices[sp] = {int(ucol[s]): (int(s), int(e)) for s, e in zip(st, en)}
            assert len(slices[sp]) == len(set(ucol.tolist())), f"{sp}: unit rows are not contiguous"
            del ucol
        for u in units:
            sp = u["split"].decode()
            m = cyc["unit"] == u["unit"]
            cu = np.sort(cyc[m], order="cycle")
            hi = cu["HI"]
            s0, s1 = slices[sp][int(u["unit"])]
            rows = slice(s0, s1)
            eol = int(u["eol"])
            Arow, Yrow = h[f"A_{sp}"][rows], h[f"Y_{sp}"][rows, 0]
            d = out / "runs" / u["family"].decode() / f"e{int(u['engine_index']):04d}"
            E = json.loads((d / "engine.json").read_text())
            hours = [json.loads((d / f"c{c:04d}.json").read_text())["hours"] for c in range(1, eol + 1)]
            P = {"K": {E["family"]: E["K"]}, "lam": R.LAM, "s_init": R.S_INIT, "L_bg": R.L_BG, "tmax": eol + 40}
            b5 = b5_engine(E["family"], 0, P, FedRng(E, hours + [hours[-1]] * 40))
            r = {"unit": int(u["unit"]), "family": E["family"], "split": sp, "Fc": int(u["Fc"]), "eol": eol,
                 "onset": int(u["onset"]), "hi0": float(u["hi0"]), "hi_first": float(hi[0]), "hi_last": float(hi[-1]),
                 "hi_positive_before_last": bool(np.all(hi[:-1] > 0)), "hi_last_le_0": bool(hi[-1] <= 0),
                 "hi_never_rises": bool(np.all(np.diff(hi) <= 1e-12)),
                 "max_hi_rise": float(np.max(np.diff(hi))) if len(hi) > 1 else 0.0,
                 "cycles_1_to_eol": bool(np.array_equal(cu["cycle"], np.arange(1, eol + 1))),
                 "rul_table_ok": bool(np.array_equal(cu["RUL"], eol - cu["cycle"])),
                 "rul_rows_ok": bool(np.array_equal(Yrow, eol - Arow[:, 1].astype(int))),
                 "rows_equal_seconds": s1 - s0 == int(cu["seconds"].sum()),
                 "b5_eol": int(b5["eol"]), "b5_hi_last": b5["hist"][-1]["hi"],
                 "life_equals_b5": int(b5["eol"]) == eol,
                 "hi_max_abs_diff_b5": float(np.max(np.abs(np.array([x["hi"] for x in b5["hist"][1:]][:len(hi)])
                                                           - hi[:len(b5["hist"]) - 1]))),
                 "hours": float(cu["hours"].sum()), "limiter_s": int(cu["limiter_s"].sum()),
                 "clipped_s": int(cu["clipped_s"].sum()), "cap_s": int(cu["cap_s"].sum()),
                 "half_step_flights": int((cu["dyn_step_s"] < 0.01).sum()), "ending_limit": u["ending_limit"].decode()}
            per_unit.append(r)
            # noise: per flight sigma / rms over the unit's flights, and X_s_clean = runner output (cycle 1)
            Xs, Xc = h[f"X_s_{sp}"][rows], h[f"X_s_clean_{sp}"][rows]
            cs = Arow[:, 1].astype(int)
            rel = []
            for c in np.unique(cs):
                k = cs == c
                rel.append(np.std(Xs[k] - Xc[k], axis=0) / np.sqrt(np.mean(Xc[k] ** 2, axis=0)))
            noise[int(u["unit"])] = np.array(rel)
            with np.load(d / "c0001.npz") as z:
                raw = z["Y"][:, [DYN_KEEP.index(k) for _, k in X_S]]
            r["X_s_clean_equals_runner"] = bool(np.array_equal(Xc[cs == 1], raw))
        allrel = np.concatenate(list(noise.values()))
        res["noise_rel_sigma_median_per_channel"] = dict(zip([n for n, _ in X_S], np.median(allrel, 0).round(7).tolist()))
        res["noise_rel_sigma_expected"] = SIG_REL
        res["noise_rel_sigma_range"] = [float(allrel.min()), float(allrel.max())]
        res["units"] = per_unit
    return res


def rerun_identical(out, spec):
    import gen_engine as G
    fam, slot, n = spec.split(":"); slot, n = int(slot), int(n)
    plan = json.loads((out / "plan.json").read_text())
    tmp = out / "rerun_check"
    shutil.rmtree(tmp, ignore_errors=True)
    cfg = dict(plan["cfg"], out=str(tmp))
    G.run_engine(cfg, fam, slot, slot, stop_after=n)
    a, b = G.engine_dir(plan["cfg"], fam, slot), G.engine_dir(cfg, fam, slot)
    same = []
    for c in range(1, n + 1):
        with np.load(a / f"c{c:04d}.npz") as za, np.load(b / f"c{c:04d}.npz") as zb:
            y = np.array_equal(za["Y"], zb["Y"]) and np.array_equal(za["flag"], zb["flag"])
        ja, jb = json.loads((a / f"c{c:04d}.json").read_text()), json.loads((b / f"c{c:04d}.json").read_text())
        keys = ["source_key", "source_cycle", "theta", "HI", "ref", "limiter_s", "clipped_s", "cap_s"]
        same.append({"cycle": c, "Y_bitwise": bool(y), "summary_equal": all(ja[k] == jb[k] for k in keys)})
    ea, eb = json.loads((a / "engine.json").read_text()), json.loads((b / "engine.json").read_text())
    return {"engine": spec, "engine_json_equal": ea == eb, "flights": same}


def speed_size(out, h5dir):
    log = (out / "run.log").read_text().splitlines()
    jobs = sorted(out.glob("job_*.json"))
    wall = sum(json.loads(j.read_text())["wall_s"] for j in jobs)
    secs, dyn, ref = 0, 0.0, 0.0
    for j in (out / "runs").glob("*/e*/c*.json"):
        s = json.loads(j.read_text())
        secs += s["seconds"]; dyn += s["wall_dyn_s"]; ref += s["wall_ref_s"]
    fh = secs / 3600
    merge = json.loads((h5dir / "merge.json").read_text())
    by = sum(r["bytes"] for r in merge)
    mh = sum(r["hours"] for r in merge)
    plan = json.loads((out / "plan.json").read_text())
    return {"flight_hours": round(fh, 2), "job_wall_h": round(wall / 3600, 3), "workers": plan["workers"],
            "flight_h_per_wall_h": round(fh / (wall / 3600), 1) if wall else None,
            "per_slot_flight_h_per_h": round(fh / (dyn + ref) * 3600, 1),
            "ref_share_of_compute": round(ref / (dyn + ref), 4),
            "bytes_per_flight_h_MB": round(by / mh / 1e6, 3), "filter": merge[0]["filter"],
            "note": "the job's wall time is set by its longest engine when engines < workers; per-slot rate x 12 "
                    "is the steady-state estimate",
            "log_lines": len(log)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--h5", default=None)
    ap.add_argument("--ref", default="DS01-005")
    ap.add_argument("--rerun", default=None, help="FAM:slot:flights")
    ap.add_argument("--skip-loader", action="store_true")
    ap.add_argument("--loader-units", type=int, default=None,
                    help="loader check on an excerpt of the first N units per split (files too large for memory)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    h5dir = pathlib.Path(a.h5) if a.h5 else out / "h5"
    files = sorted(h5dir.glob("AGTF30_*.h5"))
    C = {"files": [f.name for f in files]}
    C["raw_layout"] = {f.name: check_raw_layout(f, NC / f"N-CMAPSS_{a.ref}.h5") for f in files}
    if not a.skip_loader:
        C["loader"] = check_loader(files, a.ref, a.loader_units)
        C["loader_units_per_split"] = a.loader_units or "all"
    C["content"] = [check_file(f, out) for f in files]
    if a.rerun:
        C["rerun"] = rerun_identical(out, a.rerun)
    C["speed_size"] = speed_size(out, h5dir)
    (out / "checks.json").write_text(json.dumps(C, indent=1, default=str))
    print(json.dumps(C, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
