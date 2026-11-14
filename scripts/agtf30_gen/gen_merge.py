#!/usr/bin/env python3
"""C4: merge the per-flight checkpoints of a run into one .h5 per subset (B_RECIPE section 7).

    python gen_merge.py --out $AGTF30_WORK/agtf30_c/dry [--filter gzip4] [--fams HPT,...]
    python gen_merge.py --out $AGTF30_WORK/agtf30_c/dry --compare-filters HPT   # size / read speed per filter
    python gen_merge.py --fix-units AGTF30_HPTLPT.h5 ...                   # repair a file merged with the old
                                                                           # 4-byte family field (rewrites `units` only)

File <out>/h5/AGTF30_<FAM>.h5, N-CMAPSS layout and names (float64; Y int64 as in N-CMAPSS):
  W, X_s, X_v, T, Y, A  _dev / _test, and A_var W_var X_s_var X_v_var T_var (N-CMAPSS column order);
  X_s   = 65 dB Gaussian noise on the 14 measured channels (sigma = 10^(-65/20) * rms of the clean
          signal over the flight, per channel and flight; N-CMAPSS paper Sec. 3.3); X_s_clean = no noise;
  X_v   W31, W32 = NaN (AGTF30 has no cooling-bleed split);
  A     unit, cycle, Fc, hs, cycles_since_shop_visit, shop_visits (v1: = cycle, 0);
  X_a   AGTF30 extras (+ X_a_var);
  tables `units`, `cycles` (one row per unit and flight) and `events` (empty in v1);
  attributes: recipe version/id, git commit, C build id (sha256 of the runners), noise rule.
Rows are ordered by unit then cycle. Units are numbered 1..n by slot (dev and test share the
numbering, like N-CMAPSS). Only finished engines (done.json) are merged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_recipe as R  # noqa: E402
from gen_engine import DYN_KEEP  # noqa: E402

SNR_DB = 65.0
SIG_REL = 10 ** (-SNR_DB / 20)
A_VAR = ["unit", "cycle", "Fc", "hs", "cycles_since_shop_visit", "shop_visits"]
W_VAR = ["alt", "Mach", "TRA", "T2"]
X_S = [("T24", "eng.S24.Tt"), ("T30", "eng.S36.Tt"), ("T48", "eng.S48.Tt"), ("T50", "eng.S5.Tt"),
       ("P15", "eng.S17.Pt"), ("P2", "eng.S2.Pt"), ("P21", "eng.S21.Pt"), ("P24", "eng.S24.Pt"),
       ("Ps30", "eng.S36.Ps"), ("P40", "eng.S4.Pt"), ("P50", "eng.S5.Pt"), ("Nf", "eng.Shaft.N_Fan"),
       ("Nc", "eng.Shaft.N_HPC"), ("Wf", "act.Wfact")]
X_V = [("T40", "eng.S4.Tt"), ("P30", "eng.S36.Pt"), ("P45", "eng.S45.Pt"), ("W21", "eng.S21.W"),
       ("W22", "eng.S24.W"), ("W25", "eng.S25.W"), ("W31", None), ("W32", None), ("W48", "eng.S48.W"),
       ("W50", "eng.S5.W"), ("SmFan", "eng.SM.SMFan"), ("SmLPC", "eng.SM.SMLPC"), ("SmHPC", "eng.SM.SMHPC"),
       ("phi", "phi")]
X_A = [("PLA", "in.PLA"), ("N_LPC", "eng.Shaft.N_LPC"), ("T45", "eng.S45.Tt"), ("T45_limit", "cntrl.T45Lim"),
       ("Fnet", "eng.Perf.Fnet"), ("TSFC", "eng.Perf.TSFC"), ("VBV", "act.VBVact"), ("VAFN", "act.VAFNact"),
       ("Wbleed_HPC", "eng.Data.HPC_Data.Wbleeds"), ("RegNum", "cntrl.RegNum"), ("N1_target", "cntrl.N1cReq"),
       ("PLA_clip", "flag")]
FILTERS = {"none": {}, "lzf": {"compression": "lzf", "shuffle": True},
           "gzip1": {"compression": "gzip", "compression_opts": 1, "shuffle": True},
           "gzip4": {"compression": "gzip", "compression_opts": 4, "shuffle": True},
           "gzip6": {"compression": "gzip", "compression_opts": 6, "shuffle": True},
           "gzip9": {"compression": "gzip", "compression_opts": 9, "shuffle": True}}
CHUNK_ROWS = 16384
EVENT_COLS = R.T_COLS

CYC_DT = np.dtype([("unit", "i4"), ("cycle", "i4"), ("split", "S4"), ("RUL", "i4"), ("hs", "i1"),
                   ("src_key", "S24"), ("src_cycle", "i4"), ("seconds", "i4"), ("hours", "f8"),
                   ("HI", "f8"), ("hi_T45", "f8"), ("hi_Fan", "f8"), ("hi_LPC", "f8"), ("hi_HPC", "f8"),
                   ("margin_T45", "f8"), ("margin_SMFan", "f8"), ("margin_SMLPC", "f8"), ("margin_SMHPC", "f8"),
                   ("limit", "S4"), ("ref_T45", "f8"), ("ref_N1", "f8"), ("ref_N3", "f8"), ("ref_Fnet", "f8"),
                   ("ref_retries", "i4"), ("g", "f8"), ("tau", "i4"), ("limiter_s", "i4"), ("clipped_s", "i4"),
                   ("cap_s", "i4"), ("dyn_step_s", "f8"), ("dyn_attempts", "i4"), ("rc", "i4")])
UNIT_DT = np.dtype([("unit", "i4"), ("slot", "i4"), ("engine_index", "i4"), ("split", "S4"), ("family", "S8"),
                    ("Fc", "i4"), ("u_FAN", "f8"), ("u_LPC", "f8"), ("u_HPC", "f8"), ("u_HPT", "f8"), ("u_LPT", "f8"),
                    ("r_bg", "f8"), ("a", "f8"), ("b", "f8"), ("E_s", "f8"), ("K", "f8"), ("onset", "i4"),
                    ("eol", "i4"), ("ending_limit", "S4"), ("hi0", "f8"), ("replacements", "i4"),
                    ("seed", "i8", (3,)), ("hours", "f8")])
FAMILY_WIDTH = UNIT_DT["family"].itemsize    # S8: v1.1 names (HPTLPT, LPCHPC) need 6; v1.0 files used S4
EVENT_DT = np.dtype([("unit", "i4"), ("before_cycle", "i4"), ("type", "S12")] +
                    [(c, "f8") for c in EVENT_COLS] + [("source", "S64")])


def _bytes(names):
    w = max(len(n) for n in names)
    return np.array([n.encode() for n in names], dtype=f"S{w}")


def build_id(cfg) -> dict:
    out = {}
    for k in ("bin", "bin_fine"):
        for exe in ("cb_dyn", "cb_ss"):
            p = list(pathlib.Path(cfg[k]).glob(exe + "*"))
            if p:
                out[f"{k}/{p[0].name}"] = hashlib.sha256(p[0].read_bytes()).hexdigest()[:16]
    return out


def git_commit() -> str:
    if os.environ.get("AGTF30_GIT_COMMIT"):        # cluster copy without a git checkout (v1.1 on AI Panther)
        return os.environ["AGTF30_GIT_COMMIT"]
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE, capture_output=True,
                               text=True).stdout.strip()
        return c + ("+dirty(scripts/agtf30_gen)" if dirty else "")
    except OSError:
        return "unknown"


def noise(engine: dict, cycle: int, clean: np.ndarray) -> np.ndarray:
    """65 dB Gaussian noise for one flight (child stream (2, cycle) of the engine's seed)."""
    rng = np.random.default_rng(R.seed_seq(engine["family"], engine["engine_index"], (2, int(cycle))))
    sig = np.sqrt(np.mean(clean ** 2, axis=0)) * SIG_REL
    return clean + rng.standard_normal(clean.shape) * sig


def flight_rows(d: pathlib.Path, E: dict, unit: int, c: int, eol: int):
    s = json.loads((d / f"c{c:04d}.json").read_text())
    with np.load(d / f"c{c:04d}.npz") as z:
        Y, flag = z["Y"], z["flag"]
    key = s["source_key"]
    col = {k: i for i, k in enumerate(DYN_KEEP)}
    n = len(Y)
    F = LIB.flight(key, s["source_cycle"])
    W = np.c_[F["alt"], F["mn"], F["tra"], Y[:, col["eng.S2.Tt"]]]
    Xc = np.c_[tuple(Y[:, col[k]] for _, k in X_S)]
    Xn = noise(E, c, Xc)

    def xv(k):
        if k is None:
            return np.full(n, np.nan)
        if k == "phi":
            return 3600 * Y[:, col["act.Wfact"]] / Y[:, col["eng.S36.Ps"]]
        return Y[:, col[k]]
    Xv = np.c_[tuple(xv(k) for _, k in X_V)]
    Xa = np.c_[tuple(flag.astype(float) if k == "flag" else Y[:, col[k]] for _, k in X_A)]
    T = np.tile(R.t_row(s["theta"]), (n, 1))
    A = np.tile([unit, c, E["cls"], s["hs"], c, 0], (n, 1)).astype(float)
    Yr = np.full((n, 1), eol - c, dtype=np.int64)
    ref = s["ref"]
    hp = ref["hi_parts"]
    cyc = (unit, c, E["split"].encode(), eol - c, s["hs"], key.encode(), s["source_cycle"], n, n / 3600, ref["HI"],
           hp["T45"], hp["Fan"], hp["LPC"], hp["HPC"], ref["margin_T45"], ref["margin_SMFan"], ref["margin_SMLPC"],
           ref["margin_SMHPC"], ref["limit"].encode(), ref["T45"], ref["N1"], ref["N3"], ref["Fnet"],
           ref.get("retries") or 0, s["g"], s["tau"], s["limiter_s"], s["clipped_s"], s["cap_s"], s["dyn_step_s"],
           len(s["dyn_attempts"]), 0)
    return {"W": W, "X_s": Xn, "X_s_clean": Xc, "X_v": Xv, "X_a": Xa, "T": T, "A": A, "Y": Yr}, cyc


def finished_engines(out: pathlib.Path, fam: str, n_per_fam: int):
    """slot -> (engine dir, done record, replacements)"""
    res = {}
    for slot in range(n_per_fam):
        for k in range(10):
            d = out / "runs" / fam / f"e{slot + k * R.REPLACE_STRIDE:04d}"
            if (d / "done.json").exists():
                res[slot] = (d, json.loads((d / "done.json").read_text()), k)
                break
            if not (d / "dropped.json").exists():
                break
    return res


def merge_family(out: pathlib.Path, fam: str, n_per_fam: int, cfg: dict, filt: str, path: pathlib.Path) -> dict:
    eng = finished_engines(out, fam, n_per_fam)
    # the split depends on the subset size (n_per_fam), not on the engine: engines run for a smaller set
    # (the pilot's slots 0-9 = the main run's slots 0-9, same seeds) are re-split for this merge
    splits = R.split_plan(fam, R.class_plan(fam, n_per_fam))
    t0 = time.time()
    groups = {"W": W_VAR, "X_s": [n for n, _ in X_S], "X_s_clean": [n for n, _ in X_S], "X_v": [n for n, _ in X_V],
              "X_a": [n for n, _ in X_A], "T": R.T_COLS, "A": A_VAR, "Y": None}
    cyc_rows, unit_rows = [], []
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h:
        for g, names in groups.items():
            if names is not None:
                h.create_dataset(f"{g}_var", data=_bytes(names))
            for sp in ("dev", "test"):
                ncol = 1 if names is None else len(names)
                h.create_dataset(f"{g}_{sp}", shape=(0, ncol), maxshape=(None, ncol),
                                 dtype=np.int64 if g == "Y" else np.float64, chunks=(CHUNK_ROWS, ncol),
                                 **FILTERS[filt])
        for sp in ("dev", "test"):
            for slot in sorted(eng):
                d, done, nrep = eng[slot]
                E = json.loads((d / "engine.json").read_text())
                E["split"] = splits[slot]
                if E["split"] != sp:
                    continue
                unit, eol = slot + 1, done["eol"]
                hrs = 0.0
                for c in range(1, eol + 1):
                    rows, cy = flight_rows(d, E, unit, c, eol)
                    for g, X in rows.items():
                        ds = h[f"{g}_{sp}"]
                        n0 = ds.shape[0]
                        ds.resize(n0 + len(X), axis=0)
                        ds[n0:] = X
                    cyc_rows.append(cy)
                    hrs += cy[8]
                unit_rows.append((unit, slot, E["engine_index"], sp.encode(), fam.encode(), E["cls"],
                                  *[E["u"][m] for m in ("FAN", "LPC", "HPC", "HPT", "LPT")], E["r_bg"], E["a"], E["b"],
                                  E["E_s"], E["K"], E["onset"] if E["onset"] is not None else -1, eol,
                                  done["limit"].encode(), E["hi0"], nrep, E["seed_entropy"], hrs))
        h.create_dataset("cycles", data=np.array(cyc_rows, dtype=CYC_DT), **({} if filt == "none" else FILTERS["gzip4"]))
        h.create_dataset("units", data=np.array(unit_rows, dtype=UNIT_DT))
        h.create_dataset("events", data=np.zeros(0, dtype=EVENT_DT), maxshape=(None,), chunks=(64,))
        h.attrs.update({"recipe_version": R.RECIPE_VERSION if fam not in R.V11_FAMILIES
                        else R.RECIPE_VERSION + "; " + R.V11_RECIPE, "recipe_id": R.RECIPE_ID, "family": fam,
                        "git_commit": git_commit(), "c_build_id": json.dumps(build_id(cfg)),
                        "created": dt.datetime.now().isoformat(timespec="seconds"),
                        "noise": f"X_s: Gaussian, sigma = 10^(-{SNR_DB:g}/20) * rms(clean flight signal), per channel"
                                 " and flight; X_s_clean: none",
                        "W31_W32": "NaN: AGTF30 outputs only the total HPC bleed (X_a Wbleed_HPC)",
                        "units_note": "imperial as N-CMAPSS (degR, psia, rpm, lbm/s); P15 = AGTF30 S17.Pt; T30/Ps30/P30 ="
                                      " AGTF30 station 36",
                        "filter": filt, "generator": "scripts/agtf30_gen", "n_units": len(unit_rows),
                        "units_missing": json.dumps(sorted(set(range(n_per_fam)) - set(eng)))})
    tmp.replace(path)
    return {"family": fam, "units": len(unit_rows), "missing_slots": sorted(set(range(n_per_fam)) - set(eng)),
            "flights": len(cyc_rows), "hours": round(sum(r[8] for r in cyc_rows), 2), "bytes": path.stat().st_size,
            "wall_s": time.time() - t0, "filter": filt}


def read_speed(path: pathlib.Path) -> float:
    t0 = time.time()
    with h5py.File(path, "r") as h:
        for k in h:
            if k.endswith("_dev") or k.endswith("_test"):
                _ = h[k][:]
    return time.time() - t0


def fix_units_family(path: pathlib.Path) -> dict:
    """Rewrite the `units` table of a file merged before the family field was widened (S4 truncated HPTLPT to
    'HPTL' and LPCHPC to 'LPCH'). The full name is taken from the file attribute `family`; every other field and
    every signal dataset is left unchanged. Files whose names already fit are not touched."""
    with h5py.File(path, "r+") as h:
        fam = h.attrs["family"]
        fam = fam.decode() if isinstance(fam, bytes) else str(fam)
        old = h["units"][:]
        have = {v.decode() for v in old["family"]}
        if old.dtype["family"].itemsize >= len(fam) and have <= {fam}:
            return {"file": str(path), "family": fam, "changed": False, "had": sorted(have)}
        if have - {fam[:old.dtype["family"].itemsize]}:
            raise ValueError(f"{path}: units.family {sorted(have)} does not match file family {fam!r}")
        new = np.zeros(len(old), dtype=UNIT_DT)
        for k in old.dtype.names:
            new[k] = old[k]
        new["family"] = fam.encode()
        del h["units"]
        h.create_dataset("units", data=new)
        h.attrs["units_family_fix"] = (f"units.family widened S{old.dtype['family'].itemsize} -> S{FAMILY_WIDTH} "
                                       f"({sorted(have)} -> {fam!r}) on {dt.datetime.now().isoformat(timespec='seconds')}")
    return {"file": str(path), "family": fam, "changed": True, "had": sorted(have)}


LIB = None


def main(argv=None):
    global LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--fams", default="HPT,HPC,LPT,FAN,MIX")
    ap.add_argument("--filter", default="gzip1", choices=list(FILTERS))
    ap.add_argument("--compare-filters", default=None, help="family to write with every filter")
    ap.add_argument("--fix-units", nargs="+", default=None, metavar="H5", help="repair truncated units.family")
    a = ap.parse_args(argv)
    if a.fix_units:
        for f in a.fix_units:
            print(json.dumps(fix_units_family(pathlib.Path(f))), flush=True)
        return
    if not a.out:
        ap.error("--out is required")
    out = pathlib.Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    cfg, n = plan["cfg"], plan["n_per_fam"]
    import gen_engine as G
    LIB = G.Library(cfg["lib"])
    if a.compare_filters:
        res = []
        for f in FILTERS:
            p = out / "h5_filters" / f"AGTF30_{a.compare_filters}_{f}.h5"
            r = merge_family(out, a.compare_filters, n, cfg, f, p)
            r["read_s"] = min(read_speed(p) for _ in range(2))
            res.append(r)
            print(json.dumps(r), flush=True)
        (out / "h5_filters" / "filters.json").write_text(json.dumps(res, indent=1))
        return
    res = [merge_family(out, fam, n, cfg, a.filter, out / "h5" / f"AGTF30_{fam}.h5") for fam in a.fams.split(",")]
    for r in res:
        print(json.dumps(r), flush=True)
    (out / "h5" / "merge.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
