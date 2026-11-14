#!/usr/bin/env python3
"""Compare a C-runner output (dyn.bin) with the A1 Normal-mode reference (ref_<case>.mat).

    cb_compare_ref.py --ref ref_healthy.mat --run <run dir> [--csv out.csv] [--tol 1e-6]

Every runner column ``x`` (``x[k]`` for vector element k) is matched to the
reference signal ``out_Dyn.x`` (column k). Full-rate runs (-r full) are compared
sample by sample (time vectors must be identical); 1 Hz runs are compared with
the reference sampled the campaign way: last native sample at or before each
integer second. Metrics per channel: max_abs, max_rel = max_abs / max|ref|,
max_rel_pointwise = max |d| / max(|ref|, 1e-6). Pass rule: max_rel <= tol.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cb_run import read_out  # noqa: E402


def load_ref(file: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    h = h5py.File(file, "r")
    refs = lambda k: [h[r] for r in np.array(h[k]).ravel()]
    names = ["".join(chr(c) for c in np.array(x).ravel()) for x in refs("names")]
    times, datas = refs("time"), refs("data")
    return {n: (t, d) for n, t, d in zip(names, times, datas)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(argv)
    R = load_ref(a.ref)
    run = pathlib.Path(a.run)
    names, Y = read_out(run / "dyn.bin")
    rate = json.loads((run / "run.json").read_text()).get("rate", "full")
    tC = Y[:, 0]
    tR = np.array(R["out_Dyn.eng.S2.Tt"][0]).ravel()
    if rate == "full":
        n = min(len(tC), len(tR))
        sel = np.arange(n)
        time_info = {"rows_c": len(tC), "rows_ref": len(tR), "time_max_abs": float(np.max(np.abs(tC[:n] - tR[:n])))}
    else:
        k = np.arange(len(tC), dtype=float)                       # integer seconds 0..
        sel = np.searchsorted(tR, k, side="right") - 1            # last ref sample <= k
        time_info = {"rows_c": len(tC), "rows_ref_1hz": len(sel),
                     "time_max_abs": float(np.max(np.abs(tC - tR[sel])))}
    rows = []
    for j, nm in enumerate(names[1:], start=1):
        m = re.match(r"^(.*)\[(\d+)\]$", nm)
        base, col = (m.group(1), int(m.group(2)) - 1) if m else (nm, 0)
        key = "out_Dyn." + base
        if key not in R:
            rows.append({"channel": nm, "status": "missing_in_ref"}); continue
        ref = np.array(R[key][1]).T            # (time, cols)
        r = ref[sel, col]
        c = Y[: len(sel), j] if rate == "full" else Y[:, j]
        r = r[: len(c)]
        ok = np.isfinite(r) & np.isfinite(c)
        d = np.abs(r[ok] - c[ok])
        peak = float(np.max(np.abs(r[ok]))) if ok.any() else 0.0
        mx = float(d.max()) if d.size else float("nan")
        rows.append({"channel": nm, "status": "compared", "n": int(ok.sum()),
                     "nonfinite_mismatch": int(np.sum(np.isfinite(r) != np.isfinite(c))),
                     "ref_peak_abs": peak, "max_abs": mx,
                     "max_rel": (mx / peak if peak > 0 else (0.0 if mx == 0 else float("inf"))),
                     "max_rel_pointwise": float(np.max(d / np.maximum(np.abs(r[ok]), 1e-6))) if d.size else float("nan")})
    comp = [r for r in rows if r["status"] == "compared"]
    worst = max(comp, key=lambda r: r["max_rel"])
    S = {"run": str(run), "ref": a.ref, "rate": rate, **time_info, "channels": len(rows),
         "compared": len(comp), "missing": [r["channel"] for r in rows if r["status"] != "compared"],
         "max_rel": worst["max_rel"], "worst_channel": worst["channel"],
         "n_rel_gt_tol": sum(r["max_rel"] > a.tol for r in comp), "tol": a.tol,
         "n_identical": sum(r["max_abs"] == 0 for r in comp),
         "pass": all(r["max_rel"] <= a.tol and r["nonfinite_mismatch"] == 0 for r in comp)
         and not any(r["status"] != "compared" for r in rows) and time_info["time_max_abs"] < 1e-9}
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["channel", "status", "n", "nonfinite_mismatch", "ref_peak_abs",
                                               "max_abs", "max_rel", "max_rel_pointwise"])
            w.writeheader(); w.writerows(rows)
    print(json.dumps(S, indent=1))
    return 0 if S["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
