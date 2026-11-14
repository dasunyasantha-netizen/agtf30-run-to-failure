#!/usr/bin/env python3
"""A2/A3 check: C runner results vs the A1 Normal-mode reference (healthy + degraded).

    cb_validate.py --runs <dir with healthy_full/, degraded_full/, healthy_1hz/, degraded_1hz/>
                   --a1 $AGTF30_WORK/agtf30_cbuild/a1 [--bin <runner dir> --flight flight.csv]

With --bin the four runs are made first (cb_run.py); without it existing run
folders are compared (e.g. results copied back from AI Panther). For each case:
  * initial conditions solved by the C steady-state runner vs MATLAB's
    define_inputs/setup_InputsSS result (mws_<case>.mat);
  * every runner channel vs the Normal reference, full rate and 1 Hz
    (cb_compare_ref.py). Pass: max relative difference <= 1e-6 everywhere.
Writes <runs>/validation.json and per-run compare CSVs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import scipy.io

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cb_compare_ref  # noqa: E402
import cb_run  # noqa: E402

DEGRADED = ["MWS.HPT.s_Eff=-0.02", "MWS.HPC.s_Eff=-0.01", "MWS.HPC.s_Wc=-0.015"]   # as a1_reference.m


def ic_check(run: pathlib.Path, mws_file: pathlib.Path) -> dict:
    M = scipy.io.loadmat(mws_file, squeeze_me=True, struct_as_record=False)["MWS"].In
    R = json.loads((run / "run.json").read_text())
    ic = R.get("dyn_ic", {})
    ref = {"CB_ICcom": np.atleast_1d(M.ICcom), "CB_N2IC": M.N2IC, "CB_N3IC": M.N3IC, "CB_WfIC": M.WfIC,
           "CB_VAFNIC": M.VAFNIC, "CB_VBVIC": M.VBVIC}
    for f in M.Sensor._fieldnames:
        if f.startswith("IC"):
            ref["CB_Sensor_" + f] = getattr(M.Sensor, f)
    out = {}
    for k, v in ref.items():
        if k not in ic:
            continue
        a, b = np.atleast_1d(np.asarray(v, float)), np.atleast_1d(np.asarray(ic[k], float))
        out[k] = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))
    sg = R.get("start_guess", {})
    out_sg = {"ssNc1": abs(sg.get("ssNc1", np.nan) - M.ssNc1) / abs(M.ssNc1)}
    return {"max_rel_by_ic": out, "max_rel": max(out.values()) if out else float("nan"),
            "ssNc1_rel": out_sg["ssNc1"], "ss_converged": R.get("ss", {}).get("converged")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--bin")
    ap.add_argument("--flight")
    a = ap.parse_args(argv)
    runs, a1 = pathlib.Path(a.runs), pathlib.Path(a.a1)
    V = {"cases": {}}
    for case, theta in (("healthy", []), ("degraded", DEGRADED)):
        for rate in ("full", "1hz"):
            d = runs / f"{case}_{rate}"
            if a.bin:
                args = ["--bin", a.bin, "--flight", a.flight, "--out", str(d), "--rate", rate]
                for t in theta:
                    args += ["--theta", t]
                cb_run.main(args)
            R = json.loads((d / "run.json").read_text())
            entry = {"ok": R.get("ok"), "ss_wall_s": R.get("ss", {}).get("wall_s"),
                     "dyn_wall_s": R.get("dyn", {}).get("wall_s"), "dyn_bytes": R.get("dyn", {}).get("bytes"),
                     "dyn_stderr": R.get("dyn", {}).get("stderr"), "ic": ic_check(d, a1 / f"mws_{case}.mat")}
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cb_compare_ref.main(["--ref", str(a1 / f"ref_{case}.mat"), "--run", str(d),
                                     "--csv", str(d / "compare.csv")])
            entry["compare"] = json.loads(buf.getvalue())
            V["cases"][f"{case}_{rate}"] = entry
            c = entry["compare"]
            print(f"{case:8s} {rate:4s} ok={entry['ok']} ic_max_rel={entry['ic']['max_rel']:.3g} "
                  f"max_rel={c['max_rel']:.3g} ({c['worst_channel']}) identical={c['n_identical']}/{c['compared']} "
                  f"pass={c['pass']} dyn_s={entry['dyn_wall_s']}")
    V["pass"] = all(e["compare"]["pass"] and e["ic"]["max_rel"] <= 1e-6 for e in V["cases"].values())
    (runs / "validation.json").write_text(json.dumps(V, indent=1))
    print("VALIDATION_PASS" if V["pass"] else "VALIDATION_FAIL")
    return 0 if V["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
