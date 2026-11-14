#!/usr/bin/env python3
"""D2 pilot judgement (decision #9) from the merged .h5 files and the run folder.

    python gen_judge.py --out $AGTF30_WORK/agtf30_c/pilot

Writes <out>/judge/judge.json and trend plots (<out>/judge/trend_<FAM>.png, hi_all.png):
  * lives per subset and overall (spread, no life cap) and the ending limit;
  * HI: never rises, single-flight drop share, flights with no fall;
  * wear in the sensors: X_s operating-condition residual (agtf30_data.Normaliser, fitted on the first 10
    flights of the dev engines) per flight, new (first 5 flights) vs end of life (last 5), in % and in
    multiples of the per-flight-mean noise;
  * solver: drops (drops.jsonl), flights needing the half step, iteration-cap seconds; limiter and clipped seconds;
  * speed (job wall time, flight-h simulated in this job) and size (bytes per flight-h of the .h5 files).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_learn"))
from agtf30_data import Normaliser, iter_units  # noqa: E402

PLOT_CH = ["T30", "T48", "Wf", "Nc"]
SNR_DB = 65.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reused-flight-h", type=float, default=0.0,
                    help="flight-h copied in from an earlier run (not simulated by this job)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    jd = out / "judge"; jd.mkdir(exist_ok=True)
    files = sorted((out / "h5").glob("AGTF30_*.h5"))
    J = {"files": {}, "subsets": {}}
    lives_all, cyc_all = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            U, C = h["units"][:], h["cycles"][:]
            J["files"][f.name] = {"bytes": f.stat().st_size, "hours": float(C["hours"].sum())}
        fam = f.stem.split("_")[1]
        lives = U["eol"].astype(int)
        lives_all += lives.tolist()
        cyc_all.append(C)
        hi = {}
        for u in U["unit"]:
            m = C["unit"] == u
            hi[int(u)] = C["HI"][m][np.argsort(C["cycle"][m])]
        drops = [-np.diff(np.r_[U["hi0"][U["unit"] == u][0], v]) for u, v in hi.items()]
        J["subsets"][fam] = {
            "n_units": int(len(U)), "split": {s.decode(): int((U["split"] == s).sum()) for s in np.unique(U["split"])},
            "lives": lives.tolist(), "life_min": int(lives.min()), "life_median": float(np.median(lives)),
            "life_max": int(lives.max()), "life_class": {int(c): lives[U["Fc"] == c].tolist() for c in np.unique(U["Fc"])},
            "ending_limit": sorted({x.decode() for x in U["ending_limit"]}),
            "hi0_range": [float(U["hi0"].min()), float(U["hi0"].max())],
            "hi_rises": int(sum((d < -1e-12).sum() for d in drops)),
            "hi_flat_flights_share": float(np.mean(np.concatenate([d <= 1e-12 for d in drops]))),
            "hi_max_single_drop_share": float(max(d.max() / d.sum() for d in drops)),
            "replacements": int(U["replacements"].sum()),
            "flights": int(len(C)), "flight_h": float(C["hours"].sum()),
            "limiter_s": int(C["limiter_s"].sum()), "clipped_s": int(C["clipped_s"].sum()),
            "cap_s": int(C["cap_s"].sum()), "half_step_flights": int((C["dyn_step_s"] < 0.015 - 1e-9).sum()),
            "ref_retries": int(C["ref_retries"].sum()), "seconds": int(C["seconds"].sum())}
        J["subsets"][fam]["_hi"] = hi
    lv = np.array(lives_all)
    J["lives_all"] = {"n": int(len(lv)), "min": int(lv.min()), "p10": float(np.percentile(lv, 10)),
                      "median": float(np.median(lv)), "p90": float(np.percentile(lv, 90)), "max": int(lv.max()),
                      "in_60_100": int(((lv >= 60) & (lv <= 100)).sum()), "below_60": int((lv < 60).sum()),
                      "above_100": int((lv > 100).sum())}

    # wear in the sensors
    norm = Normaliser().fit(iter_units(files, "dev", max_cycle=10))
    sig_rel = 10 ** (-SNR_DB / 20)
    trends = {}
    for split in ("dev", "test"):
        for k, u in iter_units(files, split):
            fam = k.split(":")[0].split("_")[1]
            R = norm.residual(u["W"], u["X_s"])
            cyc = u["cycle"]
            st = np.flatnonzero(np.r_[True, np.diff(cyc) != 0]); en = np.r_[st[1:], len(cyc)]
            Fr = np.array([R[s:e].mean(0) for s, e in zip(st, en)])
            nsec = (en - st).astype(float)
            trends.setdefault(fam, {})[k] = {"res": Fr, "nsec": nsec, "names": u["xs_names"]}
    for fam, T in trends.items():
        names = next(iter(T.values()))["names"]
        new = np.array([t["res"][:5].mean(0) for t in T.values()])
        eol = np.array([t["res"][-5:].mean(0) for t in T.values()])
        # noise of a 5-flight mean of per-second residuals ~ sig_rel / sqrt(seconds in 5 flights)
        nse = np.array([sig_rel / np.sqrt(t["nsec"][-5:].sum()) for t in T.values()])
        # flight-to-flight scatter of the residual on the first 10 flights (operating condition + noise left over)
        hsd = np.array([t["res"][:10].std(0, ddof=1) for t in T.values()])
        d = (eol - new) * 100
        J["subsets"][fam]["xs_change_pct_new_to_eol"] = {
            n: {"median": float(np.median(d[:, i])), "min": float(d[:, i].min()), "max": float(d[:, i].max()),
                "median_over_5flight_noise": float(np.median(np.abs(eol - new)[:, i] / nse)),
                "median_over_healthy_flight_sd": float(np.median(np.abs(eol - new)[:, i] / hsd[:, i]))}
            for i, n in enumerate(names)}
    J["normaliser_r2"] = dict(zip(names, map(float, norm.r2)))

    # run speed
    jobs = [json.loads(p.read_text()) for p in sorted(out.glob("job_*.json"))]
    wall = max(j["wall_s"] for j in jobs) if jobs else None
    fh = sum(v["hours"] for v in J["files"].values())
    drops = [json.loads(x) for x in (out / "drops.jsonl").read_text().splitlines()] if (out / "drops.jsonl").exists() else []
    sim_h = sum(json.loads(j.read_text())["seconds"] for j in (out / "runs").glob("*/e*/c*.json")) / 3600
    J["run"] = {"wall_h": wall / 3600 if wall else None, "flight_h_in_files": fh,
                "flight_h_simulated_incl_dropped": sim_h, "flight_h_reused": a.reused_flight_h,
                "flight_h_per_wall_h": (sim_h - a.reused_flight_h) / (wall / 3600) if wall else None,
                "drops": [{k: d[k] for k in ("family", "slot", "engine_index", "cycle", "reason", "source")} for d in drops],
                "bytes_h5": sum(v["bytes"] for v in J["files"].values()),
                "mb_per_flight_h": sum(v["bytes"] for v in J["files"].values()) / 1e6 / fh}

    plot(J, trends, jd)
    for s in J["subsets"].values():
        s.pop("_hi")
    (jd / "judge.json").write_text(json.dumps(J, indent=1))
    print(json.dumps({"lives_all": J["lives_all"], "run": {k: v for k, v in J["run"].items() if k != "drops"}}, indent=1))
    print("JUDGE_DONE")


def plot(J, trends, jd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid, line = "#1f2328", "#d0d7de", "#2f6fb3"
    fams = list(J["subsets"])
    fig, axs = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True)
    for ax, fam in zip(np.atleast_1d(axs), fams):
        for v in J["subsets"][fam]["_hi"].values():
            ax.plot(np.arange(1, len(v) + 1), v, color=line, lw=1.2, alpha=0.6)
        ax.axhline(0, color=grid, lw=1)
        ax.set_title(fam, color=ink, fontsize=10); ax.set_xlabel("flight", fontsize=9)
        ax.grid(color=grid, lw=0.5); ax.tick_params(labelsize=8)
    np.atleast_1d(axs)[0].set_ylabel("HI after flight (1 engine per line)", fontsize=9)
    fig.tight_layout(); fig.savefig(jd / "hi_all.png", dpi=120); plt.close(fig)
    for fam, T in trends.items():
        names = next(iter(T.values()))["names"]
        idx = [names.index(c) for c in PLOT_CH if c in names]
        fig, axs = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 3.0))
        for ax, i in zip(axs, idx):
            for t in T.values():
                r = t["res"][:, i] * 100
                ax.plot(np.linspace(0, 1, len(r)), r, color=line, lw=1.0, alpha=0.5)
            ax.axhline(0, color=grid, lw=1)
            ax.set_title(f"{fam}: {names[i]}", color=ink, fontsize=10)
            ax.set_xlabel("life fraction (new -> end of life)", fontsize=8)
            ax.grid(color=grid, lw=0.5); ax.tick_params(labelsize=8)
        axs[0].set_ylabel("flight-mean residual vs healthy model (%)", fontsize=8)
        fig.tight_layout(); fig.savefig(jd / f"trend_{fam}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
