#!/usr/bin/env python3
"""E3 (pass 2): tables and paper figures from the per-unit statistics of e3_stats.py.

    python e3_report.py --stats $AGTF30_WORK/agtf30_v1_work/e3 --out docs/agtf30_v1

Writes <out>/stats.json (every number), <out>/stats_tables.md (side-by-side tables for the data card) and
<out>/figures/*.png: HI curves, life histograms by class, one sensor trend per subset (T50 residual), and
operating-condition coverage against N-CMAPSS. Colours: the three length classes and the two datasets use the
first three slots of the validated categorical palette (dataviz skill reference palette); density maps one hue.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import matplotlib
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical slots 1-3 (validated all-pairs, light surface)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMS = ["HPT", "HPC", "LPT", "FAN", "MIX"]
NC_KEYS = ["DS01-005", "DS02-006", "DS03-012", "DS04", "DS05", "DS06", "DS07", "DS08a-009", "DS08c-008"]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load(stats: pathlib.Path):
    D = {}
    for p in sorted(stats.glob("*.pkl")):
        if p.name.startswith("normaliser"):
            continue
        d = pickle.loads(p.read_bytes())
        D[p.stem] = d
    return D


def subset_name(stem):
    return stem.replace("AGTF30_", "").replace("N-CMAPSS_", "")


def mmm(x):
    x = np.asarray(x)
    return [int(x.min()), float(np.median(x)), int(x.max())] if len(x) else None


def summarise(stem, d):
    U = list(d["units"].values())
    meta = d["meta"]
    life = np.array([u["life"] for u in U]); fc = np.array([u["Fc"] for u in U])
    onset = np.array([u["onset"] for u in U], float); onset[onset < 0] = np.nan
    tn = U[0]["T_names"]
    Te = np.array([u["T_eol"] for u in U]) * 100
    T0 = np.array([u["T0"] for u in U]) * 100
    med = np.median(Te, 0)
    wear = {n: {"median_pct": round(float(med[i]), 3), "min": round(float(Te[:, i].min()), 3),
                "max": round(float(Te[:, i].max()), 3), "change_median_pct": round(float(np.median(Te[:, i] - T0[:, i])), 3)}
            for i, n in enumerate(tn) if abs(med[i]) > 0.05}
    xs_names = meta["xs_names"]
    xmin = np.min([u["xs_min"] for u in U], 0); xmax = np.max([u["xs_max"] for u in U], 0)
    Xsub = np.vstack([u["X_sub"] for u in U])
    q = np.percentile(Xsub, [1, 50, 99], axis=0)
    nd2 = np.median([u["noise_d2"] for u in U], 0)
    S = {"dataset": meta["dataset"], "subset": subset_name(stem), "engines": len(U),
         "dev": sum(u["split"] == "dev" for u in U), "test": sum(u["split"] == "test" for u in U),
         "flights": int(sum(u["flights"] for u in U)), "flight_hours": round(float(sum(u["hours"] for u in U)), 1),
         "first_cycle_gt1": int(sum(u["first_cycle"] > 1 for u in U)),
         "life": mmm(life), "life_by_class": {str(c): mmm(life[fc == c]) for c in (1, 2, 3) if (fc == c).any()},
         "engines_by_class": {str(c): int((fc == c).sum()) for c in (1, 2, 3)},
         "hours_per_flight_median": round(float(np.median(np.concatenate([u["flight_hours"] for u in U]))), 2),
         "onset_median": float(np.nanmedian(onset)) if np.isfinite(onset).any() else None,
         "onset_by_class": {str(c): float(np.nanmedian(onset[fc == c])) for c in (1, 2, 3)
                            if (fc == c).any() and np.isfinite(onset[fc == c]).any()},
         "onset_fraction_median": float(np.nanmedian(onset / life)) if np.isfinite(onset).any() else None,
         "no_onset": int(np.isnan(onset).sum()),
         "wear_eol": wear,
         "sensor_range": {n: {"min": float(xmin[i]), "p1": float(q[0, i]), "p50": float(q[1, i]), "p99": float(q[2, i]),
                              "max": float(xmax[i])} for i, n in enumerate(xs_names)},
         "noise_snr_db_d2": {n: round(float(-20 * np.log10(nd2[i])), 1) for i, n in enumerate(xs_names)}}
    if meta["dataset"] == "AGTF30":
        ne = np.median([u["noise_exact"] for u in U], 0)
        nc = np.median([u["noise_d2_clean"] for u in U], 0)
        S["noise_snr_db_exact"] = {n: round(float(-20 * np.log10(ne[i])), 2) for i, n in enumerate(xs_names)}
        S["noise_snr_db_d2_clean"] = {n: round(float(-20 * np.log10(nc[i])), 1) for i, n in enumerate(xs_names)}
        S["ending_limit"] = {k.decode(): int(v) for k, v in zip(*np.unique(meta["units"]["ending_limit"], return_counts=True))}
    return S


def pooled(stats, dataset):
    rows = [s for s in stats if s["dataset"] == dataset]
    return rows


def fmt_mmm(x):
    return "-" if x is None else f"{x[0]} / {x[1]:g} / {x[2]}"


def tables(stats, D):
    L = []
    L.append("| Dataset | Subset | Engines (dev/test) | Flights | Flight-h | h per flight (median) | Life min / median / max |"
             " Class 1 | Class 2 | Class 3 | Onset median (flight) | Onset / life |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in stats:
        lc = s["life_by_class"]
        L.append(f"| {s['dataset']} | {s['subset']} | {s['engines']} ({s['dev']}/{s['test']}) | {s['flights']:,} | "
                 f"{s['flight_hours']:,.0f} | {s['hours_per_flight_median']} | {fmt_mmm(s['life'])} | "
                 + " | ".join(fmt_mmm(lc.get(str(c))) for c in (1, 2, 3))
                 + f" | {s['onset_median'] if s['onset_median'] is not None else '-'} | "
                   f"{'-' if s['onset_fraction_median'] is None else round(s['onset_fraction_median'], 2)} |")
    for ds in ("AGTF30", "N-CMAPSS"):
        U = [u for k, d in D.items() if d["meta"]["dataset"] == ds for u in d["units"].values()]
        life = np.array([u["life"] for u in U]); fc = np.array([u["Fc"] for u in U])
        L.append(f"| **{ds} all** | | {len(U)} | {sum(u['flights'] for u in U):,} | {sum(u['hours'] for u in U):,.0f} | | "
                 f"{fmt_mmm(mmm(life))} | " + " | ".join(fmt_mmm(mmm(life[fc == c])) for c in (1, 2, 3)) + " | | |")
    L.append("")
    L.append("Wear at end of life (health parameters T, % change from the nominal map, median [min, max] over engines;"
             " only parameters with |median| > 0.05 %):")
    L.append("")
    L.append("| Dataset | Subset | Health parameters at end of life |")
    L.append("|---|---|---|")
    for s in stats:
        w = "; ".join(f"{k} {v['median_pct']:+.2f} [{v['min']:+.2f}, {v['max']:+.2f}]" for k, v in s["wear_eol"].items())
        L.append(f"| {s['dataset']} | {s['subset']} | {w} |")
    L.append("")
    xs = list(stats[0]["sensor_range"])
    L.append("Sensor ranges (X_s, p1 - p99 over all rows of the subset, 1/200 row sample; units as N-CMAPSS) and noise"
             " (SNR in dB from the second-difference estimator, median over engines; AGTF30 exact value in brackets):")
    L.append("")
    L.append("| Dataset | Subset | " + " | ".join(xs) + " |")
    L.append("|---|---|" + "---|" * len(xs))
    for s in stats:
        L.append(f"| {s['dataset']} | {s['subset']} | " + " | ".join(
            f"{s['sensor_range'][c]['p1']:.4g}-{s['sensor_range'][c]['p99']:.4g}" for c in xs) + " |")
    L.append("")
    L.append("| Dataset | Subset | " + " | ".join(xs) + " |")
    L.append("|---|---|" + "---|" * len(xs))
    for s in stats:
        ex = s.get("noise_snr_db_exact")
        L.append(f"| {s['dataset']} | {s['subset']} | " + " | ".join(
            f"{s['noise_snr_db_d2'][c]:.0f}" + (f" ({ex[c]:.1f})" if ex else "") for c in xs) + " |")
    L.append("")
    a = [s for s in stats if s["dataset"] == "AGTF30"]
    if a:
        L.append("AGTF30 estimator floor (second-difference SNR on X_s_clean, dB): " + "; ".join(
            f"{s['subset']} {min(s['noise_snr_db_d2_clean'].values()):.0f}-{max(s['noise_snr_db_d2_clean'].values()):.0f}"
            for s in a))
    return "\n".join(L) + "\n"


def panels(n):
    """One row for up to 5 subsets (v1.0, v1.1); two rows of 4 for all 8 (v1.0 + v1.1). Unused axes hidden."""
    ncol = n if n <= 5 else (n + 1) // 2
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2.3 * nrow + 0.3 * (nrow - 1)), sharey=True, squeeze=False)
    if nrow > 1:
        fig.subplots_adjust(hspace=0.45)
    ax = ax.ravel()
    for a in ax[n:]:
        a.set_visible(False)
    for a in ax[::ncol]:
        a.first_col = True          # y label on every row
    for a in ax[n - ncol:n]:
        a.last_row = True           # x label on the bottom row only
    return fig, ax[:n]


# ---------------------------------------------------------------- figures
def fig_hi(D, out):
    fig, ax = panels(len(FAMS))
    for i, f in enumerate(FAMS):
        d = D.get(f"AGTF30_{f}")
        if d is None:
            continue
        cyc, units = d["meta"]["cycles"], d["meta"]["units"]
        cls = {int(u["unit"]): int(u["Fc"]) for u in units}
        for u in np.unique(cyc["unit"]):
            m = cyc["unit"] == u
            ax[i].plot(cyc["cycle"][m], cyc["HI"][m], color=SLOT[cls[int(u)] - 1], lw=0.6, alpha=0.6)
        ax[i].set_title(f, color=INK); ax[i].axhline(0, color=INK2, lw=0.6)
        if getattr(ax[i], "last_row", False):
            ax[i].set_xlabel("flight")
    for a in ax:
        if getattr(a, 'first_col', False):
            a.set_ylabel("health index HI")
    for c in (1, 2, 3):
        ax[-1].plot([], [], color=SLOT[c - 1], lw=1.5, label=f"class {c}")
    ax[-1].legend(loc="lower left")                 # the empty corner: new engines, low HI
    fig.savefig(out / "fig_hi_curves.png"); plt.close(fig)


def fig_life(D, out):
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.3), sharey=True)
    bins = np.arange(40, 135, 5)
    for c in (1, 2, 3):
        for j, ds in enumerate(("AGTF30", "N-CMAPSS")):
            life = [u["life"] for d in D.values() if d["meta"]["dataset"] == ds for u in d["units"].values() if u["Fc"] == c]
            ax[c - 1].hist(life, bins=bins, density=True, histtype="step", lw=2, color=SLOT[j],
                           label=f"{ds} (n={len(life)})")
        ax[c - 1].set_title(f"flight class {c}", color=INK); ax[c - 1].set_xlabel("life (flights)")
        ax[c - 1].legend(loc="upper right")
    ax[0].set_ylabel("density")
    fig.savefig(out / "fig_life_hist.png"); plt.close(fig)


def fig_trend(D, out, channel="T50"):
    fig, ax = panels(len(FAMS))
    grid = np.linspace(0, 1, 21)
    for i, f in enumerate(FAMS):
        d = D.get(f"AGTF30_{f}")
        if d is None:
            continue
        j = d["meta"]["feature_names"].index(f"res_{channel}")
        curves = []
        for u in d["units"].values():
            x = u["F_cycle"] / u["life"]; y = u["F"][:, j] * 100
            ax[i].plot(x, y, color=SLOT[0], lw=0.4, alpha=0.25)
            curves.append(np.interp(grid, x, y))
        ax[i].plot(grid, np.median(curves, 0), color=INK, lw=1.6, label="median")
        ax[i].set_title(f, color=INK)
        if getattr(ax[i], "last_row", False):
            ax[i].set_xlabel("life fraction")
    for a in ax:
        if getattr(a, 'first_col', False):
            a.set_ylabel(f"{channel} residual (%)")
    ax[-1].legend(loc="upper left")
    fig.savefig(out / f"fig_trend_{channel}.png"); plt.close(fig)


def fig_coverage(D, out):
    fig, ax = plt.subplots(1, 5, figsize=(11, 2.8), gridspec_kw={"width_ratios": [1, 1, 0.05, 0.3, 1.1], "wspace": 0.3})
    ax[3].set_visible(False)
    cax, ax = ax[2], [ax[0], ax[1], ax[4]]
    H = {}
    xe, ye = np.linspace(0, 0.8, 81), np.linspace(0, 40000, 81)
    tr = {}
    for ds in ("AGTF30", "N-CMAPSS"):
        W = np.vstack([u["W_sub"] for d in D.values() if d["meta"]["dataset"] == ds for u in d["units"].values()])
        H[ds], _, _ = np.histogram2d(W[:, 1], W[:, 0], bins=[xe, ye], density=True)
        tr[ds] = W[:, 2]
    vmax = max(h.max() for h in H.values())
    from matplotlib.colors import LogNorm
    for k, ds in enumerate(("AGTF30", "N-CMAPSS")):
        im = ax[k].pcolormesh(xe, ye, np.ma.masked_equal(H[ds].T, 0), cmap="Blues",
                              norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax))
        ax[k].set_title(f"{ds}: altitude x Mach", color=INK); ax[k].set_xlabel("Mach")
        ax[k].grid(False)
    ax[0].set_ylabel("altitude (ft)")
    fig.colorbar(im, cax=cax)
    for j, ds in enumerate(("AGTF30", "N-CMAPSS")):
        ax[2].hist(tr[ds], bins=np.linspace(0, 100, 51), density=True, histtype="step", lw=2, color=SLOT[j], label=ds)
    ax[2].set_xlabel("TRA (%)"); ax[2].set_title("throttle (N-CMAPSS TRA)", color=INK); ax[2].legend(loc="upper left")
    fig.savefig(out / "fig_coverage.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=f"{WORK}/agtf30_v1_work/e3")
    ap.add_argument("--out", default="docs/agtf30_v1")
    ap.add_argument("--fams", default=",".join(FAMS), help="AGTF30 subsets (v1.1: HPTLPT,LPCHPC,ALL)")
    a = ap.parse_args(argv)
    FAMS[:] = a.fams.split(",")
    out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    D = load(pathlib.Path(a.stats))
    order = [f"AGTF30_{f}" for f in FAMS] + [f"N-CMAPSS_{k}" for k in NC_KEYS]
    stats = [summarise(k, D[k]) for k in order if k in D]
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "stats_tables.md").write_text(tables(stats, D))
    fig_hi(D, out / "figures"); fig_life(D, out / "figures"); fig_trend(D, out / "figures"); fig_coverage(D, out / "figures")
    print("E3_REPORT_DONE", len(stats), "subsets")


if __name__ == "__main__":
    main()
