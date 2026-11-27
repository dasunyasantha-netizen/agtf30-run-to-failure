"""Figures for the AGTF30 dataset paper, as 600 dpi PNG and vector PDF.

All seven figures are drawn here at their final printed width (W = 6.85 in = 174 mm, the text width of the A4
manuscript), so the font sizes below are the printed sizes: 8 pt ticks, legends and annotations, 9 pt axis labels,
10 pt panel titles and panel letters. One sans-serif family (DejaVu Sans) throughout. Figures are saved without
tight cropping, so every file is exactly W wide.

Inputs (all read-only; paths set below, see AGTF30_* environment variables):
- per-unit statistics of all eight subsets and the nine N-CMAPSS files: $AGTF30_WORK/agtf30_v11_work/e3/*.pkl
  (written by scripts/agtf30_release/e3_stats.py; operating conditions every 50th row, per-flight residuals)
- fault signatures: provenance judge files of the two releases ($AGTF30_WORK/agtf30_v1, $AGTF30_WORK/agtf30_v11)
- one example flight (Fig. 2): $AGTF30_WORK/agtf30_v11_work/fig_flight/flight.npz, written by extract_flight.py
- fan speed against its target, every 50th second of every engine: $AGTF30_WORK/agtf30_v11_work/fig_fan/fan_target.npz
  (written by extract_fan.py from X_s_clean.Nf, X_a.N1_target and W.T2 of the released files;
  X_a.N1_target is the controller's corrected request, so the physical target is N1_target * sqrt(T2/518.67))
"""
import json, os, pickle, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm

# Paths, as in the rest of the code (README "Paths"): AGTF30_WORK is the work root (D:/tmp in the original runs).
#   $AGTF30_WORK/agtf30_v1, $AGTF30_WORK/agtf30_v11   the two released folders (or set AGTF30_V10, AGTF30_V11)
#   $AGTF30_WORK/agtf30_v11_work/{e3, fig_fan, fig_flight}   inputs written by e3_stats.py, extract_fan.py, extract_flight.py
#   AGTF30_FIGOUT   where the figures and fig_numbers.json are written (default: this script's folder)
ROOTW = os.environ.get("AGTF30_WORK", "work")
V10 = os.environ.get("AGTF30_V10", os.path.join(ROOTW, "agtf30_v1"))
V11 = os.environ.get("AGTF30_V11", os.path.join(ROOTW, "agtf30_v11"))
WORK = os.path.join(ROOTW, "agtf30_v11_work")
OUT = os.environ.get("AGTF30_FIGOUT", os.path.dirname(os.path.abspath(__file__)))
STATS = pathlib.Path(WORK, "e3")
FAN = os.path.join(WORK, "fig_fan", "fan_target.npz")
FLIGHT = os.path.join(WORK, "fig_flight", "flight.npz")
DPI, W = 600, 6.85
ORDER = ["HPT", "HPC", "LPT", "FAN", "MIX", "HPTLPT", "LPCHPC", "ALL"]
NAMES = {"HPT": "HPT", "HPC": "HPC", "LPT": "LPT", "FAN": "Fan", "MIX": "HPC+HPT", "HPTLPT": "HPT+LPT",
         "LPCHPC": "LPC+HPC", "ALL": "All five"}
CLASS = {1: "short flights", 2: "medium flights", 3: "long flights"}
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated, light surface)
INK, MUTED, GRID = "#1f1f1e", "#52514e", "#e4e3df"
N1_REF = 2214.6                                     # AGTF30 fan speed at the throttle-mapping reference (100 %)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "axes.titleweight": "normal"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=DPI, facecolor="white")
    plt.close(fig)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, weight="bold", ha="right", va="bottom")


def load():
    D = {}
    for p in sorted(STATS.glob("*.pkl")):
        if not p.name.startswith("normaliser"):
            D[p.stem] = pickle.loads(p.read_bytes())
    return D


D = load()
AG = {f: D[f"AGTF30_{f}"] for f in ORDER}
NC = [d for k, d in D.items() if d["meta"]["dataset"] == "N-CMAPSS"]


def grid8(h):
    fig, ax = plt.subplots(2, 4, figsize=(W, h), sharex=True, sharey=True, layout="constrained")
    return fig, ax.ravel()


# ---------------------------------------------------------------- Figure 1: workflow
fig, ax = plt.subplots(figsize=(W, 3.9))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
BLUE, ORANGE, AQUA = SLOT[0], SLOT[1], SLOT[2]


def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(x + w / 2, y + h - 4.9, body, ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.3)


def arrow(x1, y1, x2, y2, text=None, rad=0.0, tx=0, ty=1.0, ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=1,
                shrinkB=1, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha=ha, fontsize=7.5, color=INK)


ax.text(1, 56, "Inputs", fontsize=9, color=BLUE, weight="bold")
box(1, 39, 17, 14, "N-CMAPSS flights", "recorded altitude,\nMach, fan speed,\n1 Hz; 99 engines", BLUE)
box(1, 21, 17, 14, "Wear recipe", "initial, background\nwear; fault onset\nand growth", BLUE)
box(1, 3, 17, 14, "AGTF30 model", "compiled C programs\n(dynamic, steady\nstate): steps 2, 3", BLUE)
ax.text(22, 56, "Per engine, one flight after another", fontsize=9, color=ORANGE, weight="bold")
box(22, 39, 16, 14, "Start engine", "seeds, flight class,\nsource flights,\ninitial wear", ORANGE)
box(42, 39, 16, 14, "1  Map throttle", "same % of rated\nfan speed as the\nsource engine", ORANGE)
box(62, 39, 16, 14, "2  Fly flight", "dynamic model,\n1 Hz; wear fixed\nduring the flight", ORANGE)
box(62, 21, 16, 14, "3  Health check", "steady state, sea-\nlevel take-off →\nmargins → HI", ORANGE)
box(42, 21, 16, 14, "4  Update wear", "wear for the next\nflight (fault\nafter onset)", ORANGE)
cx, cy = 70, 9.5
ax.add_patch(Polygon([[cx, cy + 6.5], [cx + 8, cy], [cx, cy - 6.5], [cx - 8, cy]], closed=True, fc="white",
                     ec=ORANGE, lw=1.2))
ax.text(cx, cy, "HI > 0 ?", ha="center", va="center", fontsize=8, weight="bold")
ax.text(82, 56, "Outputs", fontsize=9, color=AQUA, weight="bold")
box(82, 21, 17, 14, "Engine done", "last flight = end\nof life; RUL\ncounted back", AQUA)
box(82, 39, 17, 14, "Data files", "one HDF5 file per\nsubset, N-CMAPSS\nlayout; 480 engines", AQUA)
arrow(18.3, 46, 21.7, 46)                       # flights -> start
arrow(18.3, 34, 26, 38.7)                       # recipe -> start (initial wear)
arrow(18.3, 28, 41.7, 28)                       # recipe -> update
arrow(38.3, 46, 41.7, 46)                       # start -> map
arrow(58.3, 46, 61.7, 46)                       # map -> fly
arrow(70, 38.7, 70, 35.3)                       # fly -> check
arrow(70, 20.7, 70, 16.3)                       # check -> decision
arrow(61.8, 9.5, 50, 20.7, text="yes: next flight", rad=-0.3, tx=-9, ty=-5)
arrow(50, 35.3, 50, 38.7)                       # update -> map
arrow(78.2, 9.5, 90.5, 20.7, text="no", rad=0.3, tx=5, ty=-5)
arrow(90.5, 35.3, 90.5, 38.7)                   # done -> files
save(fig, "fig1_workflow")

# ---------------------------------------------------------------- Figure 2: one example flight
if os.path.exists(FLIGHT):
    fl = np.load(FLIGHT)
    tmin = np.arange(len(fl["alt"])) / 60
    tgt = fl["n1_target"] * np.sqrt(fl["t2"] / 518.67) / N1_REF * 100
    fig, ax = plt.subplots(2, 2, figsize=(W, 3.6), sharex=True, layout="constrained")
    ax = ax.ravel()
    ax[0].plot(tmin, fl["alt"] / 1000, color=SLOT[0], lw=1.0); ax[0].set_ylabel("altitude (1,000 ft)")
    ax[1].plot(tmin, fl["mach"], color=SLOT[0], lw=1.0); ax[1].set_ylabel("Mach number")
    ax[2].plot(tmin, fl["nf"] / N1_REF * 100, color=SLOT[1], lw=1.2, label="achieved")
    ax[2].plot(tmin, tgt, color=INK, lw=0.7, ls=(0, (3, 2)), label="target")
    ax[2].set_ylabel("fan speed (%)"); ax[2].legend(loc="upper right", ncol=2)
    ax[3].plot(tmin, fl["t48"], color=SLOT[1], lw=0.8); ax[3].set_ylabel("T48 (°R)")
    for i, a in enumerate(ax):
        letter(a, "abcd"[i])
    for a in ax[2:]:
        a.set_xlabel("time from start of record (min)")
    ax[0].set_title("Flight conditions (input, W)", loc="left", fontsize=9)
    ax[2].set_title("Engine response (measured, X_s)", loc="left", fontsize=9)
    save(fig, "fig2_flight")
    FLIGHT_INFO = {k: (fl[k].item().decode() if fl[k].dtype.kind == "S" else fl[k].item())
                   for k in ("h5", "unit", "cycle", "eol", "fc", "hours", "hi", "rul", "src_key")}
    FLIGHT_INFO.update(alt_max_ft=float(fl["alt"].max()), mach_max=float(fl["mach"].max()),
                       seconds=int(len(fl["alt"])))
else:
    FLIGHT_INFO = None

# ---------------------------------------------------------------- Figure 3: HI trajectories
fig, ax = grid8(4.0)
for i, f in enumerate(ORDER):
    cyc, units = AG[f]["meta"]["cycles"], AG[f]["meta"]["units"]
    cls = {int(u["unit"]): int(u["Fc"]) for u in units}
    for u in np.unique(cyc["unit"]):
        m = cyc["unit"] == u
        ax[i].plot(cyc["cycle"][m], cyc["HI"][m], color=SLOT[cls[int(u)] - 1], lw=0.5, alpha=0.6)
    ax[i].axhline(0, color=MUTED, lw=0.6)
    ax[i].set_title(NAMES[f]); letter(ax[i], "abcdefgh"[i])
for a in ax[4:]:
    a.set_xlabel("flight")
for a in ax[::4]:
    a.set_ylabel("health index HI")
h = [plt.Line2D([], [], color=SLOT[c - 1], lw=1.5) for c in (1, 2, 3)]
fig.legend(h, [f"{CLASS[c]} (class {c})" for c in (1, 2, 3)], loc="outside upper center", ncol=3)
save(fig, "fig3_hi_curves")

# ---------------------------------------------------------------- Figure 4: operating conditions and fan speed
HAVE_FAN = os.path.exists(FAN)
fan = np.load(FAN) if HAVE_FAN else None
if HAVE_FAN:
    tgt = fan["tgt"] * np.sqrt(fan["t2"] / 518.67)     # X_a.N1_target is the corrected request (cntrl.N1cReq)
    nf_pct, tgt_pct = fan["nf"] / N1_REF * 100, tgt / N1_REF * 100
    err = (fan["nf"] / tgt - 1) * 100
    lim = fan["reg"] != 0
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), layout="constrained")
    Wa = np.vstack([u["W_sub"] for d in AG.values() for u in d["units"].values()])
    xe, ye = np.linspace(0, 0.8, 81), np.linspace(0, 40000, 81)
    H, _, _ = np.histogram2d(Wa[:, 1], Wa[:, 0], bins=[xe, ye])
    H = H / H.sum() * 100
    e2 = np.linspace(55, 105, 101)
    H2, _, _ = np.histogram2d(tgt_pct, nf_pct, bins=[e2, e2])
    H2 = H2 / H2.sum() * 100
    norm = LogNorm(vmin=1e-4, vmax=max(H.max(), H2.max()))     # one colour scale for panels a and b
    im = ax[0].pcolormesh(xe, ye / 1000, np.ma.masked_equal(H.T, 0), cmap="Blues", norm=norm)
    ax[0].set_xlabel("Mach"); ax[0].set_ylabel("altitude (1,000 ft)"); ax[0].grid(False)
    ax[0].set_title("Flight conditions")
    ax[1].pcolormesh(e2, e2, np.ma.masked_equal(H2.T, 0), cmap="Blues", norm=norm)
    cb = fig.colorbar(im, ax=ax[:2], pad=0.01, aspect=30, shrink=0.95)
    cb.set_label("share of seconds per cell (%)", fontsize=8); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ax[1].plot([55, 105], [55, 105], color=INK, lw=0.6, ls="--")
    ax[1].set_xlabel("target fan speed (%)"); ax[1].set_ylabel("achieved fan speed (%)"); ax[1].grid(False)
    ax[1].set_title("Fan speed")
    be = np.linspace(-3, 3, 121)
    ax[2].hist(err[~lim], bins=be, color=SLOT[0], alpha=0.85, label="no limiter active")
    ax[2].hist(err[lim], bins=be, color=SLOT[1], alpha=0.85, label="limiter active")
    ax[2].set_yscale("log"); ax[2].set_xlabel("relative to target (%)"); ax[2].set_ylabel("seconds (sampled)")
    ax[2].set_ylim(top=3e8); ax[2].set_title("Fan-speed error"); ax[2].legend(loc="upper left", ncol=1, borderaxespad=0.2)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    save(fig, "fig4_coverage")
    FAN_STATS = {"n": int(len(err)), "within_1pct": float(np.mean(np.abs(err) <= 1) * 100),
                 "p5": float(np.percentile(err, 5)), "p95": float(np.percentile(err, 95)),
                 "p5_nolim": float(np.percentile(err[~lim], 5)), "p95_nolim": float(np.percentile(err[~lim], 95)),
                 "within_1pct_nolim": float(np.mean(np.abs(err[~lim]) <= 1) * 100), "limiter_share": float(lim.mean() * 100),
                 "within_1pct_lim": float(np.mean(np.abs(err[lim]) <= 1) * 100) if lim.any() else None,
                 "clip_share": float((fan["clip"] != 0).mean() * 100),
                 "outside_pm3": int((np.abs(err) > 3).sum()), "outside_pm3_pct": float((np.abs(err) > 3).mean() * 100),
                 "err_min": float(err.min()), "err_max": float(err.max()),
                 "nf_pct_p1_p99": [float(np.percentile(nf_pct, 1)), float(np.percentile(nf_pct, 99))]}

else:
    FAN_STATS = None

# ---------------------------------------------------------------- Figure 5: lives, cumulative
fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True, layout="constrained")
LIFE = {}
for c in (1, 2, 3):
    for j, (ds, src) in enumerate((("AGTF30", AG.values()), ("N-CMAPSS", NC))):
        life = np.sort([u["life"] for d in src for u in d["units"].values() if u["Fc"] == c])
        LIFE[(ds, c)] = life
        y = np.arange(1, len(life) + 1) / len(life)
        ax[c - 1].step(life, y, where="post", color=SLOT[j], lw=1.5)
        ax[c - 1].axvline(np.median(life), color=SLOT[j], lw=0.8, ls="--")
    ax[c - 1].set_title(f"{CLASS[c].capitalize()} (class {c})"); ax[c - 1].set_xlabel("life (flights)")
    ax[c - 1].text(0.97, 0.04, f"n = {len(LIFE[('AGTF30', c)])} / {len(LIFE[('N-CMAPSS', c)])}", transform=ax[c - 1].transAxes,
                   va="bottom", ha="right", fontsize=8, color=MUTED)
    letter(ax[c - 1], "abc"[c - 1])
ax[0].set_ylabel("share of engines")
fig.legend([plt.Line2D([], [], color=SLOT[0], lw=1.5), plt.Line2D([], [], color=SLOT[1], lw=1.5),
            plt.Line2D([], [], color=MUTED, lw=0.8, ls="--")], ["AGTF30 (this dataset)", "N-CMAPSS (9 files used)", "median"],
           loc="outside upper center", ncol=3)
save(fig, "fig5_life_hist")

# ---------------------------------------------------------------- Figure 6: fault signatures
subsets = {}
for p in (os.path.join(V10, "provenance", "judge__judge.json"), os.path.join(V11, "provenance", "judge.json")):
    subsets.update(json.load(open(p))["subsets"])
ch = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
M = np.array([[subsets[s]["xs_change_pct_new_to_eol"][c]["median"] for c in ch] for s in ORDER])
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#b52f2f"])
fig, ax = plt.subplots(figsize=(W, 3.0), layout="constrained")
im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=-5.5, vcenter=0, vmax=5.5), aspect="auto")
for i in range(len(ORDER)):
    for j in range(len(ch)):
        v = M[i, j]
        ax.text(j, i, f"{v:+.1f}" if abs(v) >= 0.05 else "0.0", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 3.6 else INK)
ax.set_xticks(range(len(ch)), ch); ax.set_yticks(range(len(ORDER)), [NAMES[s] for s in ORDER])
ax.tick_params(length=0, labelsize=8); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([x - 0.5 for x in range(1, len(ch))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(ORDER))], minor=True)
ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=25)
cb.set_label("median change (%)", fontsize=9); cb.outline.set_visible(False)
ax.set_xlabel("measured signal"); ax.set_ylabel("fault subset")
save(fig, "fig6_signatures")

# ---------------------------------------------------------------- Figure 7: T50 residual trends
fig, ax = grid8(4.0)
g = np.linspace(0, 1, 21)
for i, f in enumerate(ORDER):
    d = AG[f]
    j = d["meta"]["feature_names"].index("res_T50")
    curves = []
    for u in d["units"].values():
        x = u["F_cycle"] / u["life"]; y = u["F"][:, j] * 100
        ax[i].plot(x, y, color=SLOT[0], lw=0.4, alpha=0.25)
        curves.append(np.interp(g, x, y))
    ax[i].plot(g, np.median(curves, 0), color=INK, lw=1.5)
    ax[i].set_title(NAMES[f]); letter(ax[i], "abcdefgh"[i])
for a in ax[4:]:
    a.set_xlabel("life fraction")
for a in ax[::4]:
    a.set_ylabel("T50 residual (%)")
fig.legend([plt.Line2D([], [], color=SLOT[0], lw=0.8, alpha=0.6), plt.Line2D([], [], color=INK, lw=1.5)],
           ["one engine (60 per subset)", "median"], loc="outside upper center", ncol=2)
save(fig, "fig7_trend_T50")

json.dump({"fig2_flight": FLIGHT_INFO, "fan": FAN_STATS, "life_median": {f"{k[0]}_{k[1]}": float(np.median(v)) for k, v in LIFE.items()}},
          open(os.path.join(OUT, "fig_numbers.json"), "w"), indent=1)
print(json.dumps(FAN_STATS, indent=1))
