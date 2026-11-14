#!/usr/bin/env python3
"""B6: N-CMAPSS signal groups (W, X_s, X_v, T, A) -> AGTF30 C-program outputs.

Checks every mapped AGTF30 channel exists in the cb_dyn output, compares magnitudes with the SAME real N-CMAPSS
flight (the 3 B4 closed-loop healthy flights; dev unit 1 cycle 1), and gives the AGTF30 noise sigma under the
N-CMAPSS rule measured in b6_noise.py (Gaussian, sigma = rms(clean flight signal) * 10^(-65/20)).
Output $AGTF30_WORK/agtf30_b/b6/sensor_map.json and sensor_map.md."""
import sys, pathlib, json
import numpy as np, h5py
import os
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agtf30_cbuild"))
import cb_run  # noqa: E402

NC = pathlib.Path(NCMAPSS)
B4 = pathlib.Path(f"{WORK}/agtf30_b/b4"); O = pathlib.Path(f"{WORK}/agtf30_b/b6"); O.mkdir(parents=True, exist_ok=True)
FLIGHTS = (("N-CMAPSS_DS01-005", 1, 1), ("N-CMAPSS_DS04", 1, 1), ("N-CMAPSS_DS08c-008", 1, 1))
SNR_REL = 10 ** (-65 / 20)

# (group, N-CMAPSS name, AGTF30 expression, quality, note). Expressions use cb_dyn output names.
MAP = [
    ("W", "alt", "in.Alt", "exact", "N-CMAPSS input passed through"),
    ("W", "Mach", "in.MN", "exact", "N-CMAPSS input passed through"),
    ("W", "TRA", "ncmapss:TRA", "copied", "N-CMAPSS lever kept as-is; AGTF30 lever stored separately as PLA (B4)"),
    ("W", "T2", "eng.S2.Tt", "exact", "AGTF30 inlet total temperature (dT = 0, N-CMAPSS is ISA)"),
    ("X_s", "Wf", "act.Wfact", "exact", "lbm/s (TSFC*Fn/3600 check)"),
    ("X_s", "Nf", "eng.Shaft.N_Fan", "exact", "geared fan: fan speed, not LP shaft speed"),
    ("X_s", "Nc", "eng.Shaft.N_HPC", "exact", "core (HP) speed"),
    ("X_s", "T24", "eng.S24.Tt", "exact", "LPC exit"),
    ("X_s", "T30", "eng.S36.Tt", "station renamed", "AGTF30 HPC exit is station 36"),
    ("X_s", "T48", "eng.S48.Tt", "exact", "HPT exit / LPT inlet (S45 Tt == S48 Tt in AGTF30)"),
    ("X_s", "T50", "eng.S5.Tt", "exact", "LPT exit"),
    ("X_s", "P15", "eng.S17.Pt", "close", "bypass nozzle inlet (S13 not in the 83 outputs; S17 = bypass duct exit)"),
    ("X_s", "P2", "eng.S2.Pt", "exact", "fan inlet"),
    ("X_s", "P21", "eng.S21.Pt", "exact", "fan exit"),
    ("X_s", "P24", "eng.S24.Pt", "exact", "LPC exit"),
    ("X_s", "Ps30", "eng.S36.Ps", "station renamed", "HPC exit static"),
    ("X_s", "P40", "eng.S4.Pt", "exact", "burner exit"),
    ("X_s", "P50", "eng.S5.Pt", "exact", "LPT exit"),
    ("X_v", "T40", "eng.S4.Tt", "exact", "burner exit"),
    ("X_v", "P30", "eng.S36.Pt", "station renamed", "HPC exit total"),
    ("X_v", "P45", "eng.S45.Pt", "exact", "HPT exit"),
    ("X_v", "W21", "eng.S21.W", "exact", "fan flow (total)"),
    ("X_v", "W22", "eng.S24.W", "close", "core flow through LPC (LPC exit; equals LPC inlet)"),
    ("X_v", "W25", "eng.S25.W", "exact", "HPC inlet (after VBV bleed)"),
    ("X_v", "W31", "none", "partial", "AGTF30 outputs only total HPC bleed (Data.HPC_Data.Wbleeds); split needs a rebuild"),
    ("X_v", "W32", "none", "partial", "as W31"),
    ("X_v", "W48", "eng.S48.W", "exact", "HPT exit"),
    ("X_v", "W50", "eng.S5.W", "exact", "LPT exit"),
    ("X_v", "SmFan", "eng.SM.SMFan", "definition", "AGTF30 constant-flow SM in map coordinates, percent (B1)"),
    ("X_v", "SmLPC", "eng.SM.SMLPC", "definition", "as SmFan"),
    ("X_v", "SmHPC", "eng.SM.SMHPC", "definition", "as SmFan"),
    ("X_v", "phi", "3600*act.Wfact/eng.S36.Ps", "exact", "N-CMAPSS phi = Wf[pph]/Ps30 (checked: phi/(Wf/Ps30) = 3600)"),
]
EXTRA = [  # AGTF30 signals without an N-CMAPSS equivalent, proposed as an extra group X_a (not in N-CMAPSS)
    ("PLA", "in.PLA", "AGTF30 lever from the B4 throttle match"),
    ("N_LPC", "eng.Shaft.N_LPC", "LP shaft speed (geared engine)"),
    ("T45", "eng.S45.Tt", "limited temperature (death limit)"),
    ("T45_limit", "cntrl.T45Lim", "controller T45 limit signal"),
    ("Fnet", "eng.Perf.Fnet", "net thrust"),
    ("TSFC", "eng.Perf.TSFC", ""),
    ("VBV", "act.VBVact", "variable bleed valve position"),
    ("VAFN", "act.VAFNact", "variable fan nozzle area"),
    ("Wbleed_HPC", "eng.Data.HPC_Data.Wbleeds", "total HPC bleed (W31+W32 proxy)"),
    ("RegNum", "cntrl.RegNum", "active limiter (0 = none, 2 = T45 ...)"),
    ("N1_target", "cntrl.N1cReq", "commanded fan speed"),
]


def ev(expr, d):
    if expr.startswith("3600*"):
        return 3600 * d["act.Wfact"] / d["eng.S36.Ps"]
    return d[expr]


def main():
    rows = []
    for fid, u, c in FLIGHTS:
        names, Y = cb_run.read_out(B4 / f"run_{fid.replace('N-CMAPSS_', '')}_u{u}_c{c}" / "dyn.bin")
        d = dict(zip(names, Y.T))
        n = len(Y)
        with h5py.File(NC / f"{fid}.h5", "r") as f:
            A = f["A_dev"][:, :2]; idx = np.flatnonzero((A[:, 0] == u) & (A[:, 1] == c))
            s = slice(idx[0], idx[-1] + 1)
            nc = {}
            for g in ("W", "X_s", "X_v"):
                vn = [x.decode() for x in np.array(f[g + "_var"]).ravel()]
                Z = f[g + "_dev"][s]
                for j, k in enumerate(vn):
                    nc[k] = Z[:, j]
        m = min(n, len(nc["alt"])); k0 = 60  # skip the first minute (start transient, B5)
        for grp, ncn, expr, q, note in MAP:
            r = {"flight": fid, "group": grp, "ncmapss": ncn, "agtf30": expr, "quality": q}
            if expr not in ("none", "ncmapss:TRA"):
                a = ev(expr, d)[k0:m].astype(float); b = nc[ncn][k0:m].astype(float)
                r.update(agtf_median=float(np.median(a)), nc_median=float(np.median(b)),
                         ratio_median=float(np.median(a / b)), corr=float(np.corrcoef(a, b)[0, 1]),
                         sigma65=float(np.sqrt(np.mean(a ** 2)) * SNR_REL))
            rows.append(r)
        print(fid, "ok", flush=True)
    missing = [e for _, _, e, _, _ in MAP if e not in ("none", "ncmapss:TRA") and not e.startswith("3600*") and e not in names]
    missing += [e for _, e, _ in EXTRA if e not in names]
    out = {"snr_rel": SNR_REL, "missing_in_outputs": missing, "map": MAP, "extra": EXTRA, "rows": rows}
    json.dump(out, open(O / "sensor_map.json", "w"), indent=1)
    lines = ["| group | N-CMAPSS | AGTF30 | quality | AGTF30/N-CMAPSS median (3 flights) | corr (min) | AGTF30 sigma at 65 dB (median) | note |",
             "|---|---|---|---|---|---|---|---|"]
    for grp, ncn, expr, q, note in MAP:
        rr = [r for r in rows if r["ncmapss"] == ncn and "ratio_median" in r]
        if rr:
            rat = "/".join(f"{r['ratio_median']:.2f}" for r in rr); cor = f"{min(r['corr'] for r in rr):.2f}"
            sg = f"{np.median([r['sigma65'] for r in rr]):.3g}"
        else:
            rat = cor = sg = "-"
        lines.append(f"| {grp} | {ncn} | `{expr}` | {q} | {rat} | {cor} | {sg if grp == 'X_s' else '-'} | {note} |")
    (O / "sensor_map.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("missing:", missing)


if __name__ == "__main__":
    main()
