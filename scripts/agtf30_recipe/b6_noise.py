#!/usr/bin/env python3
"""B6: measure N-CMAPSS sensor noise from the data and compare it with the paper's rule (Gaussian noise on xs at
SNR 65 dB, Arias Chao et al. 2021 Sec 3.3).

White-noise std per flight is estimated from second differences, sigma = 1.4826*MAD(x[t]-2x[t-1]+x[t-2])/sqrt(6),
which removes the slow flight-driven signal. The same estimate on xv (virtual sensors, no noise added) gives the
floor caused by the real, already noisy flight inputs w. SNR_dB = 20*log10(rms(x)/sigma), rms over the flight.
Output $AGTF30_WORK/agtf30_b/b6/ncmapss_noise.json."""
import h5py, numpy as np, glob, os, json
WORK = os.environ.get("AGTF30_WORK", "work")   # work/scratch root (D:/tmp in the original runs), see README
NCMAPSS = os.environ.get("NCMAPSS_DIR", "N-CMAPSS").rstrip("/\\") + "/"   # N-CMAPSS data_set folder, see README
D = NCMAPSS
O = f"{WORK}/agtf30_b/b6/"
os.makedirs(O, exist_ok=True)


def sig2(x):
    d = x[2:] - 2 * x[1:-1] + x[:-2]
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6)


res = {}
for fn in sorted(glob.glob(D + "*.h5")):
    b = os.path.basename(fn).replace(".h5", "")
    if "DS08d" in b:
        continue  # truncated on disk
    with h5py.File(fn, "r") as f:
        av = [x.decode() for x in np.array(f["A_var"]).ravel()]
        A = f["A_dev"][:]
        iu, ic = av.index("unit"), av.index("cycle")
        for grp in ("X_s", "X_v"):
            names = [x.decode() for x in np.array(f[grp + "_var"]).ravel()]
            X = f[grp + "_dev"][:]
            for u in np.unique(A[:, iu])[:2]:  # 2 units per file
                mu = A[:, iu] == u
                cyc = np.unique(A[mu, ic])
                for c in cyc[[0, len(cyc) // 2, -1]]:  # first, middle, last flight
                    m = mu & (A[:, ic] == c)
                    for j, n in enumerate(names):
                        x = X[m, j].astype(float)
                        s = sig2(x); rms = np.sqrt(np.mean(x ** 2))
                        res.setdefault(grp + ":" + n, []).append(
                            {"file": b, "unit": int(u), "cycle": int(c), "sigma": float(s), "rms": float(rms),
                             "snr_db": float(20 * np.log10(rms / s)) if s > 0 else None})
    print(b, flush=True)

summ = {}
for k, v in res.items():
    s = np.array([r["sigma"] for r in v]); rel = np.array([r["sigma"] / r["rms"] for r in v])
    snr = np.array([r["snr_db"] for r in v if r["snr_db"] is not None])
    summ[k] = {"n": len(v), "sigma_median": float(np.median(s)), "sigma_p10": float(np.percentile(s, 10)),
               "sigma_p90": float(np.percentile(s, 90)), "rel_median": float(np.median(rel)),
               "rel_cv": float(np.std(rel) / np.mean(rel)), "sigma_cv": float(np.std(s) / np.mean(s)),
               "snr_db_median": float(np.median(snr)) if len(snr) else None,
               "rms_median": float(np.median([r["rms"] for r in v]))}
    print(f"{k:14s} sigma {summ[k]['sigma_median']:.4g} (cv {summ[k]['sigma_cv']:.2f})  rel {summ[k]['rel_median']:.3e} "
          f"(cv {summ[k]['rel_cv']:.2f})  SNR {summ[k]['snr_db_median']}")
json.dump({"rule": "sigma=1.4826*MAD(2nd diff)/sqrt(6); snr=20log10(rms/sigma)", "summary": summ, "rows": res},
          open(O + "ncmapss_noise.json", "w"), indent=1)

# ---- Independence test: if independent noise were added to each xs channel, ratios of channels that are
# physically proportional (P15/P21 bypass duct; P40 vs the noise-free virtual P30) would show white noise of
# about sqrt(2)*sigma_rel. Their second-difference sigma is reported over all files, dev+test, every unit's
# first and last flight.
rt = []
for fn in sorted(glob.glob(D + "*.h5")):
    b = os.path.basename(fn).replace(".h5", "")
    if "DS08d" in b:
        continue
    with h5py.File(fn, "r") as f:
        av = [x.decode() for x in np.array(f["A_var"]).ravel()]
        xs = [x.decode() for x in np.array(f["X_s_var"]).ravel()]; xv = [x.decode() for x in np.array(f["X_v_var"]).ravel()]
        for part in ("dev", "test"):
            A = f["A_" + part][:]; X = f["X_s_" + part][:]; V = f["X_v_" + part][:]
            iu, ic = av.index("unit"), av.index("cycle")
            for u in np.unique(A[:, iu]):
                mu = A[:, iu] == u; cyc = np.unique(A[mu, ic])
                for c in (cyc[0], cyc[-1]):
                    m = mu & (A[:, ic] == c)
                    g = lambda n: X[m, xs.index(n)].astype(float); h = lambda n: V[m, xv.index(n)].astype(float)
                    rt.append({"file": b, "part": part, "unit": int(u), "cycle": int(c),
                               "P15_P21": float(sig2(np.log(g("P15") / g("P21")))),
                               "P40_P30v": float(sig2(np.log(g("P40") / h("P30")))),
                               "Ps30_rel": float(sig2(g("Ps30")) / np.mean(g("Ps30"))),
                               "rel_65dB": 10 ** (-65 / 20)})
mx = {k: float(max(r[k] for r in rt)) for k in ("P15_P21", "P40_P30v", "Ps30_rel")}
print("ratio test over", len(rt), "flights: max sigma", mx, " 65 dB rule would give ~", 10 ** (-65 / 20) * np.sqrt(2))
json.dump({"n_flights": len(rt), "max": mx, "rows": rt}, open(O + "ncmapss_noise_ratio_test.json", "w"), indent=1)
