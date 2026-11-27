"""One representative flight for the paper's example-record figure: HPT subset (release 1.0), development engine 11
(medium flights, class 2; life 67 flights), flight 34 (half of its life). Reads W (alt, Mach, TRA), X_s (measured, with noise) and
X_s_clean (Nf, T48, Wf) and X_a N1_target, and the flight's row of the per-flight table. Read-only on the release.

  python extract_flight.py      (release $AGTF30_WORK/agtf30_v1 or AGTF30_V10; output $AGTF30_WORK/agtf30_v11_work/fig_flight/flight.npz)
Output: flight.npz in the output folder.
"""
import os, h5py, numpy as np

ROOTW = os.environ.get("AGTF30_WORK", "work")
REL = os.environ.get("AGTF30_V10", os.path.join(ROOTW, "agtf30_v1"))
OUTD = os.path.join(ROOTW, "agtf30_v11_work", "fig_flight")
FILE, UNIT, CYCLE = "AGTF30_HPT.h5", 11, 34


def names(h, k):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in h[k][:]]


with h5py.File(os.path.join(REL, FILE), "r") as h:
    a = names(h, "A_var")
    A = h["A_dev"]
    unit, cyc = A[:, a.index("unit")], A[:, a.index("cycle")]
    rows = np.flatnonzero((unit == UNIT) & (cyc == CYCLE))
    s, e = int(rows[0]), int(rows[-1]) + 1
    assert len(rows) == e - s, "rows of one flight must be contiguous"
    w, xs, xa = names(h, "W_var"), names(h, "X_s_var"), names(h, "X_a_var")
    W, X, XC, XA = h["W_dev"][s:e], h["X_s_dev"][s:e], h["X_s_clean_dev"][s:e], h["X_a_dev"][s:e]
    C = h["cycles"][:]
    c = C[(C["unit"] == UNIT) & (C["cycle"] == CYCLE)][0]
    U = h["units"][:]
    u = U[U["unit"] == UNIT][0]
out = dict(alt=W[:, w.index("alt")], mach=W[:, w.index("Mach")], tra=W[:, w.index("TRA")], t2=W[:, w.index("T2")],
           nf=X[:, xs.index("Nf")], nf_clean=XC[:, xs.index("Nf")], t48=X[:, xs.index("T48")],
           t48_clean=XC[:, xs.index("T48")], wf=X[:, xs.index("Wf")], wf_clean=XC[:, xs.index("Wf")],
           n1_target=XA[:, xa.index("N1_target")], hi=float(c["HI"]), rul=int(c["RUL"]), hours=float(c["hours"]),
           src_key=c["src_key"], fc=int(u["Fc"]), eol=int(u["eol"]), unit=UNIT, cycle=CYCLE, h5=FILE)
os.makedirs(OUTD, exist_ok=True)
np.savez_compressed(os.path.join(OUTD, "flight.npz"), **out)
print({k: (v.shape if hasattr(v, "shape") and np.ndim(v) else v) for k, v in out.items()})
