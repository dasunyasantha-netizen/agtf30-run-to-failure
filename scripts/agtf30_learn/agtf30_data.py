"""Shared data loading, per-flight features and windowing for the AGTF30 .h5 files (and N-CMAPSS files).

Kept independent of the generator so the model study can reuse it unchanged:

    from agtf30_data import load_units, Normaliser, flight_features, make_windows, Scaler
    files = ["AGTF30_HPT.h5"]
    norm = Normaliser().fit(iter_units(files, "dev", max_cycle=10))   # healthy model, training units only
    F = {k: flight_features(u, norm) for k, u in iter_units(files, "dev")}   # one unit in memory at a time
    Xw, yw, uid, cyc = make_windows(F, window=10)

Rows of a unit are contiguous and in cycle order in both N-CMAPSS and the AGTF30 files, so a unit is read as
one row slice (memory: one unit at a time, about 1 M rows x 18 columns for a long-class engine).

Per-flight features (one row per flight, `feature_names`):
  * `mean_<W/X_s>`: flight means of W and X_s, the same numbers the original project's NCMAPSSLoader produced;
  * `res_<X_s>`: flight means of the X_s residual after removing the operating condition with a healthy
    regression X_s ~ poly2(W) fitted on the first `healthy_cycles` flights of the training units only
    (relative residual, x / x_hat - 1, so channels are comparable).
"""
from __future__ import annotations

import pathlib

import h5py
import numpy as np

W_COLS = ["alt", "Mach", "TRA", "T2"]


def _names(h, key):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in h[key][:]]


def load_units(paths, split="dev", xs_key="X_s", tag_from_file=True, units=None):
    """Return {unit_key: dict(W, X_s, cycle, rul, file, unit)}; unit_key = '<file stem>:<unit>'."""
    return dict(iter_units(paths, split, xs_key, tag_from_file, units=units))


def iter_units(paths, split="dev", xs_key="X_s", tag_from_file=True, max_cycle=None, units=None, extra=()):
    """Yield (unit_key, unit dict) one unit at a time; max_cycle reads only that unit's first flights;
    units: optional collection of unit keys to read (others are skipped unread), e.g. a scaling-subset list
    from configs/agtf30_v1/scaling_subsets.json; extra: further groups to read for the unit (e.g. "A", "T",
    "X_s_clean"), returned as unit[g] with names in unit[g + "_names"]."""
    units = None if units is None else set(units)
    for p in map(pathlib.Path, paths):
        with h5py.File(p, "r") as h:
            if h[f"A_{split}"].shape[0] == 0:
                continue
            a_names = _names(h, "A_var")
            A = h[f"A_{split}"]
            unit_col = A[:, a_names.index("unit")]
            cyc_col = A[:, a_names.index("cycle")]
            starts = np.flatnonzero(np.r_[True, np.diff(unit_col) != 0])
            ends = np.r_[starts[1:], len(unit_col)]
            xs_names = _names(h, "X_s_var")
            w_names = _names(h, "W_var")
            for s, e in zip(starts, ends):
                u = int(unit_col[s])
                key = f"{p.stem}:{u}" if tag_from_file else str(u)
                if units is not None and key not in units:
                    continue
                if max_cycle is not None:
                    e = s + int(np.searchsorted(cyc_col[s:e], max_cycle, side="right"))
                d = {"W": h[f"W_{split}"][s:e], "X_s": h[f"{xs_key}_{split}"][s:e],
                     "cycle": cyc_col[s:e].astype(np.int64), "rul": h[f"Y_{split}"][s:e, 0].astype(np.int64),
                     "file": p.name, "unit": u, "xs_names": xs_names, "w_names": w_names}
                for g in extra:
                    d[g], d[g + "_names"] = h[f"{g}_{split}"][s:e], _names(h, f"{g}_var")
                yield key, d


def _poly2(W):
    n, k = W.shape
    cols = [np.ones(n)] + [W[:, i] for i in range(k)]
    cols += [W[:, i] * W[:, j] for i in range(k) for j in range(i, k)]
    return np.column_stack(cols)


class Normaliser:
    """Healthy operating-condition model: X_s ~ poly2(standardised W), least squares, per channel."""

    def __init__(self, healthy_cycles=10, max_rows=400_000, seed=0):
        self.healthy_cycles, self.max_rows, self.seed = healthy_cycles, max_rows, seed

    def fit(self, units):
        """units: dict or iterable of (key, unit) pairs (e.g. iter_units(..., max_cycle=healthy_cycles))."""
        Ws, Xs = [], []
        for _, u in (units.items() if isinstance(units, dict) else units):
            m = u["cycle"] <= self.healthy_cycles
            Ws.append(u["W"][m]); Xs.append(u["X_s"][m])
        n = sum(len(w) for w in Ws)
        rng = np.random.default_rng(self.seed)
        if n > self.max_rows:
            # the same rows, in the same order, as stacking everything and taking [i], without the stacked copy
            i = rng.choice(n, self.max_rows, replace=False)
            off = np.cumsum([0] + [len(w) for w in Ws])
            k = np.searchsorted(off, i, side="right") - 1
            W = np.empty((len(i), Ws[0].shape[1])); X = np.empty((len(i), Xs[0].shape[1]))
            for j in np.unique(k):
                m = k == j
                W[m], X[m] = Ws[j][i[m] - off[j]], Xs[j][i[m] - off[j]]
            del Ws, Xs
        else:
            W, X = np.vstack(Ws), np.vstack(Xs)
        self.w_mu, self.w_sd = W.mean(0), W.std(0) + 1e-12
        P = _poly2((W - self.w_mu) / self.w_sd)
        self.coef, *_ = np.linalg.lstsq(P, X, rcond=None)
        fit = P @ self.coef
        self.r2 = 1 - ((X - fit) ** 2).sum(0) / ((X - X.mean(0)) ** 2).sum(0)
        return self

    def residual(self, W, X):
        return X / (_poly2((W - self.w_mu) / self.w_sd) @ self.coef) - 1.0


def feature_names(unit):
    return ([f"mean_{c}" for c in unit["w_names"] + unit["xs_names"]] + [f"res_{c}" for c in unit["xs_names"]])


def flight_features(unit, norm: Normaliser | None):
    """One row per flight (cycle order): [means of W and X_s, mean relative residual of X_s]; plus rul per flight."""
    cyc = unit["cycle"]
    starts = np.flatnonzero(np.r_[True, np.diff(cyc) != 0])
    ends = np.r_[starts[1:], len(cyc)]
    WX = np.hstack([unit["W"], unit["X_s"]])
    R = norm.residual(unit["W"], unit["X_s"]) if norm is not None else np.zeros_like(unit["X_s"])
    F = np.array([np.r_[WX[s:e].mean(0), R[s:e].mean(0)] for s, e in zip(starts, ends)])
    return {"F": F, "cycle": cyc[starts], "rul": unit["rul"][starts]}


def select(F, names, which="res"):  # F unused; kept for call symmetry
    """Column subset: 'res' (residuals + W means), 'mean' (loader means only) or 'all'."""
    if which == "all":
        idx = list(range(len(names)))
    elif which == "mean":
        idx = [i for i, n in enumerate(names) if n.startswith("mean_")]
    else:
        idx = [i for i, n in enumerate(names) if n.startswith("res_") or n[5:] in W_COLS]
    return idx


def make_windows(feats, window=10, cols=None):
    """Sliding windows over flights, one per flight (left-padded with the first flight), target = RUL of the last."""
    X, y, uid, cyc = [], [], [], []
    for k, f in feats.items():
        F = f["F"] if cols is None else f["F"][:, cols]
        pad = np.vstack([np.repeat(F[:1], window - 1, 0), F])
        for i in range(len(F)):
            X.append(pad[i:i + window]); y.append(f["rul"][i]); uid.append(k); cyc.append(f["cycle"][i])
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(uid), np.asarray(cyc)


class Scaler:
    """Per-feature z-score fitted on training windows."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
