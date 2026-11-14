#!/usr/bin/env python3
"""D2 learnability check: small LSTM / GRU / transformer on per-flight windows vs the training-mean baseline.

    python learn_check.py --h5 $AGTF30_WORK/agtf30_c/pilot/h5 --out $AGTF30_WORK/agtf30_c/pilot/learn

Train on every subset's dev engines pooled, test on the test engines (whole engines, as the files split them).
Same windows, split and normalisation for the three architectures. Primary feature set 'res' (declared before
the pilot results): W flight means + X_s operating-condition residual means (agtf30_data.Normaliser, fitted on
the first 10 flights of the training engines). Secondary 'mean': the plain flight means the project's
NCMAPSSLoader gave. Pass rule (decision #9) judged on the LSTM with 'res': RMSE and N-CMAPSS score both below
the training-mean baseline. Seeds 0-2; the result is the mean over seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agtf30_data import Normaliser, Scaler, feature_names, flight_features, iter_units, make_windows, select  # noqa: E402
from rul_metrics import nasa_score, rmse  # noqa: E402  (copy of the original project's metrics)

RUL_SCALE = 100.0


class RNN(nn.Module):
    def __init__(self, n_in, cell="lstm", hidden=32):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class TF(nn.Module):
    def __init__(self, n_in, window, d=32, heads=2, layers=1):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 64, 0.1, batch_first=True), layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.enc(self.inp(x) + self.pos)[:, -1]).squeeze(-1)


def train(model, X, y, epochs, seed, lr=1e-3, bs=64):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y / RUL_SCALE)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy() * RUL_SCALE


def metrics(y, p, uid):
    per_file = {}
    for f in sorted({u.split(":")[0] for u in uid}):
        m = np.array([u.startswith(f + ":") for u in uid])
        per_file[f] = {"rmse": rmse(y[m], p[m]), "score": nasa_score(y[m], p[m]), "n": int(m.sum())}
    return {"rmse": rmse(y, p), "score": nasa_score(y, p), "n": int(len(y)), "per_file": per_file}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--xs-key", default="X_s")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(a.h5).glob("AGTF30_*.h5"))
    t0 = time.time()
    norm = Normaliser().fit(iter_units(files, "dev", a.xs_key, max_cycle=10))
    names = feature_names(next(iter_units(files, "dev", a.xs_key, max_cycle=1))[1])
    Ftr = {k: flight_features(u, norm) for k, u in iter_units(files, "dev", a.xs_key)}
    Fte = {k: flight_features(u, norm) for k, u in iter_units(files, "test", a.xs_key)}
    R = {"files": [f.name for f in files], "n_train_units": len(Ftr), "n_test_units": len(Fte),
         "normaliser_r2": dict(zip(names[len(names) - len(norm.r2):], map(float, norm.r2))),
         "window": a.window, "epochs": a.epochs, "seeds": a.seeds, "load_s": time.time() - t0, "sets": {}}
    for fs in ("res", "mean"):
        cols = select(None, names, fs)
        Xtr, ytr, _, _ = make_windows(Ftr, a.window, cols)
        Xte, yte, ute, cte = make_windows(Fte, a.window, cols)
        sc = Scaler().fit(Xtr)
        Xtr, Xte = sc(Xtr), sc(Xte)
        base = np.full_like(yte, ytr.mean())
        S = {"features": [names[i] for i in cols], "n_train_windows": int(len(ytr)),
             "baseline_train_mean": {"value": float(ytr.mean()), **metrics(yte, base, ute)}, "models": {}}
        preds = {}
        for arch in ("lstm", "gru", "transformer"):
            runs = []
            for s in range(a.seeds):
                torch.manual_seed(s)
                m = TF(Xtr.shape[-1], a.window) if arch == "transformer" else RNN(Xtr.shape[-1], arch)
                t1 = time.time()
                train(m, Xtr, ytr, a.epochs, s)
                p = predict(m, Xte)
                runs.append({"seed": s, "train_s": time.time() - t1, **metrics(yte, p, ute)})
                preds[f"{arch}_{s}"] = p
            S["models"][arch] = {"rmse_mean": float(np.mean([r["rmse"] for r in runs])),
                                 "rmse_sd": float(np.std([r["rmse"] for r in runs])),
                                 "score_mean": float(np.mean([r["score"] for r in runs])),
                                 "beats_baseline_rmse": bool(np.mean([r["rmse"] for r in runs]) < S["baseline_train_mean"]["rmse"]),
                                 "beats_baseline_score": bool(np.mean([r["score"] for r in runs]) < S["baseline_train_mean"]["score"]),
                                 "runs": runs}
            print(fs, arch, round(S["models"][arch]["rmse_mean"], 2), "baseline", round(S["baseline_train_mean"]["rmse"], 2), flush=True)
        np.savez_compressed(out / f"pred_{fs}.npz", y=yte, unit=ute, cycle=cte, baseline=base, **preds)
        R["sets"][fs] = S
    L = R["sets"]["res"]["models"]["lstm"]
    R["pass_rule_lstm_res"] = bool(L["beats_baseline_rmse"] and L["beats_baseline_score"])
    (out / "learn.json").write_text(json.dumps(R, indent=1))
    print("LEARN_DONE pass", R["pass_rule_lstm_res"], f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
