#!/usr/bin/env python3
"""Run a scenario: simulate its engines (layer 1, resumable), then write the h5 files (layer 2 overlay applied).

    python mixer_run.py --new my_scenario.json [--root <runs folder>]   # make a run folder and run it
    python mixer_run.py <root>/<name>                                   # run / resume a run folder
    python mixer_run.py <root>/<name> --export-only                     # only rebuild the h5 files
    python mixer_run.py <root>/<name> --export-only --snr 50            # new overlay, no re-simulation
    python mixer_run.py <root>/<name> --export-only --no-noise          #   -> exports/<tag>/
    (<root>: --root, or $AGTF30_MIXER_ROOT, or $AGTF30_WORK/agtf30_mixer/runs)

Run folder:
  scenario.json      the scenario of this run (the UI writes it; a resume re-reads it)
  physics.json       the frozen physics (seed base, K, lam, s_init); a resume with other physics is refused
  runs/<FAM>/eNNNN/  per-flight checkpoints, exactly gen_engine's layout (c####.npz/json, done/dropped.json)
  state.json         per-engine result of the last launch;  run.log  progress log;  pid.txt  runner process
  h5/                AGTF30_<FAM>.h5 (released N-CMAPSS layout) + provenance.json
  exports/<tag>/     the same for other overlays (--snr / --no-noise)

Engines run in a process pool (workers 1-4) at Idle CPU priority by default (gen_run.set_priority), one engine
per worker process. A dropped engine (solver failure, decision #7) is replaced by index + k*1000 as in the dataset
when replace_dropped is on. An engine that reaches max_flights without end of life is kept as censored
(RUL = -1 in the h5, units.eol = -1); raising max_flights and resuming flies it on.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import pathlib
import platform
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenario as S  # noqa: E402  (puts scripts/agtf30_gen on sys.path)
import gen_recipe as R  # noqa: E402

MAX_REPLACEMENTS = 3
WORK = os.environ.get("AGTF30_WORK", "work")     # work root: agtf30_cbuild/, agtf30_c/library, agtf30_mixer/runs
DEFAULT_ROOT = os.environ.get("AGTF30_MIXER_ROOT", f"{WORK}/agtf30_mixer/runs")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
MIXER_VERSION = "agtf30-mixer 1.0 (2026-11-10)"


def machine() -> dict:
    """Paths of this machine (not part of the scenario; recorded in provenance)."""
    return {"lib": os.environ.get("AGTF30_LIB", f"{WORK}/agtf30_c/library"),
            "bin": os.environ.get("AGTF30_BIN_B", f"{WORK}/agtf30_cbuild/bin_b"),
            "bin_fine": os.environ.get("AGTF30_BIN_FINE", f"{WORK}/agtf30_cbuild/bin_fine")}


def _log(run: pathlib.Path, msg: str):
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(run / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _atomic_json(path: pathlib.Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _cfg(run: pathlib.Path, slot: int) -> dict:
    import gen_engine as G
    m = machine()
    cfg = G.default_cfg(run, m["lib"], max(S.DATASET_N_PER_FAM, slot + 1))
    cfg["bin"], cfg["bin_fine"] = m["bin"], m["bin_fine"]
    return cfg


def run_slot(task):
    """One engine slot in a worker process: the dataset's replacement rule, the scenario's max flights."""
    sc, run, fam, slot = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    S.apply_physics(sc)
    import gen_engine as G
    run = pathlib.Path(run)
    p = S.normalise(sc)["physics"]
    mf = p["max_flights"]
    cfg = _cfg(run, slot)
    lib = G.Library(cfg["lib"])
    hist = []
    for k in range(MAX_REPLACEMENTS + 1 if p["replace_dropped"] else 1):
        idx = slot + k * R.REPLACE_STRIDE
        try:
            r = G.run_engine(cfg, fam, slot, idx, lib=lib, max_flights=mf + 1, stop_after=mf)
        except (G.EnvFault, MemoryError, OSError) as e:      # the PC failed, not the engine: resumable
            hist.append({"status": "env_error", "reason": f"{type(e).__name__}: {e}"[:2000], "family": fam,
                         "slot": slot, "engine_index": idx})
            break
        except Exception as e:  # noqa: BLE001  (logged as a drop, never silent)
            r = {"status": "dropped", "reason": f"exception {type(e).__name__}: {e}", "family": fam, "slot": slot,
                 "engine_index": idx}
            G._atomic_json(G.engine_dir(cfg, fam, idx) / "dropped.json", r)
        if r["status"] == "stopped":
            r = dict(r, status="censored", reason=f"no end of life within max_flights={mf}")
        hist.append(r)
        if r["status"] in ("done", "censored"):
            break
    return {"family": fam, "slot": slot, "final": hist[-1], "history": hist}


def check_physics(run: pathlib.Path, sc: dict):
    key = S.physics_key(sc)
    pj = run / "physics.json"
    if pj.exists():
        old = json.loads(pj.read_text())
        # families may be added; a family already in the run must keep its K
        same = all(old["key"][k] == key[k] for k in ("seed_base", "lam", "s_init")) and \
            all(old["key"]["K"].get(f, v) == v for f, v in key["K"].items())
        if not same:
            raise SystemExit(f"physics of {run} differ from the scenario ({old['key']} vs {key}); "
                             "physics changes need a new run name (overlay changes: use --export-only)")
        old["key"]["K"].update(key["K"])
        key = old["key"]
    _atomic_json(pj, {"key": key, "sha256": S.sha256(key)})


def simulate(run: pathlib.Path, sc: dict):
    s = S.normalise(sc)
    p = s["physics"]
    check_physics(run, s)
    import gen_run
    gen_run.set_priority(s["run"]["priority"])
    tasks = [(R.class_plan(f, slot + 1)[slot], f, slot) for f in p["families"] for slot in p["engine_indices"]]
    tasks.sort(key=lambda x: (-x[0], x[1], x[2]))           # long-flight class first, as gen_run
    (run / "pid.txt").write_text(f"{os.getpid()}\n")
    with open(run / "launches.jsonl", "a") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(),
                            "scenario_sha256": S.sha256(s), "engines": len(tasks), "workers": s["run"]["workers"],
                            "priority": s["run"]["priority"], "machine": machine()}) + "\n")
    _log(run, f"start {len(tasks)} engines, {s['run']['workers']} workers, priority {s['run']['priority']}")
    state = {"started": dt.datetime.now().isoformat(timespec="seconds"), "engines": len(tasks), "results": []}
    _atomic_json(run / "state.json", dict(state, status="running"))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    w = min(s["run"]["workers"], len(tasks))
    with ctx.Pool(w, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_slot, [(s, str(run), f, slot) for _, f, slot in tasks]):
            state["results"].append(r)
            for h in r["history"]:
                if h["status"] in ("dropped", "env_error"):
                    _log(run, f"{h['status'].upper()} {h['family']} slot {h['slot']} engine {h['engine_index']}: "
                              f"{str(h.get('reason'))[:300]}")
            fin = r["final"]
            _log(run, f"engine {r['family']}/{r['slot']} -> {fin['status']} (index {fin['engine_index']}, "
                      f"eol {fin.get('eol')}) {len(state['results'])}/{len(tasks)}, {time.time() - t0:.0f}s")
            _atomic_json(run / "state.json", dict(state, status="running"))
    state.update(wall_s=time.time() - t0, finished=dt.datetime.now().isoformat(timespec="seconds"))
    env = sum(r["final"]["status"] == "env_error" for r in state["results"])
    state["status"] = "env_error" if env else "simulated"
    _atomic_json(run / "state.json", state)
    return state


# ---------------------------------------------------------------- export (layer 2 overlay + h5 writing)

def final_engine(run: pathlib.Path, fam: str, slot: int):
    """The engine that fills this slot: done, or censored (checkpoints, not dropped). (dir, status, replacements)"""
    for k in range(MAX_REPLACEMENTS + 1):
        d = run / "runs" / fam / f"e{slot + k * R.REPLACE_STRIDE:04d}"
        if (d / "done.json").exists():
            return d, "done", k
        if (d / "dropped.json").exists():
            continue
        if (d / "c0001.json").exists():
            return d, "censored", k
        return None, "missing", k
    return None, "dropped", MAX_REPLACEMENTS


def overlay_tag(ov: dict) -> str:
    return "nonoise" if not ov["noise"] else f"snr{ov['snr_db']:g}"


def export(run: pathlib.Path, sc: dict, dest: pathlib.Path) -> list[dict]:
    import h5py
    import numpy as np
    import gen_engine as G
    import gen_merge as M
    s = S.normalise(sc)
    S.apply_physics(s)
    p, ov = s["physics"], s["overlay"]
    m = machine()
    M.LIB = G.Library(m["lib"])
    M.SIG_REL = 10 ** (-ov["snr_db"] / 20)
    cfg = _cfg(run, 0)
    dest.mkdir(parents=True, exist_ok=True)
    changed = S.changed_settings(s)
    groups = {"W": M.W_VAR, "X_s": [n for n, _ in M.X_S], "X_s_clean": [n for n, _ in M.X_S],
              "X_v": [n for n, _ in M.X_V], "X_a": [n for n, _ in M.X_A], "T": R.T_COLS, "A": M.A_VAR, "Y": None}
    summary = []
    for fam in p["families"]:
        dsplit = R.split_plan(fam, R.class_plan(fam, S.DATASET_N_PER_FAM))
        eng = {}
        for slot in p["engine_indices"]:
            d, st, k = final_engine(run, fam, slot)
            if d is not None:
                eng[slot] = (d, st, k)
        path = dest / f"AGTF30_{fam}.h5"
        tmp = path.with_suffix(".tmp.h5")
        cyc_rows, unit_rows, units_info = [], [], []
        filt = M.FILTERS[s["output"]["h5_filter"]]
        with h5py.File(tmp, "w") as h:
            for g, names in groups.items():
                if names is not None:
                    h.create_dataset(f"{g}_var", data=M._bytes(names))
                for sp in ("dev", "test"):
                    ncol = 1 if names is None else len(names)
                    h.create_dataset(f"{g}_{sp}", shape=(0, ncol), maxshape=(None, ncol),
                                     dtype=np.int64 if g == "Y" else np.float64, chunks=(M.CHUNK_ROWS, ncol), **filt)
            for sp in ("dev", "test"):
                for slot in sorted(eng):
                    d, st, nrep = eng[slot]
                    split = dsplit[slot] if slot < S.DATASET_N_PER_FAM else "dev"
                    if split != sp:
                        continue
                    E = json.loads((d / "engine.json").read_text())
                    E["split"] = split
                    if st == "done":
                        done = json.loads((d / "done.json").read_text())
                        last, eol, limit = done["eol"], done["eol"], done["limit"]
                    else:
                        last = max(int(j.stem[1:]) for j in d.glob("c[0-9][0-9][0-9][0-9].json"))
                        eol, limit = -1, "none"
                    unit, hrs = slot + 1, 0.0
                    for c in range(1, last + 1):
                        rows, cy = M.flight_rows(d, E, unit, c, last)
                        if not ov["noise"]:
                            rows["X_s"] = rows["X_s_clean"]
                        if eol < 0:                                  # censored: RUL unknown
                            rows["Y"] = np.full_like(rows["Y"], -1)
                            cy = cy[:3] + (-1,) + cy[4:]
                        for g, X in rows.items():
                            ds = h[f"{g}_{sp}"]
                            n0 = ds.shape[0]
                            ds.resize(n0 + len(X), axis=0)
                            ds[n0:] = X
                        cyc_rows.append(cy)
                        hrs += cy[8]
                    unit_rows.append((unit, slot, E["engine_index"], sp.encode(), fam.encode(), E["cls"],
                                      *[E["u"][mm] for mm in ("FAN", "LPC", "HPC", "HPT", "LPT")], E["r_bg"], E["a"],
                                      E["b"], E["E_s"], E["K"], E["onset"] if E["onset"] is not None else -1, eol,
                                      limit.encode(), E["hi0"], nrep, E["seed_entropy"], hrs))
                    units_info.append({"unit": unit, "slot": slot, "engine_index": E["engine_index"], "split": sp,
                                       "status": st, "flights": last, "eol": eol, "limit": limit,
                                       "replacements": nrep, "dir": str(d)})
            h.create_dataset("cycles", data=np.array(cyc_rows, dtype=M.CYC_DT),
                             **({} if s["output"]["h5_filter"] == "none" else M.FILTERS["gzip4"]))
            h.create_dataset("units", data=np.array(unit_rows, dtype=M.UNIT_DT))
            h.create_dataset("events", data=np.zeros(0, dtype=M.EVENT_DT), maxshape=(None,), chunks=(64,))
            missing = sorted(set(p["engine_indices"]) - set(eng))
            h.attrs.update({
                "dataset": "AGTF30 scenario mixer output (N-CMAPSS layout)",
                "mixer": MIXER_VERSION, "family": fam, "recipe_id": S.RECIPE["seed_base"],
                "recipe_version": R.RECIPE_VERSION if fam not in R.V11_FAMILIES
                else R.RECIPE_VERSION + "; " + R.V11_RECIPE,
                "recipe_default": bool(S.is_recipe_default(s)),
                "scenario": json.dumps(s, sort_keys=True), "scenario_sha256": S.sha256(s),
                "changed_settings": json.dumps(changed), "seed_base": p["seed_base"],
                "git_commit": M.git_commit(), "c_build_id": json.dumps(M.build_id(cfg)),
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "noise": (f"X_s: Gaussian, sigma = 10^(-{ov['snr_db']:g}/20) * rms(clean flight signal), per channel"
                          " and flight; X_s_clean: none") if ov["noise"] else "none: X_s = X_s_clean",
                "split_rule": "dataset split (n = 60 per subset) for engine indices 0-59; dev for higher indices",
                "W31_W32": "NaN: AGTF30 outputs only the total HPC bleed (X_a Wbleed_HPC)",
                "units_note": "imperial as N-CMAPSS (degR, psia, rpm, lbm/s); P15 = AGTF30 S17.Pt; T30/Ps30/P30 ="
                              " AGTF30 station 36; unit = engine slot + 1",
                "censored_units": json.dumps([u["unit"] for u in units_info if u["status"] == "censored"]),
                "units_missing": json.dumps(missing), "filter": s["output"]["h5_filter"],
                "generator": "scripts/agtf30_gen + scripts/agtf30_mixer", "n_units": len(unit_rows)})
        tmp.replace(path)
        summary.append({"family": fam, "file": str(path), "bytes": path.stat().st_size, "units": units_info,
                        "missing_slots": missing, "flights": len(cyc_rows)})
    prov = {"mixer": MIXER_VERSION, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "scenario": s, "scenario_sha256": S.sha256(s), "recipe_default": S.is_recipe_default(s),
            "changed_settings": changed, "physics_key": S.physics_key(s), "overlay_tag": overlay_tag(ov),
            "machine": dict(machine(), host=platform.node(), python=sys.version.split()[0], numpy=np.__version__,
                            h5py=h5py.__version__, c_build_id=M.build_id(cfg)),
            "git_commit": M.git_commit(), "files": summary}
    _atomic_json(dest / "provenance.json", prov)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run folder")
    ap.add_argument("--new", help="scenario file: make <root>/<name> and run it")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--snr", type=float, default=None, help="overlay: noise SNR in dB (export to exports/<tag>/)")
    ap.add_argument("--no-noise", action="store_true", help="overlay: no noise (export to exports/nonoise/)")
    a = ap.parse_args(argv)
    if a.new:
        sc = S.load(a.new)
        name = sc["name"] or pathlib.Path(a.new).stem
        if not NAME_RE.match(name):
            raise SystemExit(f"bad run name {name!r}")
        run = pathlib.Path(a.root) / name
        run.mkdir(parents=True, exist_ok=True)
        S.save(dict(sc, name=name), run / "scenario.json")
    elif a.run:
        run = pathlib.Path(a.run)
    else:
        ap.error("give a run folder or --new scenario.json")
    sc = S.load(run / "scenario.json")
    err = S.validate(sc)
    if err:
        raise SystemExit("scenario errors:\n  " + "\n  ".join(err))
    if a.export_only:
        if not (run / "physics.json").exists():
            raise SystemExit("nothing simulated yet in this run folder")
        if a.snr is not None or a.no_noise:
            sc["overlay"] = {"noise": not a.no_noise, "snr_db": a.snr if a.snr is not None else sc["overlay"]["snr_db"]}
            dest = run / "exports" / overlay_tag(sc["overlay"])
        else:
            dest = run / "h5"
        _log(run, f"export (overlay {overlay_tag(sc['overlay'])}) -> {dest}")
        for r in export(run, sc, dest):
            _log(run, f"wrote {r['file']} ({len(r['units'])} units, {r['flights']} flights, {r['bytes']/1e6:.1f} MB)")
        _log(run, "MIXER_EXPORT_DONE")
        return
    state = simulate(run, sc)
    if state["status"] != "simulated":
        _log(run, "MIXER_RUN_ENV_ERROR: some engines stopped on PC faults (not drops); resume the run")
        return
    _atomic_json(run / "state.json", dict(state, status="exporting"))
    _log(run, "export h5")
    for r in export(run, sc, run / "h5"):
        _log(run, f"wrote {r['file']} ({len(r['units'])} units, {r['flights']} flights, {r['bytes']/1e6:.1f} MB)")
    _atomic_json(run / "state.json", dict(state, status="finished"))
    _log(run, "MIXER_RUN_DONE")


if __name__ == "__main__":
    main()
