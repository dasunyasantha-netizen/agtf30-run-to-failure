#!/usr/bin/env python3
"""Scenario file of the AGTF30 scenario mixer: defaults, validation, changed settings, and how a scenario is
applied to the generator (scripts/agtf30_gen).

A scenario is a JSON file with three data sections, following the two-layer design in
docs/synthetic/agtf30_feasibility_plan.md ("Scenario mixer design"):

  physics  (layer 1, needs simulation)   families, engine indices, replacement rule, max flights, seed base,
                                         K per family, lam, s_init
  overlay  (layer 2, instant, no re-run) sensor noise on X_s (SNR in dB, or off)
  paper2   (fixed off in v1)             climate, maintenance, sensor faults, ... (present so the file format
                                         already has them; any value other than the default is refused)

plus `run` (workers, priority) and `output` (h5 filter), which do not change the data.

Defaults are the released recipe (B_RECIPE.md, G_V11_FAULTS.md), read from gen_recipe so there is one source:
with defaults, engine index i of family F gives exactly the dataset's engine i of F (v1.0 ids 1-5, v1.1 ids 6-8).
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
GEN = HERE.parent / "agtf30_gen"
sys.path.insert(0, str(GEN))
import gen_recipe as R  # noqa: E402

FORMAT = "agtf30-scenario/1"
DATASET_N_PER_FAM = 60            # v1.0 and v1.1 subsets have 60 engines each (split rule uses this n)
DATASET_MAX_FLIGHTS = 400         # gen_engine.run_engine default: no end of life within 400 flights -> dropped
MAX_ENGINES = 200
FAMILY_NOTE = {"HPT": "HPT efficiency + flow", "HPC": "HPC efficiency + flow", "LPT": "LPT efficiency + flow",
               "FAN": "fan efficiency + flow", "MIX": "HPC + HPT", "HPTLPT": "HPT + LPT (v1.1)",
               "LPCHPC": "LPC + HPC (v1.1)", "ALL": "all five modules (v1.1)"}

# Original recipe values, captured before anything is applied (apply_physics changes the module globals).
RECIPE = {"seed_base": int(R.RECIPE_ID), "K": dict(R.K), "lam": float(R.LAM), "s_init": float(R.S_INIT)}

PAPER2_DEFAULTS = {
    "climate": "standard_day", "airport_altitude": "ncmapss_flights", "fleet_scatter": False,
    "maintenance": "none", "abrupt_events": False, "gearbox_fault": False, "nozzle_fault": False,
    "actuator_fault": False, "combustor_fault": False, "sensor_faults": "none", "label_noise": False,
}


def defaults() -> dict:
    return {
        "format": FORMAT,
        "name": "",
        "physics": {
            "families": ["HPT"],
            "engine_indices": [0],
            "replace_dropped": True,
            "max_flights": DATASET_MAX_FLIGHTS,
            "seed_base": RECIPE["seed_base"],
            "K": dict(RECIPE["K"]),
            "lam": RECIPE["lam"],
            "s_init": RECIPE["s_init"],
        },
        "overlay": {"noise": True, "snr_db": 65.0},
        "paper2": dict(PAPER2_DEFAULTS),
        "run": {"workers": 1, "priority": "idle"},
        "output": {"h5_filter": "gzip1"},
    }


# keys whose change does not change the data (they are still recorded)
NO_DATA_EFFECT = {"name", "run.workers", "run.priority", "output.h5_filter"}
# physics keys that may differ between launches of the same run folder (they do not change any trajectory)
PHYSICS_RESUMABLE = {"engine_indices", "max_flights", "replace_dropped"}


def parse_indices(spec) -> list[int]:
    """'0-4, 43' or [0, 1] -> sorted unique ints."""
    if isinstance(spec, list):
        vals = [int(x) for x in spec]
    else:
        vals = []
        for part in re.split(r"[,\s]+", str(spec).strip()):
            if not part:
                continue
            a, _, b = part.partition("-")
            a, b = int(a), int(b or a)
            if b < a or b - a > 10000:
                raise ValueError(f"bad range '{part}'")
            vals += range(a, b + 1)
    return sorted(set(vals))


def format_indices(vals: list[int]) -> str:
    out, i = [], 0
    vals = sorted(vals)
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[j] + 1:
            j += 1
        out.append(str(vals[i]) if i == j else f"{vals[i]}-{vals[j]}")
        i = j + 1
    return ", ".join(out)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def normalise(sc: dict) -> dict:
    """Fill missing keys from the defaults (a partial scenario file is fine) and canonicalise types."""
    s = _merge(defaults(), sc or {})
    p = s["physics"]
    p["engine_indices"] = parse_indices(p["engine_indices"])
    if isinstance(p["families"], str):
        p["families"] = [f.strip().upper() for f in p["families"].split(",") if f.strip()]
    p["families"] = [f for f in R.FAMILIES if f in {x.upper() for x in p["families"]}] + \
                    [f for f in p["families"] if f.upper() not in R.FAMILIES]      # unknown kept for the error
    for k in ("lam", "s_init"):
        p[k] = float(p[k])
    p["K"] = {f: float(v) for f, v in p["K"].items()}
    p["max_flights"] = int(p["max_flights"])
    p["seed_base"] = int(p["seed_base"])
    p["replace_dropped"] = bool(p["replace_dropped"])
    s["overlay"]["noise"] = bool(s["overlay"]["noise"])
    s["overlay"]["snr_db"] = float(s["overlay"]["snr_db"])
    s["run"]["workers"] = int(s["run"]["workers"])
    return s


def validate(sc: dict) -> list[str]:
    s = normalise(sc)
    e = []
    if s.get("format") != FORMAT:
        e.append(f"format must be {FORMAT}")
    p = s["physics"]
    bad = [f for f in p["families"] if f not in R.FAMILIES]
    if bad:
        e.append(f"unknown families {bad}; known: {list(R.FAMILIES)}")
    if not p["families"]:
        e.append("choose at least one fault family")
    if not p["engine_indices"]:
        e.append("choose at least one engine index")
    if any(i < 0 or i > 99_999 for i in p["engine_indices"]):
        e.append("engine indices must be 0-99999")
    if p["replace_dropped"] and any(i >= R.REPLACE_STRIDE for i in p["engine_indices"]):
        e.append(f"with 'replace dropped engines' on, indices must be below {R.REPLACE_STRIDE} "
                 f"(replacements use index + k*{R.REPLACE_STRIDE})")
    n = len(p["families"]) * len(p["engine_indices"])
    if n > MAX_ENGINES:
        e.append(f"{n} engines requested; v1 allows at most {MAX_ENGINES} per run")
    if not 1 <= p["max_flights"] <= R.N_SEQ:
        e.append(f"max flights must be 1-{R.N_SEQ} (each engine draws {R.N_SEQ} flights)")
    if not 0 <= p["seed_base"] < 2 ** 63:
        e.append("seed base must be a non-negative integer")
    unk = set(p["K"]) - set(R.FAMILIES)
    if unk:
        e.append(f"K given for unknown families {sorted(unk)}")
    for f, v in p["K"].items():
        if not 0.05 <= v <= 20:
            e.append(f"K[{f}] = {v} outside 0.05-20")
    if not 0 <= p["lam"] <= 2:
        e.append("lam must be 0-2")
    if not 0 <= p["s_init"] <= 2:
        e.append("s_init must be 0-2")
    o = s["overlay"]
    if o["noise"] and not 10 <= o["snr_db"] <= 150:
        e.append("noise SNR must be 10-150 dB")
    for k, v in s["paper2"].items():
        if k not in PAPER2_DEFAULTS:
            e.append(f"unknown paper2 knob '{k}'")
        elif v != PAPER2_DEFAULTS[k]:
            e.append(f"paper2.{k} is not available in v1 (must stay {PAPER2_DEFAULTS[k]!r})")
    if not 1 <= s["run"]["workers"] <= 4:
        e.append("workers must be 1-4")
    if s["run"]["priority"] not in ("idle", "below_normal", "normal"):
        e.append("priority must be idle, below_normal or normal")
    import gen_merge as M
    if s["output"]["h5_filter"] not in M.FILTERS:
        e.append(f"h5 filter must be one of {list(M.FILTERS)}")
    return e


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def changed_settings(sc: dict) -> list[dict]:
    """Every setting that differs from the released recipe defaults. K is compared only for chosen families,
    and the family/engine selection is reported as 'selection' (it chooses engines, it does not alter them)."""
    s, d = normalise(sc), defaults()
    fams = set(s["physics"]["families"])
    a, b = _flat(s), _flat(d)
    out = []
    for k in sorted(set(a) | set(b)):
        if k in ("format",) or k.startswith("physics.K.") and k.split(".")[-1] not in fams:
            continue
        if a.get(k) != b.get(k):
            kind = ("selection" if k in ("physics.families", "physics.engine_indices", "name") else
                    "no data effect" if k in NO_DATA_EFFECT else
                    "overlay" if k.startswith("overlay.") else "physics")
            out.append({"setting": k, "default": b.get(k), "value": a.get(k), "kind": kind})
    return out


def is_recipe_default(sc: dict) -> bool:
    """True if every data-changing setting is the released recipe's (the engines are the dataset's engines)."""
    return not [c for c in changed_settings(sc) if c["kind"] in ("physics", "overlay")
                and c["setting"] not in ("physics.max_flights", "physics.replace_dropped")] \
        and normalise(sc)["physics"]["max_flights"] >= DATASET_MAX_FLIGHTS


def physics_key(sc: dict) -> dict:
    """The part of the physics that fixes every trajectory (a run folder keeps it for life)."""
    p = normalise(sc)["physics"]
    return {"seed_base": p["seed_base"], "K": {f: p["K"][f] for f in p["families"]}, "lam": p["lam"],
            "s_init": p["s_init"], "recipe_id": RECIPE["seed_base"], "recipe_version": R.RECIPE_VERSION}


def sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def apply_physics(sc: dict):
    """Set the generator's recipe constants for this process. Called once in every worker process (each runs
    one scenario only) and in the exporting process. The random draws (wear parameters, flights, noise) depend
    only on (seed base, family id, engine index), so changing K, lam or s_init keeps the same draws."""
    p = normalise(sc)["physics"]
    R.RECIPE_ID = p["seed_base"]
    R.LAM = p["lam"]
    R.S_INIT = p["s_init"]
    R.K.clear()
    R.K.update(RECIPE["K"])
    R.K.update(p["K"])


def load(path) -> dict:
    return normalise(json.loads(pathlib.Path(path).read_text()))


def save(sc: dict, path):
    pathlib.Path(path).write_text(json.dumps(normalise(sc), indent=1))
