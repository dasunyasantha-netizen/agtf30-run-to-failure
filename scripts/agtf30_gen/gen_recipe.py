#!/usr/bin/env python3
"""C1 engine sampler and C2 flight sequence for the AGTF30 dataset (B_RECIPE.md, approved 2026-11-03).

Every number here is the approved recipe's; the wear vectors and constants are imported from the
phase B script that produced them (scripts/agtf30_recipe/b5_life.py), so there is one source.

Seeds: SeedSequence([RECIPE_ID, family_id, engine_index]) with child streams (spawn keys)
  (0,) wear parameters, (1,) flight sequence, (2, cycle) sensor noise of one flight.
Same (family, engine_index) -> same engine, same flights and same noise, on any machine.

Wear (B_RECIPE section 6), t = flights flown, frozen within a flight:
  theta(t) = u*0.2*SHORT + r_bg*0.12*(t/80)*LONG + g(tau)*DIR_family,  g = expm1(K*a*tau**b) [% eff]
  onset = first t with sum_{j<=t} (1.48 + hours_j) > E_s;  tau = t - onset (0 before onset).
Flight c (1-based) is flown, and checked at the reference point, with theta(c): the wear including
that flight's usage (the B5 life simulation's check t = c). The new engine (t = 0) is HI0.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agtf30_recipe"))
sys.path.insert(0, str(HERE.parent / "agtf30_cbuild"))
from b5_life import ALPHA, DIR, E_MU, E_SD, LONG, MODS, SHORT, add, scale, vec  # noqa: E402

RECIPE_ID = 20261103                     # B_RECIPE.md approval date
RECIPE_VERSION = "B_RECIPE.md approved 2026-11-03 (sections 2-9, 10a)"
FAMILIES = {"HPT": 1, "HPC": 2, "LPT": 3, "FAN": 4, "MIX": 5,    # = b5_life.FAM_SEED
            "HPTLPT": 6, "LPCHPC": 7, "ALL": 8}                 # v1.1 (G_V11_FAULTS.md)
S_INIT, LAM, L_BG = 0.2, 0.12, 80.0                               # B5 v2 / recipe section 6
K = {"HPT": 1.9, "HPC": 1.8, "LPT": 2.6, "FAN": 2.6, "MIX": 1.25,  # growth time scale, recipe section 6
     "HPTLPT": 1.6, "LPCHPC": 1.4, "ALL": 0.72}                     # v1.1, G_V11_FAULTS.md section 4
V11_FAMILIES = ("HPTLPT", "LPCHPC", "ALL")
V11_RECIPE = "G_V11_FAULTS.md (2026-11-09): directions section 2, K section 4; otherwise B_RECIPE.md as v1.0"
SPLIT_KEY = 999_999                       # SeedSequence([RECIPE_ID, family_id, SPLIT_KEY]) for the test draw
REPLACE_STRIDE = 1000                     # replacement engine index = slot + k*1000 (decision #7)
N_SEQ = 600                               # flights drawn per engine (life is ~50-130)
T_COLS = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"]           # N-CMAPSS T_var order
T_SRC = [("FAN", "Eff"), ("FAN", "Wc"), ("LPC", "Eff"), ("LPC", "Wc"), ("HPC", "Eff"), ("HPC", "Wc"),
         ("HPT", "Eff"), ("HPT", "Wc"), ("LPT", "Eff"), ("LPT", "Wc")]


def seed_seq(fam: str, engine_index: int, key: tuple = ()) -> np.random.SeedSequence:
    return np.random.SeedSequence([RECIPE_ID, FAMILIES[fam], int(engine_index)], spawn_key=key)


def class_plan(fam: str, n: int) -> list[int]:
    """Length class of each slot 0..n-1. Balanced; the remainder goes to classes in a rotation by
    family (pilot n=10: 4/3/3 with the 4 on class 1 for HPT, 2 for HPC, 3 for LPT, 1 for FAN, 2 for MIX).
    Slots are interleaved so any prefix is balanced too."""
    first = (FAMILIES[fam] - 1) % 3
    order = [1 + (first + j) % 3 for j in range(3)]
    return [order[i % 3] for i in range(n)]


def split_plan(fam: str, classes: list[int], test_frac: float = 0.3) -> list[str]:
    """70/30 by engine within the subset, stratified by class (recipe section 9)."""
    rng = np.random.default_rng(np.random.SeedSequence([RECIPE_ID, FAMILIES[fam], SPLIT_KEY]))
    out = ["dev"] * len(classes)
    for c in (1, 2, 3):
        slots = [i for i, k in enumerate(classes) if k == c]
        n_test = int(round(test_frac * len(slots)))
        for i in sorted(rng.choice(slots, n_test, replace=False)):
            out[int(i)] = "test"
    return out


def sample_engine(fam: str, engine_index: int, cls: int) -> dict:
    """C1: initial wear u (per module), background rate r_bg, onset threshold E_s, growth a, b."""
    rng = np.random.default_rng(seed_seq(fam, engine_index, (0,)))
    u = {m: float(rng.uniform(0, 1)) for m in MODS}
    r_bg = float(rng.uniform(0.8, 1.2))
    a, b = float(rng.uniform(0.001, 0.003)), float(rng.uniform(1.4, 1.6))
    E_s = float(rng.normal(E_MU, E_SD))
    return {"family": fam, "family_id": FAMILIES[fam], "engine_index": int(engine_index), "cls": int(cls),
            "u": u, "r_bg": r_bg, "a": a, "b": b, "E_s": E_s, "K": K[fam],
            "seed_entropy": [RECIPE_ID, FAMILIES[fam], int(engine_index)]}


def flight_sequence(fam: str, engine_index: int, cls: int, index: dict, n: int = N_SEQ) -> list[tuple[str, int]]:
    """C2: a random source unit of the class from a random start cycle, its flights in cycle order;
    when it runs out, another random unit of the same class from cycle 1. Returns [(unit key, cycle)]."""
    rng = np.random.default_rng(seed_seq(fam, engine_index, (1,)))
    pool = sorted((u for u in index["units"] if u["Fc"] == cls), key=lambda u: u["key"])
    seq: list[tuple[str, int]] = []
    j = int(rng.integers(len(pool)))
    start = int(rng.integers(1, pool[j]["n_cycles"] + 1))
    while len(seq) < n:
        seq += [(pool[j]["key"], c) for c in range(start, pool[j]["n_cycles"] + 1)]
        j = int((j + 1 + rng.integers(len(pool) - 1)) % len(pool))     # another unit (never the same one)
        start = 1
    return seq[:n]


def flight_hours(seq, index: dict) -> np.ndarray:
    by = {u["key"]: u for u in index["units"]}
    return np.array([by[k]["seconds"][c - 1] / 3600.0 for k, c in seq])


def onset_flight(E: dict, hours: np.ndarray) -> int | None:
    """First t (flights flown) with cumulative (ALPHA + hours) above E_s (b5_life rule)."""
    cum = np.cumsum(ALPHA + hours)
    k = np.flatnonzero(cum > E["E_s"])
    return int(k[0] + 1) if len(k) else None


def theta(E: dict, t: int, onset: int | None) -> dict[str, float]:
    """Map-modifier changes (fractions) after t flights (b5_life.engine, same formula)."""
    fam = E["family"]
    init = vec({m: (SHORT[m][0] * E["u"][m] * S_INIT, SHORT[m][1] * E["u"][m] * S_INIT) for m in MODS})
    tau = (t - onset) if onset is not None and t > onset else 0
    g = float(np.expm1(E["K"] * E["a"] * tau ** E["b"])) if tau > 0 else 0.0
    bg = vec(scale(LONG, E["r_bg"] * (t / L_BG) * LAM))
    return add(init, bg, vec(scale(DIR[fam], g))), g, tau


def t_row(th: dict) -> list[float]:
    """theta dict -> N-CMAPSS T columns (fraction, 0 = new)."""
    return [float(th.get(f"MWS.{m}.s_{s}", 0.0)) for m, s in T_SRC]


def load_index(lib: str | pathlib.Path) -> dict:
    return json.loads((pathlib.Path(lib) / "index.json").read_text())
