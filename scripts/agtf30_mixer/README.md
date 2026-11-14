# AGTF30 scenario mixer (v1)

A small local web page to run AGTF30 geared-turbofan engines to failure, either with the released dataset's
recipe (and get exactly the released engines) or with your own settings. It edits a scenario file; the
generator in `scripts/agtf30_gen/` runs from that file.

## Start

```bash
python scripts/agtf30_mixer/mixer.py
```

This opens <http://127.0.0.1:8730> in your browser (local only). Ctrl+C stops the page, **not** the runs: each run
is its own background process, and you can close the page while it works. Options: `--port`, `--root <folder for
runs>` (default `$AGTF30_MIXER_ROOT`, else `$AGTF30_WORK/agtf30_mixer/runs`), `--no-browser`.

Needs: Python with numpy and h5py, the C runners and the flight library. Paths come from the environment:
`AGTF30_BIN_B` (default `$AGTF30_WORK/agtf30_cbuild/bin_b`), `AGTF30_BIN_FINE` (`$AGTF30_WORK/agtf30_cbuild/bin_fine`),
`AGTF30_LIB` (`$AGTF30_WORK/agtf30_c/library`, 4.3 GB). `AGTF30_WORK` defaults to `work`; see the main README
("Scenario mixer") for how to build the runners and the library.

## Use

The page has four numbered steps on the left and your runs on the right.

1. **Which engines?** Click one or more fault-type tiles, then choose how many engines: 1, 5, 10, all 60, or
   **Pick numbers…** for your own list (for example `0-4, 43`).
2. **Wear physics.** Keep **Dataset recipe** to get the released engines exactly, or choose **Custom** to change
   the sliders (fault growth K, background wear, initial wear, seed). Every changed value is highlighted and has a
   **reset** link.
3. **Sensor noise.** Turn noise on or off and set the SNR.
4. **Run settings.** Set the name and how many engines run at once. The bar at the bottom shows the number of
   engines, the estimated time and disk space, and whether the run is the dataset recipe or custom. Press **Run**.

On the right, each run shows its progress, a health-index chart (1 = new, 0 = end of life; hover for values,
colour = fault type), a table of engines, the output files (**Copy path**, **Open folder**) and the log.
**Pause** stops a run, and **Resume** continues it from the last finished flight (every flight is checkpointed).
**Reuse settings** loads a run's settings back into the form. **Open** and **Save** (top left) read and write
scenario files. The page runs one run at a time, to keep memory use low. To run engines in
parallel, set **Engines at the same time** (1-4) within one run.

Without the page:

```bash
python scripts/agtf30_mixer/mixer_run.py --new my_scenario.json
```

```bash
python scripts/agtf30_mixer/mixer_run.py <root>/<name>
```

The first makes `<root>/<name>/` and runs it; the second resumes a run folder. A scenario file may be partial:
missing keys take the defaults.

## Knobs

| Layer | Setting | Default (released recipe) | Notes |
|---|---|---|---|
| 1 physics | families | HPT | 1 HPT, 2 HPC, 3 LPT, 4 FAN, 5 MIX (v1.0); 6 HPTLPT, 7 LPCHPC, 8 ALL (v1.1) |
| 1 | engine indices | 0 | 0-59 are the dataset's engines (unit = index + 1); 60 and up are new engines of the same recipe |
| 1 | max flights | 400 | an engine alive at this point is kept as **censored** (RUL = −1); raise it and resume to fly on |
| 1 | replace dropped | on | as the dataset: an engine the solver cannot fly is replaced by index + 1000·k (k ≤ 3) |
| 1 | seed base | 20261103 | seeds are `SeedSequence([seed base, family id, engine index])`; change it for new random engines |
| 1 | K (per family) | HPT 1.9, HPC 1.8, LPT 2.6, FAN 2.6, MIX 1.25, HPTLPT 1.6, LPCHPC 1.4, ALL 0.72 | fault growth time scale |
| 1 | lam | 0.12 | background wear speed |
| 1 | s_init | 0.2 | initial wear size |
| 2 overlay | noise, SNR | on, 65 dB | Gaussian on X_s, σ = 10^(−SNR/20) × rms of the clean flight signal; X_s_clean always stored |
| — | engines at once, CPU priority, h5 compression | 1, idle, gzip1 | do not change the data |

Paper 2 knobs (climate, maintenance, sensor faults, gearbox, nozzle, actuators, combustor, ...) are shown greyed
out. They are in the file format (`paper2` section) but must stay off in v1.

Changing K, lam or s_init keeps each engine's random draws (initial wear, onset threshold, growth shape, flights,
noise): a custom engine is the same engine with different wear physics. Changing the seed base gives new draws.

The physics of a run folder are fixed when it starts. Running again under the same name resumes or extends it
(more engine indices, more families, higher max flights). Other physics need a new name.

## Output

```
<root>/<name>/
  scenario.json          the scenario
  h5/AGTF30_<FAM>.h5     one file per family, the released N-CMAPSS layout
  h5/provenance.json     scenario + hash, every changed setting, code commit, C build ids, machine, per-unit table
  exports/<tag>/         the same for another noise overlay
  runs/<FAM>/eNNNN/      per-flight checkpoints (gen_engine's layout); run.log, state.json, physics.json
```

The h5 files have the released layout: `W, X_s, X_s_clean, X_v, X_a, T, A, Y` (`_dev`/`_test`, plus `*_var`), and
the `cycles`, `units` and `events` tables. Engines 0-59 keep the dataset's dev/test split; higher indices go to dev.
File attributes `scenario`, `scenario_sha256`, `changed_settings` and `recipe_default` record what was changed.

**Overlay without re-simulation.** **Need different noise? → Write a copy** on the page (or `mixer_run.py <run> --export-only --snr 50`
or `--no-noise`) writes `exports/snr50/` or `exports/nonoise/` from the stored simulation, with no new simulation.

## Reproducing the released dataset

With every data setting at its default, engine index *i* of family *F* is the dataset's engine *i* of *F*. Check a
run against the release, bit for bit (all signal groups, the `cycles` rows and the `units` row of each unit):

```bash
python scripts/agtf30_mixer/mixer_verify.py <root>/<name> --release <folder with the v1.0 AGTF30_*.h5>
```

Checked on 2026-11-10 (`MIXER_VERIFY_PASS`): LPT engine 43 (the shortest v1.0 engine, 82 flights), run from the
page's defaults, is identical to unit 44 of the v1.0 `AGTF30_LPT.h5`. That covers all 184,242 rows of every signal
group, plus its `cycles` and `units` rows. The run was interrupted by a memory error at flight 18 and resumed,
which shows that a resume does not change the result.

Checked again on 2026-11-14 from a clean copy of the code release, with the C programs and the flight library
rebuilt by the steps in the main README (Windows 11, MATLAB R2023b, MSVC 2022): LPT engine 43 again
`MIXER_VERIFY_PASS`, and the rebuilt flight library was byte-identical to the original one.

Bit-identity needs C programs built with the same compiler as the release you compare with: v1.0 (FAN, HPC,
HPT, LPT, MIX) was flown with Windows MSVC builds, v1.1 (HPTLPT, LPCHPC, ALL) with Linux gcc builds. A Windows
build run against v1.1 (LPCHPC engine 36) gave the same flights, life and labels (`A`, `Y` identical), but the
floating-point signals differ in the last bits (largest absolute difference 9.4e-10), so `mixer_verify.py`
reports `MIXER_VERIFY_FAIL` there.

## Cost

One engine flies about 150 flight hours per hour on the development PC (Windows 11, MSVC builds), less while you use it, because runs are at Idle
priority. The verification engine (82 short flights, 51 flight hours) took 20 min. A typical engine is 60-100
flights and about 180 flight hours, so about 1-1.5 h. Disk: about 1.1 MB of checkpoints per flight hour plus
1.2 MB of h5 rows (gzip1), so about 400 MB for a typical engine. If the PC runs short of memory, a flight stops
with an environment error. That is not a drop: press **Resume**.

## Files

`mixer.py` web server and page API · `ui.html` the page · `scenario.py` scenario format, defaults, validation,
changed settings · `mixer_run.py` runner and h5 export · `mixer_verify.py` bit-for-bit check against a release.
