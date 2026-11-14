# AGTF30 run-to-failure dataset: generation code

Code that made the AGTF30 run-to-failure dataset: 480 simulated NASA AGTF30 geared-turbofan engines, each flown from
new to end of life on recorded N-CMAPSS flights, in eight subsets of 60 engines (v1.0: FAN, HPC, HPT, LPT, MIX;
v1.1: HPTLPT, LPCHPC, ALL). The files use the N-CMAPSS layout.

- Paper: *Run-to-failure simulations of 480 geared turbofan engines under recorded flight conditions* (Scientific Data,
  in preparation).
- Data: Zenodo (DOI to be added).

## Contents

| Folder | What it does | Needs |
|---|---|---|
| `scripts/agtf30_cbuild/` | Turns the AGTF30 Simulink models into two standalone C programs (dynamic flight, steady state): code generation, `cb_main.c`, compile, validation against MATLAB, cluster build and speed test | MATLAB, Simulink, Simulink Coder, AGTF30, T-MATS; a C compiler |
| `scripts/agtf30_recipe/` | Recipe studies: throttle mapping, limits sweep, wear and life model, fault onset, sensor noise, sensor map, v1.1 sweeps | the C programs, N-CMAPSS |
| `scripts/agtf30_gen/` | Generator: flight library, engine sampler, per-flight engine loop with checkpoints, parallel/SLURM runs, merge into `.h5`, checks, drop analysis, reference-point repair | the C programs, N-CMAPSS |
| `scripts/agtf30_release/` | Freezing the release, nested scaling subsets, statistics and figures, baseline, release check, licence files | the generated files |
| `scripts/agtf30_learn/` | Loader (`agtf30_data.py`) and learnability check (`learn_check.py`) | numpy, h5py, torch |
| `scripts/agtf30_mixer/` | Scenario mixer: local web page and command line to run engines to failure with the dataset recipe or your own settings, and a bit-for-bit check against the release (see "Scenario mixer") | the C programs, the flight library; numpy, h5py, scipy |
| `notebooks/` | `example_read_one_engine.ipynb`: read one engine and plot it | numpy, h5py, matplotlib |

Each script starts with a docstring that gives its purpose and command line. The letters in the file names are the
work phases: A = C build, B = recipe, C = generator, D = pilot checks, E = release, G = v1.1 additions.

Generated C code is not included. It is made from your own copy of AGTF30 and T-MATS (both Apache-2.0, see NOTICE).

## Quick start: read the data

```bash
pip install -r requirements.txt
```

Open `notebooks/example_read_one_engine.ipynb` and set `PATH` (or the environment variable `AGTF30_FILE`) to one
downloaded file, for example `AGTF30_ALL.h5`. It reads only the rows of one engine.

Re-run the learnability check of the paper (small LSTM, GRU and transformer against the training-mean baseline):

```bash
python scripts/agtf30_learn/learn_check.py --h5 <folder with AGTF30_*.h5> --out learn_out
```

It trains on the development engines of every file in the folder and tests on the test engines. Results go to
`learn_out/learn.json`; compare with `provenance/learn*.json` in the dataset.

## Paths and settings

Nothing is hard-coded to one computer. Set these environment variables, or pass the folders as arguments where a
script has them (`--out`, `--h5`, `--lib`, ...).

| Variable | Used for | Default |
|---|---|---|
| `AGTF30_WORK` | Root of all work folders (`agtf30_cbuild`, `agtf30_b`, `agtf30_c`, `agtf30_g`, `agtf30_v1`, `agtf30_v11`, ...). The original runs used `D:/tmp`. | `work` (relative to the current folder) |
| `NCMAPSS_DIR` | N-CMAPSS `data_set` folder (the `N-CMAPSS_DS*.h5` files) | `N-CMAPSS` |
| `AGTF30_SRC` | NASA AGTF30 v2.0 package (MATLAB; original runs used commit `a1521a76`) | `AGTF30` |
| `TMATS_DIR` | NASA T-MATS v1.3.3 `Trunk` folder (MATLAB) | `T-MATS/Trunk` |
| `AGTF30_BIN_B`, `AGTF30_BIN_FINE` | Folders with the built C programs (normal and fine time step) | `$AGTF30_WORK/agtf30_cbuild/bin_b`, `.../bin_fine` |
| `AGTF30_TMP` | Temporary folder for the C programs' files | system temp folder |
| `VCVARS` | MSVC `vcvars64.bat` (Windows builds only) | Visual Studio 2022 Build Tools path |
| `AGTF30_SCRATCH` | Cluster scratch folder (SLURM scripts) | none, must be set |
| `AGTF30_CONDA_ENV` | Conda environment on the cluster | `agtf30` |
| `AGTF30_HOST` | SSH alias of the cluster (`make_linux_pkg.sh` comment) | none |
| `AGTF30_GIT_COMMIT` | Code version written into the `.h5` files when there is no git checkout (cluster copy) | `git rev-parse HEAD` |

MATLAB scripts read the same variables through `cb_env.m`.

## Making the dataset (outline)

1. **C programs** (MATLAB): `cb_make_models.m`, `cb_build.m` generate the code; `c/build_all.py` compiles both
   programs (`--cc gcc` on Linux); `a1_reference.m` and `cb_validate.py` compare them with MATLAB. Linux and SLURM:
   `make_linux_pkg.sh`, `slurm/a3_build_run.sh`, `slurm/a4_throughput.sh`.
2. **Recipe** (no MATLAB): `b4_throttle.py`, `b3_sweep.py`, `b5_life.py`, `b6_noise.py`, `b6_sensor_map.py`,
   `b8_checks.py`; v1.1: `g1_v11_sweep.py`, `g2_v11_life.py`.
3. **Generator**: `gen_library.py` (flight library), then `gen_run.py` (engines, resumable; SLURM array support),
   then `gen_post.py` (merge, checks, judge, learnability check), `gen_drops.py` for the drop analysis. v1.1 on a
   SLURM cluster: `slurm/make_v11_pkg.sh`, then `v11_build.sh`, `v11_run.sh`, `v11_merge.sh`, `v11_drops.sh`
   (submit from `$AGTF30_SCRATCH/agtf30_v11`).
4. **Release**: `e1_freeze.py`, `e2_scaling.py`, `e3_stats.py`, `e3_report.py`, `e4_baseline.py`,
   `e6_release_check.py`, `e7_licence.py`; v1.1: `v11_release.py`.

Runs on Windows default to Idle CPU priority (`--priority`).

## Scenario mixer

`scripts/agtf30_mixer/` is a small local web page (and a command line) that runs AGTF30 engines to failure, either
with the dataset recipe or with your own settings (fault growth, background wear, initial wear, seed, sensor noise).
With the dataset recipe it reproduces the released engines bit for bit. How to use the page:
`scripts/agtf30_mixer/README.md`.

The mixer needs two things that are not in this repository: the C engine programs, built from NASA's AGTF30 and
T-MATS code, and the flight library, built from N-CMAPSS. The steps below make both from a fresh clone. They were
tested on Windows 11 from a clean folder (see "Tested" at the end). Run the commands from the repository root in
Git Bash (Windows) or a normal shell (Linux, macOS).

**You need**

- Python 3 (tested with 3.14): `pip install -r requirements.txt` (numpy, h5py and scipy are enough for the mixer).
- MATLAB R2023b with Simulink and Simulink Coder (only for steps 2-4; other releases are untested and may change the
  generated code), and a C compiler that MATLAB can use for MEX files (`mex -setup C`).
- A C compiler for the engine programs: on Windows the Visual Studio 2022 Build Tools (MSVC; set `VCVARS` if
  `vcvars64.bat` is not in the default place), on Linux gcc.
- N-CMAPSS (NASA Prognostics Data Repository, "Turbofan Engine Degradation Simulation Data Set 2"), unpacked:
  the folder with the `N-CMAPSS_DS*.h5` files.
- About 7 GB of free disk (flight library 4.3 GB, build folders 0.8 GB, sources 0.2 GB, a short engine run
  0.1-0.2 GB).

**1. Sources and paths**

```bash
ROOT="$(pwd -W 2>/dev/null || pwd)"         # Git Bash: D:/... form that MATLAB and Python understand
export AGTF30_WORK="$ROOT/work"             # all build and run folders go here (ignored by git)
git clone https://github.com/nasa/AGTF30 work/src/AGTF30
git -C work/src/AGTF30 checkout a1521a76
git clone https://github.com/nasa/T-MATS work/src/T-MATS
git -C work/src/T-MATS checkout v1.3.3
export AGTF30_SRC="$AGTF30_WORK/src/AGTF30"
export TMATS_DIR="$AGTF30_WORK/src/T-MATS/Trunk"
export NCMAPSS_DIR="/path/to/N-CMAPSS/data_set"
```

Keep these variables set for every later step (MATLAB reads them too). On Windows the paths must be in the
`D:/...` form (forward slashes), not Git Bash's `/d/...`; `pwd -W` gives that form.

**2. T-MATS MEX files** (about 2 min)

```bash
matlab -batch "cd(fullfile(getenv('TMATS_DIR'),'TMATS_Library','MEX')); make_file_TMATS"
```

**3. MATLAB reference and C code generation** (about 12 min)

```bash
mkdir -p "$AGTF30_WORK/agtf30_cbuild/inputs"
cp scripts/agtf30_cbuild/inputs/flight1800.csv "$AGTF30_WORK/agtf30_cbuild/inputs/flight_u1c1.csv"
matlab -batch "addpath('scripts/agtf30_cbuild'); a1_reference"
matlab -batch "addpath('scripts/agtf30_cbuild'); cb_build"
matlab -batch "addpath('scripts/agtf30_cbuild'); cb_build('pkgs', 'gen_ss', true)"
matlab -batch "addpath('scripts/agtf30_cbuild'); cb_build('pkgf', 'gen_fine', false, 0.0075)"
```

`a1_reference` simulates the first 1800 s of an N-CMAPSS flight in MATLAB (the reference for step 5); the three
`cb_build` calls generate C code for the normal model, the steady-state model with extra reference-point outputs, and
the dynamic model with a halved time step. All work happens in copies under `$AGTF30_WORK/agtf30_cbuild`; the AGTF30
and T-MATS folders are only read. If MATLAB stops with a licence error, run the same command again.

**4. Compile the two engine programs** (about 1 min)

```bash
CB="$AGTF30_WORK/agtf30_cbuild"
mkdir -p "$CB/gen_b" "$CB/gen_f"
cp -r "$CB/gen/AGTF30SysDyn" "$CB/gen_ss/AGTF30SysSS" "$CB/gen/ic_tables.json" "$CB/gen_b/"
cp -r "$CB/gen_fine/AGTF30SysDyn" "$CB/gen/AGTF30SysSS" "$CB/gen/ic_tables.json" "$CB/gen_f/"
python scripts/agtf30_cbuild/c/build_all.py --gen "$CB/gen_b" --out "$CB/bin_b" --work "$CB/work_b"
python scripts/agtf30_cbuild/c/build_all.py --gen "$CB/gen_f" --out "$CB/bin_fine" --work "$CB/work_f"
```

`bin_b` (0.015 s step) flies every flight; `bin_fine` (0.0075 s step) is the generator's retry for a flight the
normal program cannot finish. This is the same pairing as the dataset (`scripts/agtf30_gen/slurm/make_v11_pkg.sh`).

**5. Check the programs against MATLAB** (optional, about 1 min)

```bash
python scripts/agtf30_cbuild/cb_validate.py --runs "$CB/a2" --a1 "$CB/a1" --bin "$CB/bin_b" \
    --flight "$CB/inputs/flight_u1c1.csv"
```

Pass: every channel within a relative difference of 1e-6 of MATLAB (`$CB/a2/validation.json`).

**6. Flight library** (about 1-2 min with 4 workers, 4.3 GB)

```bash
python scripts/agtf30_gen/gen_library.py --workers 4
```

It reads N-CMAPSS and the throttle tables in `bin_b`, and writes `$AGTF30_WORK/agtf30_c/library` (one `.npz` per
N-CMAPSS unit and `index.json`). Each worker needs a few GB of memory; use fewer workers on a small computer.

**7. Start the mixer**

```bash
python scripts/agtf30_mixer/mixer.py
```

It opens <http://127.0.0.1:8730> (local only; `--no-browser`, `--port`). It finds the programs and the library from
`AGTF30_WORK` (or set `AGTF30_BIN_B`, `AGTF30_BIN_FINE`, `AGTF30_LIB`); runs go to `$AGTF30_WORK/agtf30_mixer/runs`
(or `AGTF30_MIXER_ROOT`). Without the page:

```bash
echo '{"format": "agtf30-scenario/1", "name": "lpt43", "physics": {"families": ["LPT"], "engine_indices": [43]}}' > lpt43.json
python scripts/agtf30_mixer/mixer_run.py --new lpt43.json
```

**8. Check a run against the released data**

```bash
python scripts/agtf30_mixer/mixer_verify.py "$AGTF30_WORK/agtf30_mixer/runs/lpt43" --release <folder with AGTF30_LPT.h5>
```

`MIXER_VERIFY_PASS` means every signal group, the `cycles` rows and the `units` row of each engine in the run equal
the released file bit for bit. LPT engine 43 (unit 44 of the v1.0 file, 82 flights, 51 flight hours) is the shortest
v1.0 engine; its run takes about 25 min.

**Tested.** On 2026-11-14 (Windows 11, MATLAB R2023b, Visual Studio 2022 Build Tools, Python 3.14) from a new folder:
a copy of this repository, new AGTF30 (a1521a76) and T-MATS (v1.3.3) clones, a new Python environment from
`requirements.txt`, then the commands of steps 2-8 (run by a script, same arguments; the page in step 7 with
`--no-browser`, checked by one request). Build times: MEX 98 s, `a1_reference` 400 s, the three
`cb_build` calls 301 s, compile 12 s, validation 48 s (`VALIDATION_PASS`), flight library 69 s; the rebuilt
library was byte-identical to the one used for the dataset (100 of 100 files). LPT engine 43 took 23 min at Idle
priority and gave `MIXER_VERIFY_PASS` against v1.0. Disk: library 4.3 GB, build folders 0.8 GB, AGTF30 and
T-MATS clones 0.2 GB, run 0.1 GB (Python environment with torch 1 GB).

**Other systems.** Bit-identity needs the compiler the release was flown with: v1.0 (FAN, HPC, HPT, LPT, MIX) used
Windows MSVC builds, v1.1 (HPTLPT, LPCHPC, ALL) Linux gcc builds (C code generated on Windows, compiled with gcc
`-O2 -ffp-contract=off`, see `scripts/agtf30_gen/slurm/make_v11_pkg.sh`). With the other compiler the
floating-point signals can differ in the last bits, and `mixer_verify.py` then reports `MIXER_VERIFY_FAIL`. In the
clean test, the Windows build run against v1.1 (LPCHPC engine 36, 79 flights, 38 min) gave the same flights, life
and labels (`A`, `Y` identical) and a largest signal difference of 9.4e-10. On Linux,
`build_all.py` uses gcc by default; generating the C code with MATLAB on Linux and anything on macOS were not
tested.

## Notes

- This is a snapshot of the scripts after the v1.1 release (development commit `457ccdc`, 2026-11-14), with only
  these changes for publication: paths moved to the environment variables above, the two RUL metrics copied into
  `scripts/agtf30_learn/rul_metrics.py`, and the N-CMAPSS loader check in `gen_checks.py` made optional. The commit
  ids in the paper and in the dataset's provenance files (`a52a306`, `b3f88b8`, `69e7016`, `2528540`, `4215483`)
  refer to the development repository.
- `gen_checks.py` step 1 used the original project's N-CMAPSS loader (`ml_pipeline`), which is not part of this
  release. Without it the step is skipped and marked as skipped. The released files were checked with it (see
  `provenance/checks.json` in the dataset).
- The release scripts (`e1`, `e6`, `e7`, `v11_release.py`) also copy the data card, statistics and licence texts
  from the development repository's `docs/` and `configs/` folders. Those files are in the dataset itself
  (`index/`, `LICENSE.txt`, `NOTICE.txt`); the scripts are included as a record of how the release was made.
- Text in the scripts refers to internal decision numbers and reports (for example "decision #7", `B_RECIPE.md`);
  the paper describes the same rules.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`. The dataset itself is CC BY 4.0.

## Citation

Please cite the paper and the dataset (references to be added after publication).
