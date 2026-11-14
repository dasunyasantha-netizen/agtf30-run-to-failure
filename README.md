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
