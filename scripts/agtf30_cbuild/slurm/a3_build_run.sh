#!/bin/bash
#SBATCH --job-name=agtf30_a3
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=a3_%j.out
# A3: compile the AGTF30 runners with gcc on a compute node and run the two
# validation cases (healthy, degraded) at full rate and 1 Hz.
# Run from ~/agtf30_cbuild (package unpacked there: gen/, scripts/, inputs/).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/agtf30_cbuild}"
echo "host $(hostname) cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) gcc $(gcc -dumpfullversion)"
rm -rf work bin a3_runs
python3 scripts/c/build_all.py --gen gen --out bin --work work --cc gcc
DEG="--theta MWS.HPT.s_Eff=-0.02 --theta MWS.HPC.s_Eff=-0.01 --theta MWS.HPC.s_Wc=-0.015"
for rate in full 1hz; do
  python3 scripts/cb_run.py --bin bin --flight inputs/flight1800.csv --out a3_runs/healthy_$rate --rate $rate
  python3 scripts/cb_run.py --bin bin --flight inputs/flight1800.csv --out a3_runs/degraded_$rate --rate $rate $DEG
done
grep -h '"wall_s"' a3_runs/*/run.json | head -20
echo A3_DONE
