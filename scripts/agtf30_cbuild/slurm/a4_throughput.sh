#!/bin/bash
#SBATCH --job-name=agtf30_a4
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=a4_%j.out
# A4: throughput of the standalone AGTF30 runners on one whole node.
#   1) one run alone (baseline speed), 1800-s flight
#   2) N = number of cores simultaneous runs, 1800-s flight
#   3) N simultaneous runs of the whole 4.03-h flight (14498 s, unit_00001 c00001)
# Run from ~/agtf30_cbuild after a3_build_run.sh (bin/ exists). N = cores in the allocation
# (nproc), so the same script also runs on part of a node:
#   sbatch -p short -N1 -n1 -c 8 --mem=4G -t 00:30:00 --wrap "bash a4_throughput.sh"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/agtf30_cbuild}"
N=$(nproc)
echo "host $(hostname) nproc $N mem $(free -g | awk '/Mem:/{print $2}') GB"
lscpu | grep -E 'Model name|Socket|Thread|Core' || true
OUT=a4_runs_${SLURM_JOB_ID:-x}; rm -rf $OUT
python3 scripts/cb_throughput.py --bin bin --flight inputs/flight1800.csv --n 1 --out $OUT/n1_1800
python3 scripts/cb_throughput.py --bin bin --flight inputs/flight1800.csv --n "$N" --out $OUT/n${N}_1800
python3 scripts/cb_throughput.py --bin bin --flight inputs/flight_u1c1.csv --n "$N" --out $OUT/n${N}_full
du -sh $OUT/*
echo A4_DONE
