#!/bin/bash
#SBATCH --job-name=v11_merge
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_%j.out
# Merge the finished engines into AGTF30_<FAM>.h5 (gzip 1 + shuffle, as v1.0), then the generator checks.
set -euo pipefail
source "${AGTF30_SCRATCH:?set AGTF30_SCRATCH to the cluster scratch folder}/agtf30_v11/pkg/v11_env.sh"
python $GEN/gen_run.py --out $B/main --status
python $GEN/gen_merge.py --out $B/main --fams $FAMS
cd $B/main/h5 && sha256sum AGTF30_*.h5 > SHA256SUMS_v11.txt && cat SHA256SUMS_v11.txt
echo V11_MERGE_DONE
