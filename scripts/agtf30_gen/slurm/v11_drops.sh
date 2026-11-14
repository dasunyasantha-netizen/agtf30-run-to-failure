#!/bin/bash
#SBATCH --job-name=v11_drops
#SBATCH --partition=med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/drops_%j.out
# Drop analysis of the v1.1 main run, as for v1.0 (gen_drops.py: failure point, re-flights with the decision #7
# ladder, success of the same source flight on other engines, would-be lives; then the wear ladder pass).
set -euo pipefail
source "${AGTF30_SCRATCH:?set AGTF30_SCRATCH to the cluster scratch folder}/agtf30_v11/pkg/v11_env.sh"
python $GEN/gen_drops.py --out $B/main --fams $FAMS --workers 30
python $GEN/gen_drops.py --out $B/main --fams $FAMS --workers 30 --ladder
echo V11_DROPS_DONE
