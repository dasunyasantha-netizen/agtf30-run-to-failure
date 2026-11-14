#!/bin/bash
#SBATCH --job-name=v11_run
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/run_%A_%a.out
# Main run: 60 engines per new subset (180 slots), split over the array tasks (every n-th slot, longest
# class first). Re-submitting resumes from the per-flight checkpoints.
set -euo pipefail
source "${AGTF30_SCRATCH:?set AGTF30_SCRATCH to the cluster scratch folder}/agtf30_v11/pkg/v11_env.sh"
python $GEN/gen_run.py --out $B/main --lib $B/library --fams $FAMS --n-per-fam 60 \
  --workers $SLURM_CPUS_PER_TASK --array-index $SLURM_ARRAY_TASK_ID --array-count $SLURM_ARRAY_TASK_COUNT --priority normal
