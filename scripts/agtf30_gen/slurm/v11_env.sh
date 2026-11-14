# sourced by the v1.1 SLURM scripts. Set AGTF30_SCRATCH (cluster scratch folder) and optionally
# AGTF30_CONDA_ENV (conda environment with numpy, scipy, h5py) before sbatch; submit from $AGTF30_SCRATCH/agtf30_v11.
B="${AGTF30_SCRATCH:?set AGTF30_SCRATCH to the cluster scratch folder}/agtf30_v11"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate "${AGTF30_CONDA_ENV:-agtf30}"
export AGTF30_BIN_B=$B/bin_b AGTF30_BIN_FINE=$B/bin_fine AGTF30_GIT_COMMIT="$(cat $B/pkg/GIT_COMMIT) (AI Panther copy)"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
GEN=$B/pkg/scripts/agtf30_gen
FAMS=HPTLPT,LPCHPC,ALL
