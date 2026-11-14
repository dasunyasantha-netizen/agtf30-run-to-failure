#!/bin/bash
#SBATCH --job-name=v11_build
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_%j.out
# gcc build of the two runner pairs, then a 3-flight cross-check engine per new subset (compare with the
# Windows smoke run $AGTF30_WORK/agtf30_g/smoke_win).
set -euo pipefail
source "${AGTF30_SCRATCH:?set AGTF30_SCRATCH to the cluster scratch folder}/agtf30_v11/pkg/v11_env.sh"
cd $B
echo "host $(hostname) gcc $(gcc -dumpfullversion) python $(python -V)"
rm -rf work_b work_f bin_b bin_fine smoke_linux
python $B/pkg/scripts/agtf30_cbuild/c/build_all.py --gen pkg/gen_b --out bin_b --work work_b --cc gcc
python $B/pkg/scripts/agtf30_cbuild/c/build_all.py --gen pkg/gen_f --out bin_fine --work work_f --cc gcc
sha256sum bin_b/cb_* bin_fine/cb_*
for f in HPTLPT LPCHPC ALL; do
  python $GEN/gen_engine.py --out $B/smoke_linux --lib $B/library --fam $f --slot 0 --n-per-fam 60 --stop-after 3 &
done
wait
echo V11_BUILD_DONE
