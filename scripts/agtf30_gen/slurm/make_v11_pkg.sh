#!/bin/bash
# v1.1 on AI Panther: assemble the code package (runner C sources for bin_b and bin_fine, generator scripts,
# SLURM scripts) as $AGTF30_WORK/agtf30_g/pkg_v11.tgz. The flight library is copied separately (4.3 GB).
#   bash scripts/agtf30_gen/slurm/make_v11_pkg.sh
# Runner pairs exactly as the v1.0 Windows builds (C_GENERATOR, E_DATASET C build ids):
#   bin_b    = cb_dyn from gen/AGTF30SysDyn (0.015 s step), cb_ss from gen_ss/AGTF30SysSS (extended outputs)
#   bin_fine = cb_dyn from gen_fine/AGTF30SysDyn (0.0075 s step), cb_ss from gen/AGTF30SysSS
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$(cd "$HERE/../.." && pwd)"
W="${AGTF30_WORK:-work}"; CB="$W/agtf30_cbuild"; P="$W/agtf30_g/pkg_v11"
rm -rf "$P"; mkdir -p "$P/gen_b" "$P/gen_f" "$P/scripts"
cp -r "$CB/gen/AGTF30SysDyn" "$P/gen_b/"; cp -r "$CB/gen_ss/AGTF30SysSS" "$P/gen_b/"
cp -r "$CB/gen_fine/AGTF30SysDyn" "$P/gen_f/"; cp -r "$CB/gen/AGTF30SysSS" "$P/gen_f/"
cp "$CB/gen/ic_tables.json" "$P/gen_b/"; cp "$CB/gen/ic_tables.json" "$P/gen_f/"
for d in agtf30_gen agtf30_recipe agtf30_cbuild; do
  mkdir -p "$P/scripts/$d"; (cd "$S/$d" && find . -name '*.py' -not -path '*__pycache__*' -exec cp --parents {} "$P/scripts/$d/" \;)
done
cp "$S/agtf30_cbuild/c/cb_main.c" "$P/scripts/agtf30_cbuild/c/"
cp "$HERE"/v11_*.sh "$P/"
git -C "$S" rev-parse HEAD > "$P/GIT_COMMIT"
tar czf "$W/agtf30_g/pkg_v11.tgz" -C "$P" .
ls -l "$W/agtf30_g/pkg_v11.tgz"; cat "$P/GIT_COMMIT"
