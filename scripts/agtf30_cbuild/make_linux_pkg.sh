#!/bin/bash
# Assemble the Linux package of the standalone AGTF30 runners (A3) from the
# generated code (cb_build.m output) and the runner scripts, as a tgz to copy to
# AI Panther:
#   bash scripts/agtf30_cbuild/make_linux_pkg.sh [ROOT]      (ROOT default $AGTF30_WORK/agtf30_cbuild)
#   scp ROOT/pkg_linux.tgz $AGTF30_HOST:agtf30_cbuild/ && ssh $AGTF30_HOST 'cd agtf30_cbuild && tar xzf pkg_linux.tgz'
# Contents: gen/ (packNGo zips, output names, defines, patches, ic_tables.json),
# scripts/ (cb_run.py, cb_throughput.py, c/), inputs/ (flight CSVs), SLURM scripts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-${AGTF30_WORK:-work}/agtf30_cbuild}"
P="$ROOT/pkg_linux"
rm -rf "$P"; mkdir -p "$P/gen" "$P/scripts/c" "$P/inputs"
for m in AGTF30SysDyn AGTF30SysSS; do
  mkdir -p "$P/gen/$m"
  cp "$ROOT/gen/$m/${m}_cb.zip" "$ROOT/gen/$m/defines.txt" "$ROOT/gen/$m/outputs_$m.txt" \
     "$ROOT/gen/$m/sources.txt" "$P/gen/$m/"
  [ -f "$ROOT/gen/$m/patches.json" ] && cp "$ROOT/gen/$m/patches.json" "$P/gen/$m/"
done
cp "$ROOT/gen/ic_tables.json" "$P/gen/"
cp "$HERE/cb_run.py" "$HERE/cb_throughput.py" "$P/scripts/"
cp "$HERE/c/build_all.py" "$HERE/c/cb_compile.py" "$HERE/c/cb_main.c" "$HERE/c/gen_param_table.py" "$P/scripts/c/"
cp "$ROOT/inputs/"*.csv "$P/inputs/"
cp "$HERE/slurm/"*.sh "$P/"
tar czf "$ROOT/pkg_linux.tgz" -C "$P" .
ls -l "$ROOT/pkg_linux.tgz"
