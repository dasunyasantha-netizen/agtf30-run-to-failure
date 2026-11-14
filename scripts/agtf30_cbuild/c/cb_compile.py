#!/usr/bin/env python3
"""Compile one standalone runner from a packNGo (flat) zip of the GRT code.

    cb_compile.py --zip AGTF30SysDyn_cb.zip --outputs outputs_AGTF30SysDyn.txt \
                  --model AGTF30SysDyn --exe cb_dyn --work <dir> --out <bin dir> [--cc gcc|cl]

Unzips the generated code into <work>/<model>, writes cb_params_table.h
(gen_param_table.py), and compiles every generated .c except GRT's own
rt_main.c, plus cb_main.c, with optimisation but strict IEEE arithmetic
(gcc: -O2 -ffp-contract=off, no -ffast-math; cl: /O2 /fp:precise).
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
VCVARS = os.environ.get("VCVARS",   # MSVC environment script (Windows builds only), see README
                        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cc", default="cl" if os.name == "nt" else "gcc")
    ap.add_argument("-D", dest="extra", action="append", default=[],
                    help="extra define, e.g. CB_FIXED_STEP=1 CB_DEFAULT_STOP=128 for AGTF30SysSS")
    a = ap.parse_args(argv)
    src = pathlib.Path(a.work) / a.model
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(src)
    shutil.copy(HERE / "cb_main.c", src / "cb_main.c")
    # Source patches written by cb_build.m: values that Simulink code generation folds into
    # literals although the block parameter is a per-run CB_* global (Transport Delay
    # initial output). Each patch must match exactly 'count' times or the build stops.
    pfile = pathlib.Path(a.outputs).with_name("patches.json")
    if pfile.exists():
        for pt in json.loads(pfile.read_text()):
            f = src / pt["file"]
            txt = f.read_text()
            n = txt.count(pt["find"])
            if n != pt["count"]:
                sys.exit(f"cb_compile: patch {pt} matched {n} times (expected {pt['count']})")
            f.write_text(txt.replace(pt["find"], pt["replace"]))
            print(f"cb_compile: patched {pt['file']}: {pt['find']} -> {pt['replace']} ({n}x)")
    subprocess.run([sys.executable, str(HERE / "gen_param_table.py"), str(src), a.outputs,
                    str(src / "cb_params_table.h")], check=True)
    cfiles = sorted(p.name for p in src.glob("*.c") if p.name not in ("rt_main.c", "grt_main.c", "classic_main.c"))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    defs = ["CB_MODEL=" + a.model, "MODEL=" + a.model, "NUMST=2", "NCSTATES=0", "HAVESTDIO",
            "RT", "USE_RTMODEL", "MAT_FILE=0", "INTEGER_CODE=0", "MT=0", "CLASSIC_INTERFACE=0",
            "ALLOCATIONFCN=0", "TID01EQ=0", "ONESTEPFCN=1", "TERMFCN=1", "MULTI_INSTANCE_CODE=0"] + a.extra
    if a.cc == "gcc":
        exe = out / a.exe
        cmd = ["gcc", "-O2", "-ffp-contract=off", "-fno-fast-math", "-std=gnu99", "-w", "-I."] + \
              ["-D" + d for d in defs] + cfiles + ["-o", str(exe), "-lm"]
        p = subprocess.run(cmd, cwd=src)
    else:
        exe = out / (a.exe + ".exe")
        cl = " ".join(["cl", "/nologo", "/O2", "/fp:precise", "/W0", "/I."] + ["/D" + d for d in defs] + cfiles +
                      ["/Fe:" + str(exe)])
        p = subprocess.run(f'call "{VCVARS}" >nul && {cl} >compile_log.txt 2>&1', cwd=src, shell=True)
        if p.returncode:
            print((src / "compile_log.txt").read_text(errors="replace")[-6000:])
    if p.returncode:
        sys.exit(f"cb_compile: compile failed for {a.model} (rc {p.returncode})")
    print(f"cb_compile: built {exe} from {len(cfiles)} C files")


if __name__ == "__main__":
    main()
