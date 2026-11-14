#!/usr/bin/env python3
"""Build both runners (cb_dyn, cb_ss) from the generated-code folder.

    build_all.py --gen <gen dir> --out <bin dir> --work <work dir> [--cc gcc|cl]

<gen dir> holds AGTF30SysDyn/ and AGTF30SysSS/ (packNGo zip, outputs_<model>.txt,
defines.txt) and ic_tables.json, as written by cb_build.m. ic_tables.json is
copied next to the executables (cb_run.py --bin <bin dir>).
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cc", default="cl" if os.name == "nt" else "gcc")
    a = ap.parse_args(argv)
    gen = pathlib.Path(a.gen)
    for model, exe in (("AGTF30SysDyn", "cb_dyn"), ("AGTF30SysSS", "cb_ss")):
        g = gen / model
        cmd = [sys.executable, str(HERE / "cb_compile.py"), "--zip", str(g / f"{model}_cb.zip"),
               "--outputs", str(g / f"outputs_{model}.txt"), "--model", model, "--exe", exe,
               "--work", a.work, "--out", a.out, "--cc", a.cc]
        dfile = g / "defines.txt"
        if dfile.exists():
            for d in dfile.read_text().split():
                cmd += ["-D", d]
        subprocess.run(cmd, check=True)
    shutil.copy(gen / "ic_tables.json", pathlib.Path(a.out) / "ic_tables.json")
    print("build_all: done ->", a.out)


if __name__ == "__main__":
    main()
