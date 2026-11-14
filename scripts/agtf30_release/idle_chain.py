#!/usr/bin/env python3
"""Run phase E jobs one after another at Idle CPU priority, each only when enough memory is free.

    python idle_chain.py --log $AGTF30_WORK/agtf30_v1_work/chain.log --min-free-gb 6 -- \
        "python scripts/agtf30_learn/learn_check.py --h5 $AGTF30_WORK/agtf30_v1 --out $AGTF30_WORK/agtf30_v1_work/e4" \
        "python scripts/agtf30_release/e3_stats.py"

The owner works on this PC: before each step the chain waits until the free commit charge (Windows
ullAvailPageFile) is at least --min-free-gb, so a step never pushes the owner's programs out of memory.
A step that fails is retried (after waiting for memory again) up to --retries times: on this PC the only
failures seen are memory ones (numpy _ArrayMemoryError, HDF5 'filter returned failure' when a decompression
buffer cannot be allocated). Last log line: CHAIN_DONE, or CHAIN_FAILED <step>.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agtf30_gen"))
from gen_run import set_priority  # noqa: E402  (Idle priority class, checked)


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb() -> float:
    if os.name != "nt":
        return 1e9
    m = MEMSTAT(); m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def log(path, msg):
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    with open(path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("steps", nargs="+")
    a = ap.parse_args()
    set_priority("idle")                                         # children inherit IDLE_PRIORITY_CLASS
    log_path = pathlib.Path(a.log)
    log(log_path, f"chain start pid {os.getpid()}, {len(a.steps)} steps, min free commit {a.min_free_gb} GB")
    for n, cmd in enumerate(a.steps, 1):
        for attempt in range(a.retries + 1):
            waited = 0
            while free_commit_gb() < a.min_free_gb:
                if waited % 1800 == 0:
                    log(log_path, f"step {n} waiting for memory: free commit {free_commit_gb():.1f} GB")
                time.sleep(a.poll_s); waited += a.poll_s
            log(log_path, f"step {n} start (attempt {attempt + 1}, free commit {free_commit_gb():.1f} GB): {cmd}")
            with open(log_path.with_name(log_path.stem + f"_step{n}.log"), "a") as out:
                rc = subprocess.call(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT)
            log(log_path, f"step {n} end rc {rc}")
            if rc == 0:
                break
        else:
            log(log_path, f"CHAIN_FAILED {n}")
            return
    log(log_path, "CHAIN_DONE")


if __name__ == "__main__":
    main()
