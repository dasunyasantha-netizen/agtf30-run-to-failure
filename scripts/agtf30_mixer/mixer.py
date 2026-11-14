#!/usr/bin/env python3
"""AGTF30 scenario mixer: a local web page to run engines with the dataset's defaults or your own settings.

    python scripts/agtf30_mixer/mixer.py            # opens http://127.0.0.1:8730 in the browser
    python scripts/agtf30_mixer/mixer.py --port 8731 --no-browser --root <runs folder>

Only Python's standard library plus the generator's own dependencies (numpy, h5py). Listens on 127.0.0.1 only.
The page edits a scenario (scenario.py); "Run" saves it into <root>/<name>/scenario.json and starts mixer_run.py
as a detached background process (Idle CPU priority), so closing the page or this server does not stop a run.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenario as S  # noqa: E402
import mixer_run as MR  # noqa: E402
import gen_recipe as R  # noqa: E402

ROOT = pathlib.Path(MR.DEFAULT_ROOT)
_cache: dict = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ run folder status

def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True,
                           text=True, creationflags=0x08000000).stdout
        return f'"{pid}"' in r
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _engine_series(d: pathlib.Path) -> dict:
    """HI per flight of one engine dir, cached by file count (checkpoint files never change once written)."""
    files = sorted(d.glob("c[0-9][0-9][0-9][0-9].json"))
    key = (str(d), len(files))
    with _lock:
        if key in _cache:
            return _cache[key]
    hi, hrs, onset = [], 0.0, None
    for f in files:
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            break
        hi.append(round(s["HI"], 5))
        hrs += s["hours"]
        onset = s["onset"]
    hi0 = None
    try:
        hi0 = json.loads((d / "engine.json").read_text()).get("hi0")
    except (OSError, ValueError):
        pass
    out = {"hi": hi, "hours": round(hrs, 2), "onset": onset, "hi0": hi0}
    if hi0 is not None or not files:
        with _lock:
            _cache[key] = out
    return out


def run_status(run: pathlib.Path) -> dict:
    sc = S.load(run / "scenario.json")
    p = sc["physics"]
    pid = int((run / "pid.txt").read_text().split()[0]) if (run / "pid.txt").exists() else None
    alive = bool(pid) and _pid_alive(pid)
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {}
    engines = []
    for fam in p["families"]:
        for slot in p["engine_indices"]:
            hist = []
            for k in range(MR.MAX_REPLACEMENTS + 1):
                idx = slot + k * R.REPLACE_STRIDE
                d = run / "runs" / fam / f"e{idx:04d}"
                if not d.exists():
                    break
                e = {"engine_index": idx, **_engine_series(d)}
                if (d / "done.json").exists():
                    dj = json.loads((d / "done.json").read_text())
                    e.update(status="done", eol=dj["eol"], limit=dj["limit"])
                elif (d / "dropped.json").exists():
                    e.update(status="dropped", reason=json.loads((d / "dropped.json").read_text()).get("reason"))
                elif len(e["hi"]) >= p["max_flights"]:
                    e.update(status="censored")
                else:
                    e.update(status="running" if alive else "stopped")
                hist.append(e)
                if e["status"] != "dropped":
                    break
            cur = hist[-1] if hist else {"engine_index": slot, "status": "queued" if alive else "not started",
                                          "hi": [], "hours": 0}
            engines.append({"family": fam, "slot": slot, "unit": slot + 1, "cls": R.class_plan(fam, slot + 1)[slot],
                            "replacements": max(0, len(hist) - 1), "dropped": hist[:-1], **cur})
    st = state.get("status", "new")
    if alive and st in ("finished", "simulated", "env_error", "new"):
        st = "running"
    if not alive and st in ("running", "exporting"):
        st = "stopped"
    outputs = []
    for dpath in [run / "h5"] + sorted((run / "exports").glob("*")) if (run / "exports").exists() else [run / "h5"]:
        for f in sorted(dpath.glob("*.h5")) if dpath.exists() else []:
            if not f.name.endswith(".tmp.h5"):
                outputs.append({"path": str(f), "mb": round(f.stat().st_size / 1e6, 1)})
    log = (run / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-40:] \
        if (run / "run.log").exists() else []
    launches = []
    if (run / "launches.jsonl").exists():
        launches = [json.loads(x) for x in (run / "launches.jsonl").read_text().splitlines() if x.strip()]
    return {"name": run.name, "dir": str(run), "status": st, "alive": alive, "pid": pid, "scenario": sc,
            "changed": S.changed_settings(sc), "recipe_default": S.is_recipe_default(sc), "engines": engines,
            "outputs": outputs, "log": log, "first_start": launches[0]["time"] if launches else None,
            "last_start": launches[-1]["time"] if launches else None, "finished": state.get("finished"),
            "wall_s": state.get("wall_s"),
            "totals": {"flights": sum(len(e["hi"]) for e in engines),
                       "hours": round(sum(e.get("hours") or 0 for e in engines), 1),
                       "done": sum(e["status"] in ("done", "censored") for e in engines), "engines": len(engines)}}


def list_runs() -> list[dict]:
    out = []
    if ROOT.exists():
        for d in sorted(ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if (d / "scenario.json").exists():
                try:
                    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
                    pid = int((d / "pid.txt").read_text().split()[0]) if (d / "pid.txt").exists() else None
                    s = st.get("status", "new")
                    alive = bool(pid) and _pid_alive(pid)
                    if alive:
                        s = "running"
                    elif s in ("running", "exporting"):
                        s = "stopped"
                    ph = json.loads((d / "scenario.json").read_text()).get("physics", {})
                    out.append({"name": d.name, "status": s, "modified": d.stat().st_mtime,
                                "families": ph.get("families", []),
                                "engines": len(ph.get("families", [])) * len(S.parse_indices(ph.get("engine_indices", []))),
                                "recipe_default": S.is_recipe_default(json.loads((d / "scenario.json").read_text())),
                                "done": sum(1 for r in st.get("results", [])
                                            if r["final"]["status"] in ("done", "censored"))})
                except (OSError, ValueError):
                    out.append({"name": d.name, "status": "unreadable"})
    return out


# ------------------------------------------------------------------ actions

def launch(run: pathlib.Path, extra: list[str] | None = None) -> int:
    args = [sys.executable, str(HERE / "mixer_run.py"), str(run)] + (extra or [])
    log = open(run / "stdout.log", "a")
    if os.name == "nt":
        flags = 0x00000200 | 0x08000000   # NEW_PROCESS_GROUP, NO_WINDOW: hidden console that workers and runners inherit
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=flags, close_fds=True)
    else:
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    (run / "pid.txt").write_text(f"{p.pid}\n")
    return p.pid


def stop(run: pathlib.Path):
    if not (run / "pid.txt").exists():
        return
    pid = int((run / "pid.txt").read_text().split()[0])
    if _pid_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)
        else:
            import signal
            os.killpg(pid, signal.SIGTERM)
    MR._log(run, "stopped from the mixer page (resume continues from the last finished flight)")


def active_runs() -> list[str]:
    """Run folders whose runner process is alive (one run at a time, to keep memory use low)."""
    out = []
    if ROOT.exists():
        for d in ROOT.iterdir():
            try:
                if (d / "pid.txt").exists() and _pid_alive(int((d / "pid.txt").read_text().split()[0])):
                    out.append(d.name)
            except (OSError, ValueError):
                pass
    return out


def busy_error() -> list[str]:
    act = active_runs()
    return [f"run '{act[0]}' is still working; one run at a time (raise 'engines at once' to run more engines "
            "in parallel within one run)"] if act else []


def start_run(sc: dict) -> dict:
    sc = S.normalise(sc)
    err = S.validate(sc)
    name = sc["name"].strip()
    if not MR.NAME_RE.match(name):
        err.append("run name: letters, digits, '_', '-', '.' only (max 80)")
    err += busy_error()
    if err:
        return {"ok": False, "errors": err}
    run = ROOT / name
    if run.exists() and (run / "physics.json").exists():
        old = json.loads((run / "physics.json").read_text())["key"]
        new = S.physics_key(sc)
        if any(old[k] != new[k] for k in ("seed_base", "lam", "s_init")) or \
                any(old["K"].get(f, v) != v for f, v in new["K"].items()):
            return {"ok": False, "errors": [f"run '{name}' already exists with other physics; choose a new name"]}
    if run.exists() and (run / "pid.txt").exists():
        pid = int((run / "pid.txt").read_text().split()[0])
        if _pid_alive(pid):
            return {"ok": False, "errors": [f"run '{name}' is running; stop it first"]}
    run.mkdir(parents=True, exist_ok=True)
    S.save(sc, run / "scenario.json")
    pid = launch(run)
    return {"ok": True, "name": name, "pid": pid}


# ------------------------------------------------------------------ HTTP

class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _run(self, name: str) -> pathlib.Path | None:
        name = unquote(name)
        if not MR.NAME_RE.match(name) or not (ROOT / name / "scenario.json").exists():
            return None
        return ROOT / name

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self._send(200, {
                    "defaults": S.defaults(), "families": [{"name": f, "id": i, "note": S.FAMILY_NOTE[f],
                                                            "K": S.RECIPE["K"][f]} for f, i in R.FAMILIES.items()],
                    "paper2": S.PAPER2_DEFAULTS, "root": str(ROOT), "machine": MR.machine(),
                    "max_flights_cap": R.N_SEQ, "dataset_n": S.DATASET_N_PER_FAM, "mixer": MR.MIXER_VERSION})
            if u.path == "/api/runs":
                return self._send(200, list_runs())
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = self._run(parts[2])
                return self._send(200, run_status(run)) if run else self._send(404, {"error": "no such run"})
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "scenario.json":
                run = self._run(parts[2])
                if not run:
                    return self._send(404, {"error": "no such run"})
                return self._send(200, (run / "scenario.json").read_bytes())
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            body = self._body()
            if u.path == "/api/check":
                sc = S.normalise(body)
                return self._send(200, {"errors": S.validate(sc), "changed": S.changed_settings(sc),
                                        "recipe_default": S.is_recipe_default(sc), "scenario": sc,
                                        "engines": len(sc["physics"]["families"]) * len(sc["physics"]["engine_indices"]),
                                        "exists": bool(sc["name"]) and MR.NAME_RE.match(sc["name"]) is not None
                                        and (ROOT / sc["name"] / "scenario.json").exists()})
            if u.path == "/api/run":
                return self._send(200, start_run(body))
            if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                run = self._run(parts[2])
                if not run:
                    return self._send(404, {"error": "no such run"})
                act = parts[3]
                if act == "stop":
                    stop(run)
                    return self._send(200, {"ok": True})
                if act == "resume":
                    if busy_error():
                        return self._send(200, {"ok": False, "errors": busy_error()})
                    return self._send(200, {"ok": True, "pid": launch(run)})
                if act == "export":            # layer 2 only: new overlay from the stored simulation
                    if busy_error():
                        return self._send(200, {"ok": False, "errors": busy_error()})
                    ov = {"noise": bool(body.get("noise", True)), "snr_db": float(body.get("snr_db", 65))}
                    extra = ["--export-only"] + (["--no-noise"] if not ov["noise"] else ["--snr", str(ov["snr_db"])])
                    return self._send(200, {"ok": True, "pid": launch(run, extra), "tag": MR.overlay_tag(ov)})
                if act == "open":
                    target = run / "h5" if (run / "h5").exists() else run
                    if os.name == "nt":
                        os.startfile(str(target))  # noqa: S606
                    elif shutil.which("open" if sys.platform == "darwin" else "xdg-open"):
                        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(target)])
                    return self._send(200, {"ok": True, "dir": str(target)})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main(argv=None):
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8730)
    ap.add_argument("--root", default=MR.DEFAULT_ROOT, help="folder for run folders")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    ROOT = pathlib.Path(a.root)
    ROOT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"AGTF30 scenario mixer at {url}  (runs in {ROOT}; Ctrl+C stops the page, not the runs)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
