"""Run-directory helpers: atomic JSON writes, CSV logging, single-instance lock."""
import csv
import json
import os
from typing import Any, Sequence


def write_json(path: str, obj: Any) -> None:
    """Write to a temp file then rename, so a kill mid-write never leaves a
    truncated checkpoint behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


class CsvLog:
    def __init__(self, path: str, fields: Sequence[str], resume: bool) -> None:
        new = not (resume and os.path.exists(path))
        self.f = open(path, "w" if new else "a", newline="", buffering=1)
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def write(self, row: dict) -> None:
        self.w.writerow(row)

    def close(self):
        self.f.close()


class RunLock:
    def __init__(self, run_dir: str) -> None:
        self.path = os.path.join(run_dir, "run.lock")

    def __enter__(self):
        if os.path.exists(self.path) and os.name != "nt":  # os.kill(pid, 0) sends CTRL_C on Windows
            pid = int(open(self.path).read().strip() or 0)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass  # stale lock from a crashed run
            else:
                raise SystemExit(f"Another run (PID {pid}) is using this run directory. "
                                 f"Delete {self.path} if that is wrong.")
        with open(self.path, "w") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        if os.path.exists(self.path):
            os.remove(self.path)
