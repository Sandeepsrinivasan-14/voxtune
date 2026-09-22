"""Runs training jobs one at a time (single GPU) in worker subprocesses and records the outcome in the DB."""
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..core import db
from ..core.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Orchestrator:
    def __init__(self):
        self.q: queue.Queue[int] = queue.Queue()
        self.procs: dict[int, subprocess.Popen] = {}
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        with db.SessionLocal() as s:  # runs left over from a previous server process can never finish
            for r in s.query(db.Run).filter(db.Run.status.in_(["queued", "running"])).all():
                r.status, r.error = "failed", "Interrupted by server restart."
            s.commit()
        if not self.thread:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def run_dir(self, run_id: int) -> Path:
        return settings.runs_dir / f"run_{run_id:04d}"

    def submit(self, run_id: int) -> None:
        self.q.put(run_id)

    def stop(self, run_id: int) -> None:
        d = self.run_dir(run_id)
        if d.exists():
            (d / "STOP").write_text("stop")
        with db.SessionLocal() as s:
            r = s.get(db.Run, run_id)
            if r and r.status == "queued":
                r.status = "stopped"
                s.commit()

    def kill(self, run_id: int) -> None:
        p = self.procs.get(run_id)
        if p and p.poll() is None:
            p.kill()

    def _loop(self) -> None:
        while True:
            run_id = self.q.get()
            try:
                self._run(run_id)
            except Exception as e:  # noqa: BLE001
                self._finish(run_id, "failed", error=f"Orchestrator error: {e}")

    def _finish(self, run_id: int, status: str, metrics=None, seconds=None, peak=None, error=None) -> None:
        with db.SessionLocal() as s:
            r = s.get(db.Run, run_id)
            r.status, r.metrics, r.train_seconds, r.peak_gpu_gb, r.error = status, metrics, seconds, peak, error
            s.commit()

    def _run(self, run_id: int) -> None:
        with db.SessionLocal() as s:
            r = s.get(db.Run, run_id)
            if r.status == "stopped":
                return
            r.status = "running"
            job = dict(r.config)
            s.commit()
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        with open(d / "worker.log", "w", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-m", "app.training.worker", str(d)], cwd=BACKEND_DIR,
                                 stdout=log, stderr=subprocess.STDOUT)
            self.procs[run_id] = p
            while p.poll() is None:
                time.sleep(1)
        self.procs.pop(run_id, None)
        if (d / "result.json").exists():
            res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            train = res.get("train", {})
            self._finish(run_id, train.get("status", "completed"), res, train.get("train_seconds"), train.get("peak_gpu_gb"))
        else:
            err = "Worker exited without a result."
            if (d / "error.json").exists():
                err = json.loads((d / "error.json").read_text(encoding="utf-8"))["error"]
            elif (d / "STOP").exists():
                self._finish(run_id, "stopped")
                return
            self._finish(run_id, "failed", error=err)


orchestrator = Orchestrator()
