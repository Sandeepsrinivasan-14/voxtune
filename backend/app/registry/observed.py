"""VRAM actually used by completed runs on THIS machine, read from their progress logs.

Registry estimates are only estimates; a finished run on the current GPU is proof.
"""
import json

from ..core import db
from ..core.config import settings
from ..core.gpu import GpuInfo

_cache: dict[tuple[str, int], dict | None] = {}


def _run_usage(run_id: int) -> dict | None:
    key = (str(settings.runs_dir), run_id)
    if key in _cache:
        return _cache[key]
    f = settings.runs_dir / f"run_{run_id:04d}" / "progress.jsonl"
    usage = None
    if f.exists():
        ev = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        tr = [e for e in ev if e["type"] == "train" and e.get("gpu_used_gb") is not None]
        if tr:
            usage = {"used_gb": max(e["gpu_used_gb"] for e in tr), "total_gb": tr[-1]["gpu_total_gb"]}
    _cache[key] = usage
    return usage


def observed_vram(session, gpu: GpuInfo) -> dict[str, dict[str, float]]:
    """{model_id: {method: GB used}} for completed runs made on a GPU with the same memory size as the current one."""
    out: dict[str, dict[str, float]] = {}
    if not gpu.available:
        return out
    for r in session.query(db.Run).filter_by(status="completed"):
        u = _run_usage(r.id)
        if u and abs(u["total_gb"] - gpu.total_gb) <= 0.15:
            per = out.setdefault(r.model_id, {})
            per[r.method] = max(per.get(r.method, 0.0), u["used_gb"])
    return out
