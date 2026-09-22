from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core import db
from ..evaluation.metrics import compare_columns
from ..registry import METHODS

router = APIRouter(prefix="/api/experiments", tags=["experiments"])



def _row(r: db.Run) -> dict:
    m = r.metrics or {}
    return {"run_id": r.id, "name": r.name, "model_id": r.model_id, "method": METHODS.get(r.method, r.method), "status": r.status,
            "metrics": {k: v for k, v in (m.get("tuned") or {}).items() if k != "examples"},
            "train_seconds": r.train_seconds, "peak_gpu_gb": r.peak_gpu_gb, "dataset_id": r.dataset_id,
            "trainable_params": (m.get("train") or {}).get("trainable_params"), "config": (r.config or {}).get("train_config"),
            "quantized_base": bool(m.get("quantized_base")),
            "base_metrics": {k: v for k, v in (m.get("base") or {}).items() if k != "examples"}}


@router.get("/compare")
def compare(run_ids: str | None = None, group: str | None = None, s: Session = Depends(db.get_session)):
    q = s.query(db.Run)
    if group:
        runs = q.filter_by(group=group).order_by(db.Run.id).all()
    elif run_ids:
        ids = [int(x) for x in run_ids.split(",") if x.strip()]
        runs = [r for r in q.filter(db.Run.id.in_(ids)).order_by(db.Run.id).all()]
    else:
        raise HTTPException(422, "Give run_ids or group.")
    if not runs:
        raise HTTPException(404, "No runs found.")
    done = [r for r in runs if r.status == "completed"]
    kind = ((done[0].metrics or {}).get("kind")) if done else None
    cols = compare_columns(kind)  # only used to flag each column's leader, never an overall "best"
    base = None
    if done:
        base = {k: v for k, v in (done[0].metrics.get("base") or {}).items() if k != "examples"}
    rows = [_row(r) for r in runs]
    leaders = {}
    for col, higher in cols.items():
        vals = [(r["metrics"].get(col), r["run_id"]) for r in rows if r["status"] == "completed" and r["metrics"].get(col) is not None]
        if len(vals) > 1:
            leaders[col] = (max if higher else min)(vals)[1]
    for col, key in (("train_seconds", False), ("peak_gpu_gb", False)):
        vals = [(r[col], r["run_id"]) for r in rows if r["status"] == "completed" and r[col] is not None]
        if len(vals) > 1:
            leaders[col] = min(vals)[1]
    return {"kind": kind, "columns": list(cols), "base": base, "rows": rows, "leaders": leaders,
            "note": "Leaders are per-column only. Which trade-off is right depends on accuracy, compute, time and deployment needs."}
