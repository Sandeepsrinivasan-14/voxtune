import json
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core import db, hostinfo
from ..core.gpu import detect_gpu
from ..evaluation.metrics import METRICS
from ..registry import METHODS, MODELS, get_model, observed_vram
from ..training.orchestrator import orchestrator

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# auxiliary models the pipelines load besides the model being tuned
SUPPORT_MODELS = {"microsoft/speecht5_hifigan": "SpeechT5 vocoder", "openai/whisper-base": "ASR used to score TTS intelligibility"}


def _active_run(s: Session) -> dict | None:
    r = s.query(db.Run).filter_by(status="running").order_by(db.Run.id).first()
    if not r:
        return None
    f = orchestrator.run_dir(r.id) / "progress.jsonl"
    ev = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
    train = [e for e in ev if e["type"] == "train"]
    val = [e for e in ev if e["type"] == "eval"]
    status = next((e["message"] for e in reversed(ev) if e["type"] == "status"), None)
    last = train[-1] if train else None
    return {"id": r.id, "name": r.name, "model_id": r.model_id, "method": r.method, "phase": status,
            "step": last["step"] if last else None, "total_steps": last["total_steps"] if last else None,
            "epoch": last["epoch"] if last else None, "loss": last["loss"] if last else None,
            "val_loss": val[-1]["val_loss"] if val else None, "elapsed": last["elapsed"] if last else None,
            "eta": last["eta"] if last else None}


def _summary(r: db.Run) -> dict:
    task = get_model(r.model_id).task
    primary = next(m for m in METRICS[task] if m.primary)
    m = r.metrics or {}
    return {"id": r.id, "name": r.name, "status": r.status, "model_id": r.model_id, "method": r.method, "task": task,
            "seconds": r.train_seconds, "peak_gpu_gb": r.peak_gpu_gb, "created_at": r.created_at.isoformat(),
            "metric": {"key": primary.key, "label": primary.label, "fmt": primary.fmt, "higher_better": primary.higher_better,
                       "base": (m.get("base") or {}).get(primary.key), "tuned": (m.get("tuned") or {}).get(primary.key)}}


@router.get("")
def dashboard(project_id: int | None = None, s: Session = Depends(db.get_session)):
    gpu = detect_gpu()
    dq, rq = s.query(db.Dataset), s.query(db.Run)
    if project_id:
        dq, rq = dq.filter_by(project_id=project_id), rq.filter_by(project_id=project_id)
    datasets, runs = dq.all(), rq.order_by(db.Run.id.desc()).all()
    audio_h = sum((d.analysis or {}).get("total_hours", 0) for d in datasets)
    prep = [d.prepared["hours"] for d in datasets if d.prepared]
    trained_h = sum(r.train_seconds or 0 for r in runs) / 3600

    cached: dict[str, dict] = {}
    for hf_id, label in [(m.hf_id, m.id) for m in MODELS.values()] + list(SUPPORT_MODELS.items()):
        if hf_id in cached:
            cached[hf_id]["label"] += f" / {label}"  # same weights serve more than one role (e.g. whisper-base is also the TTS-scoring ASR)
            continue
        st = hostinfo.model_cache_status(hf_id)
        if st["cached"]:
            cached[hf_id] = {"hf_id": hf_id, "label": label, **st}
    cached = list(cached.values())
    storage = hostinfo.storage()
    storage["hf_cache"] = round(hostinfo.gb(hostinfo.dir_size(hostinfo._hf_cache_dir())), 2)

    return {
        "counts": {"projects": s.query(db.Project).count(), "datasets": len(datasets), "runs": len(runs),
                   "registered_models": s.query(db.RegisteredModel).count()},
        "datasets": {"by_status": dict(Counter(d.status for d in datasets)), "audio_hours": round(audio_h, 3),
                     "recordings": sum((d.analysis or {}).get("total_recordings", 0) for d in datasets),
                     "train_hours": round(sum(h["train"] for h in prep), 3)},
        "runs": {"by_status": dict(Counter(r.status for r in runs)), "train_hours": round(trained_h, 3),
                 "by_method": {METHODS.get(k, k): v for k, v in Counter(r.method for r in runs).items()}},
        "queue": {"active": _active_run(s), "queued": sum(1 for r in runs if r.status == "queued")},
        "recent_runs": [_summary(r) for r in runs[:8]],
        "storage_gb": storage,
        "cached_models": cached,
        "observed_vram": [{"model_id": m, "method": METHODS.get(meth, meth), "used_gb": gb}
                          for m, per in observed_vram(s, gpu).items() for meth, gb in per.items()],
    }
