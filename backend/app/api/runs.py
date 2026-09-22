import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core import db
from ..core.gpu import detect_gpu
from ..registry import METHODS, check_support, get_model, observed_vram
from ..strategy import recommend
from ..training.config import TrainConfig
from ..training.methods.validate import validate
from ..training.orchestrator import orchestrator

router = APIRouter(prefix="/api", tags=["training"])


def run_out(r: db.Run) -> dict:
    return {"id": r.id, "project_id": r.project_id, "dataset_id": r.dataset_id, "model_id": r.model_id, "method": r.method,
            "name": r.name, "group": r.group, "status": r.status, "config": (r.config or {}).get("train_config"),
            "autotune": (r.config or {}).get("autotune"), "warnings": (r.config or {}).get("warnings", []),
            "metrics": r.metrics, "train_seconds": r.train_seconds, "peak_gpu_gb": r.peak_gpu_gb, "error": r.error,
            "created_at": r.created_at.isoformat()}


class RecommendIn(BaseModel):
    model_id: str
    dataset_id: int
    goal: str = "general"
    domain_terms: list[str] = []
    method: str | None = None


@router.post("/strategy/recommend")
def strategy_recommend(body: RecommendIn, s: Session = Depends(db.get_session)):
    d = _prepared_dataset(s, body.dataset_id)
    try:
        m = get_model(body.model_id)
    except KeyError:
        raise HTTPException(404, "Unknown model")
    if m.task != d.task:
        raise HTTPException(422, f"{m.id} is a {m.task.upper()} model but the dataset is {d.task.upper()}.")
    gpu = detect_gpu()
    return recommend(m, gpu, d.prepared["hours"]["train"], d.prepared["counts"]["train"], body.goal,
                     body.domain_terms, force_method=body.method, observed=observed_vram(s, gpu).get(m.id),
                     n_val=d.prepared["counts"]["val"])


def _prepared_dataset(s: Session, ds_id: int) -> db.Dataset:
    d = s.get(db.Dataset, ds_id)
    if not d:
        raise HTTPException(404, "Dataset not found")
    if not d.prepared:
        raise HTTPException(409, "Dataset must be analyzed and prepared before training.")
    return d


class RunIn(BaseModel):
    dataset_id: int
    model_id: str
    method: str | None = None
    goal: str = "general"
    auto: bool = False
    config: dict = {}
    name: str | None = None
    autotune: dict | None = None  # {"trials": 8, "budget_steps": 24, "metric": null|"val_loss"|"val_wer"}


class BatchIn(RunIn):
    methods: list[str]


def _create(s: Session, body: RunIn, method: str | None, group: str | None) -> db.Run:
    d = _prepared_dataset(s, body.dataset_id)
    try:
        m = get_model(body.model_id)
    except KeyError:
        raise HTTPException(404, "Unknown model")
    if m.task != d.task:
        raise HTTPException(422, f"{m.id} is a {m.task.upper()} model but the dataset is {d.task.upper()}.")
    raw = dict(body.config)
    gpu = detect_gpu()
    obs = observed_vram(s, gpu).get(m.id, {})
    if body.auto:
        rec = recommend(m, gpu, d.prepared["hours"]["train"], d.prepared["counts"]["train"], body.goal,
                        raw.get("domain_terms"), force_method=method, observed=obs, n_val=d.prepared["counts"]["val"])
        if not rec["method"]:
            raise HTTPException(422, f"{METHODS.get(method, method)} is unsupported for {m.id}: "
                                     f"{rec['alternatives'].get(method, {}).get('reason') or rec['explanation'][0]}")
        method = rec["method"]
        raw = {**rec["config"], **raw}
    if method not in METHODS:
        raise HTTPException(422, f"Unknown or missing method '{method}'.")
    sup = check_support(m, method, gpu, obs.get(method))
    if not sup.supported:
        raise HTTPException(422, f"{METHODS[method]} is unsupported for {m.id}: {sup.reason}")
    raw.setdefault("language", d.language)
    try:
        cfg = TrainConfig(**raw)
    except Exception as e:
        raise HTTPException(422, f"Invalid training config: {e}")
    errors, warnings = validate(method, cfg, family=m.family.lower())
    if errors:
        raise HTTPException(422, " ".join(errors))
    tune = None
    if body.autotune:
        tune = {"trials": body.autotune.get("trials", 8), "budget_steps": body.autotune.get("budget_steps", 24),
                "metric": body.autotune.get("metric")}
        if not (2 <= tune["trials"] <= 32 and 4 <= tune["budget_steps"] <= 500) or tune["metric"] not in (None, "val_loss", "val_wer"):
            raise HTTPException(422, "Auto-tune needs 2-32 trials, 4-500 steps per trial, and metric val_loss or val_wer.")
        if tune["metric"] == "val_wer" and m.family != "Whisper":
            raise HTTPException(422, "Ranking by validation WER is only available for speech-to-text models.")
    job = {"family": m.family, "model_id": m.id, "hf_id": m.hf_id, "method": method, "task": m.task,
           "dataset_dir": d.prepared["dir"], "train_config": cfg.model_dump(), "autotune": tune, "warnings": warnings}
    r = db.Run(project_id=d.project_id, dataset_id=d.id, model_id=m.id, method=method, config=job, group=group,
               name=body.name or f"{m.id} + {METHODS[method]}", status="queued")
    s.add(r)
    s.commit()
    orchestrator.submit(r.id)
    return r


@router.post("/runs", status_code=201)
def create_run(body: RunIn, s: Session = Depends(db.get_session)):
    return run_out(_create(s, body, body.method, None))


@router.post("/runs/batch", status_code=201)
def create_batch(body: BatchIn, s: Session = Depends(db.get_session)):
    """Same dataset and shared settings, one run per method, executed sequentially for a fair comparison."""
    if len(set(body.methods)) < 2:
        raise HTTPException(422, "Pick at least two different methods to compare.")
    group = uuid.uuid4().hex[:8]
    runs = [_create(s, body, meth, group) for meth in dict.fromkeys(body.methods)]
    return {"group": group, "runs": [run_out(r) for r in runs]}


@router.get("/runs")
def list_runs(project_id: int | None = None, group: str | None = None, s: Session = Depends(db.get_session)):
    q = s.query(db.Run)
    if project_id:
        q = q.filter_by(project_id=project_id)
    if group:
        q = q.filter_by(group=group)
    return [run_out(r) for r in q.order_by(db.Run.id.desc()).all()]


def _get(s: Session, run_id: int) -> db.Run:
    r = s.get(db.Run, run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@router.get("/runs/{run_id}")
def get_run(run_id: int, s: Session = Depends(db.get_session)):
    return run_out(_get(s, run_id))


@router.get("/runs/{run_id}/progress")
def progress(run_id: int, after: int = 0, s: Session = Depends(db.get_session)):
    r = _get(s, run_id)
    f = orchestrator.run_dir(run_id) / "progress.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
    return {"status": r.status, "next": len(lines), "events": [json.loads(x) for x in lines[after:]]}


@router.get("/runs/{run_id}/log")
def log(run_id: int, tail: int = 60, s: Session = Depends(db.get_session)):
    _get(s, run_id)
    f = orchestrator.run_dir(run_id) / "worker.log"
    return {"log": "\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]) if f.exists() else ""}


@router.post("/runs/{run_id}/stop")
def stop(run_id: int, s: Session = Depends(db.get_session)):
    r = _get(s, run_id)
    if r.status not in ("queued", "running"):
        raise HTTPException(409, f"Run is already {r.status}.")
    orchestrator.stop(run_id)
    return {"stopping": True}


@router.get("/runs/{run_id}/files/{path:path}")
def run_file(run_id: int, path: str, s: Session = Depends(db.get_session)):
    _get(s, run_id)
    base = (orchestrator.run_dir(run_id) / "samples").resolve()
    target = (orchestrator.run_dir(run_id) / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)
