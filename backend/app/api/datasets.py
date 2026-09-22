import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core import db
from ..core.config import settings
from ..datasets import analyzer, loader, prepare, text
from ..datasets.augment import AUGMENTATIONS

router = APIRouter(prefix="/api", tags=["datasets"])


def _out(d: db.Dataset) -> dict:
    return {"id": d.id, "project_id": d.project_id, "name": d.name, "task": d.task, "status": d.status, "language": d.language,
            "analysis": d.analysis, "prepared": d.prepared, "error": d.error, "created_at": d.created_at.isoformat()}


def _root(ds_id: int):
    return settings.datasets_dir / f"ds_{ds_id:04d}"


def _get(s: Session, ds_id: int) -> db.Dataset:
    d = s.get(db.Dataset, ds_id)
    if not d:
        raise HTTPException(404, "Dataset not found")
    return d


@router.post("/projects/{project_id}/datasets", status_code=201)
def upload_dataset(project_id: int, name: str = Form(...), language: str = Form("en"), file: UploadFile = File(...),
                   s: Session = Depends(db.get_session)):
    p = s.get(db.Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    fname = (file.filename or "").lower()
    if p.task == "llm":
        if not fname.endswith((".zip",) + text.TEXT_EXTENSIONS):
            raise HTTPException(422, "Upload a .jsonl/.csv/.tsv file (or a .zip containing one) with a prompt and a response column.")
    elif not fname.endswith(".zip"):
        raise HTTPException(422, "Upload a .zip containing audio files and a metadata.csv/.jsonl (or paired .txt files).")
    d = db.Dataset(project_id=project_id, name=name, task=p.task, language=language, path="")
    s.add(d)
    s.commit()
    root = _root(d.id)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zpath = settings.uploads_dir / f"{uuid.uuid4().hex}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(zpath, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if p.task == "llm":
            if fname.endswith(".zip"):
                loader.safe_extract(zpath, raw)
            else:
                shutil.copy(zpath, raw / f"metadata{Path(fname).suffix}")
            items = text.load_text_items(raw)
            if not items:
                raise ValueError("The file has no rows.")
        else:
            loader.safe_extract(zpath, raw)
            items = loader.load_items(raw)
            if not items:
                raise ValueError("No audio/transcript pairs were found in the archive.")
    except Exception as e:
        shutil.rmtree(root, ignore_errors=True)
        s.delete(d)
        s.commit()
        raise HTTPException(422, f"Could not read dataset: {e}")
    finally:
        zpath.unlink(missing_ok=True)
    (root / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    d.path = str(root)
    s.commit()
    return _out(d)


@router.get("/projects/{project_id}/datasets")
def list_datasets(project_id: int, s: Session = Depends(db.get_session)):
    return [_out(d) for d in s.query(db.Dataset).filter_by(project_id=project_id).order_by(db.Dataset.id.desc())]


@router.get("/datasets/{ds_id}")
def get_dataset(ds_id: int, s: Session = Depends(db.get_session)):
    return _out(_get(s, ds_id))


@router.delete("/datasets/{ds_id}", status_code=204)
def delete_dataset(ds_id: int, s: Session = Depends(db.get_session)):
    d = _get(s, ds_id)
    shutil.rmtree(_root(ds_id), ignore_errors=True)
    s.delete(d)
    s.commit()


def _background(ds_id: int, status_during: str, status_after: str, fn) -> None:
    def work():
        with db.SessionLocal() as s:
            d = s.get(db.Dataset, ds_id)
            try:
                fn(d)
                d.status, d.error = status_after, None
            except Exception as e:  # noqa: BLE001
                d.status, d.error = "failed", str(e)
            s.commit()

    threading.Thread(target=work, daemon=True).start()


@router.post("/datasets/{ds_id}/analyze", status_code=202)
def analyze(ds_id: int, s: Session = Depends(db.get_session)):
    d = _get(s, ds_id)
    if d.status in ("analyzing", "preparing"):
        raise HTTPException(409, f"Dataset is busy ({d.status}).")
    d.status = "analyzing"
    s.commit()

    def fn(d):
        items = json.loads((_root(d.id) / "items.json").read_text(encoding="utf-8"))
        summary, results = text.analyze_text_dataset(items) if d.task == "llm" else analyzer.analyze_dataset(items, d.task)
        (_root(d.id) / "analysis_items.json").write_text(json.dumps(results), encoding="utf-8")
        d.analysis, d.prepared = summary, None

    _background(ds_id, "analyzing", "analyzed", fn)
    return _out(d)


@router.get("/datasets/{ds_id}/items")
def items(ds_id: int, flag: str | None = None, limit: int = 200, s: Session = Depends(db.get_session)):
    _get(s, ds_id)
    f = _root(ds_id) / "analysis_items.json"
    if not f.exists():
        raise HTTPException(409, "Dataset has not been analyzed yet.")
    rows = json.loads(f.read_text(encoding="utf-8"))
    if flag:
        rows = [r for r in rows if flag in r["flags"]]
    return {"total": len(rows), "items": [{k: v for k, v in r.items() if k != "md5"} for r in rows[:limit]]}


class PrepareIn(BaseModel):
    exclude_flags: list[str] | None = None  # default depends on the task (audio vs text flags)
    val_frac: float = Field(0.1, gt=0, lt=0.4)
    test_frac: float = Field(0.1, gt=0, lt=0.4)
    augment: list[str] = []
    seed: int = 42


@router.post("/datasets/{ds_id}/prepare", status_code=202)
def prepare_ds(ds_id: int, body: PrepareIn, s: Session = Depends(db.get_session)):
    d = _get(s, ds_id)
    if not d.analysis:
        raise HTTPException(409, "Analyze the dataset before preparing it.")
    if d.status in ("analyzing", "preparing"):
        raise HTTPException(409, f"Dataset is busy ({d.status}).")
    allowed = text.TEXT_FLAGS if d.task == "llm" else analyzer.ALL_FLAGS
    exclude = body.exclude_flags if body.exclude_flags is not None else list(text.DEFAULT_EXCLUDE if d.task == "llm" else analyzer.DEFAULT_EXCLUDE)
    bad = [a for a in body.augment if a not in AUGMENTATIONS] + [f for f in exclude if f not in allowed]
    if bad:
        raise HTTPException(422, f"Unknown option(s): {bad}")
    if body.augment and d.task != "stt":
        raise HTTPException(422, "Augmentation is only available for STT datasets.")
    d.status = "preparing"
    s.commit()

    def fn(d):
        root = _root(d.id)
        items = json.loads((root / "items.json").read_text(encoding="utf-8"))
        results = json.loads((root / "analysis_items.json").read_text(encoding="utf-8"))
        shutil.rmtree(root / "processed", ignore_errors=True)
        if d.task == "llm":
            d.prepared = text.prepare_text_dataset(items, results, root / "processed", exclude_flags=exclude,
                                                   val_frac=body.val_frac, test_frac=body.test_frac, seed=body.seed)
        else:
            d.prepared = prepare.prepare_dataset(items, results, root / "processed", d.task, exclude_flags=exclude,
                                                 val_frac=body.val_frac, test_frac=body.test_frac, augment=body.augment,
                                                 seed=body.seed)

    _background(ds_id, "preparing", "prepared", fn)
    return _out(d)
