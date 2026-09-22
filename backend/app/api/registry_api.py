from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core import db
from ..modelstore import store

router = APIRouter(prefix="/api/registry", tags=["registry"])


def _out(m: db.RegisteredModel) -> dict:
    return {"id": m.id, "name": m.name, "version": m.version, "base_model_id": m.base_model_id, "method": m.method,
            "run_id": m.run_id, "dataset_name": m.dataset_name, "meta": m.meta, "created_at": m.created_at.isoformat()}


class RegisterIn(BaseModel):
    run_id: int
    name: str | None = None


@router.post("", status_code=201)
def register(body: RegisterIn, s: Session = Depends(db.get_session)):
    r = s.get(db.Run, body.run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    try:
        return _out(store.register(s, r, body.name))
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.get("")
def list_models(s: Session = Depends(db.get_session)):
    return [_out(m) for m in s.query(db.RegisteredModel).order_by(db.RegisteredModel.id.desc())]


@router.get("/{model_id}/export")
def export(model_id: int, format: str = "adapter", s: Session = Depends(db.get_session)):
    m = s.get(db.RegisteredModel, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    if format not in ("adapter", "merged"):
        raise HTTPException(422, "format must be 'adapter' or 'merged'")
    if format == "merged" and m.method not in store.PEFT_METHODS:
        raise HTTPException(422, f"'merged' export only applies to adapter-style methods; {m.method} is already a full model.")
    path = store.export_zip(m, format)
    return FileResponse(path, filename=path.name, media_type="application/zip")


@router.delete("/{model_id}", status_code=204)
def delete(model_id: int, s: Session = Depends(db.get_session)):
    import shutil

    m = s.get(db.RegisteredModel, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    shutil.rmtree(m.path, ignore_errors=True)
    s.delete(m)
    s.commit()
