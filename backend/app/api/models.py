from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core import db, hostinfo
from ..core.gpu import detect_gpu
from ..registry import METHODS, check_support, get_model, list_models, observed_vram

router = APIRouter(prefix="/api/models", tags=["models"])


def describe(m, gpu, observed: dict[str, float]) -> dict:
    d = m.to_dict()
    d["support"] = {meth: check_support(m, meth, gpu, observed.get(meth)).to_dict() for meth in METHODS if meth in m.methods}
    d["observed_gb"] = observed
    d["local"] = hostinfo.model_cache_status(m.hf_id)
    return d


@router.get("")
def models(task: str | None = None, s: Session = Depends(db.get_session)):
    if task not in (None, "stt", "tts", "llm"):
        raise HTTPException(422, "task must be 'stt', 'tts' or 'llm'")
    gpu = detect_gpu()
    obs = observed_vram(s, gpu)
    return [describe(m, gpu, obs.get(m.id, {})) for m in list_models(task)]


@router.get("/{model_id}")
def model(model_id: str, s: Session = Depends(db.get_session)):
    try:
        m = get_model(model_id)
    except KeyError:
        raise HTTPException(404, "Unknown model")
    gpu = detect_gpu()
    return describe(m, gpu, observed_vram(s, gpu).get(m.id, {}))
