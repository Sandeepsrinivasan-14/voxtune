from fastapi import APIRouter

from ..core import hostinfo
from ..core.gpu import detect_gpu

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/gpu")
def gpu():
    return detect_gpu().to_dict()


@router.get("/live")
def live():
    """Cheap, poll-friendly readings: CPU, RAM, disk and per-GPU utilisation/memory/temperature/power."""
    return hostinfo.live()


@router.get("/info")
def info():
    """Hardware and software environment of the machine running VoxTune."""
    return hostinfo.info()


@router.get("/health")
def health():
    return {"status": "ok"}
