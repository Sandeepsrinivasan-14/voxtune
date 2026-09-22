"""Real statistics about the machine VoxTune is running on and the workspace it manages."""
import os
import platform
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path

import psutil

from .config import settings
from .gpu import gpus_live

PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "speechbrain", "librosa", "jiwer",
            "soundfile", "fastapi", "sqlalchemy")
_size_cache: dict[str, tuple[float, int]] = {}
psutil.cpu_percent(interval=None)  # prime the counter so the first real reading isn't 0


def dir_size(path: Path, ttl: float = 15.0) -> int:
    """Total bytes under path (cached briefly; partial downloads excluded)."""
    key = str(path)
    hit = _size_cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".part", ".incomplete")):
                continue
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    _size_cache[key] = (time.time(), total)
    return total


def gb(n: float) -> float:
    return round(n / 1024**3, 2)


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def os_name() -> str:
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        # platform.release() reports "10" on Windows 11; builds >= 22000 are Windows 11
        return f"{'Windows 11' if build >= 22000 else 'Windows 10'} (build {build})"
    return f"{platform.system()} {platform.release()}"


def live() -> dict:
    settings.ensure_dirs()
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(settings.data_dir)
    freq = psutil.cpu_freq()
    return {
        "t": time.time(),
        "cpu": {"percent": psutil.cpu_percent(interval=None), "freq_mhz": round(freq.current) if freq else None},
        "ram": {"used_gb": gb(vm.used), "total_gb": gb(vm.total), "percent": vm.percent},
        "disk": {"path": str(settings.data_dir), "used_gb": gb(du.used), "total_gb": gb(du.total), "free_gb": gb(du.free),
                 "percent": round(du.used / du.total * 100, 1)},
        "gpus": gpus_live(),
    }


def info() -> dict:
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    torch_info: dict = {"installed": False}
    try:
        import torch

        cuda = torch.cuda.is_available()
        torch_info = {
            "installed": True, "version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": cuda,
            "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
            "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                         "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(i)))}
                        for i in range(torch.cuda.device_count())],
        }
    except Exception:
        pass
    return {
        "os": os_name(),
        "python": platform.python_version(),
        "cpu": {"name": cpu_name(), "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count()},
        "ram_total_gb": gb(psutil.virtual_memory().total),
        "torch": torch_info,
        "packages": pkgs,
        "data_dir": str(settings.data_dir),
    }


def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_status(hf_id: str) -> dict:
    """Where (if anywhere) this model's weights are already on disk."""
    local = settings.data_dir / "models_cache" / hf_id.replace("/", "--")
    if (local / "COMPLETE").exists():
        return {"cached": True, "where": "VoxTune mirror", "size_gb": gb(dir_size(local))}
    repo = _hf_cache_dir() / f"models--{hf_id.replace('/', '--')}"
    snaps = repo / "snapshots"
    if snaps.exists() and any(snaps.glob("*/model.safetensors")) | any(snaps.glob("*/pytorch_model.bin")):
        return {"cached": True, "where": "Hugging Face cache", "size_gb": gb(dir_size(repo / "blobs"))}
    return {"cached": False, "where": None, "size_gb": 0.0}


def storage() -> dict:
    parts = {
        "uploads": settings.uploads_dir, "datasets": settings.datasets_dir, "runs": settings.runs_dir,
        "registry": settings.registry_dir, "model_mirror": settings.data_dir / "models_cache",
    }
    sizes = {k: gb(dir_size(p)) if p.exists() else 0.0 for k, p in parts.items()}
    db_file = settings.data_dir / "voxtune.db"
    sizes["database"] = gb(db_file.stat().st_size) if db_file.exists() else 0.0
    sizes["total"] = round(sum(sizes.values()), 2)
    return sizes
