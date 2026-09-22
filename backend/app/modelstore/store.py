"""Versioned model registry (checkpoints copied out of run dirs) and export."""
import json
import re
import shutil
import tempfile
from pathlib import Path

from ..core import db
from ..core.config import model_source, settings
from ..core.gpu import detect_gpu
from ..registry import get_model

PEFT_METHODS = {"lora", "qlora", "adalora", "adapter", "domain_adaptation"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-").lower() or "model"


def register(s, run: db.Run, name: str | None) -> db.RegisteredModel:
    if run.status != "completed":
        raise ValueError("Only completed runs can be registered.")
    src = settings.runs_dir / f"run_{run.id:04d}" / "checkpoint"
    if not src.exists():
        raise ValueError("Run has no checkpoint.")
    ds = s.get(db.Dataset, run.dataset_id)
    base_name = name or f"{run.model_id}-{run.method}"
    version = s.query(db.RegisteredModel).filter_by(name=base_name).count() + 1
    dest = settings.registry_dir / f"{_slug(base_name)}-v{version}"
    shutil.copytree(src, dest)
    m = run.metrics or {}
    gpu = detect_gpu()
    meta = {
        "task": m.get("kind"), "base_hf_id": get_model(run.model_id).hf_id, "config": m.get("final_config"),
        "base_metrics": {k: v for k, v in (m.get("base") or {}).items() if k != "examples"},
        "tuned_metrics": {k: v for k, v in (m.get("tuned") or {}).items() if k != "examples"},
        "train_seconds": run.train_seconds, "peak_gpu_gb": run.peak_gpu_gb, "gpu": gpu.name if gpu.available else "CPU",
        "dataset_hours": (ds.prepared or {}).get("hours") if ds else None, "language": m.get("final_config", {}).get("language"),
    }
    rm = db.RegisteredModel(name=base_name, base_model_id=run.model_id, method=run.method, version=version, run_id=run.id,
                            dataset_name=ds.name if ds else "", path=str(dest), meta=meta)
    s.add(rm)
    s.commit()
    return rm


def export_zip(rm: db.RegisteredModel, fmt: str) -> Path:
    """fmt='adapter' zips the checkpoint as-is; fmt='merged' bakes PEFT weights into the base model (fp16)."""
    src = Path(rm.path)
    tmp = Path(tempfile.mkdtemp(prefix="voxtune_export_"))
    out = tmp / "model"
    if fmt == "merged" and rm.method in PEFT_METHODS:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, SpeechT5ForTextToSpeech, WhisperForConditionalGeneration

        task = get_model(rm.base_model_id).task
        cls = AutoModelForCausalLM if task == "llm" else SpeechT5ForTextToSpeech if task == "tts" else WhisperForConditionalGeneration
        base = cls.from_pretrained(model_source(rm.meta["base_hf_id"]), dtype=torch.float16)
        merged = PeftModel.from_pretrained(base, str(src)).merge_and_unload()
        merged.save_pretrained(out)
        for f in src.glob("*.json"):
            if f.name != "adapter_config.json":
                shutil.copy(f, out / f.name)
    else:
        shutil.copytree(src, out)
    (out / "voxtune_metadata.json").write_text(json.dumps({"name": rm.name, "version": rm.version, "method": rm.method,
                                                          "base_model": rm.base_model_id, "dataset": rm.dataset_name, **rm.meta}, indent=2))
    archive = shutil.make_archive(str(tmp / f"{_slug(rm.name)}-v{rm.version}-{fmt}"), "zip", out)
    return Path(archive)
