"""Everything the UI needs to label and lay itself out, defined once in the backend."""
from fastapi import APIRouter

from ..datasets import analyzer, text
from ..datasets.augment import AUG_LABELS
from ..evaluation.metrics import METRICS
from ..modelstore.store import PEFT_METHODS
from ..registry import METHODS, MODELS
from ..strategy.engine import GOALS
from ..training.config import ui_field_specs
from ..training.methods.info import METHOD_INFO

router = APIRouter(prefix="/api/meta", tags=["meta"])

TASK_LABELS = {"stt": "Speech-to-Text", "tts": "Text-to-Speech", "llm": "Text LLM (Gemma-style)"}


@router.get("")
def meta():
    return {
        "tasks": [{"id": t, "label": lab, "families": sorted({m.family for m in MODELS.values() if m.task == t})}
                  for t, lab in TASK_LABELS.items()],
        "methods": METHODS,
        "method_info": METHOD_INFO,
        "merged_export_methods": sorted(PEFT_METHODS),
        "goals": [{"id": k, **v} for k, v in GOALS.items()],
        "augmentations": [{"id": k, "label": v} for k, v in AUG_LABELS.items()],
        "flags": {t: text.flag_specs() if t == "llm" else analyzer.flag_specs(t) for t in TASK_LABELS},
        "quality_labels": analyzer.QUALITY_LABELS,
        "score_labels": analyzer.SCORE_LABELS,
        "train_fields": ui_field_specs(),
        "metrics": {t: [m.to_dict() for m in ms] for t, ms in METRICS.items()},
    }
