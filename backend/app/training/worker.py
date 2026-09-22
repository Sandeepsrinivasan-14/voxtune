"""Training worker: `python -m app.training.worker <run_dir>`. Runs in its own process so a run can be
killed cleanly and never takes the API server down with it."""
import gc
import importlib
import json
import sys
import traceback
from pathlib import Path

import torch

from .common import Progress
from .config import TrainConfig


# model family -> plugin module (families that share an implementation)
PLUGINS = {"gemma": "causal_lm", "smollm2": "causal_lm", "qwen2": "causal_lm"}


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def main(run_dir: str) -> int:
    run_dir = Path(run_dir)
    job = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**job["train_config"])
    progress = Progress(run_dir)
    fam = job["family"].lower()
    plugin = importlib.import_module(f"app.training.plugins.{PLUGINS.get(fam, fam)}")
    try:
        for attempt in range(4):
            try:
                result = plugin.run(job, cfg, run_dir, progress)
                break
            except Exception as e:
                if not _is_oom(e) or cfg.batch_size == 1 or attempt == 3:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                new_bs = max(1, cfg.batch_size // 2)
                cfg.grad_accum *= -(-cfg.batch_size // new_bs)  # keep the effective batch size
                progress.emit("oom_retry", old_batch=cfg.batch_size, new_batch=new_bs, grad_accum=cfg.grad_accum)
                cfg.batch_size = new_bs
        result["final_config"] = cfg.model_dump()
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress.emit("done", status=result.get("train", {}).get("status", "completed"))
        return 0
    except BaseException as e:  # noqa: BLE001 - the parent process reads this file
        (run_dir / "error.json").write_text(json.dumps({"error": str(e), "trace": traceback.format_exc()}), encoding="utf-8")
        progress.emit("error", message=str(e)[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
