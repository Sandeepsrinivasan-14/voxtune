"""Run one real training job directly (no server), e.g. for experiments and ablations.

usage: python scripts/train_cli.py <prepared_dataset_dir> <model_id> <method> '<json overrides>' [--autotune trials,steps]
prints a single JSON line: base vs tuned metrics, training info, peak GPU memory, auto-tune report (if any).
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.registry import get_model  # noqa: E402
from app.training.config import TrainConfig  # noqa: E402
from app.training.methods.validate import validate  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds, model_id, method = args[0], args[1], args[2]
    overrides = json.loads(args[3]) if len(args) > 3 else {}
    tune = None
    if "--autotune" in sys.argv:
        t, s = sys.argv[sys.argv.index("--autotune") + 1].split(",")
        tune = {"trials": int(t), "budget_steps": int(s)}
    spec = get_model(model_id)
    cfg = TrainConfig(**overrides)
    errors, warnings = validate(method, cfg, family=spec.family.lower())
    if errors:
        print(json.dumps({"error": errors}))
        sys.exit(2)
    out_root = Path(tempfile.gettempdir()) / "voxtune_cli"
    rd = out_root / f"run_{int(time.time() * 1000)}"
    rd.mkdir(parents=True)
    job = {"family": spec.family, "model_id": model_id, "hf_id": spec.hf_id, "method": method, "task": spec.task,
           "dataset_dir": ds, "train_config": cfg.model_dump(), "autotune": tune}
    (rd / "config.json").write_text(json.dumps(job))
    r = subprocess.run([sys.executable, "-m", "app.training.worker", str(rd)], cwd=ROOT / "backend", capture_output=True, text=True)
    if (rd / "error.json").exists():
        print(json.dumps({"error": json.loads((rd / "error.json").read_text())["error"][:500], "warnings": warnings}))
        sys.exit(1)
    res = json.loads((rd / "result.json").read_text())
    ev = [json.loads(x) for x in (rd / "progress.jsonl").read_text().splitlines()]
    used = [e["gpu_used_gb"] for e in ev if e["type"] == "train" and "gpu_used_gb" in e]
    lr = next((e for e in ev if e["type"] == "lr_find"), None)
    tuned = {k: v for k, v in res["tuned"].items() if k != "examples"}
    base = {k: v for k, v in res["base"].items() if k != "examples"}
    print(json.dumps({"method": method, "overrides": overrides, "base": base, "tuned": tuned,
                      "train": res["train"], "gpu_used_gb": max(used) if used else None,
                      "lr_find": None if not lr else {k: lr[k] for k in ("default", "chosen", "raw_suggestion", "clamped")},
                      "autotune": res.get("autotune"), "run_dir": str(rd), "rc": r.returncode}))


if __name__ == "__main__":
    main()
