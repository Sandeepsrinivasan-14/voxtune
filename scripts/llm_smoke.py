"""Real-GPU check of the text-LLM path: each variant is a full worker run (base eval -> train -> tuned eval) on the synthetic
lab-note task from make_llm_sample_dataset.py.

usage: python scripts/llm_smoke.py [model_id] [variant-name ...]
Prints one OK/FAIL line per variant: perplexity, next-token accuracy, ROUGE-L and exact match (base -> tuned), peak GPU, time.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = str(ROOT / "data" / "llm_sample" / "processed")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "smollm2-135m-instruct"
BASE = {"epochs": 3, "batch_size": 4, "grad_accum": 2, "precision": "bf16", "early_stopping": False, "max_eval_samples": 60,
        "max_seq_len": 256, "max_new_tokens": 32, "learning_rate": 2e-4, "lora_targets": "all_linear"}
VARIANTS = {
    "lora": ("lora", {}),
    "lora + DoRA": ("lora", {"use_dora": True}),
    "lora + auto LR + EMA + cosine": ("lora", {"auto_lr": True, "ema_decay": 0.99, "lr_scheduler": "cosine", "warmup_steps": 5}),
    "lora + label smoothing": ("lora", {"label_smoothing": 0.1}),
    "lora + PiSSA init": ("lora", {"lora_init": "pissa"}),
    "qlora (4-bit)": ("qlora", {}),
    "full fine-tuning (LLRD)": ("full", {"learning_rate": 2e-5, "llrd": 0.9}),
}
only = sys.argv[2:]
for name, (method, ov) in VARIANTS.items():
    if only and name not in only:
        continue
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_cli.py"), DS, MODEL, method, json.dumps({**BASE, **ov})],
                       capture_output=True, text=True)
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
        if "error" in r:
            raise RuntimeError(r["error"])
        b, u = r["base"], r["tuned"]
        print(f"OK   {name:32s} ppl {b['perplexity']:.2f}->{u['perplexity']:.2f}  tok-acc {b['token_accuracy']:.3f}->{u['token_accuracy']:.3f}  "
              f"rougeL {b['rouge_l']:.3f}->{u['rouge_l']:.3f}  exact {b['exact_match']:.3f}->{u['exact_match']:.3f}  "
              f"params {r['train']['trainable_params']:,}  gpu {r['gpu_used_gb']}GB  {time.time() - t:.0f}s"
              + (f"  lr {r['lr_find']['default']:g}->{r['lr_find']['chosen']:.2g}" if r.get("lr_find") else ""), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name:32s} {str(e)[:200] or (p.stderr or p.stdout)[-300:]}", flush=True)
