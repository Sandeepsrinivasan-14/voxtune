"""Ablation study on the larger dataset from make_ablation_dataset.py: every variant x every seed, real training each time.

usage:
  python scripts/ablation.py run [seeds=1,2,3] [max_steps=20]   # resumable; appends to data/ablation/results_<steps>steps.jsonl
  python scripts/ablation.py report [max_steps=20]              # mean/spread per variant and paired differences vs the LoRA baseline

Why a step budget: in the first 3-epoch run plain LoRA already reached WER 0.000 on this dataset (a ceiling: variants would tie), so the
study compares variants at a small fixed budget, where differences in sample efficiency can show. Each choice is therefore
"which setting gets furthest in N optimizer steps", not "which is best when trained to convergence".

Variants differ from the baseline in one idea each (the regularisation bundle is a deliberate exception). Same split, same
test set, same step budget; only the seed (adapter init, data order) changes between repeats.
"""
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ablation"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[1] == "run" else int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "report" else 20
RESULTS = DATA / f"results_{STEPS}steps.jsonl"
MODEL = "whisper-tiny"
TERMS = json.loads((DATA / "terms.json").read_text()) if (DATA / "terms.json").exists() else []
BASE = {"epochs": 3, "max_steps": STEPS, "batch_size": 4, "grad_accum": 2, "precision": "bf16", "early_stopping": False, "max_eval_samples": 200,
        "learning_rate": 5e-4, "domain_terms": TERMS}
VARIANTS = {  # name -> (method, overrides)
    "lora (baseline)": ("lora", {}),
    "lora + DoRA": ("lora", {"use_dora": True}),
    "lora + rsLoRA r32": ("lora", {"use_rslora": True, "lora_r": 32, "lora_alpha": 11}),
    "lora decoder, all linear": ("lora", {"lora_scope": "decoder", "lora_targets": "all_linear"}),
    "lora + regularisation bundle": ("lora", {"spec_augment": True, "ema_decay": 0.99, "label_smoothing": 0.1, "lr_scheduler": "cosine"}),
    "domain adaptation (x3 oversample, 2x loss)": ("domain_adaptation", {"domain_loss_weight": 2.0}),
    "qlora (4-bit base)": ("qlora", {"learning_rate": 2e-4}),
    "full fine-tuning": ("full", {"learning_rate": 1e-5}),
}
KEYS = ("wer", "cer", "domain_term_accuracy")


def run(seeds: list[int]) -> None:
    done = {(r["variant"], r["seed"]) for r in map(json.loads, RESULTS.read_text().splitlines())} if RESULTS.exists() else set()
    ds = str(DATA / "processed")
    for seed in seeds:
        for name, (method, ov) in VARIANTS.items():
            if (name, seed) in done:
                continue
            t = time.time()
            cfg = {**BASE, **ov, "seed": seed}
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_cli.py"), ds, MODEL, method, json.dumps(cfg)],
                               capture_output=True, text=True)
            try:
                r = json.loads(p.stdout.strip().splitlines()[-1])
                if "error" in r:
                    raise RuntimeError(r["error"])
                row = {"variant": name, "seed": seed, "method": method, "base": {k: r["base"].get(k) for k in KEYS},
                       "tuned": {k: r["tuned"].get(k) for k in KEYS}, "val_loss": r["train"]["best_val_loss"],
                       "params": r["train"]["trainable_params"], "gpu_gb": r["gpu_used_gb"], "seconds": round(time.time() - t), "n": r["tuned"]["n"]}
            except Exception as e:  # noqa: BLE001 - record and continue so one failure doesn't lose the study
                row = {"variant": name, "seed": seed, "method": method, "error": str(e)[:300] or (p.stderr or "")[-300:]}
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(name, seed, row.get("tuned") or row.get("error"), flush=True)


def report() -> None:
    rows = [json.loads(x) for x in RESULTS.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    by: dict[str, dict[int, dict]] = {}
    for r in ok:
        by.setdefault(r["variant"], {})[r["seed"]] = r
    base_rows = [r for v in by.values() for r in v.values()]
    n = base_rows[0]["n"]
    print(f"model {MODEL}; budget: {STEPS} optimizer steps (effective batch 8); test clips per run: {n}; seeds: {sorted({r['seed'] for r in ok})}; failed runs: {len(bad)}\n")
    b = {k: [r["base"][k] for r in base_rows if r["base"][k] is not None] for k in KEYS}
    print("no fine-tuning (base model):  " + "  ".join(f"{k} {st.mean(v):.4f}" for k, v in b.items() if v) + "\n")
    ref = by.get("lora (baseline)", {})

    def fmt(vals: list[float]) -> str:
        return f"{st.mean(vals):.4f} ± {st.pstdev(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"

    print(f"{'variant':44s} {'WER (mean ± sd)':>18s} {'domain-term acc':>18s} {'val loss':>9s} {'trainable':>11s} {'GPU GB':>7s} {'sec':>5s}  paired dWER vs lora")
    for name, seeds in by.items():
        wer = [r["tuned"]["wer"] for r in seeds.values()]
        dom = [r["tuned"]["domain_term_accuracy"] for r in seeds.values() if r["tuned"]["domain_term_accuracy"] is not None]
        paired = [r["tuned"]["wer"] - ref[s]["tuned"]["wer"] for s, r in seeds.items() if s in ref and name != "lora (baseline)"]
        pd = "" if not paired else f"{st.mean(paired):+.4f} (better in {sum(d < 0 for d in paired)}/{len(paired)} seeds)"
        first = next(iter(seeds.values()))
        print(f"{name:44s} {fmt(wer):>18s} {fmt(dom) if dom else '—':>18s} {st.mean(r['val_loss'] for r in seeds.values()):9.4f} "
              f"{first['params']:11,d} {st.mean(r['gpu_gb'] for r in seeds.values()):7.2f} {st.mean(r['seconds'] for r in seeds.values()):5.0f}  {pd}")
    for r in bad:
        print("FAILED", r["variant"], r["seed"], r["error"][:150])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        run([int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3").split(",")])
