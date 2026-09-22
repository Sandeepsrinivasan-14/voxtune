"""Auto-tune: measure which technique/hyper-parameter combination works best on THIS data, instead of trusting defaults.

Successive halving: sample candidates (the untuned baseline is always included), train each for a small step budget,
keep the best third, give the survivors a 3x budget, repeat, then hand the winner back for the full training run.
Candidates that are invalid combinations are never trained. The report compares the winner to the baseline at equal budget.
"""
import math
import random
import time
import traceback

from ..config import TrainConfig
from .validate import validate

LR_MULT = [0.5, 1.0, 2.0]

# per-mechanism search dimensions: field -> options ("lr_mult" scales the configured learning rate)
SPACES = {
    "lora": {"lora_r": [8, 16, 32], "lora_targets": ["attention", "all_linear"], "lora_scope": ["all", "decoder"],
             "use_dora": [False, True], "use_rslora": [False, True], "loraplus_ratio": [0.0, 8.0, 16.0],
             "lora_init": ["default", "pissa"], "spec_augment": [False, True], "lr_mult": LR_MULT},
    "qlora": {"lora_r": [8, 16, 32], "lora_targets": ["attention", "all_linear"], "lora_scope": ["all", "decoder"],
              "use_dora": [False, True], "use_rslora": [False, True], "loraplus_ratio": [0.0, 8.0, 16.0],
              "spec_augment": [False, True], "lr_mult": LR_MULT},
    "adalora": {"lora_r": [12, 16, 24], "lora_targets": ["attention", "all_linear"], "lora_scope": ["all", "decoder"],
                "spec_augment": [False, True], "lr_mult": LR_MULT},
    "ia3": {"spec_augment": [False, True], "lr_mult": [0.3, 1.0, 3.0]},
    "full": {"llrd": [1.0, 0.9, 0.8], "freeze_encoder": [False, True], "label_smoothing": [0.0, 0.1],
             "spec_augment": [False, True], "lr_mult": [0.3, 1.0, 3.0]},
    "freeze_text_encoder": {"lr_mult": [0.3, 1.0, 3.0]},
}


def _apply(cfg: TrainConfig, overrides: dict) -> TrainConfig:
    o = dict(overrides)
    lr_mult = o.pop("lr_mult", 1.0)
    upd = {**o, "learning_rate": cfg.learning_rate * lr_mult}
    if "lora_r" in o:  # keep the effective adapter scale comparable across ranks
        upd["lora_alpha"] = max(1, round(2 * math.sqrt(o["lora_r"]))) if o.get("use_rslora") else 2 * o["lora_r"]
    return cfg.model_copy(update=upd)


def sample_candidates(mech: str, method: str, cfg: TrainConfig, family: str, n: int, seed: int) -> list[dict]:
    space = SPACES[mech]
    rng = random.Random(seed)
    cands: list[dict] = [{}]  # the baseline: exactly what the user/recommendation configured
    seen = {frozenset()}
    attempts = 0
    while len(cands) < n and attempts < 400:
        attempts += 1
        k = rng.randint(1, min(3, len(space)))  # change a few dimensions at once, keep candidates interpretable
        o = {f: rng.choice(space[f]) for f in rng.sample(list(space), k)}
        key = frozenset(o.items())
        if key in seen:
            continue
        errors, _ = validate(method, _apply(cfg, o), family=family)
        if errors:
            continue
        seen.add(key)
        cands.append(o)
    return cands


def _score(m: dict, primary: str) -> tuple:
    other = "val_loss" if primary == "val_wer" else "val_wer"
    return (m.get(primary, math.inf), m.get(other, math.inf))


def autotune(ctx, method: str, cfg: TrainConfig, opts: dict, trial_fn):
    """Returns (best TrainConfig, report dict). `trial_fn(cfg, steps) -> {"val_loss", "val_wer"?, "seconds", ...}`."""
    from .info import MECHANISM

    mech = MECHANISM[method]
    family = ctx.job["family"].lower()
    n = int(opts.get("trials", 8))
    steps = int(opts.get("budget_steps", 24))
    primary = opts.get("metric") or ("val_wer" if family == "whisper" and len(ctx.val) >= 30 else "val_loss")
    cands = sample_candidates(mech, method, cfg, family, n, cfg.seed)
    progress = ctx.progress
    progress.emit("autotune_plan", candidates=len(cands), budget_steps=steps, metric=primary, space=list(SPACES[mech]))

    trials: list[dict] = []
    pool = list(range(len(cands)))
    rnd, budget, t0 = 1, steps, time.time()
    while True:
        results = []
        for ci in pool:
            overrides = cands[ci]
            tid = len(trials)
            progress.emit("status", message=f"Auto-tune round {rnd}: candidate {ci + 1}/{len(cands)} ({budget} steps)")
            try:
                m = trial_fn(_apply(cfg, overrides), budget)
                status = "ok"
            except Exception as e:  # noqa: BLE001 - one bad candidate (e.g. OOM) must not sink the search
                m, status = {"error": str(e)[:200], "trace": traceback.format_exc()[-400:]}, "failed"
            t = {"trial": tid, "candidate": ci, "round": rnd, "budget_steps": budget, "overrides": overrides, "status": status,
                 "metrics": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k != "trace"}}
            trials.append(t)
            results.append((_score(m, primary) if status == "ok" else (math.inf, math.inf), ci, t))
            progress.emit("autotune_trial", **{k: t[k] for k in ("trial", "candidate", "round", "budget_steps", "overrides", "status", "metrics")})
        results.sort(key=lambda r: r[0])
        keep = max(1, math.ceil(len(pool) / 3))
        survivors = [ci for _, ci, _ in results[:keep]]
        for _, ci, t in results:
            t["survived"] = ci in survivors
        if len(survivors) == 1 or rnd >= 3 or results[0][0][0] == math.inf:
            winner = results[0][1]
            break
        pool, rnd, budget = survivors, rnd + 1, budget * 3

    best = cands[winner]
    r1 = {t["candidate"]: t for t in trials if t["round"] == 1}
    base_t, win_t = r1[0], r1[winner]
    report = {
        "metric": primary, "trials": trials, "winner": {"candidate": winner, "overrides": best},
        "baseline_round1": base_t["metrics"], "winner_round1": win_t["metrics"],
        "improved_over_baseline": winner != 0 and _score(win_t["metrics"], primary) < _score(base_t["metrics"], primary),
        "seconds": round(time.time() - t0, 1),
        "note": "Trials are short and validation sets small, so treat small gaps as noise; the baseline is always a candidate.",
    }
    progress.emit("autotune_done", winner=best, improved=report["improved_over_baseline"], seconds=report["seconds"], metric=primary)
    return _apply(cfg, best), report
