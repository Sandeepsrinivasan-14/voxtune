"""Strategy engine: rule-based recommendation of a method + training config, with a plain-language
explanation. It only ever recommends methods that the capability registry says can run here."""
import math

from ..core.gpu import GpuInfo
from ..registry import METHODS, ModelSpec, check_support

GOALS = {
    "general": {"label": "General accuracy improvement", "tasks": ["stt", "tts", "llm"]},
    "domain": {"label": "Domain vocabulary (e.g. medical terms)", "tasks": ["stt"]},
    "accent": {"label": "Accent / recording conditions", "tasks": ["stt"]},
    "speaker": {"label": "Custom voice / speaker", "tasks": ["tts"]},
}


def _bf16_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _pick_method(model: ModelSpec, gpu: GpuInfo, hours: float, goal: str, why: list[str], obs: dict[str, float]) -> str | None:
    ok = {m: check_support(model, m, gpu, obs.get(m)).supported for m in model.methods}
    if model.task == "llm":
        n = obs.get("_n_train", 0)
        if ok.get("full") and model.params_m <= 300 and n >= 5000:
            why.append(f"{n} examples is plenty for a {model.params_m:g}M-parameter model and the GPU can hold every weight, "
                       "so full fine-tuning gives the most capacity.")
            return "full"
        if ok.get("lora"):
            tight = gpu.total_gb < obs.get("lora", model.methods["lora"]) * 1.3
            if tight and ok.get("qlora"):
                why.append("GPU memory is tight for LoRA on this model, so QLoRA (4-bit base) is used.")
                return "qlora"
            why.append("LoRA trains small adapter matrices on the attention and feed-forward layers: little forgetting, low memory, "
                       "and a small adapter file per custom version.")
            return "lora"
        for m in ("qlora", "full"):
            if ok.get(m):
                why.append(f"{METHODS[m]} is the only method that fits your hardware for this model.")
                return m
        return None
    if model.task == "tts":
        if goal == "speaker" and ok.get("speaker_adaptation"):
            why.append("Goal is a custom voice, so speaker adaptation is preferred: it freezes the text encoder and only "
                       "adapts the decoder to the target speaker, which needs less data and memory.")
            return "speaker_adaptation"
        if ok.get("full") and hours >= 2:
            why.append(f"{hours:.1f} h of speech is enough for full fine-tuning of the whole TTS model.")
            return "full"
        if ok.get("speaker_adaptation"):
            why.append("Limited data or memory: adapting only the decoder is safer than updating every weight.")
            return "speaker_adaptation"
        return "full" if ok.get("full") else None
    if goal == "domain" and ok.get("domain_adaptation"):
        why.append("Goal is domain vocabulary, so domain adaptation is used: LoRA plus oversampling of utterances that "
                   "contain your domain terms, evaluated with domain-term accuracy.")
        return "domain_adaptation"
    if ok.get("full") and hours >= 30 and model.params_m <= 300:
        why.append(f"{hours:.0f} h is a large dataset relative to a {model.params_m:g}M-parameter model, and the GPU can hold "
                   "full fine-tuning, so updating all weights gives the most capacity.")
        return "full"
    if ok.get("lora"):
        tight = gpu.total_gb < obs.get("lora", model.methods["lora"]) * 1.3
        if tight and ok.get("qlora"):
            why.append("GPU memory is tight for LoRA on this model, so QLoRA (4-bit base) is used to cut memory.")
            return "qlora"
        why.append(f"With {hours:.1f} h of data, LoRA trains a small set of adapter weights: less overfitting, lower memory, "
                   "and a small adapter file per custom version.")
        return "lora"
    for m in ("qlora", "adapter", "full"):
        if ok.get(m):
            why.append(f"{METHODS[m]} is the only method that fits your hardware for this model.")
            return m
    return None


def recommend(model: ModelSpec, gpu: GpuInfo, dataset_hours: float, n_train: int, goal: str = "general",
              domain_terms: list[str] | None = None, force_method: str | None = None,
              observed: dict[str, float] | None = None, n_val: int = 0) -> dict:
    """With `force_method` the method is the user's choice and only the settings are tuned for it.
    `observed` maps method -> VRAM (GB) that real runs of this model used on this GPU."""
    obs = observed or {}
    alts = {m: check_support(model, m, gpu, obs.get(m)).to_dict() for m in model.methods}
    why: list[str] = []
    warnings: list[str] = []
    if model.task == "llm":
        obs = {**obs, "_n_train": n_train}
    if force_method:
        method = force_method if alts.get(force_method, {}).get("supported") else None
        if method:
            why.append(f"{METHODS[method]} was chosen manually; the settings below are tuned for it.")
    else:
        method = _pick_method(model, gpu, dataset_hours, goal, why, obs)
    if method is None:
        return {"method": None, "config": None, "explanation": ["No fine-tuning method for this model can run on the detected hardware."],
                "warnings": [], "alternatives": alts, "metric": None}
    if goal == "domain" and not domain_terms:
        warnings.append("Domain goal selected but no domain terms were given; domain adaptation needs them.")
    if model.task != "llm" and dataset_hours < 0.25:
        warnings.append(f"Only {dataset_hours * 60:.0f} minutes of audio: expect noisy metrics and possible overfitting.")

    if model.task == "llm":
        return _finish_llm(model, gpu, method, n_train, n_val, why, warnings, alts)

    need = model.methods[method]
    ratio = gpu.total_gb / need if gpu.available else 0
    batch = 2 if ratio < 2 else 4 if ratio < 3 else 8 if ratio < 6 else 16
    batch = max(1, min(batch, n_train))
    eff = min(16, max(2, n_train // 8))
    accum = max(1, math.ceil(eff / batch))
    steps_per_epoch = max(1, math.ceil(n_train / (batch * accum)))
    epochs = max(3, min(15 if method == "full" else 30, math.ceil(300 / steps_per_epoch)))
    total = steps_per_epoch * epochs
    lr = {"full": 1e-5, "lora": 5e-4, "domain_adaptation": 5e-4, "qlora": 2e-4, "adapter": 3e-3, "adalora": 5e-4,
          "speaker_adaptation": 5e-5}[method]
    if model.task == "tts" and method == "lora":
        lr = 5e-4
    if model.task == "tts" and method == "full":
        lr = 2e-5
    precision = "bf16" if _bf16_ok() else "fp16" if gpu.available else "fp32"
    opt = "adamw_8bit" if method == "full" and gpu.total_gb < 12 and gpu.available else "adamw"
    config = {
        "epochs": epochs, "learning_rate": lr, "batch_size": batch, "grad_accum": accum,
        "warmup_steps": min(500, int(0.1 * total)), "precision": precision, "optimizer": opt,
        "eval_steps": 0, "early_stopping": True, "patience": 3,
        "gradient_checkpointing": model.params_m >= 200 or gpu.total_gb < 8,
        "language": "en", "domain_terms": domain_terms or [], "max_eval_samples": 12 if model.task == "tts" else 50,
    }
    why.append(f"Batch {batch} x gradient accumulation {accum} (effective {batch * accum}) is sized for {gpu.total_gb:g} GB of VRAM; "
               "if it still runs out of memory the worker halves the batch automatically.")
    why.append(f"{epochs} epochs (~{total} optimizer steps) with early stopping and best-checkpoint saving "
               "(the model you get back is exactly the checkpoint that was saved).")
    if opt == "adamw_8bit":
        why.append("8-bit AdamW is used to shrink optimizer state on a small GPU.")
    updates, notes = _techniques(model, method, dataset_hours, total, n_val, domain_terms)
    config.update(updates)
    why += notes
    tune_steps = min(60, max(16, steps_per_epoch))
    autotune = {"suggested": dataset_hours >= 0.25 and n_val >= 8, "trials": 8, "budget_steps": tune_steps,
                "reason": "Everything above is a sensible default from common practice. Auto-tune measures candidates "
                          "(ranks, adapted layers, DoRA/rsLoRA/LoRA+, LR, regularisers...) on YOUR validation data and keeps the best; "
                          "the untuned baseline always competes, so it can only match or beat the defaults on that data."}
    return {"method": method, "config": config, "explanation": why, "warnings": warnings, "alternatives": alts, "autotune": autotune,
            "metric": "WER" if model.task == "stt" else "MFCC distance / speaker similarity / intelligibility (ASR WER)"}


def _finish_llm(model: ModelSpec, gpu: GpuInfo, method: str, n_train: int, n_val: int, why: list[str], warnings: list[str],
                alts: dict) -> dict:
    """Settings for a text LLM. Same principle as the speech path: sensible defaults with the reasoning shown; Auto-tune measures."""
    if n_train < 50:
        warnings.append(f"Only {n_train} training examples: expect noisy metrics and overfitting.")
    ratio = gpu.total_gb / model.methods[method] if gpu.available else 0
    batch = max(1, min(2 if ratio < 2 else 4 if ratio < 3 else 8 if ratio < 6 else 16, n_train))
    accum = max(1, math.ceil(min(16, max(4, n_train // 8)) / batch))
    spe = max(1, math.ceil(n_train / (batch * accum)))
    epochs = max(2, min(5, math.ceil(150 / spe)))
    total = spe * epochs
    lr = {"full": 2e-5, "lora": 2e-4, "qlora": 2e-4}[method]
    big_vocab = model.family == "Gemma"
    config = {
        "epochs": epochs, "learning_rate": lr, "batch_size": batch, "grad_accum": accum,
        "warmup_steps": min(100, int(0.1 * total)), "precision": "bf16" if _bf16_ok() else "fp16" if gpu.available else "fp32",
        "optimizer": "adamw_8bit" if method == "full" and gpu.total_gb < 12 and gpu.available else "adamw",
        "eval_steps": 0, "early_stopping": True, "patience": 3, "gradient_checkpointing": model.params_m >= 400 or gpu.total_gb < 8,
        "max_seq_len": 512, "max_new_tokens": 64, "max_eval_samples": 50, "auto_lr": True, "lr_scheduler": "cosine",
    }
    why.append(f"Batch {batch} x gradient accumulation {accum} (effective {batch * accum}) is sized for {gpu.total_gb:g} GB of VRAM"
               + ("; Gemma's 262k-token vocabulary makes the output layer large, so a smaller batch is used" if big_vocab else "")
               + "; if it still runs out of memory the worker halves the batch automatically.")
    why.append(f"{epochs} epochs (~{total} optimizer steps) with early stopping and best-checkpoint saving; the loss covers the "
               "response tokens only, not the prompt.")
    if method in ("lora", "qlora"):
        r = 16 if n_train < 2000 else 32
        config.update(lora_r=r, lora_alpha=2 * r, lora_targets="all_linear")
        why.append(f"LoRA rank {r} on attention and feed-forward layers (the usual choice for instruction tuning).")
        if r >= 32:
            config.update(use_rslora=True, lora_alpha=round(2 * math.sqrt(r)))
            why.append("rsLoRA scaling keeps the higher rank from stalling.")
    if method == "full":
        config["llrd"] = 0.9
        why.append("Layer-wise LR decay 0.9 protects pretrained layers during full fine-tuning.")
    why.append("Auto learning rate: a short LR range test on your own batches picks the LR (kept within 0.1x-5x of the default). Cosine decay.")
    if total >= 150:
        config["ema_decay"] = 0.999
        why.append("EMA of the weights (0.999): the evaluated and saved model is the smoothed one.")
    autotune = {"suggested": n_train >= 200 and n_val >= 8, "trials": 8, "budget_steps": min(60, max(16, spe)),
                "reason": "Everything above is a sensible default from common practice. Auto-tune measures candidates (rank, adapted layers, "
                          "DoRA/rsLoRA/LoRA+, LR) on YOUR validation data and keeps the best; the untuned baseline always competes."}
    return {"method": method, "config": config, "explanation": why, "warnings": warnings, "alternatives": alts, "autotune": autotune,
            "metric": "Perplexity on held-out responses (lower is better), plus next-token accuracy, ROUGE-L and exact match"}


def _techniques(model: ModelSpec, method: str, hours: float, total_steps: int, n_val: int,
                terms: list[str] | None) -> tuple[dict, list[str]]:
    """Technique defaults with the reasoning behind each. These follow common practice; Auto-tune is what measures them."""
    up: dict = {"auto_lr": True, "lr_scheduler": "cosine"}
    notes = ["Auto learning rate: a short LR range test on your own batches picks the LR (kept within 0.1x-5x of the method's default, "
             "weights restored afterwards). Cosine decay after warmup."]
    stt = model.task == "stt"
    if method in ("lora", "qlora", "domain_adaptation", "adalora"):
        r = 16 if hours < 5 else 32
        up.update(lora_r=r, lora_alpha=2 * r, lora_targets="all_linear" if hours >= 3 else "attention")
        notes.append(f"LoRA rank {r} on {'attention and feed-forward' if hours >= 3 else 'attention'} layers: "
                     f"{'more data supports more capacity' if hours >= 3 else 'little data favours a compact adapter'}.")
        if r >= 32 and method != "adalora":
            up.update(use_rslora=True, lora_alpha=round(2 * math.sqrt(r)))
            notes.append("rsLoRA scaling (alpha/sqrt(r)) keeps the higher rank from stalling.")
    if method == "full":
        up["llrd"] = 0.9
        notes.append("Layer-wise LR decay 0.9 (lower layers move less) protects pretrained features during full fine-tuning.")
    if stt and hours < 10:
        up["spec_augment"] = True
        notes.append("SpecAugment masking in the encoder regularises small ASR datasets.")
    if total_steps >= 150:
        up["ema_decay"] = 0.999
        notes.append("EMA of the weights (0.999): the evaluated and saved model is the smoothed one.")
    if stt and n_val >= 30:
        up["select_by"] = "val_wer"
        notes.append(f"Best checkpoint is chosen by validation WER ({n_val} validation clips), the metric you care about.")
    elif stt:
        notes.append(f"Best checkpoint is chosen by validation loss: {n_val} validation clips are too few for WER to be a stable selector.")
    if method == "domain_adaptation" and terms:
        up["domain_loss_weight"] = 2.0
        notes.append("Loss on tokens of your domain terms is weighted 2x, on top of oversampling utterances that contain them.")
    return up, notes
