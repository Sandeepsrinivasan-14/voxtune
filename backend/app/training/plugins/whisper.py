"""Whisper fine-tuning on the shared methods core.

Methods: full FT, LoRA (+DoRA / rsLoRA / PiSSA / OLoRA / LoRA+), QLoRA (4-bit), AdaLoRA, IA3 adapter and domain adaptation.
Techniques: LR range test, layer-wise LR decay, EMA, SpecAugment, label smoothing, domain-term loss weighting,
selection of the best checkpoint by validation WER, and Auto-tune (search over technique/hyper-parameter candidates).
"""
import gc
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor

from ...core.config import model_source
from ...evaluation import stt
from ..common import Progress, amp_dtype, device, fit, plan_steps, read_jsonl
from ..config import TrainConfig
from ..methods.apply import apply_method, needs_fp32_base
from ..methods.autotune import autotune
from ..methods.ema import EMA
from ..methods.info import MECHANISM
from ..methods.losses import domain_weights, weighted_ce
from ..methods.lrfind import find_lr
from ..methods.params import param_groups

FAMILY = "whisper"


@dataclass
class Ctx:
    job: dict
    cfg: TrainConfig
    run_dir: Path
    progress: Progress
    dev: str
    train: list[dict]
    val: list[dict]
    test: list[dict]
    processor: WhisperProcessor
    patterns: list[list[int]] = field(default_factory=list)


def _audio(item: dict):
    return sf.read(item["audio"], dtype="float32")[0]


@torch.no_grad()
def transcribe(model, processor, items: list[dict], lang: str, dev: str, beams: int = 1, bs: int = 8) -> list[str]:
    model.eval()
    outs: list[str] = []
    for i in range(0, len(items), bs):
        feats = processor.feature_extractor([_audio(it) for it in items[i:i + bs]], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            ids = model.generate(input_features=feats, language=lang, task="transcribe", max_new_tokens=200,
                                 num_beams=beams, use_cache=True)
        outs += processor.batch_decode(ids, skip_special_tokens=True)
    return outs


def _evaluate(model, ctx: Ctx, cfg: TrainConfig) -> dict:
    items = ctx.test
    hyps = transcribe(model, ctx.processor, items, cfg.language, ctx.dev, cfg.num_beams)
    m = stt.compute_metrics([it["text"] for it in items], hyps, cfg.domain_terms)
    m["examples"] = [{"ref": it["text"], "hyp": h} for it, h in list(zip(items, hyps))[:5]]
    return m


def _load_base(hf_id: str, method: str, cfg: TrainConfig, dev: str):
    mech = MECHANISM[method]
    kw = {}
    if mech == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    elif not (mech == "full" or needs_fp32_base(method, cfg)) and dev == "cuda":
        kw["dtype"] = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(model_source(hf_id), **kw)
    if "quantization_config" not in kw:
        model.to(dev)
    model.config.use_cache = False
    try:
        model.generation_config.forced_decoder_ids = None
    except Exception:
        pass
    if cfg.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.02
        model.config.mask_feature_length = 10
    return model


def _wrap(model, method: str, cfg: TrainConfig, total_steps: int):
    gc_kw = {"use_reentrant": False}
    if MECHANISM[method] == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing,
                                                gradient_checkpointing_kwargs=gc_kw)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kw)
    return apply_method(model, method, cfg, FAMILY, total_steps)


def _term_patterns(tok, terms: list[str]) -> list[list[int]]:
    pats: set[tuple[int, ...]] = set()
    for t in terms:
        for v in {t, t.lower(), t.capitalize(), t.upper()}:
            for s in (v, " " + v):
                ids = tok.encode(s, add_special_tokens=False)
                if ids:
                    pats.add(tuple(ids))
    return [list(p) for p in pats]


def _collate_fn(ctx: Ctx, cfg: TrainConfig):
    processor = ctx.processor
    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    weighted = cfg.domain_loss_weight > 1 and bool(ctx.patterns)

    def collate(items):
        feats = processor.feature_extractor([_audio(it) for it in items], sampling_rate=16000, return_tensors="pt")
        lab = processor.tokenizer([it["text"] for it in items], padding=True, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == start).all():
            labels = labels[:, 1:]
        batch = {"input_features": feats.input_features, "labels": labels}
        if weighted:
            batch["weights"] = domain_weights(labels.tolist(), ctx.patterns, cfg.domain_loss_weight, labels.shape[1])
        return batch

    return collate


def _forward_fns(method: str, cfg: TrainConfig):
    plain = MECHANISM[method] == "adalora" or cfg.label_smoothing == 0  # AdaLoRA needs the model's own loss (orthogonality reg.)

    def val_forward(m, b):  # always plain CE so val loss is comparable across runs and settings
        return m(input_features=b["input_features"], labels=b["labels"]).loss

    def forward(m, b):
        if plain and "weights" not in b:
            return val_forward(m, b)
        out = m(input_features=b["input_features"], labels=b["labels"])
        return weighted_ce(out.logits, b["labels"], b.get("weights"), cfg.label_smoothing)

    return forward, val_forward


def _oversample(items: list[dict], terms: list[str], factor: int) -> list[dict]:
    hit = [it for it in items if any(stt.count_term(stt.normalize(it["text"]), t) for t in terms)]
    return items + hit * (factor - 1)


def _val_wer_fn(ctx: Ctx, cfg: TrainConfig):
    items = ctx.val[:cfg.max_val_wer_samples]

    def val_wer(m) -> dict:
        hyps = transcribe(m, ctx.processor, items, cfg.language, ctx.dev, cfg.num_beams)
        return {"val_wer": stt.compute_metrics([i["text"] for i in items], hyps)["wer"]}

    return val_wer


def train_model(ctx: Ctx, method: str, cfg: TrainConfig, *, save_to: Path | None = None, trial: bool = False) -> tuple[object, dict]:
    """Load, wrap and train. Returns (model with the best weights loaded, training info).
    Checkpoints (best by the chosen metric) are written to `save_to` when given."""
    _, total = plan_steps(len(ctx.train), cfg)
    model, post_step = _wrap(_load_base(ctx.job["hf_id"], method, cfg, ctx.dev), method, cfg, total)
    groups = param_groups(model, cfg)
    collate = _collate_fn(ctx, cfg)
    forward, val_forward = _forward_fns(method, cfg)
    if cfg.auto_lr and not trial:
        ctx.progress.emit("status", message="Running LR range test")
        res = find_lr(model, ctx.train, collate, forward, cfg, groups)
        lo, hi = cfg.learning_rate / 10, cfg.learning_rate * 5
        chosen = min(max(res["suggested"], lo), hi) if res["suggested"] else cfg.learning_rate
        ctx.progress.emit("lr_find", default=cfg.learning_rate, chosen=chosen, raw_suggestion=res["suggested"],
                          clamped=bool(res["suggested"]) and chosen != res["suggested"], curve=res["curve"], steps=res["steps"])
        cfg.learning_rate = chosen
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    extra = _val_wer_fn(ctx, cfg) if cfg.select_by == "val_wer" else None

    def save():
        if save_to:
            model.save_pretrained(save_to)
            ctx.processor.save_pretrained(save_to)

    info = fit(model, ctx.train, ctx.val, collate, forward, cfg, ctx.run_dir, ctx.progress, save, groups=groups, ema=ema,
               post_step=post_step, extra_eval=extra, val_forward=val_forward)
    return model, info


def run(job: dict, cfg: TrainConfig, run_dir: Path, progress: Progress) -> dict:
    dev = device()
    method = job["method"]
    data = Path(job["dataset_dir"])
    train, val, test = (read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test"))
    test = test[:cfg.max_eval_samples]
    if method == "domain_adaptation":
        if not cfg.domain_terms:
            raise ValueError("Domain adaptation needs at least one domain term.")
        train = _oversample(train, cfg.domain_terms, cfg.oversample_domain)
    processor = WhisperProcessor.from_pretrained(model_source(job["hf_id"]))
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task="transcribe")
    ctx = Ctx(job, cfg, run_dir, progress, dev, train, val, test, processor, _term_patterns(processor.tokenizer, cfg.domain_terms))

    progress.emit("status", message="Loading base model and evaluating it on the test split")
    base_model = _load_base(job["hf_id"], method, cfg, dev)
    base = _evaluate(base_model, ctx, cfg)
    del base_model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    progress.emit("base_eval", **{k: v for k, v in base.items() if k != "examples"})

    tune_report = None
    if job.get("autotune"):
        cfg, tune_report = autotune(ctx, method, cfg, job["autotune"], lambda c, steps: _trial(ctx, method, c, steps))

    progress.emit("status", message=f"Training ({method})")
    model, info = train_model(ctx, method, cfg, save_to=run_dir / "checkpoint")
    model.config.use_cache = True
    progress.emit("status", message="Evaluating fine-tuned model")
    tuned = _evaluate(model, ctx, cfg)
    progress.emit("tuned_eval", **{k: v for k, v in tuned.items() if k != "examples"})
    total = sum(p.numel() for p in model.parameters())
    del model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    out = {"kind": "stt", "base": base, "tuned": tuned, "train": info, "total_params": total,
           "quantized_base": MECHANISM[method] == "qlora", "mechanism": MECHANISM[method],
           "samples": {"train": len(ctx.train), "val": len(val), "test": len(test)}, "amp": str(amp_dtype(cfg))}
    if tune_report:
        out["autotune"] = tune_report
    return out


def _trial(ctx: Ctx, method: str, cfg: TrainConfig, steps: int) -> dict:
    """One short Auto-tune trial: train for `steps` optimizer steps and report validation metrics."""
    c = cfg.model_copy(update={"max_steps": steps, "early_stopping": False, "epochs": 1000, "auto_lr": False})
    model, info = train_model(ctx, method, c, trial=True)
    metrics = dict(info["best_metrics"] or {})
    if "val_wer" not in metrics and ctx.val:
        items = ctx.val[:c.max_val_wer_samples]
        hyps = transcribe(model, ctx.processor, items, c.language, ctx.dev, c.num_beams)
        metrics["val_wer"] = stt.compute_metrics([i["text"] for i in items], hyps)["wer"]
    peak = info["peak_gpu_gb"]
    del model
    gc.collect()
    if ctx.dev == "cuda":
        torch.cuda.empty_cache()
    return {**metrics, "seconds": info["train_seconds"], "peak_gpu_gb": peak}
