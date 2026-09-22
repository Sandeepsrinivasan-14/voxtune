"""SpeechT5 TTS fine-tuning on the shared methods core: full FT, speaker adaptation (text encoder frozen) and LoRA
(decoder attention/feed-forward, speech prenet and postnet), with LR range test, layer-wise LR decay, EMA and Auto-tune."""
import gc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from ...core.config import model_source
from ...evaluation import tts
from ..common import Progress, device, fit, plan_steps, read_jsonl
from ..config import TrainConfig
from ..methods.apply import apply_method
from ..methods.autotune import autotune
from ..methods.ema import EMA
from ..methods.lrfind import find_lr
from ..methods.params import param_groups

FAMILY = "speecht5"


@dataclass
class Ctx:
    job: dict
    cfg: TrainConfig
    run_dir: Path
    progress: Progress
    dev: str
    processor: SpeechT5Processor
    train: list[dict]
    val: list[dict]
    test: list[dict]
    means: dict
    global_mean: torch.Tensor

    def spk_of(self, it):
        return self.means.get(it["speaker"], self.global_mean)


def _wav(p: str) -> np.ndarray:
    return sf.read(p, dtype="float32")[0]


def _speaker_means(splits: dict[str, list[dict]], dev: str) -> tuple[dict, torch.Tensor]:
    embs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for it in splits["train"]:
        embs[it["speaker"]].append(tts.embed(_wav(it["audio"]), dev))
    means = {k: torch.stack(v).mean(0) for k, v in embs.items()}
    return means, torch.stack(list(means.values())).mean(0)


def _collate_fn(processor, reduction: int):
    def collate(items):
        ids = [{"input_ids": it["input_ids"]} for it in items]
        labels = [{"input_values": it["labels"]} for it in items]
        b = processor.pad(input_ids=ids, labels=labels, return_tensors="pt")
        b["labels"] = b["labels"].masked_fill(b.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
        del b["decoder_attention_mask"]
        if reduction > 1:
            n = b["labels"].shape[1] - b["labels"].shape[1] % reduction
            b["labels"] = b["labels"][:, :n]
        b["speaker_embeddings"] = torch.stack([it["spk"] for it in items])
        return dict(b)

    return collate


def _base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


@torch.no_grad()
def _synthesize(model, vocoder, processor, items, spk_of, dev, out_dir: Path) -> list[np.ndarray]:
    model.eval()
    core = _base(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = []
    for i, it in enumerate(items):
        ids = processor(text=it["text"], return_tensors="pt")["input_ids"].to(dev)
        w = core.generate_speech(ids, spk_of(it).unsqueeze(0).to(dev), vocoder=vocoder).cpu().numpy()
        sf.write(out_dir / f"{i:03d}.wav", w, 16000)
        wavs.append(w)
    return wavs


def _load(ctx: Ctx, cfg: TrainConfig):
    model = SpeechT5ForTextToSpeech.from_pretrained(model_source(ctx.job["hf_id"])).to(ctx.dev)
    model.config.use_cache = False
    return model


def _forward(m, b):
    return m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"],
             speaker_embeddings=b["speaker_embeddings"]).loss


def train_model(ctx: Ctx, method: str, cfg: TrainConfig, prepared: tuple, *, save_to: Path | None = None, trial: bool = False):
    train, val = prepared
    model = _load(ctx, cfg)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _, total = plan_steps(len(train), cfg)
    model, post_step = apply_method(model, method, cfg, FAMILY, total)
    groups = param_groups(model, cfg)
    collate = _collate_fn(ctx.processor, _base(model).config.reduction_factor)
    if cfg.auto_lr and not trial:
        ctx.progress.emit("status", message="Running LR range test")
        res = find_lr(model, train, collate, _forward, cfg, groups)
        lo, hi = cfg.learning_rate / 10, cfg.learning_rate * 5
        chosen = min(max(res["suggested"], lo), hi) if res["suggested"] else cfg.learning_rate
        ctx.progress.emit("lr_find", default=cfg.learning_rate, chosen=chosen, raw_suggestion=res["suggested"],
                          clamped=bool(res["suggested"]) and chosen != res["suggested"], curve=res["curve"], steps=res["steps"])
        cfg.learning_rate = chosen
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    def save():
        if save_to:
            model.save_pretrained(save_to)
            ctx.processor.save_pretrained(save_to)
            torch.save({"means": ctx.means, "global": ctx.global_mean}, Path(save_to) / "speaker_embeddings.pt")

    info = fit(model, train, val, collate, _forward, cfg, ctx.run_dir, ctx.progress, save, groups=groups, ema=ema, post_step=post_step)
    return model, info


def _trial(ctx: Ctx, method: str, cfg: TrainConfig, prepared: tuple, steps: int) -> dict:
    c = cfg.model_copy(update={"max_steps": steps, "early_stopping": False, "epochs": 1000, "auto_lr": False})
    model, info = train_model(ctx, method, c, prepared, trial=True)
    out = {**(info["best_metrics"] or {}), "seconds": info["train_seconds"], "peak_gpu_gb": info["peak_gpu_gb"]}
    del model
    gc.collect()
    if ctx.dev == "cuda":
        torch.cuda.empty_cache()
    return out


def run(job: dict, cfg: TrainConfig, run_dir: Path, progress: Progress) -> dict:
    dev = device()
    method = job["method"]
    data = Path(job["dataset_dir"])
    splits = {s: read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test")}
    splits["test"] = splits["test"][:cfg.max_eval_samples]
    progress.emit("status", message="Computing speaker embeddings")
    means, global_mean = _speaker_means(splits, dev)
    processor = SpeechT5Processor.from_pretrained(model_source(job["hf_id"]))
    ctx = Ctx(job, cfg, run_dir, progress, dev, processor, splits["train"], splits["val"], splits["test"], means, global_mean)

    def prep(items):
        out = []
        for it in items:
            e = processor(text=it["text"], audio_target=_wav(it["audio"]), sampling_rate=16000, return_attention_mask=False)
            out.append({"input_ids": e["input_ids"], "labels": e["labels"][0] if np.ndim(e["labels"]) == 3 else e["labels"],
                        "spk": tts.embed(_wav(it["audio"]), dev) if it["speaker"] not in means else means[it["speaker"]]})
        return out

    prepared = (prep(splits["train"]), prep(splits["val"]))
    vocoder = SpeechT5HifiGan.from_pretrained(model_source("microsoft/speecht5_hifigan")).to(dev)

    progress.emit("status", message="Synthesizing test sentences with the base model")
    base_model = _load(ctx, cfg)
    base_model.config.use_cache = True
    base_wavs = _synthesize(base_model, vocoder, processor, ctx.test, ctx.spk_of, dev, run_dir / "samples" / "base")
    del base_model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    tune_report = None
    if job.get("autotune"):
        cfg, tune_report = autotune(ctx, method, cfg, job["autotune"], lambda c, steps: _trial(ctx, method, c, prepared, steps))

    progress.emit("status", message=f"Training ({method})")
    model, info = train_model(ctx, method, cfg, prepared, save_to=run_dir / "checkpoint")
    _base(model).config.use_cache = True
    progress.emit("status", message="Synthesizing test sentences with the fine-tuned model")
    tuned_wavs = _synthesize(model, vocoder, processor, ctx.test, ctx.spk_of, dev, run_dir / "samples" / "tuned")
    total = sum(p.numel() for p in model.parameters())
    del model, vocoder
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    progress.emit("status", message="Scoring audio (speaker similarity, ASR round-trip, MFCC distance)")
    refs = [_wav(it["audio"]) for it in ctx.test]
    texts = [it["text"] for it in ctx.test]
    base = tts.evaluate(refs, base_wavs, texts, dev, cfg.language)
    tuned = tts.evaluate(refs, tuned_wavs, texts, dev, cfg.language)
    progress.emit("base_eval", **base)
    progress.emit("tuned_eval", **tuned)
    samples = [{"text": t, "reference": it["audio"], "base": f"samples/base/{i:03d}.wav", "tuned": f"samples/tuned/{i:03d}.wav"}
               for i, (t, it) in enumerate(zip(texts, ctx.test))][:5]
    out = {"kind": "tts", "base": base, "tuned": tuned, "train": info, "total_params": total, "samples_audio": samples,
           "samples": {"train": len(prepared[0]), "val": len(prepared[1]), "test": len(texts)}}
    if tune_report:
        out["autotune"] = tune_report
    return out
