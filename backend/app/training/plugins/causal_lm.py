"""Text LLM fine-tuning (Gemma-style decoder-only models) on the shared methods core.

One implementation serves every decoder-only family registered with the causal-LM layout (Gemma, SmolLM2, Qwen2): models are
loaded with AutoModelForCausalLM and prompts are rendered with the tokenizer's own chat template when it has one (Gemma-it,
Qwen, SmolLM2-Instruct all do), else a plain "### Response:" layout. The loss covers response tokens only.

Methods: full FT, LoRA (+DoRA / rsLoRA / PiSSA / OLoRA / LoRA+), QLoRA (4-bit). Techniques: LR range test, layer-wise LR decay,
EMA, label smoothing, Auto-tune. Evaluation on the held-out test split: perplexity and next-token accuracy over the reference
responses (teacher-forced), plus exact match and ROUGE-L on greedy generations.
"""
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ...core.config import model_source
from ...evaluation import llm as ev
from ..common import Progress, amp_dtype, device, fit, plan_steps, read_jsonl, to_device
from ..config import TrainConfig
from ..methods.apply import apply_method, needs_fp32_base
from ..methods.autotune import autotune
from ..methods.ema import EMA
from ..methods.info import MECHANISM
from ..methods.losses import weighted_ce
from ..methods.lrfind import find_lr
from ..methods.params import param_groups


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
    tok: object
    fmt: str  # "chat" (tokenizer chat template) or "plain"


def _hub_error(e: Exception, hf_id: str) -> Exception:
    name = type(e).__name__
    if "Gated" in name or "401" in str(e) or "403" in str(e):
        return RuntimeError(f"{hf_id} is gated. Accept its license at https://huggingface.co/{hf_id} while logged in as the account whose "
                            f"Hugging Face token is configured on this machine, then run again. ({name})")
    return e


def _load_tokenizer(hf_id: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_source(hf_id))
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e, hf_id) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def detect_format(tok) -> str:
    """'chat' only if the chat template renders a prompt that is an exact prefix of the prompt+response rendering."""
    if getattr(tok, "chat_template", None):
        try:
            user = [{"role": "user", "content": "PROMPT"}]
            head = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
            full = tok.apply_chat_template(user + [{"role": "assistant", "content": "RESPONSE"}], tokenize=False)
            if full.startswith(head) and "RESPONSE" in full[len(head):]:
                return "chat"
        except Exception:  # noqa: BLE001 - odd templates fall back to the plain layout
            pass
    return "plain"


def render(tok, fmt: str, prompt: str, response: str | None = None) -> tuple[str, str]:
    """(prompt text the model is conditioned on, response text incl. end-of-turn marker). Response is '' when None."""
    if fmt == "chat":
        user = [{"role": "user", "content": prompt}]
        head = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        if response is None:
            return head, ""
        full = tok.apply_chat_template(user + [{"role": "assistant", "content": response}], tokenize=False)
        return head, full[len(head):]
    head = f"{prompt}\n\n### Response:\n"
    return head, "" if response is None else response + (tok.eos_token or "")


def encode(tok, fmt: str, item: dict, max_len: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels) with -100 on the prompt part; the response is cut first when the example is too long."""
    head, tail = render(tok, fmt, item["prompt"], item["response"])
    p = tok(head, add_special_tokens=(fmt == "plain"))["input_ids"]
    r = tok(tail, add_special_tokens=False)["input_ids"]
    if len(p) > max_len - 8:  # keep room for at least a few response tokens
        p = p[: max_len - 8]
    r = r[: max_len - len(p)]
    return p + r, [-100] * len(p) + r


def _load_base(job: dict, method: str, cfg: TrainConfig, dev: str):
    mech = MECHANISM[method]
    kw: dict = {}
    if job["family"].lower() == "gemma":
        kw["attn_implementation"] = "eager"  # HF recommends eager attention for training Gemma models
    if mech == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    elif mech == "full" or needs_fp32_base(method, cfg):
        kw["dtype"] = torch.float32  # master weights stay fp32; autocast does the low-precision math
    elif dev == "cuda":
        kw["dtype"] = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_source(job["hf_id"]), **kw)
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e, job["hf_id"]) from e
    if "quantization_config" not in kw:
        model.to(dev)
    model.config.use_cache = False
    return model


def _wrap(model, method: str, cfg: TrainConfig, total_steps: int, family: str):
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if MECHANISM[method] != "full":
            model.enable_input_require_grads()
    return apply_method(model, method, cfg, family, total_steps)


def _collate_fn(ctx: Ctx, cfg: TrainConfig):
    pad = ctx.tok.pad_token_id

    def collate(items):
        enc = [encode(ctx.tok, ctx.fmt, it, cfg.max_seq_len) for it in items]
        width = max(len(i) for i, _ in enc)
        ids = torch.full((len(enc), width), pad, dtype=torch.long)
        lab = torch.full((len(enc), width), -100, dtype=torch.long)
        att = torch.zeros((len(enc), width), dtype=torch.long)
        for k, (i, lb) in enumerate(enc):
            ids[k, :len(i)] = torch.tensor(i)
            lab[k, :len(lb)] = torch.tensor(lb)
            att[k, :len(i)] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    return collate


def _forward_fns(cfg: TrainConfig):
    def val_forward(m, b):  # plain CE so validation loss is comparable across runs and settings
        return m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss

    def forward(m, b):
        if cfg.label_smoothing == 0:
            return val_forward(m, b)
        logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        return weighted_ce(logits[:, :-1], b["labels"][:, 1:], None, cfg.label_smoothing)

    return forward, val_forward


@torch.no_grad()
def _teacher_forced(model, ctx: Ctx, cfg: TrainConfig, items: list[dict]) -> dict:
    model.eval()
    collate = _collate_fn(ctx, cfg)
    ce_sum, n_tok, correct = 0.0, 0, 0
    for i in range(0, len(items), max(1, cfg.batch_size)):
        b = to_device(collate(items[i:i + max(1, cfg.batch_size)]), ctx.dev)
        with torch.autocast(ctx.dev, dtype=torch.bfloat16, enabled=ctx.dev == "cuda"):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        logits, tgt = logits[:, :-1].float(), b["labels"][:, 1:]
        mask = tgt != -100
        ce_sum += F.cross_entropy(logits[mask], tgt[mask], reduction="sum").item()
        correct += (logits.argmax(-1)[mask] == tgt[mask]).sum().item()
        n_tok += int(mask.sum().item())
    return ev.teacher_forced_metrics(ce_sum, n_tok, correct)


@torch.no_grad()
def generate(model, ctx: Ctx, cfg: TrainConfig, items: list[dict], bs: int = 8) -> list[str]:
    model.eval()
    tok = ctx.tok
    tok.padding_side, tok.truncation_side = "left", "left"
    outs: list[str] = []
    try:
        for i in range(0, len(items), bs):
            prompts = [render(tok, ctx.fmt, it["prompt"])[0] for it in items[i:i + bs]]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len,
                      add_special_tokens=(ctx.fmt == "plain")).to(ctx.dev)
            with torch.autocast(ctx.dev, dtype=torch.bfloat16, enabled=ctx.dev == "cuda"):
                out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id, use_cache=True)
            width = enc["input_ids"].shape[1]
            outs += [tok.decode(o[width:], skip_special_tokens=True).strip() for o in out]
    finally:
        tok.padding_side, tok.truncation_side = "right", "right"
    return outs


def _evaluate(model, ctx: Ctx, cfg: TrainConfig) -> dict:
    was_cache = model.config.use_cache
    model.config.use_cache = True
    items = ctx.test
    m = _teacher_forced(model, ctx, cfg, items)
    hyps = generate(model, ctx, cfg, items)
    m.update(ev.generation_metrics([it["response"] for it in items], hyps))
    m["n"] = len(items)
    m["examples"] = [{"prompt": it["prompt"], "ref": it["response"], "hyp": h} for it, h in list(zip(items, hyps))[:5]]
    model.config.use_cache = was_cache
    return m


def train_model(ctx: Ctx, method: str, cfg: TrainConfig, *, save_to: Path | None = None, trial: bool = False) -> tuple[object, dict]:
    """Load, wrap and train. Returns (model with the best weights loaded, training info)."""
    _, total = plan_steps(len(ctx.train), cfg)
    model, post_step = _wrap(_load_base(ctx.job, method, cfg, ctx.dev), method, cfg, total, ctx.job["family"].lower())
    groups = param_groups(model, cfg)
    collate = _collate_fn(ctx, cfg)
    forward, val_forward = _forward_fns(cfg)
    if cfg.auto_lr and not trial:
        ctx.progress.emit("status", message="Running LR range test")
        res = find_lr(model, ctx.train, collate, forward, cfg, groups)
        lo, hi = cfg.learning_rate / 10, cfg.learning_rate * 5
        chosen = min(max(res["suggested"], lo), hi) if res["suggested"] else cfg.learning_rate
        ctx.progress.emit("lr_find", default=cfg.learning_rate, chosen=chosen, raw_suggestion=res["suggested"],
                          clamped=bool(res["suggested"]) and chosen != res["suggested"], curve=res["curve"], steps=res["steps"])
        cfg.learning_rate = chosen
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    def save():
        if save_to:
            model.save_pretrained(save_to)
            ctx.tok.save_pretrained(save_to)

    info = fit(model, ctx.train, ctx.val, collate, forward, cfg, ctx.run_dir, ctx.progress, save, groups=groups, ema=ema,
               post_step=post_step, val_forward=val_forward)
    return model, info


def _free(dev: str) -> None:
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


def run(job: dict, cfg: TrainConfig, run_dir: Path, progress: Progress) -> dict:
    dev = device()
    method = job["method"]
    data = Path(job["dataset_dir"])
    train, val, test = (read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test"))
    test = test[:cfg.max_eval_samples]
    tok = _load_tokenizer(job["hf_id"])
    ctx = Ctx(job, cfg, run_dir, progress, dev, train, val, test, tok, detect_format(tok))

    progress.emit("status", message=f"Loading base model and evaluating it on the test split (prompt format: {ctx.fmt})")
    base_model = _load_base(job, method, cfg, dev)
    base = _evaluate(base_model, ctx, cfg)
    del base_model
    _free(dev)
    progress.emit("base_eval", **{k: v for k, v in base.items() if k != "examples"})

    tune_report = None
    if job.get("autotune"):
        cfg, tune_report = autotune(ctx, method, cfg, job["autotune"], lambda c, steps: _trial(ctx, method, c, steps))

    progress.emit("status", message=f"Training ({method})")
    model, info = train_model(ctx, method, cfg, save_to=run_dir / "checkpoint")
    progress.emit("status", message="Evaluating fine-tuned model")
    tuned = _evaluate(model, ctx, cfg)
    progress.emit("tuned_eval", **{k: v for k, v in tuned.items() if k != "examples"})
    total = sum(p.numel() for p in model.parameters())
    del model
    _free(dev)
    out = {"kind": "llm", "base": base, "tuned": tuned, "train": info, "total_params": total,
           "quantized_base": MECHANISM[method] == "qlora", "mechanism": MECHANISM[method], "prompt_format": ctx.fmt,
           "samples": {"train": len(ctx.train), "val": len(val), "test": len(test)}, "amp": str(amp_dtype(cfg))}
    if tune_report:
        out["autotune"] = tune_report
    return out


def _trial(ctx: Ctx, method: str, cfg: TrainConfig, steps: int) -> dict:
    """One short Auto-tune trial: train for `steps` optimizer steps and report validation loss."""
    c = cfg.model_copy(update={"max_steps": steps, "early_stopping": False, "epochs": 1000, "auto_lr": False})
    model, info = train_model(ctx, method, c, trial=True)
    metrics = dict(info["best_metrics"] or {})
    peak = info["peak_gpu_gb"]
    del model
    _free(ctx.dev)
    return {**metrics, "seconds": info["train_seconds"], "peak_gpu_gb": peak}
