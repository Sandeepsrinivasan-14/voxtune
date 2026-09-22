from typing import Literal

from pydantic import BaseModel, Field

LORA_FAMILY = ["lora", "qlora", "domain_adaptation", "adalora"]
LORA_VARIANTS = ["lora", "qlora", "domain_adaptation"]  # methods that accept DoRA / rsLoRA / init / LoRA+ (AdaLoRA does not)
EVERY = ["full", "lora", "qlora", "adapter", "adalora", "domain_adaptation", "speaker_adaptation"]
ALL_TASKS = ["stt", "tts", "llm"]


def opt(**extra):
    """`applies`: methods that read the field. `tasks`: task types whose models read it (default: all)."""
    return {"applies": extra.pop("applies", EVERY), "tasks": extra.pop("tasks", ALL_TASKS), **extra}


class TrainConfig(BaseModel):
    # ---- core optimisation ----
    epochs: int = Field(5, ge=1, le=200, title="Epochs", json_schema_extra=opt())
    learning_rate: float = Field(1e-4, gt=0, title="Learning rate", json_schema_extra=opt())
    batch_size: int = Field(4, ge=1, title="Batch size", json_schema_extra=opt())
    grad_accum: int = Field(4, ge=1, title="Gradient accumulation", json_schema_extra=opt())
    warmup_steps: int = Field(0, ge=0, title="Warmup steps", json_schema_extra=opt())
    lr_scheduler: Literal["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"] = Field(
        "linear", title="LR schedule", json_schema_extra=opt())
    precision: Literal["fp32", "fp16", "bf16"] = Field("bf16", title="Precision", json_schema_extra=opt())
    optimizer: Literal["adamw", "adamw_8bit"] = Field("adamw", title="Optimizer", json_schema_extra=opt())
    weight_decay: float = 0.01
    auto_lr: bool = Field(False, title="Auto learning rate (LR range test)", json_schema_extra=opt(),
                          description="Sweeps the LR on a few real batches, then picks it from the loss curve (restores weights afterwards)")
    gradient_checkpointing: bool = Field(False, title="Gradient checkpointing", json_schema_extra=opt())
    # ---- validation, selection, stopping ----
    eval_steps: int = Field(0, ge=0, title="Eval every N steps", description="0 = evaluate once per epoch", json_schema_extra=opt())
    select_by: Literal["val_loss", "val_wer"] = Field(
        "val_loss", title="Keep best checkpoint by", json_schema_extra=opt(applies=["full", "lora", "qlora", "adapter", "adalora", "domain_adaptation"], tasks=["stt"]),
        description="val_wer decodes the validation set at every eval (slower, but selects on the metric you care about)")
    early_stopping: bool = Field(True, title="Early stopping", json_schema_extra=opt())
    patience: int = Field(3, ge=1, title="Early-stop patience", json_schema_extra=opt())
    max_val_wer_samples: int = Field(24, ge=1)
    # ---- LoRA family ----
    lora_r: int = Field(16, ge=1, title="LoRA rank", json_schema_extra=opt(applies=LORA_FAMILY))
    lora_alpha: int = Field(32, ge=1, title="LoRA alpha", json_schema_extra=opt(applies=LORA_FAMILY))
    lora_dropout: float = 0.05
    lora_scope: Literal["all", "encoder", "decoder"] = Field(
        "all", title="Adapt which part", json_schema_extra=opt(applies=LORA_FAMILY, tasks=["stt", "tts"]),
        description="decoder-only is cheaper and usually enough for vocabulary; encoder helps acoustic shifts (accent, noise)")
    lora_targets: Literal["attention", "all_linear"] = Field(
        "attention", title="Adapted layers", json_schema_extra=opt(applies=LORA_FAMILY),
        description="all_linear also adapts the feed-forward blocks: more capacity, more memory")
    use_dora: bool = Field(False, title="DoRA (weight-decomposed LoRA)", json_schema_extra=opt(applies=LORA_VARIANTS),
                           description="Learns magnitude and direction separately; often closer to full FT at low rank")
    use_rslora: bool = Field(False, title="rsLoRA (rank-stabilised scaling)", json_schema_extra=opt(applies=LORA_VARIANTS),
                             description="alpha/sqrt(r) scaling so higher ranks keep learning instead of stalling")
    lora_init: Literal["default", "pissa", "olora"] = Field(
        "default", title="Adapter initialisation", json_schema_extra=opt(applies=["lora", "domain_adaptation"]),
        description="pissa/olora start from the principal singular directions (faster convergence); needs an fp32 base")
    loraplus_ratio: float = Field(0.0, ge=0, title="LoRA+ (LR ratio for B matrices, 0 = off)", json_schema_extra=opt(applies=LORA_VARIANTS),
                                  description="Trains B with a higher LR than A (typically 8-16x) for faster, better feature learning")
    # ---- full fine-tuning ----
    freeze_encoder: bool = Field(False, title="Freeze the encoder", json_schema_extra=opt(applies=["full"], tasks=["stt", "tts"]),
                                 description="Train only the decoder: far less memory and less forgetting")
    llrd: float = Field(1.0, ge=0.3, le=1.0, title="Layer-wise LR decay (1 = off)", json_schema_extra=opt(applies=["full"]),
                        description="Lower layers get smaller LRs (e.g. 0.9^depth), which protects pretrained features")
    # ---- regularisation / stability (all methods) ----
    ema_decay: float = Field(0.0, ge=0, lt=1, title="EMA of weights (0 = off)", json_schema_extra=opt(),
                             description="Evaluates and saves a smoothed copy of the trained weights (e.g. 0.999)")
    label_smoothing: float = Field(0.0, ge=0, le=0.3, title="Label smoothing", json_schema_extra=opt(
        applies=["full", "lora", "qlora", "adapter", "domain_adaptation"], tasks=["stt", "llm"]))
    spec_augment: bool = Field(False, title="SpecAugment (time/feature masking)", json_schema_extra=opt(
        applies=["full", "lora", "qlora", "adapter", "adalora", "domain_adaptation"], tasks=["stt"]),
        description="Random masking inside the encoder: strong regulariser for small ASR datasets")
    # ---- domain adaptation ----
    domain_terms: list[str] = []
    oversample_domain: int = Field(3, ge=1)
    domain_loss_weight: float = Field(1.0, ge=1.0, title="Domain-term loss weight", json_schema_extra=opt(applies=["domain_adaptation"], tasks=["stt"]),
                                      description="Multiplies the loss on tokens of your domain terms (1 = off)")
    # ---- text LLMs ----
    max_seq_len: int = Field(512, ge=32, le=8192, title="Max sequence length (tokens)", json_schema_extra=opt(
        applies=["full", "lora", "qlora"], tasks=["llm"]),
        description="Prompt + response tokens per example; longer examples are cut. Memory grows with this and the vocabulary size")
    max_new_tokens: int = Field(64, ge=1, le=1024, title="Max new tokens when generating for evaluation", json_schema_extra=opt(
        applies=["full", "lora", "qlora"], tasks=["llm"]))
    # ---- evaluation ----
    num_beams: int = Field(1, ge=1, le=8, title="Beam size for decoding", json_schema_extra=opt(
        applies=["full", "lora", "qlora", "adapter", "adalora", "domain_adaptation"], tasks=["stt"]))
    max_steps: int = Field(0, ge=0)
    language: str = "en"
    max_eval_samples: int = Field(50, ge=1)
    seed: int = 42


# Fields the UI exposes in Advanced mode, in display order. Labels, help, choices and applicability come from the model above.
UI_FIELDS = ("epochs", "learning_rate", "auto_lr", "batch_size", "grad_accum", "warmup_steps", "lr_scheduler", "precision",
             "optimizer", "eval_steps", "select_by", "patience", "early_stopping", "gradient_checkpointing",
             "lora_r", "lora_alpha", "lora_scope", "lora_targets", "use_dora", "use_rslora", "lora_init", "loraplus_ratio",
             "freeze_encoder", "llrd", "ema_decay", "label_smoothing", "spec_augment", "domain_loss_weight", "num_beams",
             "max_seq_len", "max_new_tokens")


def ui_field_specs() -> list[dict]:
    props = TrainConfig.model_json_schema()["properties"]
    out = []
    for name in UI_FIELDS:
        p = props[name]
        kind = "select" if "enum" in p else {"integer": "number", "number": "number", "boolean": "boolean"}[p["type"]]
        out.append({"key": name, "label": p.get("title", name), "kind": kind, "options": p.get("enum"),
                    "help": p.get("description"), "default": p.get("default"),
                    "min": p.get("minimum", p.get("exclusiveMinimum")), "max": p.get("maximum"),
                    "applies": p.get("applies", EVERY), "tasks": p.get("tasks", ALL_TASKS)})
    return out
