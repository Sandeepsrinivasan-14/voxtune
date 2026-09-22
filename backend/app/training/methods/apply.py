"""Turn a loaded base model into a trainable one for a given method: PEFT wrapping or freezing."""
from peft import AdaLoraConfig, IA3Config, LoraConfig, get_peft_model

from ..config import TrainConfig
from .families import FAMILIES, target_regex
from .info import MECHANISM


def needs_fp32_base(method: str, cfg: TrainConfig) -> bool:
    """PiSSA/OLoRA decompose the base weights (SVD/QR), which must not happen in bf16."""
    return MECHANISM[method] == "lora" and cfg.lora_init != "default"


def apply_method(model, method: str, cfg: TrainConfig, family: str, total_steps: int = 0):
    """Returns (model, post_step) where post_step(step) is called after every optimizer step (or None)."""
    mech = MECHANISM[method]
    fam = FAMILIES[family]
    post_step = None
    if mech in ("lora", "qlora"):
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target_regex(family, cfg.lora_scope, cfg.lora_targets), bias="none",
            use_dora=cfg.use_dora, use_rslora=cfg.use_rslora,
            init_lora_weights=True if cfg.lora_init == "default" else cfg.lora_init))
    elif mech == "adalora":
        total = max(total_steps, 20)
        model = get_peft_model(model, AdaLoraConfig(
            init_r=cfg.lora_r, target_r=max(2, cfg.lora_r // 2), lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target_regex(family, cfg.lora_scope, cfg.lora_targets), total_step=total,
            tinit=int(0.1 * total), tfinal=int(0.2 * total), deltaT=max(1, total // 40), orth_reg_weight=0.5))
        base = model.base_model
        post_step = base.update_and_allocate
    elif mech == "ia3":
        model = get_peft_model(model, IA3Config(target_modules=["k_proj", "v_proj", "fc2"], feedforward_modules=["fc2"]))
    elif mech == "full":
        if cfg.freeze_encoder:
            for n, p in model.named_parameters():
                if n.startswith(fam.enc_prefix):
                    p.requires_grad = False
    elif mech == "freeze_text_encoder":
        for n, p in model.named_parameters():
            p.requires_grad = not n.startswith(fam.enc_prefix)
    else:  # pragma: no cover - guarded by MECHANISM
        raise ValueError(f"Unknown mechanism {mech}")
    return model, post_step
