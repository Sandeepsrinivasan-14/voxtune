"""Reject impossible technique combinations up front (before any GPU time is spent) and flag ignored settings."""
from ..config import TrainConfig, ui_field_specs
from .families import FAMILIES
from .info import MECHANISM

DEFAULTS = TrainConfig()


def validate(method: str, cfg: TrainConfig, *, family: str) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block the run; warnings are shown to the user."""
    errors: list[str] = []
    warnings: list[str] = []
    mech = MECHANISM[method]

    if mech == "qlora" and cfg.lora_init != "default":
        errors.append("PiSSA/OLoRA initialisation needs full-precision base weights and cannot be combined with QLoRA's 4-bit base.")
    if mech == "adalora":
        clash = [n for n, bad in (("DoRA", cfg.use_dora), ("rsLoRA", cfg.use_rslora), ("LoRA+", cfg.loraplus_ratio > 0),
                                  ("adapter initialisation", cfg.lora_init != "default"), ("label smoothing", cfg.label_smoothing > 0),
                                  ("domain-term loss weight", cfg.domain_loss_weight > 1)) if bad]
        if clash:
            errors.append(f"AdaLoRA (adaptive rank) cannot be combined with: {', '.join(clash)}.")
    if cfg.use_dora and cfg.lora_init != "default":
        errors.append("DoRA cannot be combined with PiSSA/OLoRA initialisation.")
    lm = FAMILIES[family].causal_lm if family in FAMILIES else False
    if lm:
        if method not in ("full", "lora", "qlora"):
            errors.append(f"{method} is not available for text language models (use full, LoRA or QLoRA).")
        if cfg.lora_scope != "all":
            errors.append("Adapting only the encoder or decoder does not apply to a decoder-only language model.")
        if cfg.freeze_encoder:
            errors.append("Decoder-only language models have no encoder to freeze.")
    if family == "gemma" and cfg.precision == "fp16":
        warnings.append("Gemma models overflow in fp16; use bf16 (or fp32).")
    if cfg.select_by == "val_wer" and family != "whisper":
        errors.append("Selecting checkpoints by validation WER is only available for speech-to-text models.")
    if cfg.spec_augment and family != "whisper":
        errors.append("SpecAugment is only available for speech-to-text models.")
    if method == "adapter" and family != "whisper":
        errors.append("The IA³ adapter is only implemented for Whisper.")
    if method == "domain_adaptation" and cfg.domain_loss_weight > 1 and not cfg.domain_terms:
        errors.append("Domain-term loss weighting needs at least one domain term.")
    if cfg.precision == "fp16" and cfg.ema_decay > 0:
        warnings.append("EMA with fp16 loss scaling works, but bf16 is more stable on GPUs that support it.")

    # anything the user changed that this method never reads
    for spec in ui_field_specs():
        k = spec["key"]
        if method not in spec["applies"] and getattr(cfg, k) != getattr(DEFAULTS, k):
            warnings.append(f"'{spec['label']}' does not apply to {method} and is ignored.")
    if family == "speecht5" and cfg.label_smoothing > 0:
        warnings.append("Label smoothing is ignored for TTS (the loss is L1 + stop-token BCE).")
    return errors, warnings
