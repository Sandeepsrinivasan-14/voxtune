"""Where each model family keeps the layers we adapt. LoRA target modules are built as regexes over module names."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    attn: tuple[str, ...]
    ffn: tuple[str, ...]
    enc: str  # regex prefix that matches modules inside the encoder stack
    dec: str  # ... and the decoder stack
    enc_prefix: str  # parameter-name prefix of the whole encoder (for freezing)
    extra_decoder: tuple[str, ...] = ()  # additional decoder-side linears worth adapting (regex, full names)
    all_prefix: str = ".*"  # what "adapt everything" may match (text LLMs exclude any vision/audio tower)
    causal_lm: bool = False  # decoder-only text model: no encoder, so encoder/decoder scoping and encoder freezing do not apply


# Llama-style decoder-only stacks share these module names: model.layers.N.self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj.
# Gemma 3's larger checkpoints are multimodal and wrap the text stack in `language_model`; the lookahead keeps LoRA off the
# vision tower and multimodal projector so only the language model is adapted.
_TEXT_ONLY = r"(?!.*(?:vision|audio|multi_modal|mm_projector))"


def _causal_lm() -> "Family":
    return Family(attn=("q_proj", "k_proj", "v_proj", "o_proj"), ffn=("gate_proj", "up_proj", "down_proj"),
                  enc=r"(?!)", dec=_TEXT_ONLY + r".*layers\.\d+", enc_prefix="<no-encoder>", all_prefix=_TEXT_ONLY + ".*", causal_lm=True)


FAMILIES = {
    "gemma": _causal_lm(),
    "smollm2": _causal_lm(),
    "qwen2": _causal_lm(),
    "whisper": Family(
        attn=("q_proj", "k_proj", "v_proj", "out_proj"), ffn=("fc1", "fc2"),
        enc=r".*encoder\.layers\.\d+", dec=r".*decoder\.layers\.\d+", enc_prefix="model.encoder."),
    "speecht5": Family(
        attn=("q_proj", "k_proj", "v_proj", "out_proj"), ffn=("intermediate_dense", "output_dense"),
        enc=r".*wrapped_encoder\.layers\.\d+", dec=r".*wrapped_decoder\.layers\.\d+", enc_prefix="speecht5.encoder.",
        extra_decoder=(r"speecht5\.decoder\.prenet\.(layers\.\d+|final_layer|speaker_embeds_layer)",
                       r"speech_decoder_postnet\.(feat_out|prob_out)")),
}


def target_regex(family: str, scope: str, targets: str) -> str:
    fam = FAMILIES[family]
    mods = fam.attn + (fam.ffn if targets == "all_linear" else ())
    tail = r"\.(" + "|".join(mods) + ")"
    # inside a layer, modules sit either directly under it (fc1) or one level down (self_attn.q_proj)
    prefix = {"all": fam.all_prefix, "encoder": fam.enc + r"(?:\..+)?", "decoder": fam.dec + r"(?:\..+)?"}[scope]
    parts = [prefix + tail]
    if scope in ("all", "decoder"):
        parts += list(fam.extra_decoder)
    return "|".join(f"(?:{p})" for p in parts)
