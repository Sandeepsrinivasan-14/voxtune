"""Model capability registry.

The platform never claims a model supports a method unless it is declared here.
`min_vram_gb` values are conservative *estimates* (small batch, fp16/4-bit, gradient
checkpointing) used only to gate options against detected hardware; they are not measurements.
`methods` maps method -> minimum VRAM in GB; a method absent from the map is unsupported for
that model. `implemented` lists methods whose training code actually exists in this build.
"""
from dataclasses import dataclass

from ..core.gpu import GpuInfo
from ..training.methods.info import METHOD_INFO

METHODS = {k: v["label"] for k, v in METHOD_INFO.items()}  # one catalogue: training/methods/info.py


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: str  # "stt" | "tts" | "llm"
    family: str
    hf_id: str
    params_m: float
    methods: dict[str, float]  # method -> min VRAM GB (estimate)
    implemented: tuple[str, ...] = ()
    license: str = ""
    notes: str = ""
    gated: bool = False  # weights sit behind a license the user must accept on Hugging Face

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "family": self.family,
            "hf_id": self.hf_id,
            "params_m": self.params_m,
            "methods": self.methods,
            "implemented": list(self.implemented),
            "license": self.license,
            "notes": self.notes,
            "gated": self.gated,
        }


@dataclass
class Support:
    supported: bool
    reason: str = ""
    verified: bool = False  # proven by a real completed run on this machine
    observed_gb: float | None = None

    def to_dict(self) -> dict:
        return {"supported": self.supported, "reason": self.reason, "verified": self.verified, "observed_gb": self.observed_gb}


def _whisper(size: str, params: float, full: float, lora: float, qlora: float, impl: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        id=f"whisper-{size}",
        task="stt",
        family="Whisper",
        hf_id=f"openai/whisper-{size}",
        params_m=params,
        methods={"full": full, "lora": lora, "qlora": qlora, "adalora": lora, "adapter": lora * 0.9, "domain_adaptation": lora},
        implemented=impl,
        license="MIT",
    )


_WHISPER_IMPL = ("full", "lora", "qlora", "adalora", "adapter", "domain_adaptation")

MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        _whisper("tiny", 39, 2.5, 1.5, 1.5, _WHISPER_IMPL),
        _whisper("base", 74, 3.5, 2.0, 1.8, _WHISPER_IMPL),
        # small LoRA/QLoRA measured on an RTX 3050: 1.89 / 1.49 GB used (bf16, batch 2, grad checkpointing)
        _whisper("small", 244, 10.0, 2.5, 2.0, _WHISPER_IMPL),
        _whisper("medium", 769, 24.0, 12.0, 6.0, _WHISPER_IMPL),
        _whisper("large-v3", 1550, 40.0, 20.0, 10.0, _WHISPER_IMPL),
        ModelSpec(
            id="speecht5-tts",
            task="tts",
            family="SpeechT5",
            hf_id="microsoft/speecht5_tts",
            params_m=144,
            # measured on an RTX 3050 (bf16, batch 2, grad checkpointing; full FT with 8-bit AdamW): 2.67 GB / 2.51 GB used; LoRA 1.98 GB
            methods={"full": 3.0, "speaker_adaptation": 3.0, "lora": 2.3},
            implemented=("full", "speaker_adaptation", "lora"),
            license="MIT",
            notes="Uses HiFi-GAN vocoder and x-vector speaker embeddings.",
        ),
        # Text LLMs. Gemma is the target family; the ungated SmolLM2 / Qwen2.5 entries use the same decoder-only layer names
        # and exist so the LLM path can be run and verified without a gated download. VRAM figures are estimates until
        # a real run measures them (Gemma has not been run here: its license is not accepted on this machine's HF account).
        ModelSpec(
            id="smollm2-135m-instruct", task="llm", family="SmolLM2", hf_id="HuggingFaceTB/SmolLM2-135M-Instruct", params_m=135,
            methods={"full": 3.0, "lora": 1.5, "qlora": 1.5}, implemented=("full", "lora", "qlora"), license="Apache-2.0",
            notes="Llama-style decoder-only model; small enough for full fine-tuning on 4 GB.",
        ),
        ModelSpec(
            id="qwen2.5-0.5b-instruct", task="llm", family="Qwen2", hf_id="Qwen/Qwen2.5-0.5B-Instruct", params_m=494,
            methods={"full": 8.0, "lora": 2.5, "qlora": 2.0}, implemented=("full", "lora", "qlora"), license="Apache-2.0",
            notes="Same layer layout as Gemma (q/k/v/o and gate/up/down projections).",
        ),
        ModelSpec(
            id="gemma-3-270m-it", task="llm", family="Gemma", hf_id="google/gemma-3-270m-it", params_m=270,
            methods={"full": 6.0, "lora": 2.5, "qlora": 2.0}, implemented=("full", "lora", "qlora"), license="Gemma Terms of Use",
            gated=True, notes="Gated: accept the license at huggingface.co/google/gemma-3-270m-it with the account whose token is "
            "configured. 262k-token vocabulary, so the output layer dominates memory at long sequence lengths.",
        ),
        ModelSpec(
            id="gemma-3-1b-it", task="llm", family="Gemma", hf_id="google/gemma-3-1b-it", params_m=1000,
            methods={"full": 16.0, "lora": 4.5, "qlora": 2.8}, implemented=("full", "lora", "qlora"), license="Gemma Terms of Use",
            gated=True, notes="Gated: accept the license at huggingface.co/google/gemma-3-1b-it with the account whose token is configured.",
        ),
        ModelSpec(
            id="xtts-v2",
            task="tts",
            family="XTTS",
            hf_id="coqui/XTTS-v2",
            params_m=467,
            methods={"full": 14.0, "speaker_adaptation": 12.0},
            implemented=(),
            license="Coqui Public Model License (non-commercial)",
            notes="GPT-style decoder; fine-tuning needs a large GPU. No trainer is included in this build "
            "because it cannot be validated on small GPUs.",
        ),
    ]
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model: {model_id}") from None


def list_models(task: str | None = None) -> list[ModelSpec]:
    return [m for m in MODELS.values() if task is None or m.task == task]


def check_support(model: ModelSpec, method: str, gpu: GpuInfo, observed_gb: float | None = None) -> Support:
    """Decide honestly whether `method` can run for `model` on this machine, right now.
    `observed_gb` is the VRAM a real run of this model+method used on this GPU; it overrides the estimate."""
    if method not in METHODS:
        return Support(False, f"Unknown method '{method}'.")
    if method not in model.methods:
        return Support(False, f"{model.family} does not support {METHODS[method]}.")
    if method not in model.implemented:
        return Support(False, f"{METHODS[method]} for {model.id} is not implemented in this build yet.")
    need = model.methods[method]
    if not gpu.available:
        return Support(False, f"No CUDA GPU detected; needs about {need:g} GB VRAM.")
    if observed_gb is not None and observed_gb <= gpu.total_gb:
        return Support(True, f"Verified on this machine: a real run used {observed_gb:g} GB.", True, observed_gb)
    if gpu.total_gb < need:
        return Support(False, f"Needs about {need:g} GB VRAM (estimate); detected {gpu.total_gb:g} GB ({gpu.name}).")
    return Support(True, observed_gb=observed_gb)
