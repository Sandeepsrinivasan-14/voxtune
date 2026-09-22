"""Catalogue of fine-tuning methods: what each one trains, when it is the right tool, and its cost profile.
Served to the UI via /api/meta and used to explain recommendations."""

# method id -> underlying mechanism the plugins implement
MECHANISM = {
    "full": "full", "lora": "lora", "domain_adaptation": "lora", "qlora": "qlora", "adapter": "ia3",
    "adalora": "adalora", "speaker_adaptation": "freeze_text_encoder",
}

METHOD_INFO = {
    "full": {
        "label": "Full fine-tuning", "memory": "high", "artifact": "full model",
        "trains": "Every weight (optionally with the encoder frozen and layer-wise LR decay).",
        "best_for": "Large datasets (tens of hours) or a strong distribution shift, when the GPU can hold weights, gradients and optimizer state.",
    },
    "lora": {
        "label": "LoRA", "memory": "low", "artifact": "small adapter",
        "trains": "Low-rank update matrices on attention (optionally feed-forward) layers; base frozen. Variants: DoRA, rsLoRA, PiSSA/OLoRA init, LoRA+.",
        "best_for": "Small to medium data, limited GPU, several custom versions of one base model.",
    },
    "qlora": {
        "label": "QLoRA", "memory": "lowest", "artifact": "small adapter",
        "trains": "LoRA on a 4-bit NF4 (double-quantized) base model.",
        "best_for": "Models that barely fit in memory. Slightly slower steps; the base is quantized, so compare against a quantized baseline.",
    },
    "adalora": {
        "label": "AdaLoRA (adaptive rank)", "memory": "low", "artifact": "small adapter",
        "trains": "LoRA whose rank budget is re-allocated across layers during training by importance scores.",
        "best_for": "When you don't know which layers need capacity: it prunes rank from unimportant matrices for you.",
    },
    "adapter": {
        "label": "Adapter (IA³)", "memory": "lowest", "artifact": "tiny adapter (kilobytes)",
        "trains": "Learned scaling vectors on keys, values and feed-forward activations (~0.01% of weights).",
        "best_for": "Very small datasets or many tiny per-customer adapters; limited capacity, so expect smaller gains than LoRA.",
    },
    "domain_adaptation": {
        "label": "Domain adaptation", "memory": "low", "artifact": "small adapter",
        "trains": "LoRA plus domain-focused training: oversampling of utterances with your terms and optional extra loss weight on those tokens.",
        "best_for": "Teaching an ASR model specialised vocabulary (medical, legal, product names).",
    },
    "speaker_adaptation": {
        "label": "Speaker adaptation", "memory": "medium", "artifact": "full model",
        "trains": "TTS decoder, prenets and postnet with the text encoder frozen, conditioned on speaker embeddings.",
        "best_for": "Cloning or adapting a voice from a small single-speaker set.",
    },
}
