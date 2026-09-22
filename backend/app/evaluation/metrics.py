"""Single source of truth for evaluation metrics: labels, formatting and direction. The UI reads this via /api/meta."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    fmt: str  # "pct" (0-1 shown as %) or "num"
    higher_better: bool
    compare: bool = False  # shown as a column in method comparison
    primary: bool = False  # headline metric for run lists

    def to_dict(self) -> dict:
        return asdict(self)


METRICS: dict[str, list[Metric]] = {
    "stt": [
        Metric("wer", "WER", "pct", False, compare=True, primary=True),
        Metric("cer", "CER", "pct", False, compare=True),
        Metric("word_accuracy", "Word accuracy", "pct", True),
        Metric("char_accuracy", "Character accuracy", "pct", True),
        Metric("domain_term_accuracy", "Domain-term accuracy", "pct", True, compare=True),
    ],
    "tts": [
        Metric("mfcc_dist", "MFCC distance", "num", False, compare=True, primary=True),
        Metric("speaker_similarity", "Speaker similarity", "pct", True, compare=True),
        Metric("intelligibility_wer", "Intelligibility (ASR WER)", "pct", False, compare=True),
    ],
    "llm": [
        Metric("perplexity", "Perplexity (reference responses)", "num", False, compare=True, primary=True),
        Metric("token_accuracy", "Next-token accuracy", "pct", True, compare=True),
        Metric("rouge_l", "ROUGE-L (generated vs reference)", "pct", True, compare=True),
        Metric("exact_match", "Exact match", "pct", True, compare=True),
    ],
}


def compare_columns(kind: str) -> dict[str, bool]:
    return {m.key: m.higher_better for m in METRICS.get(kind, []) if m.compare}
