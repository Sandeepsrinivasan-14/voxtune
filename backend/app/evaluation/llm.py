"""Text-LLM evaluation. Teacher-forced metrics (perplexity, next-token accuracy) are computed over the reference response
tokens only; generation metrics (exact match, ROUGE-L) decode greedily and compare against the reference response."""
import math
import re

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(ref: str, hyp: str) -> float:
    """ROUGE-L F1 over normalized words (longest common subsequence)."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r or not h:
        return 1.0 if r == h else 0.0
    lcs = _lcs(r, h)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def generation_metrics(refs: list[str], hyps: list[str]) -> dict:
    n = max(len(refs), 1)
    return {"exact_match": round(sum(normalize(r) == normalize(h) for r, h in zip(refs, hyps)) / n, 4),
            "rouge_l": round(sum(rouge_l(r, h) for r, h in zip(refs, hyps)) / n, 4)}


def teacher_forced_metrics(ce_sum: float, n_tokens: int, correct: int) -> dict:
    """Token-weighted mean cross-entropy -> perplexity; accuracy of the argmax next token on response tokens."""
    if n_tokens == 0:
        return {"perplexity": None, "token_accuracy": None, "loss": None}
    loss = ce_sum / n_tokens
    return {"perplexity": round(math.exp(min(loss, 50)), 4), "token_accuracy": round(correct / n_tokens, 4), "loss": round(loss, 5)}
