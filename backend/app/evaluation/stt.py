import re

import jiwer


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(text.split())


def count_term(text: str, term: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(normalize(term))}(?!\w)", text))


def compute_metrics(refs: list[str], hyps: list[str], domain_terms: list[str] | None = None) -> dict:
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]
    if not pairs:
        raise ValueError("No non-empty reference transcripts to evaluate against.")
    r, h = zip(*pairs)
    wer = jiwer.wer(list(r), list(h))
    cer = jiwer.cer(list(r), list(h))
    out = {"wer": round(wer, 4), "cer": round(cer, 4), "word_accuracy": round(max(0.0, 1 - wer), 4),
           "char_accuracy": round(max(0.0, 1 - cer), 4), "n": len(r), "domain_term_accuracy": None}
    if domain_terms:
        found = total = 0
        for ref, hyp in zip(r, h):
            for t in domain_terms:
                c = count_term(ref, t)
                total += c
                found += min(c, count_term(hyp, t))
        out["domain_term_accuracy"] = round(found / total, 4) if total else None
        out["domain_term_occurrences"] = total
    return out
