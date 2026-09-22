"""Build a small synthetic instruction dataset for checking the text-LLM path end to end.

Task: turn a sentence about a lab result into ONE structured line, e.g.
  prompt   "Extract the result: The patient's hemoglobin was 11.2 g/dL, which is low."
  response "HEMOGLOBIN | 11.2 g/dL | LOW"
A base instruct model answers in free text, so exact match starts near zero; the mapping itself is a deterministic rule, so
gains from fine-tuning can be checked exactly. It says nothing about real-world instruction data.

Writes data/samples/sample_llm.jsonl (upload it as a bare file) and a prepared split in data/llm_sample/processed.
usage: python scripts/make_llm_sample_dataset.py [n_examples]
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.datasets import text  # noqa: E402

ANALYTES = {  # name -> (unit, normal range low, high)
    "hemoglobin": ("g/dL", 12.0, 16.0), "glucose": ("mg/dL", 70, 100), "creatinine": ("mg/dL", 0.6, 1.2),
    "sodium": ("mmol/L", 135, 145), "potassium": ("mmol/L", 3.5, 5.0), "cholesterol": ("mg/dL", 125, 200),
    "platelets": ("k/uL", 150, 400), "bilirubin": ("mg/dL", 0.1, 1.2), "albumin": ("g/dL", 3.5, 5.0), "ferritin": ("ng/mL", 30, 300),
}
SENTENCES = [
    "The patient's {a} was {v} {u}, which is {s}.", "Lab results show {a} at {v} {u} ({s}).", "On admission the {a} measured {v} {u}; this is {s}.",
    "Her {a} came back as {v} {u}, considered {s}.", "The {a} level is {v} {u} and reads {s}.", "Today's {a}: {v} {u}, flagged {s}.",
]
INSTRUCTIONS = ["Extract the result:", "Convert to a structured line:", "Structure this lab note:"]


def make(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        a = rng.choice(list(ANALYTES))
        unit, lo, hi = ANALYTES[a]
        v = round(rng.uniform(lo * 0.6, hi * 1.5), 1 if hi < 50 else 0)
        status = "LOW" if v < lo else "HIGH" if v > hi else "NORMAL"
        val = f"{v:g}"
        prompt = f"{rng.choice(INSTRUCTIONS)} {rng.choice(SENTENCES).format(a=a, v=val, u=unit, s=status.lower())}"
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({"prompt": prompt, "response": f"{a.upper()} | {val} {unit} | {status}"})
    return rows


def main(n: int) -> None:
    rows = make(n)
    (ROOT / "data" / "samples").mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "samples" / "sample_llm.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    raw = ROOT / "data" / "llm_sample" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "metadata.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    items = text.load_text_items(raw)  # the same loader an upload goes through
    summary, results = text.analyze_text_dataset(items)
    info = text.prepare_text_dataset(items, results, ROOT / "data" / "llm_sample" / "processed", exclude_flags=text.DEFAULT_EXCLUDE,
                                     val_frac=0.1, test_frac=0.15, seed=7)
    print(src)
    print("analysis:", {k: summary[k] for k in ("examples", "avg_prompt_chars", "avg_response_chars", "warnings")})
    print("prepared:", info["counts"], info["dir"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
