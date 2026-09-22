"""Text (instruction / prompt -> response) datasets for LLM fine-tuning: load, analyze, clean, split.

Accepted files (a .zip, or a bare file): metadata/manifest/train .jsonl / .csv / .tsv, or any single such file, with a prompt
column (prompt | instruction | question | query | source | input) and a response column (response | output | answer |
completion | target | reply). Alpaca-style rows with both `instruction` and `input` are joined into one prompt.
"""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROMPT_KEYS = ("prompt", "instruction", "question", "query", "source")
RESPONSE_KEYS = ("response", "output", "answer", "completion", "target", "reply")
EXTRA_PROMPT_KEYS = ("input", "context")  # appended to the instruction when present (Alpaca layout)
MANIFEST_NAMES = ("metadata", "manifest", "train", "data", "dataset")
SUFFIXES = (".jsonl", ".csv", ".tsv")
TEXT_EXTENSIONS = (".jsonl", ".csv", ".tsv")
MAX_CHARS = 12000  # beyond this an example cannot survive typical sequence limits
TEXT_FLAGS = ["empty_prompt", "empty_response", "duplicate", "too_long", "duplicate_prompt"]
DEFAULT_EXCLUDE = ["empty_prompt", "empty_response", "duplicate", "too_long"]
FLAG_LABELS = {
    "empty_prompt": "Empty prompt", "empty_response": "Empty response", "duplicate": "Duplicate example",
    "too_long": "Too long", "duplicate_prompt": "Same prompt, different response",
}


def flag_specs() -> list[dict]:
    return [{"id": f, "label": FLAG_LABELS[f], "default_exclude": f in DEFAULT_EXCLUDE, "kind": "text",
             "threshold": f"{MAX_CHARS} chars" if f == "too_long" else None} for f in TEXT_FLAGS]


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def _find_manifest(root: Path) -> Path | None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
    for p in files:
        if p.stem.lower() in MANIFEST_NAMES:
            return p
    return files[0] if len(files) == 1 else None


def _rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delim))


def load_text_items(root: Path) -> list[dict]:
    manifest = _find_manifest(Path(root))
    if not manifest:
        raise ValueError("Found no dataset file. Provide one .jsonl/.csv/.tsv (or name it metadata/train.*) with a prompt and a response column.")
    items = []
    for i, row in enumerate(_rows(manifest)):
        row = {str(k).strip().lower(): ("" if v is None else str(v)) for k, v in row.items() if k}
        pk = next((k for k in PROMPT_KEYS if k in row), None)
        rk = next((k for k in RESPONSE_KEYS if k in row), None)
        if pk is None or rk is None:
            raise ValueError(f"Row {i + 1} needs a prompt column ({'/'.join(PROMPT_KEYS)}) and a response column ({'/'.join(RESPONSE_KEYS)}); "
                             f"found {sorted(row)}.")
        prompt = row[pk].strip()
        extra = next((row[k].strip() for k in EXTRA_PROMPT_KEYS if row.get(k, "").strip()), "")
        if extra and pk == "instruction":
            prompt = f"{prompt}\n\n{extra}"
        items.append({"id": f"row{i}", "prompt": prompt, "response": row[rk].strip()})
    return items


def analyze_text_dataset(items: list[dict]) -> tuple[dict, list[dict]]:
    res, seen, prompts = [], {}, {}
    for it in items:
        r = {"id": it["id"], "prompt": it["prompt"], "response": it["response"], "flags": [],
             "chars": len(it["prompt"]) + len(it["response"])}
        if not any(c.isalnum() for c in it["prompt"]):
            r["flags"].append("empty_prompt")
        if not any(c.isalnum() for c in it["response"]):
            r["flags"].append("empty_response")
        if r["chars"] > MAX_CHARS:
            r["flags"].append("too_long")
        key = hashlib.md5((_norm(it["prompt"]) + "\x1f" + _norm(it["response"])).encode()).hexdigest()
        if key in seen:
            r["flags"].append("duplicate")
        else:
            seen[key] = it["id"]
        prompts.setdefault(_norm(it["prompt"]), set()).add(_norm(it["response"]))
        res.append(r)
    for r in res:
        if len(prompts[_norm(r["prompt"])]) > 1:
            r["flags"].append("duplicate_prompt")
    n = len(res)
    counts = Counter(f for r in res for f in r["flags"])
    words = [w for r in res for w in (r["prompt"] + " " + r["response"]).lower().split()]
    chars = sorted(r["chars"] for r in res)
    uniq = len({_norm(r["prompt"]) for r in res}) / max(n, 1)
    lexical = min(1.0, len(set(words)) / max(1.0, len(words) ** 0.75)) if words else 0.0
    bad = sum(1 for r in res if set(r["flags"]) & {"empty_prompt", "empty_response", "duplicate", "too_long"})
    text_quality = 1 - bad / max(n, 1)
    diversity = 0.6 * uniq + 0.4 * lexical
    scores = {"text_quality": text_quality, "diversity": diversity, "overall": 0.6 * text_quality + 0.4 * diversity}
    warnings = []
    labels = {"empty_prompt": "examples have an empty prompt", "empty_response": "examples have an empty response",
              "duplicate": "examples are exact duplicates", "too_long": f"examples are longer than {MAX_CHARS} characters",
              "duplicate_prompt": "examples reuse a prompt with a different response (the model will see conflicting targets)"}
    for f, lab in labels.items():
        if counts.get(f):
            warnings.append(f"{counts[f]} {lab}.")
    if n < 100:
        warnings.append(f"Only {n} examples; fine-tuning results will be noisy and easy to overfit.")
    summary = {
        "task": "llm", "total_recordings": n, "examples": n,
        "avg_prompt_chars": round(sum(len(r["prompt"]) for r in res) / max(n, 1), 1),
        "avg_response_chars": round(sum(len(r["response"]) for r in res) / max(n, 1), 1),
        "p95_chars": chars[int(0.95 * (n - 1))] if n else 0, "vocabulary": len(set(words)),
        "quality": {"good": n - bad, "empty": counts["empty_prompt"] + counts["empty_response"], "duplicate": counts["duplicate"],
                    "too_long": counts["too_long"], "duplicate_prompt": counts["duplicate_prompt"]},
        "flag_counts": dict(counts), "scores": {k: round(v, 3) for k, v in scores.items()}, "warnings": warnings,
    }
    return summary, res


def split_by_prompt(items: list[dict], val_frac: float, test_frac: float, seed: int) -> dict[str, list[dict]]:
    """Group by normalized prompt so the same prompt never appears in more than one split."""
    n = len(items)
    if n < 3:
        raise ValueError(f"Need at least 3 usable examples to split, found {n}.")
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(_norm(it["prompt"]), []).append(it)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    want = {"test": max(1, round(n * test_frac)), "val": max(1, round(n * val_frac))}
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for k in keys:
        for split in ("test", "val"):
            if len(out[split]) < want[split]:
                out[split].extend(groups[k])
                break
        else:
            out["train"].extend(groups[k])
    if not out["train"]:
        flat = [it for k in keys for it in groups[k]]
        out = {"test": flat[:want["test"]], "val": flat[want["test"]:want["test"] + want["val"]], "train": flat[want["test"] + want["val"]:]}
    return out


def prepare_text_dataset(items: list[dict], results: list[dict], out_dir: Path, *, exclude_flags: list[str],
                         val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42) -> dict:
    bad = set(exclude_flags)
    by_id = {r["id"]: r for r in results}
    kept = [it for it in items if not (bad & set(by_id[it["id"]]["flags"]))]
    splits = split_by_prompt(kept, val_frac, test_frac, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, chars = {}, {}
    for split, rows in splits.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False) + "\n" for r in rows)
        counts[split] = len(rows)
        chars[split] = sum(len(r["prompt"]) + len(r["response"]) for r in rows)
    info = {"dir": str(out_dir), "counts": counts, "hours": {k: 0.0 for k in counts}, "chars": chars,
            "dropped": len(items) - len(kept), "excluded_flags": sorted(bad), "augmentations": [], "seed": seed}
    (out_dir / "prepared.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

