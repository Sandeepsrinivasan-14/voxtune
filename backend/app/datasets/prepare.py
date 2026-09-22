"""Clean -> preprocess (16 kHz mono) -> leakage-safe train/val/test split -> optional augmentation."""
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import augment as aug


def split_items(items: list[dict], val_frac: float, test_frac: float, seed: int) -> dict[str, list[dict]]:
    """Group by normalized transcript so identical sentences never straddle splits."""
    n = len(items)
    if n < 3:
        raise ValueError(f"Need at least 3 usable recordings to split, found {n}.")
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(" ".join(it["text"].lower().split()), []).append(it)
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
    if not out["train"]:  # tiny datasets with few unique sentences: fall back to a plain split
        flat = [it for k in keys for it in groups[k]]
        out = {"test": flat[: want["test"]], "val": flat[want["test"] : want["test"] + want["val"]],
               "train": flat[want["test"] + want["val"] :]}
    return out


def prepare_dataset(items: list[dict], results: list[dict], out_dir: Path, task: str, *, exclude_flags: list[str],
                    val_frac: float = 0.1, test_frac: float = 0.1, augment: list[str] | None = None,
                    seed: int = 42) -> dict:
    augment = augment or []
    if augment and task != "stt":
        raise ValueError("Augmentation is only available for STT datasets.")
    bad = set(exclude_flags)
    by_id = {r["id"]: r for r in results}
    kept = [it for it in items if it["path"] and not (bad & set(by_id[it["id"]]["flags"]))]
    dropped = len(items) - len(kept)
    splits = split_items(kept, val_frac, test_frac, seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    counts, hours = {}, {}
    for split, sitems in splits.items():
        wav_dir = out_dir / split
        wav_dir.mkdir(parents=True, exist_ok=True)
        rows, total = [], 0.0
        for i, it in enumerate(sitems):
            y, _ = librosa.load(it["path"], sr=16000, mono=True)
            variants = [("orig", y)]
            if split == "train":
                variants += [(a, aug.apply(a, y, rng)) for a in augment]
            for tag, wav in variants:
                p = wav_dir / f"{i:06d}_{tag}.wav"
                sf.write(p, wav, 16000, subtype="PCM_16")
                d = len(wav) / 16000
                total += d
                rows.append({"audio": str(p), "text": it["text"], "speaker": it["speaker"], "duration": round(d, 3), "aug": tag})
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        counts[split], hours[split] = len(rows), round(total / 3600, 4)
    info = {"dir": str(out_dir), "counts": counts, "hours": hours, "dropped": dropped, "excluded_flags": sorted(bad),
            "augmentations": augment, "sample_rate": 16000, "seed": seed}
    (out_dir / "prepared.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
