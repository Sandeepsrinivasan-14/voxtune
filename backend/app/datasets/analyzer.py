"""Dataset analysis and quality scoring.

Noise/silence/mismatch checks are signal heuristics (frame-energy based), reported as estimates.
"""
import hashlib
import math
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MIN_SEC = 1.0
MAX_SEC = {"stt": 30.0, "tts": 15.0}
AUDIO_FLAGS = {"too_noisy", "silent", "corrupted", "clipped", "too_short", "too_long", "low_sample_rate", "missing_audio"}
TRANSCRIPT_FLAGS = {"empty_transcript", "transcript_mismatch"}
DEFAULT_EXCLUDE = ["missing_audio", "corrupted", "silent", "empty_transcript", "duplicate", "too_short", "too_long"]
ALL_FLAGS = sorted(AUDIO_FLAGS | TRANSCRIPT_FLAGS | {"duplicate"})
FLAG_LABELS = {
    "missing_audio": "Missing audio file", "corrupted": "Corrupted / unreadable", "silent": "Silent",
    "empty_transcript": "Empty transcript", "duplicate": "Duplicate audio", "too_short": "Too short",
    "too_long": "Too long", "too_noisy": "Too noisy", "transcript_mismatch": "Transcript/audio mismatch",
    "clipped": "Clipped", "low_sample_rate": "Sample rate below 16 kHz",
}
QUALITY_LABELS = {
    "good": "Good", "too_noisy": "Too noisy", "too_short": "Too short", "too_long": "Too long",
    "silent_or_corrupted": "Silent / corrupted / missing", "duplicate": "Duplicate", "transcript_issues": "Transcript issues",
    "empty": "Empty prompt or response", "duplicate_prompt": "Same prompt, different response",
}
SCORE_LABELS = {"audio_quality": "Audio quality", "transcript_quality": "Transcript quality", "text_quality": "Text quality",
                "speaker_balance": "Speaker balance", "diversity": "Diversity", "overall": "Overall"}


def flag_specs(task: str) -> list[dict]:
    return [{"id": f, "label": FLAG_LABELS[f], "default_exclude": f in DEFAULT_EXCLUDE,
             "kind": "transcript" if f in TRANSCRIPT_FLAGS else "audio",
             "threshold": (f"{MIN_SEC:g}s" if f == "too_short" else f"{MAX_SEC[task]:g}s" if f == "too_long" else None)}
            for f in ALL_FLAGS]


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_item(it: dict, task: str) -> dict:
    r = {"id": it["id"], "speaker": it.get("speaker", "unknown"), "text": it["text"], "flags": [], "duration": 0.0,
         "orig_sr": None, "snr_db": None, "silence_ratio": None, "md5": None}
    if not any(c.isalnum() for c in it["text"]):
        r["flags"].append("empty_transcript")
    p = it.get("path")
    if not p or not Path(p).exists():
        r["flags"].append("missing_audio")
        return r
    try:
        r["orig_sr"] = sf.info(p).samplerate
        y, _ = librosa.load(p, sr=16000, mono=True)
        r["md5"] = _file_md5(p)
    except Exception as e:  # unreadable / truncated file
        r["flags"].append("corrupted")
        r["error"] = str(e)[:200]
        return r
    if len(y) == 0 or not np.isfinite(y).all():
        r["flags"].append("corrupted")
        return r
    dur = len(y) / 16000
    r["duration"] = round(dur, 3)
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-8)
    if rms.mean() < 1e-3 or np.abs(y).max() < 1e-3:
        r["flags"].append("silent")
        return r
    srt = np.sort(db)
    noise_floor = float(srt[: max(1, len(srt) // 10)].mean())
    speech_level = float(srt[-max(1, len(srt) * 3 // 10):].mean())
    r["snr_db"] = round(speech_level - noise_floor, 1)
    r["silence_ratio"] = round(float((db < db.max() - 40).mean()), 3)
    if r["silence_ratio"] > 0.9:
        r["flags"].append("silent")
    if noise_floor > -45 and r["snr_db"] < 15:
        r["flags"].append("too_noisy")
    if float((np.abs(y) > 0.999).mean()) > 0.01:
        r["flags"].append("clipped")
    if dur < MIN_SEC:
        r["flags"].append("too_short")
    if dur > MAX_SEC[task]:
        r["flags"].append("too_long")
    if r["orig_sr"] < 16000:
        r["flags"].append("low_sample_rate")
    voiced = max(dur * (1 - r["silence_ratio"]), 0.1)
    cps = len(it["text"]) / voiced
    r["chars_per_sec"] = round(cps, 1)
    if it["text"] and (cps < 4 or cps > 30):
        r["flags"].append("transcript_mismatch")
    return r


def _script(text: str) -> str:
    c = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                c[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    return c.most_common(1)[0][0].title() if c else "Unknown"


def _entropy_balance(weights: list[float]) -> float:
    total = sum(weights)
    if len(weights) < 2 or total == 0:
        return 1.0
    ps = [w / total for w in weights if w > 0]
    return round(-sum(p * math.log(p) for p in ps) / math.log(len(weights)), 3)


def analyze_dataset(items: list[dict], task: str, workers: int = 4) -> tuple[dict, list[dict]]:
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda it: analyze_item(it, task), items))
    seen: dict[str, str] = {}
    for r in res:
        if r["md5"]:
            if r["md5"] in seen:
                r["flags"].append("duplicate")
            else:
                seen[r["md5"]] = r["id"]
    n = len(res)
    good = [r for r in res if not r["flags"]]
    flag_counts = Counter(f for r in res for f in r["flags"])
    total_dur = sum(r["duration"] for r in res)
    spk_dur: Counter = Counter()
    for r in res:
        spk_dur[r["speaker"]] += r["duration"]
    words = [w for r in res for w in r["text"].lower().split()]
    uniq_text = len({r["text"].strip().lower() for r in res}) / max(n, 1)
    lexical = min(1.0, len(set(words)) / max(1.0, len(words) ** 0.75)) if words else 0.0

    audio_score = 1 - sum(1 for r in res if AUDIO_FLAGS & set(r["flags"])) / max(n, 1)
    text_score = 1 - sum(1 for r in res if (TRANSCRIPT_FLAGS | {"duplicate"}) & set(r["flags"])) / max(n, 1)
    balance = _entropy_balance(list(spk_dur.values()))
    n_spk = len(spk_dur)
    diversity = (0.5 * uniq_text + 0.3 * lexical + 0.2 * min(1.0, n_spk / 4)) if task == "stt" else (0.6 * uniq_text + 0.4 * lexical)
    scores = {"audio_quality": audio_score, "transcript_quality": text_score, "speaker_balance": balance, "diversity": diversity}
    scores["overall"] = 0.35 * audio_score + 0.25 * text_score + 0.2 * balance + 0.2 * diversity
    scores = {k: round(v, 3) for k, v in scores.items()}

    warnings: list[str] = []
    labels = {
        "too_noisy": "recordings contain excessive background noise",
        "too_short": f"recordings are shorter than {MIN_SEC:g}s",
        "too_long": f"recordings are longer than {MAX_SEC[task]:g}s",
        "silent": "recordings are silent or almost entirely silence",
        "corrupted": "audio files are corrupted or unreadable",
        "missing_audio": "transcript rows point to missing audio files",
        "duplicate": "recordings are exact duplicates of another file",
        "empty_transcript": "recordings have an empty transcript",
        "transcript_mismatch": "recordings have a transcript length that looks inconsistent with the audio",
        "clipped": "recordings are clipped",
        "low_sample_rate": "recordings have a sample rate below 16 kHz",
    }
    for f, lab in labels.items():
        if flag_counts.get(f):
            warnings.append(f"{flag_counts[f]} {lab}.")
    if n_spk > 1 and total_dur:
        top, d = spk_dur.most_common(1)[0]
        if d / total_dur > 0.6:
            warnings.append(f"Dataset is heavily concentrated around speaker '{top}' ({d / total_dur:.0%} of audio).")
    if total_dur < 1800:
        warnings.append(f"Only {total_dur / 60:.1f} minutes of audio; results will be noisy and easy to overfit.")

    quality = {
        "good": len(good),
        "too_noisy": flag_counts["too_noisy"],
        "too_short": flag_counts["too_short"],
        "too_long": flag_counts["too_long"],
        "silent_or_corrupted": flag_counts["silent"] + flag_counts["corrupted"] + flag_counts["missing_audio"],
        "duplicate": flag_counts["duplicate"],
        "transcript_issues": flag_counts["empty_transcript"] + flag_counts["transcript_mismatch"],
    }
    summary = {
        "task": task,
        "total_recordings": n,
        "total_hours": round(total_dur / 3600, 3),
        "avg_duration_sec": round(total_dur / max(n, 1), 2),
        "speakers": n_spk,
        "speaker_hours": {k: round(v / 3600, 3) for k, v in spk_dur.most_common()},
        "scripts": _script(" ".join(r["text"] for r in res)),
        "sample_rates": dict(Counter(r["orig_sr"] for r in res if r["orig_sr"])),
        "vocabulary": len(set(words)),
        "quality": quality,
        "flag_counts": dict(flag_counts),
        "scores": scores,
        "warnings": warnings,
    }
    return summary, res
