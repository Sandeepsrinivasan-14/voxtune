"""Read a dataset directory into a list of {id, path, text, speaker} items.

Supported layouts (found anywhere under the root):
  * metadata.csv / .tsv / .jsonl with a header containing an audio-path column and a text column
    (optional `speaker`), or LJSpeech-style `id|text` rows without a header
  * audio files with a same-named .txt transcript next to them
"""
import csv
import json
import zipfile
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
PATH_KEYS = ("path", "audio", "file", "filename", "file_name", "wav", "id")
TEXT_KEYS = ("text", "transcript", "transcription", "sentence", "normalized_text")
MANIFEST_NAMES = ("metadata", "manifest", "transcripts", "train")


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise ValueError(f"Unsafe path in zip: {m.filename}")
        zf.extractall(dest)


def _find_manifest(root: Path) -> Path | None:
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in (".csv", ".tsv", ".jsonl") and p.stem.lower() in MANIFEST_NAMES:
            return p
    return None


def _resolve(base: Path, root: Path, ref: str) -> str | None:
    ref = ref.strip()
    names = [ref] if Path(ref).suffix.lower() in AUDIO_EXT else [ref + e for e in (".wav", ".flac", ".mp3", ".ogg")]
    for n in names:
        for d in (base, base / "wavs", base / "audio", root):
            if (d / n).is_file():
                return str((d / n).resolve())
    return None


def _rows(manifest: Path) -> list[dict]:
    text = manifest.read_text(encoding="utf-8-sig")
    if manifest.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    delim = max("|\t,;", key=lambda d: lines[0].count(d))
    header = [h.strip().lower() for h in next(csv.reader([lines[0]], delimiter=delim))]
    if any(k in header for k in PATH_KEYS) and any(k in header for k in TEXT_KEYS):
        return [{h: v for h, v in zip(header, r)} for r in csv.reader(lines[1:], delimiter=delim)]
    # LJSpeech style: id|text[|normalized]
    return [{"id": r[0], "text": r[-1] if len(r) > 1 else ""} for r in csv.reader(lines, delimiter=delim)]


def load_items(root: Path) -> list[dict]:
    root = Path(root)
    items: list[dict] = []
    manifest = _find_manifest(root)
    if manifest:
        base = manifest.parent
        for i, row in enumerate(_rows(manifest)):
            row = {str(k).strip().lower(): v for k, v in row.items() if k}
            ref = next((row[k] for k in PATH_KEYS if row.get(k)), "")
            text = next((row[k] for k in TEXT_KEYS if k in row), "")
            items.append(
                {
                    "id": str(ref) or f"row{i}",
                    "path": _resolve(base, root, str(ref)) if ref else None,
                    "text": (text or "").strip(),
                    "speaker": (row.get("speaker") or row.get("speaker_id") or "").strip() or "unknown",
                }
            )
        return items
    for a in sorted(root.rglob("*")):
        if a.suffix.lower() in AUDIO_EXT:
            t = a.with_suffix(".txt")
            items.append(
                {
                    "id": a.stem,
                    "path": str(a.resolve()),
                    "text": t.read_text(encoding="utf-8").strip() if t.exists() else "",
                    "speaker": a.parent.name if a.parent != root else "unknown",
                }
            )
    return items
