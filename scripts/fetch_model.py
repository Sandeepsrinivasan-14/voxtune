"""Download a Hugging Face model repo into data/models_cache/<org>--<name> using parallel ranged requests.

Useful on slow or flaky networks where the hub client stalls. VoxTune loads models from that folder
automatically when it exists (see app.training.common.model_source).

Gated repos (e.g. google/gemma-3-1b-it) work once the license is accepted on Hugging Face with the account whose token is
configured (HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or the token saved by `huggingface-cli login`).

usage: python scripts/fetch_model.py microsoft/speecht5_tts [more repos...]
"""
import os
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1] / "data" / "models_cache"
WORKERS = 4


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    f = Path.home() / ".cache" / "huggingface" / "token"
    return f.read_text().strip() if f.exists() else None


TOKEN = _token()
AUTH = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}  # httpx drops it on the cross-host redirect to the CDN


def retry(fn, tries: int = 12):
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001 - flaky networks drop connections
            print(f"  retry {i + 1}/{tries}: {type(e).__name__}", flush=True)
            time.sleep(min(2 * (i + 1), 15))
    raise RuntimeError("giving up after repeated network errors")


def fetch_range(url: str, dest: Path, lo: int, hi: int) -> None:
    """Download bytes [lo, hi] into dest at offset lo, resuming from the last written byte on errors."""
    pos = lo
    while pos <= hi:
        try:
            with httpx.stream("GET", url, headers={"Range": f"bytes={pos}-{hi}", **AUTH}, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "r+b") as f:
                    f.seek(pos)
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
                        pos += len(chunk)
        except Exception as e:  # noqa: BLE001 - retry from where we stopped
            print(f"  retry at {pos}: {type(e).__name__}", flush=True)
            time.sleep(3)


def _size(url: str) -> int:
    with httpx.stream("GET", url, headers={"Range": "bytes=0-0", **AUTH}, follow_redirects=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise PermissionError(f"{url} returned {r.status_code}: this repo is gated. Accept its license at "
                                  f"https://huggingface.co/{url.split('/resolve/')[0].split('huggingface.co/')[1]} with the account "
                                  "whose token is configured (and make sure a token is set), then run again.")
        return int(r.headers["Content-Range"].split("/")[1])


def fetch_file(repo: str, name: str, out: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    dest, part = out / name, out / f"{name}.part"
    if dest.exists():
        return
    size = retry(lambda: _size(url))
    with open(part, "wb") as f:
        f.truncate(size)
    step = -(-size // WORKERS)
    threads = [threading.Thread(target=fetch_range, args=(url, part, i * step, min(size, (i + 1) * step) - 1)) for i in range(WORKERS)]
    print(f"{repo}/{name}: {size / 1e6:.0f} MB", flush=True)
    [t.start() for t in threads]
    [t.join() for t in threads]
    part.rename(dest)


def main(repos: list[str]) -> None:
    for repo in repos:
        out = ROOT / repo.replace("/", "--")
        out.mkdir(parents=True, exist_ok=True)
        files = retry(lambda: [s["rfilename"] for s in httpx.get(f"https://huggingface.co/api/models/{repo}", headers=AUTH, timeout=60).json()["siblings"]])
        skip_ext = (".h5", ".msgpack", ".onnx", ".ot", ".tflite")
        for name in files:
            if name in (".gitattributes", "README.md") or "/" in name or name.endswith(skip_ext):
                continue
            if name == "pytorch_model.bin" and "model.safetensors" in files:
                continue
            fetch_file(repo, name, out)
        (out / "COMPLETE").write_text("ok")  # the loader ignores mirrors without this marker
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
