import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from app.core.gpu import GpuInfo
from app.datasets import analyzer, loader, prepare
from app.evaluation import stt
from app.registry import get_model
from app.strategy import recommend

RNG = np.random.default_rng(0)


def speech_like(sec: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # syllable-like amplitude modulation with real pauses
    return (0.4 * env * np.sin(2 * np.pi * 200 * t) + 0.005 * RNG.standard_normal(len(t))).astype(np.float32)


@pytest.fixture()
def ds(tmp_path):
    raw = tmp_path / "raw"
    (raw / "audio").mkdir(parents=True)
    rows = ["audio,text,speaker"]

    def add(name, y, text, spk="a", sr=16000):
        sf.write(raw / "audio" / f"{name}.wav", y, sr)
        rows.append(f"audio/{name}.wav,{text},{spk}")

    for i in range(6):
        add(f"ok{i}", speech_like(3.0 + i * 0.1), f"this is sentence number {i} for testing", "a" if i < 4 else "b")
    add("short", speech_like(0.3), "hi")
    add("silent", np.zeros(48000, dtype=np.float32), "nothing said here at all")
    add("noisy", (0.3 * RNG.standard_normal(48000)).astype(np.float32), "this recording is full of noise")
    add("dup", speech_like(3.0), "this is sentence number 0 for testing")
    sf.write(raw / "audio" / "dupcopy.wav", speech_like(3.0), 16000)
    (raw / "audio" / "dup.wav").write_bytes((raw / "audio" / "dupcopy.wav").read_bytes())
    (raw / "audio" / "ok0.wav").write_bytes((raw / "audio" / "dupcopy.wav").read_bytes())
    add("empty", speech_like(3.0), "", "b")
    (raw / "audio" / "broken.wav").write_bytes(b"not audio")
    rows.append("audio/broken.wav,broken file text,a")
    rows.append("audio/missing.wav,file does not exist,a")
    (raw / "metadata.csv").write_text("\n".join(rows), encoding="utf-8")
    return raw


def test_loader_reads_manifest(ds):
    items = loader.load_items(ds)
    assert len(items) == 13
    assert next(i for i in items if i["id"] == "audio/missing.wav")["path"] is None


def test_loader_paired_txt(tmp_path):
    sf.write(tmp_path / "a.wav", speech_like(2), 16000)
    (tmp_path / "a.txt").write_text("hello there", encoding="utf-8")
    assert loader.load_items(tmp_path)[0]["text"] == "hello there"


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        loader.safe_extract(z, tmp_path / "out")


def test_analyzer_flags_and_scores(ds):
    summary, res = analyzer.analyze_dataset(loader.load_items(ds), "stt")
    flags = {r["id"].split("/")[-1]: set(r["flags"]) for r in res}
    assert "too_short" in flags["short.wav"]
    assert "silent" in flags["silent.wav"]
    assert "too_noisy" in flags["noisy.wav"]
    assert "corrupted" in flags["broken.wav"]
    assert "missing_audio" in flags["missing.wav"]
    assert "empty_transcript" in flags["empty.wav"]
    assert "duplicate" in flags["dup.wav"]  # byte-identical to ok0.wav, seen later in manifest order
    assert not flags["ok3.wav"]
    assert summary["speakers"] == 2 and summary["total_recordings"] == 13
    assert 0 <= summary["scores"]["overall"] <= 1
    assert any("background noise" in w for w in summary["warnings"])


def test_prepare_splits_without_text_leakage_and_augments(ds, tmp_path):
    items = loader.load_items(ds)
    _, res = analyzer.analyze_dataset(items, "stt")
    info = prepare.prepare_dataset(items, res, tmp_path / "p", "stt", exclude_flags=analyzer.DEFAULT_EXCLUDE, augment=["noise", "speed"])
    assert info["dropped"] > 0 and info["counts"]["val"] >= 1 and info["counts"]["test"] >= 1
    texts = {s: {json.loads(line)["text"] for line in (tmp_path / "p" / f"{s}.jsonl").read_text().splitlines()} for s in ("train", "val", "test")}
    assert not (texts["train"] & texts["test"]) and not (texts["train"] & texts["val"])
    train = [json.loads(line) for line in (tmp_path / "p" / "train.jsonl").read_text().splitlines()]
    assert {r["aug"] for r in train} == {"orig", "noise", "speed"}
    with pytest.raises(ValueError):
        prepare.prepare_dataset(items, res, tmp_path / "q", "tts", exclude_flags=[], augment=["noise"])


def test_stt_metrics():
    m = stt.compute_metrics(["The patient has gastroparesis."], ["the patient has gastro paresis"], ["gastroparesis"])
    assert m["wer"] == 0.5 and m["domain_term_accuracy"] == 0.0
    ok = stt.compute_metrics(["Take omeprazole daily"], ["take Omeprazole daily."], ["omeprazole"])
    assert ok["wer"] == 0 and ok["domain_term_accuracy"] == 1.0 and ok["word_accuracy"] == 1.0


def test_mfcc_distance_zero_for_identical_and_positive_for_different():
    from app.evaluation import tts

    a, b = speech_like(2.0), speech_like(2.0)[::-1] * 0.5
    assert tts.mfcc_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert tts.mfcc_distance(a, b) > 1.0


def test_strategy_small_gpu_prefers_lora_family():
    r = recommend(get_model("whisper-small"), GpuInfo(True, "RTX 3050", 4, 4), 0.5, 100)
    assert r["method"] in ("lora", "qlora") and r["config"]["batch_size"] <= 2
    assert r["explanation"] and r["alternatives"]["full"]["supported"] is False


def test_strategy_domain_goal_and_missing_terms_warning():
    r = recommend(get_model("whisper-base"), GpuInfo(True, "X", 24, 24), 8, 2000, goal="domain")
    assert r["method"] == "domain_adaptation" and any("domain terms" in w for w in r["warnings"])


def test_strategy_large_data_full_ft_when_it_fits():
    r = recommend(get_model("whisper-base"), GpuInfo(True, "X", 24, 24), 60, 30000)
    assert r["method"] == "full"


def test_strategy_forced_method_gets_its_own_settings():
    gpu = GpuInfo(True, "X", 24, 24)
    lora = recommend(get_model("whisper-tiny"), gpu, 1, 200, force_method="lora")["config"]
    full = recommend(get_model("whisper-tiny"), gpu, 1, 200, force_method="full")["config"]
    assert lora["learning_rate"] > 10 * full["learning_rate"]
    assert recommend(get_model("whisper-large-v3"), GpuInfo(True, "X", 4, 4), 1, 200, force_method="full")["method"] is None


def test_strategy_none_when_nothing_fits():
    r = recommend(get_model("xtts-v2"), GpuInfo(True, "RTX 3050", 4, 4), 5, 1000)
    assert r["method"] is None


def test_strategy_tts_speaker_goal():
    r = recommend(get_model("speecht5-tts"), GpuInfo(True, "X", 24, 24), 1, 300, goal="speaker")
    assert r["method"] == "speaker_adaptation"
    small = recommend(get_model("speecht5-tts"), GpuInfo(True, "RTX 3050", 4, 4), 5, 300)
    assert small["method"] == "full" and small["config"]["optimizer"] == "adamw_8bit"
