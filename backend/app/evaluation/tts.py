"""TTS evaluation with real, reproducible metrics.

* mfcc_dist: mean DTW-aligned Euclidean distance between MFCC frames (c1-c12) of the generated audio and the
  held-out recording of the same sentence (lower is better). It is a *relative* measure for comparing runs on the
  same data; it is not scaled like published MCD (dB) numbers and must not be compared with them.
* speaker_similarity: cosine similarity of x-vector embeddings, generated vs recorded (higher is better)
* intelligibility_wer: Whisper ASR round-trip WER on the generated audio (lower is better)
MOS is NOT computed: it requires human raters or a trained predictor that is not bundled.
"""
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import model_source, settings
from . import stt

_clf = None


def speaker_encoder(device: str):
    global _clf
    if _clf is None:
        from speechbrain.inference.speaker import EncoderClassifier

        try:
            from speechbrain.utils.fetching import LocalStrategy

            kw = {"local_strategy": LocalStrategy.COPY}
        except Exception:
            kw = {}
        _clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=str(settings.data_dir / "models_cache" / "xvect"),
            run_opts={"device": device},
            **kw,
        )
    return _clf


def release_speaker_encoder() -> None:
    global _clf
    _clf = None


@torch.no_grad()
def embed(wav: np.ndarray, device: str) -> torch.Tensor:
    e = speaker_encoder(device).encode_batch(torch.tensor(wav, dtype=torch.float32)[None].to(device))
    return F.normalize(e, dim=2).squeeze().cpu()


def mfcc_distance(ref: np.ndarray, gen: np.ndarray, sr: int = 16000) -> float:
    a = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=13)[1:]
    b = librosa.feature.mfcc(y=gen, sr=sr, n_mfcc=13)[1:]
    _, wp = librosa.sequence.dtw(a, b, metric="euclidean")
    return float(np.mean([np.linalg.norm(a[:, i] - b[:, j]) for i, j in wp]))


@torch.no_grad()
def asr_roundtrip(wavs: list[np.ndarray], device: str, model_id: str = "openai/whisper-base", language: str = "en") -> list[str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_source(model_id))
    model = WhisperForConditionalGeneration.from_pretrained(model_source(model_id)).to(device).eval()
    outs = []
    for w in wavs:
        feats = proc.feature_extractor(w, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        ids = model.generate(input_features=feats, language=language, task="transcribe", max_new_tokens=200)
        outs.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    del model
    return outs


def evaluate(refs: list[np.ndarray], gens: list[np.ndarray], texts: list[str], device: str, language: str = "en") -> dict:
    sims = [float(torch.dot(embed(r, device), embed(g, device))) for r, g in zip(refs, gens)]
    hyps = asr_roundtrip(gens, device, language=language)
    w = stt.compute_metrics(texts, hyps)
    return {
        "mfcc_dist": round(float(np.mean([mfcc_distance(r, g) for r, g in zip(refs, gens)])), 3),
        "speaker_similarity": round(float(np.mean(sims)), 4),
        "intelligibility_wer": w["wer"],
        "intelligibility_cer": w["cer"],
        "n": len(gens),
        "mos": None,
        "mos_note": "Not computed: needs human raters or a trained MOS predictor.",
    }
