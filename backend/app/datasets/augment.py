"""Waveform augmentations (16 kHz float32). STT only: degrading TTS targets would hurt synthesis."""
import librosa
import numpy as np

SR = 16000
AUGMENTATIONS = ("noise", "reverb", "speed", "volume", "pitch", "time_mask")
AUG_LABELS = {"noise": "Background noise", "reverb": "Room reverberation", "speed": "Speed perturbation",
              "volume": "Volume variation", "pitch": "Pitch variation", "time_mask": "Time masking"}


def noise(y, rng):
    snr = rng.uniform(8, 20)
    n = rng.standard_normal(len(y)).astype(np.float32)
    spec = np.fft.rfft(n)
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))  # pink-ish noise
    n = np.fft.irfft(spec, len(y)).astype(np.float32)
    n *= np.sqrt(np.mean(y**2) / (10 ** (snr / 10))) / (np.std(n) + 1e-9)
    return y + n


def reverb(y, rng):
    rt60 = rng.uniform(0.2, 0.6)
    t = np.arange(int(SR * rt60)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    out = np.convolve(y, ir / np.linalg.norm(ir), mode="full")[: len(y)]
    return out * (np.abs(y).max() / (np.abs(out).max() + 1e-9))


def speed(y, rng):
    rate = rng.choice([0.9, 1.1])
    return librosa.resample(y, orig_sr=int(SR * rate), target_sr=SR)


def volume(y, rng):
    return y * 10 ** (rng.uniform(-6, 6) / 20)


def pitch(y, rng):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=float(rng.choice([-2, -1, 1, 2])))


def time_mask(y, rng):
    y = y.copy()
    for _ in range(rng.integers(1, 4)):
        w = int(rng.uniform(0.05, 0.15) * SR)
        s = int(rng.integers(0, max(1, len(y) - w)))
        y[s : s + w] = 0
    return y


_FNS = {"noise": noise, "reverb": reverb, "speed": speed, "volume": volume, "pitch": pitch, "time_mask": time_mask}


def apply(name: str, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = _FNS[name](y.astype(np.float32), rng)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
