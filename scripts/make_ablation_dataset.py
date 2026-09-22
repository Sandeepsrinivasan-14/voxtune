"""Build a LARGER real-speech STT dataset (Windows SAPI voices) for ablation studies, then run it through VoxTune's own
analyze + prepare pipeline (no server needed). Output: data/ablation/processed (train/val/test.jsonl + wavs).

Sentences are template x medical-term combinations, so the test split contains sentences (and term/context pairings)
never seen in training while the rare terms themselves recur: that is what makes domain-term accuracy a meaningful metric.
Still synthetic speech from two voices: it shows relative effects of fine-tuning choices, not real-world accuracy.

usage: python scripts/make_ablation_dataset.py [n_templates]
"""
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.datasets import analyzer, loader, prepare  # noqa: E402

OUT = ROOT / "data" / "ablation"
TERMS = ["gastroparesis", "esophagitis", "omeprazole", "endoscopy", "colonoscopy", "pantoprazole", "cholecystectomy",
         "hepatomegaly", "dyspepsia", "sigmoidoscopy", "cirrhosis", "pancreatitis"]
TEMPLATES = [
    "The patient was diagnosed with {t} last year.", "We suspect {t} in this case.", "Please explain {t} to the family.",
    "The report mentions {t} twice.", "A history of {t} was noted at admission.", "The doctor ordered a test for {t}.",
    "Symptoms of {t} began in the spring.", "The nurse asked about {t} this morning.", "Treatment for {t} will start today.",
    "There was no sign of {t} on the scan.", "The clinic specialises in {t} and related conditions.",
    "Her mother also had {t} at a young age.", "The team discussed {t} during rounds.", "He was referred for {t} last week.",
    "Notes on {t} are in the patient file.", "The pharmacy stocked medicine for {t}.", "An update on {t} arrived by email.",
    "The student wrote an essay about {t}.", "Recovery from {t} usually takes a few weeks.", "The lecture covered {t} in detail.",
]
VOICES = {"david": "Microsoft David Desktop", "zira": "Microsoft Zira Desktop"}


def synth(rows: list[dict]) -> None:
    jobs = OUT / "jobs.json"
    jobs.write_text(json.dumps(rows), encoding="utf-8")
    ps = r"""
Add-Type -AssemblyName System.Speech
$rows = Get-Content -Raw -Encoding UTF8 'JOBS' | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($r in $rows) {
  $s.SelectVoice($r.voice)
  $s.Rate = $r.rate
  $s.SetOutputToWaveFile($r.path, $fmt)
  $s.Speak($r.text)
  $s.SetOutputToNull()
}
""".replace("JOBS", str(jobs))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main(n_templates: int) -> None:
    raw = OUT / "raw"
    (raw / "audio").mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    for ti, tpl in enumerate(TEMPLATES[:n_templates]):
        for wi, term in enumerate(TERMS):
            text = tpl.format(t=term)
            for spk, voice in VOICES.items():
                fn = f"{spk}_{ti:02d}_{wi:02d}.wav"
                rows.append({"voice": voice, "path": str(raw / "audio" / fn), "text": text, "rate": (ti + wi) % 3 - 1})
                meta.append({"audio": f"audio/{fn}", "text": text, "speaker": spk})
    print(f"synthesising {len(rows)} clips ...", flush=True)
    synth(rows)
    with open(raw / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio", "text", "speaker"])
        w.writeheader()
        w.writerows(meta)
    items = loader.load_items(raw)
    summary, results = analyzer.analyze_dataset(items, "stt")
    print("analysis:", {k: summary[k] for k in ("total_recordings", "total_hours", "flag_counts", "warnings")}, flush=True)
    info = prepare.prepare_dataset(items, results, OUT / "processed", "stt", exclude_flags=analyzer.DEFAULT_EXCLUDE,
                                   val_frac=0.1, test_frac=0.15, seed=7)
    (OUT / "terms.json").write_text(json.dumps(TERMS))
    print("prepared:", info["counts"], info["hours"], flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else len(TEMPLATES))
