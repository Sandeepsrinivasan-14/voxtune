"""Build small REAL-speech sample datasets with the Windows built-in speech engine (SAPI).

Outputs (in data/samples/): sample_stt.zip (2 voices) and sample_tts.zip (single voice), each with
audio/*.wav + metadata.csv (audio,text,speaker). Used for end-to-end checks; it is not training data
you would ship, and it says nothing about accuracy on real human recordings.
"""
import csv
import json
import subprocess
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"

SENTENCES = [
    "The patient has abdominal pain after meals.",
    "Blood pressure is normal today.",
    "We suspect gastroparesis in this patient.",
    "The endoscopy showed mild esophagitis.",
    "Please take omeprazole every morning.",
    "A colonoscopy is scheduled for next week.",
    "The patient reports nausea and early satiety.",
    "Omeprazole reduces stomach acid production.",
    "Severe esophagitis requires further treatment.",
    "Gastroparesis delays the emptying of the stomach.",
    "The colonoscopy revealed no polyps.",
    "An endoscopy will examine the upper digestive tract.",
    "Take the omeprazole capsule before breakfast.",
    "The doctor recommended a gastric emptying study.",
    "The patient has a history of acid reflux.",
    "Esophagitis can cause pain when swallowing.",
    "Your appointment is tomorrow at nine.",
    "Good morning, how can I help you today?",
    "Please describe your symptoms in detail.",
    "The heart rate is seventy two beats per minute.",
    "We need to repeat the endoscopy in six months.",
    "Diet changes help manage gastroparesis symptoms.",
    "The colonoscopy preparation begins the night before.",
    "Omeprazole should be taken on an empty stomach.",
    "The patient denies any chest pain.",
    "There is mild inflammation of the esophagus.",
    "Gastroparesis is common in patients with diabetes.",
    "The nurse will check your temperature.",
    "Your test results are ready.",
    "Avoid fatty foods if you have gastroparesis.",
    "The endoscopy was performed under light sedation.",
    "We will schedule a follow up colonoscopy.",
    "Please continue the omeprazole for eight weeks.",
    "The pain is worse at night.",
    "Esophagitis was confirmed by biopsy.",
    "The patient feels bloated after eating.",
    "Thank you for coming in today.",
    "The doctor will see you shortly.",
    "Please fill out this form before your visit.",
    "The endoscopy report is normal.",
    "Take one tablet twice a day with water.",
    "The colonoscopy is a routine screening test.",
    "Symptoms of gastroparesis include vomiting and fullness.",
    "Omeprazole is a proton pump inhibitor.",
    "The patient is allergic to penicillin.",
    "We will monitor the blood sugar closely.",
    "Severe esophagitis may cause bleeding.",
    "Your next visit is in two weeks.",
    "The endoscopy found a small ulcer.",
    "Please lie on your left side for the colonoscopy.",
    "The patient was diagnosed with gastroparesis last year.",
    "Drink plenty of water after the procedure.",
    "The blood test shows mild anemia.",
    "Omeprazole may cause headaches in some patients.",
    "The esophagitis improved after treatment.",
    "We recommend a soft diet for a few days.",
    "The patient will need a repeat endoscopy.",
    "Your prescription has been sent to the pharmacy.",
    "The abdominal ultrasound was normal.",
    "Colonoscopy findings will be discussed at your next visit.",
]


def synth(rows: list[dict]) -> None:
    (OUT / "jobs.json").write_text(json.dumps(rows), encoding="utf-8")
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
""".replace("JOBS", str(OUT / "jobs.json"))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def build(name: str, voices: dict[str, str]) -> Path:
    d = OUT / name / "audio"
    d.mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    for i, text in enumerate(SENTENCES):
        for spk, voice in voices.items():
            fn = f"{spk}_{i:03d}.wav"
            rows.append({"voice": voice, "path": str(d / fn), "text": text, "rate": (i % 3) - 1})
            meta.append({"audio": f"audio/{fn}", "text": text, "speaker": spk})
    synth(rows)
    with open(OUT / name / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio", "text", "speaker"])
        w.writeheader()
        w.writerows(meta)
    zp = OUT / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / name / "metadata.csv", "metadata.csv")
        for r in meta:
            z.write(OUT / name / r["audio"], r["audio"])
    return zp


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(build("sample_stt", {"david": "Microsoft David Desktop", "zira": "Microsoft Zira Desktop"}))
    print(build("sample_tts", {"zira": "Microsoft Zira Desktop"}))
