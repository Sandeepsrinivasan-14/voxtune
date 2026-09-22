import io
import time
import zipfile

import numpy as np
import soundfile as sf


def make_zip(n=12, sr=16000) -> bytes:
    rng = np.random.default_rng(1)
    buf = io.BytesIO()
    rows = ["audio,text,speaker"]
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(n):
            t = np.arange(int(3 * sr)) / sr
            y = (0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * (180 + 10 * i) * t)
                 + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
            b = io.BytesIO()
            sf.write(b, y, sr, format="WAV")
            z.writestr(f"audio/{i}.wav", b.getvalue())
            rows.append(f"audio/{i}.wav,sentence number {i} about the patient,{'a' if i % 2 else 'b'}")
        z.writestr("metadata.csv", "\n".join(rows))
    return buf.getvalue()


def wait(client, ds_id, status):
    for _ in range(120):
        d = client.get(f"/api/datasets/{ds_id}").json()
        if d["status"] == status:
            return d
        assert d["status"] != "failed", d["error"]
        time.sleep(0.5)
    raise AssertionError(f"dataset never reached {status}")


def test_upload_analyze_prepare_and_run_gating(client):
    pid = client.post("/api/projects", json={"name": "p", "task": "stt"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/datasets", data={"name": "d", "language": "en"}, files={"file": ("d.zip", make_zip())})
    assert r.status_code == 201
    ds = r.json()["id"]

    # cannot train or prepare before analysis
    assert client.post(f"/api/datasets/{ds}/prepare", json={}).status_code == 409
    assert client.post("/api/runs", json={"dataset_id": ds, "model_id": "whisper-tiny", "method": "lora"}).status_code == 409

    assert client.post(f"/api/datasets/{ds}/analyze").status_code == 202
    an = wait(client, ds, "analyzed")["analysis"]
    assert an["total_recordings"] == 12 and an["speakers"] == 2
    assert client.get(f"/api/datasets/{ds}/items").json()["total"] == 12

    assert client.post(f"/api/datasets/{ds}/prepare", json={"augment": ["bogus"]}).status_code == 422
    assert client.post(f"/api/datasets/{ds}/prepare", json={"augment": ["noise"]}).status_code == 202
    prep = wait(client, ds, "prepared")["prepared"]
    assert prep["counts"]["train"] > 0 and "noise" in prep["augmentations"]

    # honest gating: wrong task, unsupported-on-this-hardware and unimplemented methods are all refused with a reason
    assert client.post("/api/runs", json={"dataset_id": ds, "model_id": "speecht5-tts", "method": "full"}).status_code == 422
    big = client.post("/api/runs", json={"dataset_id": ds, "model_id": "whisper-large-v3", "method": "lora"})
    assert big.status_code == 422 and "unsupported" in big.json()["detail"].lower()
    xtts = client.post("/api/runs", json={"dataset_id": ds, "model_id": "xtts-v2", "method": "full"})
    assert xtts.status_code == 422
    assert client.post("/api/runs/batch", json={"dataset_id": ds, "model_id": "whisper-tiny", "methods": ["lora"]}).status_code == 422

    # impossible technique combinations and bad Auto-tune options are refused before any GPU time is spent
    tiny = {"dataset_id": ds, "model_id": "whisper-tiny", "method": "lora"}
    clash = client.post("/api/runs", json={**tiny, "config": {"use_dora": True, "lora_init": "pissa"}})
    assert clash.status_code == 422 and "DoRA" in clash.json()["detail"]
    ada = client.post("/api/runs", json={**tiny, "method": "adalora", "config": {"use_dora": True}})
    assert ada.status_code == 422 and "AdaLoRA" in ada.json()["detail"]
    assert client.post("/api/runs", json={**tiny, "config": {"lora_scope": "nonsense"}}).status_code == 422
    for bad in ({"trials": 1}, {"trials": 8, "budget_steps": 2}, {"metric": "bleu"}):
        assert client.post("/api/runs", json={**tiny, "autotune": bad}).status_code == 422

    rec = client.post("/api/strategy/recommend", json={"model_id": "whisper-tiny", "dataset_id": ds}).json()
    assert rec["method"] and rec["explanation"] and rec["config"]["batch_size"] >= 1
    assert rec["config"]["auto_lr"] is True and rec["autotune"]["trials"] >= 2 and rec["autotune"]["reason"]


def test_meta_exposes_method_catalogue_and_field_applicability(client):
    m = client.get("/api/meta").json()
    assert set(m["methods"]) <= set(m["method_info"])
    for info in m["method_info"].values():
        assert info["trains"] and info["best_for"] and info["label"]
    keys = {f["key"]: f for f in m["train_fields"]}
    assert "lora" in keys["use_dora"]["applies"] and "full" not in keys["use_dora"]["applies"]
    assert "full" in keys["llrd"]["applies"] and "lora" not in keys["llrd"]["applies"]
    assert all(f["applies"] for f in keys.values())


def test_bad_uploads_rejected(client):
    pid = client.post("/api/projects", json={"name": "p", "task": "stt"}).json()["id"]
    assert client.post(f"/api/projects/{pid}/datasets", data={"name": "d"}, files={"file": ("d.txt", b"x")}).status_code == 422
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "no audio here")
    assert client.post(f"/api/projects/{pid}/datasets", data={"name": "d"}, files={"file": ("d.zip", empty.getvalue())}).status_code == 422
    assert client.get(f"/api/projects/{pid}/datasets").json() == []


def test_registry_rejects_unfinished_run(client):
    assert client.post("/api/registry", json={"run_id": 999}).status_code == 404
    assert client.get("/api/experiments/compare").status_code == 422
