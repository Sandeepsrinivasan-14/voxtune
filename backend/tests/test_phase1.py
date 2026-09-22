from app.core.gpu import GpuInfo
from app.registry import check_support, get_model, list_models


def test_health_and_gpu(client):
    assert client.get("/api/system/health").json() == {"status": "ok"}
    g = client.get("/api/system/gpu").json()
    assert set(g) >= {"available", "name", "total_gb", "free_gb"}


def test_models_filtered_by_task(client):
    stt = client.get("/api/models", params={"task": "stt"}).json()
    assert stt and all(m["task"] == "stt" for m in stt)
    assert {m["id"] for m in client.get("/api/models", params={"task": "tts"}).json()} == {"speecht5-tts", "xtts-v2"}
    assert client.get("/api/models", params={"task": "video"}).status_code == 422


def test_unknown_model_404(client):
    assert client.get("/api/models/nope").status_code == 404


def test_unsupported_when_not_declared():
    s = check_support(get_model("speecht5-tts"), "qlora", GpuInfo(True, "X", 24, 24))
    assert not s.supported and "does not support" in s.reason


def test_unsupported_when_not_implemented():
    s = check_support(get_model("xtts-v2"), "speaker_adaptation", GpuInfo(True, "X", 80, 80))
    assert not s.supported and "not implemented" in s.reason


def test_vram_gating():
    m = get_model("whisper-large-v3")
    small = check_support(m, "lora", GpuInfo(True, "RTX 3050", 4, 4))
    big = check_support(m, "lora", GpuInfo(True, "A100", 40, 40))
    cpu = check_support(m, "lora", GpuInfo(False))
    assert not small.supported and "4 GB" in small.reason
    assert big.supported
    assert not cpu.supported and "No CUDA" in cpu.reason


def test_project_crud(client):
    r = client.post("/api/projects", json={"name": "Medical STT", "task": "stt"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["name"] == "Medical STT"
    assert len(client.get("/api/projects").json()) == 1
    assert client.post("/api/projects", json={"name": "x", "task": "bad"}).status_code == 422
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_registry_lists_all_tasks():
    assert {m.task for m in list_models()} == {"stt", "tts", "llm"}
