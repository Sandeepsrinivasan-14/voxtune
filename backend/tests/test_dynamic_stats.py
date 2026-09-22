import json
import shutil

import psutil

from app.core import db
from app.core.config import settings
from app.core.gpu import GpuInfo, detect_gpu
from app.registry import check_support, get_model, observed_vram


def test_live_stats_match_independent_sources(client):
    d = client.get("/api/system/live").json()
    assert abs(d["ram"]["total_gb"] - psutil.virtual_memory().total / 1024**3) < 0.05
    assert 0 <= d["cpu"]["percent"] <= 100
    du = shutil.disk_usage(settings.data_dir)
    assert abs(d["disk"]["total_gb"] - du.total / 1024**3) < 0.05
    gpu = detect_gpu()
    if gpu.available:
        assert d["gpus"] and abs(d["gpus"][0]["mem_total_gb"] - gpu.total_gb) < 0.05
        assert d["gpus"][0]["name"] == gpu.name


def test_info_reports_this_machine(client):
    i = client.get("/api/system/info").json()
    assert i["cpu"]["logical_cores"] == psutil.cpu_count()
    assert i["torch"]["installed"] and i["torch"]["version"]
    assert i["packages"]["fastapi"]


def test_meta_is_the_single_source_of_ui_labels(client):
    m = client.get("/api/meta").json()
    assert {t["id"] for t in m["tasks"]} == {"stt", "tts", "llm"}
    assert m["methods"]["lora"] == "LoRA" and "domain_adaptation" in m["merged_export_methods"]
    assert any(f["id"] == "too_noisy" and not f["default_exclude"] for f in m["flags"]["stt"])
    assert {a["id"] for a in m["augmentations"]} >= {"noise", "reverb", "time_mask"}
    keys = {f["key"]: f for f in m["train_fields"]}
    assert keys["precision"]["kind"] == "select" and "bf16" in keys["precision"]["options"]
    assert keys["epochs"]["kind"] == "number" and keys["early_stopping"]["kind"] == "boolean"
    assert [x["key"] for x in m["metrics"]["stt"] if x["primary"]] == ["wer"]
    assert all(g["tasks"] for g in m["goals"])


def test_dashboard_reflects_workspace_state(client):
    empty = client.get("/api/dashboard").json()
    assert empty["counts"] == {"projects": 0, "datasets": 0, "runs": 0, "registered_models": 0}
    assert empty["queue"]["active"] is None and empty["recent_runs"] == []
    client.post("/api/projects", json={"name": "p", "task": "stt"})
    assert client.get("/api/dashboard").json()["counts"]["projects"] == 1
    assert set(empty["storage_gb"]) >= {"uploads", "datasets", "runs", "registry", "hf_cache", "total"}


def test_observed_vram_overrides_estimate_only_when_real_and_fits():
    m = get_model("whisper-medium")  # estimate says lora needs 12 GB
    gpu = GpuInfo(True, "RTX 3050", 4, 4)
    assert not check_support(m, "lora", gpu).supported
    s = check_support(m, "lora", gpu, observed_gb=3.4)
    assert s.supported and s.verified and s.observed_gb == 3.4
    assert not check_support(m, "lora", gpu, observed_gb=5.0).supported  # observed larger than the card
    assert not check_support(m, "lora", GpuInfo(False), observed_gb=3.4).supported


def test_observed_vram_read_from_completed_run_logs(client):
    gpu = GpuInfo(True, "X", 4.0, 4.0)
    with db.SessionLocal() as s:
        p = db.Project(name="p", task="stt")
        s.add(p)
        s.commit()
        ds = db.Dataset(project_id=p.id, name="d", task="stt", path="")
        s.add(ds)
        s.commit()
        r = db.Run(project_id=p.id, dataset_id=ds.id, model_id="whisper-small", method="lora", status="completed")
        other = db.Run(project_id=p.id, dataset_id=ds.id, model_id="whisper-small", method="qlora", status="completed")
        s.add_all([r, other])
        s.commit()
        for run, total, used in ((r, 4.0, 1.9), (other, 24.0, 1.2)):  # `other` ran on a different (bigger) GPU
            d = settings.runs_dir / f"run_{run.id:04d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "progress.jsonl").write_text(json.dumps({"type": "train", "gpu_used_gb": used, "gpu_total_gb": total}) + "\n")
        obs = observed_vram(s, gpu)
    assert obs == {"whisper-small": {"lora": 1.9}}
