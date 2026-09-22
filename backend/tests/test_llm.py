"""Text-LLM path: dataset loading/analysis/split, layer targeting, validation rules, evaluation metrics, strategy, API flow,
and a real (CPU, tiny random Llama) training run through the plugin's collate/forward/eval code."""
import json
import re
import time

import pytest
import torch

from app.core.config import settings
from app.datasets import text
from app.evaluation import llm as ev
from app.registry import get_model, list_models
from app.strategy import recommend
from app.training.config import TrainConfig, ui_field_specs
from app.training.methods.families import FAMILIES, target_regex
from app.training.methods.validate import validate

ROWS = [{"prompt": f"Q{i}: what is {i}?", "response": f"answer {i} | {i * 2}"} for i in range(120)]


def jsonl(rows) -> bytes:
    return "\n".join(json.dumps(r) for r in rows).encode()


def test_loader_accepts_column_aliases_and_alpaca_layout(tmp_path):
    (tmp_path / "metadata.jsonl").write_bytes(jsonl([{"instruction": "Translate", "input": "hello", "output": "hola"},
                                                     {"instruction": "Say hi", "input": "", "output": "hi"}]))
    items = text.load_text_items(tmp_path)
    assert items[0]["prompt"] == "Translate\n\nhello" and items[0]["response"] == "hola" and items[1]["prompt"] == "Say hi"
    (tmp_path / "metadata.jsonl").unlink()
    (tmp_path / "train.csv").write_text("question,answer\nwhat?,this\n", encoding="utf-8")
    assert text.load_text_items(tmp_path)[0]["response"] == "this"
    (tmp_path / "train.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt column"):
        text.load_text_items(tmp_path)


def test_analysis_flags_and_leakage_safe_split(tmp_path):
    items = [{"id": f"r{i}", **r} for i, r in enumerate(ROWS)]
    items += [{"id": "e1", "prompt": "  ", "response": "x"}, {"id": "e2", "prompt": "ok", "response": ""},
              {"id": "d1", **ROWS[0]}, {"id": "c1", "prompt": ROWS[1]["prompt"], "response": "a different answer"}]
    summary, res = text.analyze_text_dataset(items)
    flags = {r["id"]: r["flags"] for r in res}
    assert "empty_prompt" in flags["e1"] and "empty_response" in flags["e2"] and "duplicate" in flags["d1"]
    assert "duplicate_prompt" in flags["c1"] and summary["examples"] == len(items)
    kept_info = text.prepare_text_dataset(items, res, tmp_path / "prep", exclude_flags=text.DEFAULT_EXCLUDE, seed=1)
    assert kept_info["dropped"] == 3
    sets = {s: {json.loads(line)["prompt"] for line in (tmp_path / "prep" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()}
            for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert kept_info["counts"]["train"] > kept_info["counts"]["val"] > 0


def test_causal_lm_targets_hit_text_layers_only():
    assert FAMILIES["gemma"].causal_lm and not FAMILIES["whisper"].causal_lm
    for scope in ("all", "decoder"):
        rx = target_regex("gemma", scope, "all_linear")
        yes = ["model.layers.3.self_attn.q_proj", "model.layers.0.mlp.down_proj", "language_model.model.layers.5.self_attn.o_proj"]
        no = ["vision_tower.vision_model.encoder.layers.2.self_attn.q_proj", "multi_modal_projector.proj", "lm_head", "model.embed_tokens"]
        assert all(re.fullmatch(rx, n) for n in yes) and not any(re.fullmatch(rx, n) for n in no)
    assert not re.fullmatch(target_regex("gemma", "all", "attention"), "model.layers.0.mlp.down_proj")


def test_validation_rules_for_language_models():
    ok, _ = validate("lora", TrainConfig(), family="gemma")
    assert ok == []
    for bad in ({"lora_scope": "encoder"}, {"freeze_encoder": True}, {"spec_augment": True}, {"select_by": "val_wer"}):
        errors, _ = validate("lora", TrainConfig(**bad), family="smollm2")
        assert errors, bad
    assert validate("adalora", TrainConfig(), family="gemma")[0]
    assert validate("lora", TrainConfig(precision="fp16"), family="gemma")[1]  # warns


def test_field_specs_are_task_aware():
    f = {x["key"]: x for x in ui_field_specs()}
    assert f["max_seq_len"]["tasks"] == ["llm"] and "llm" not in f["spec_augment"]["tasks"] and "llm" not in f["num_beams"]["tasks"]
    assert "llm" in f["label_smoothing"]["tasks"] and "llm" in f["learning_rate"]["tasks"]
    assert all(x["tasks"] for x in f.values())


def test_metrics():
    assert ev.rouge_l("the cat sat", "the cat sat") == 1.0 and ev.rouge_l("a b c", "x y z") == 0.0
    assert ev.generation_metrics(["Hello, World!"], ["hello world"]) == {"exact_match": 1.0, "rouge_l": 1.0}
    m = ev.teacher_forced_metrics(10.0, 10, 7)
    assert m["perplexity"] == pytest.approx(2.7183, abs=1e-3) and m["token_accuracy"] == 0.7
    assert ev.teacher_forced_metrics(0.0, 0, 0)["perplexity"] is None


def test_registry_and_strategy_gate_llms_honestly():
    from app.core.gpu import GpuInfo

    small = GpuInfo(available=True, name="test", total_gb=4.0, free_gb=4.0, utilization_pct=0)
    ids = {m.id for m in list_models("llm")}
    assert {"smollm2-135m-instruct", "gemma-3-1b-it"} <= ids and get_model("gemma-3-1b-it").gated
    r = recommend(get_model("gemma-3-1b-it"), small, 0.0, 400, "general", None, n_val=50)
    assert r["method"] == "qlora" and not r["alternatives"]["lora"]["supported"] and not r["alternatives"]["full"]["supported"]
    r = recommend(get_model("smollm2-135m-instruct"), small, 0.0, 400, "general", None, n_val=50)
    assert r["method"] == "lora" and r["config"]["max_seq_len"] == 512 and r["config"]["lora_targets"] == "all_linear"
    assert r["autotune"]["trials"] >= 2
    assert TrainConfig(**r["config"]).max_seq_len == 512  # recommendation is a valid config
    assert validate(r["method"], TrainConfig(**r["config"]), family="smollm2")[0] == []


def test_api_llm_project_upload_analyze_prepare_and_gating(client):
    pid = client.post("/api/projects", json={"name": "llm", "task": "llm"}).json()["id"]
    assert client.get("/api/models?task=llm").status_code == 200
    bad = client.post(f"/api/projects/{pid}/datasets", data={"name": "d"}, files={"file": ("d.wav", b"x")})
    assert bad.status_code == 422
    nocols = client.post(f"/api/projects/{pid}/datasets", data={"name": "d"}, files={"file": ("d.jsonl", jsonl([{"foo": 1}]))})
    assert nocols.status_code == 422 and "prompt column" in nocols.json()["detail"]
    ds = client.post(f"/api/projects/{pid}/datasets", data={"name": "d"}, files={"file": ("d.jsonl", jsonl(ROWS))})
    assert ds.status_code == 201
    ds_id = ds.json()["id"]
    assert client.post(f"/api/datasets/{ds_id}/analyze").status_code == 202
    for _ in range(60):
        d = client.get(f"/api/datasets/{ds_id}").json()
        if d["status"] == "analyzed":
            break
        assert d["status"] != "failed", d["error"]
        time.sleep(0.25)
    assert d["analysis"]["examples"] == 120
    assert client.post(f"/api/datasets/{ds_id}/prepare", json={"exclude_flags": ["missing_audio"]}).status_code == 422
    assert client.post(f"/api/datasets/{ds_id}/prepare", json={"augment": ["noise"]}).status_code == 422
    assert client.post(f"/api/datasets/{ds_id}/prepare", json={}).status_code == 202
    for _ in range(60):
        d = client.get(f"/api/datasets/{ds_id}").json()
        if d["status"] == "prepared":
            break
        assert d["status"] != "failed", d["error"]
        time.sleep(0.25)
    assert d["prepared"]["counts"]["train"] > 0
    # speech models refuse a text dataset; unsupported / invalid LLM settings are refused with reasons
    assert client.post("/api/runs", json={"dataset_id": ds_id, "model_id": "whisper-tiny", "method": "lora"}).status_code == 422
    big = client.post("/api/runs", json={"dataset_id": ds_id, "model_id": "gemma-3-1b-it", "method": "full"})
    assert big.status_code == 422 and "unsupported" in big.json()["detail"].lower()
    clash = client.post("/api/runs", json={"dataset_id": ds_id, "model_id": "smollm2-135m-instruct", "method": "lora",
                                           "config": {"lora_scope": "encoder"}})
    assert clash.status_code == 422 and "decoder-only" in clash.json()["detail"]
    rec = client.post("/api/strategy/recommend", json={"model_id": "smollm2-135m-instruct", "dataset_id": ds_id}).json()
    assert rec["method"] == "lora" and "Perplexity" in rec["metric"]
    meta = client.get("/api/meta").json()
    assert any(t["id"] == "llm" for t in meta["tasks"]) and [f["id"] for f in meta["flags"]["llm"]] == text.TEXT_FLAGS
    assert [m["key"] for m in meta["metrics"]["llm"]][0] == "perplexity"


def test_plugin_trains_and_evaluates_a_tiny_llama_on_cpu(tmp_path):
    """Real training through causal_lm's collate / forward / fit / evaluation code, on whichever device is available, with a tiny random Llama."""
    tok_dir = settings.data_dir / "models_cache" / "HuggingFaceTB--SmolLM2-135M-Instruct"
    if not (tok_dir / "COMPLETE").exists():
        pytest.skip("SmolLM2 tokenizer not mirrored locally (python scripts/fetch_model.py HuggingFaceTB/SmolLM2-135M-Instruct)")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from app.training.common import Progress, device, fit, to_device
    from app.training.methods.apply import apply_method
    from app.training.methods.params import param_groups
    from app.training.plugins import causal_lm as plug

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256, pad_token_id=tok.pad_token_id or 0, tie_word_embeddings=True)
    dev = device()
    model = LlamaForCausalLM(cfg).to(dev)
    tc = TrainConfig(epochs=25, batch_size=8, grad_accum=1, learning_rate=3e-3, precision="fp32", early_stopping=False,
                     lora_r=8, lora_alpha=16, lora_targets="all_linear", max_seq_len=96, max_new_tokens=16, warmup_steps=2,
                     lr_scheduler="cosine", ema_decay=0.9, max_eval_samples=8)
    model, _ = apply_method(model, "lora", tc, "smollm2", 0)
    assert any("lora_B" in n and "mlp.down_proj" in n for n, _ in model.named_parameters())  # adapted feed-forward layers
    data = [{"prompt": f"say {i % 6}", "response": f"number {i % 6}"} for i in range(48)]
    ctx = plug.Ctx({"family": "SmolLM2"}, tc, tmp_path, Progress(tmp_path), dev, data[:36], data[36:42], data[42:], tok, plug.detect_format(tok))
    assert ctx.fmt == "chat"
    batch = plug._collate_fn(ctx, tc)(data[:2])
    assert (batch["labels"] != -100).sum() < batch["attention_mask"].sum()  # the prompt is masked out of the loss
    before = plug._teacher_forced(model, ctx, tc, data[42:])
    forward, val_forward = plug._forward_fns(tc)
    info = fit(model, ctx.train, ctx.val, plug._collate_fn(ctx, tc), forward, tc, tmp_path, Progress(tmp_path), lambda: None,
               groups=param_groups(model, tc), val_forward=val_forward,
               ema=plug.EMA(model, tc.ema_decay))
    after = plug._teacher_forced(model, ctx, tc, data[42:])
    assert after["perplexity"] < before["perplexity"] * 0.5 and after["token_accuracy"] > before["token_accuracy"]
    assert info["trainable_params"] > 0 and info["best_val_loss"] is not None
    ev_all = plug._evaluate(model, ctx, tc)
    assert {"perplexity", "token_accuracy", "exact_match", "rouge_l", "examples"} <= set(ev_all) and ev_all["n"] == len(data[42:])
    smooth = TrainConfig(**{**tc.model_dump(), "label_smoothing": 0.1})  # the label-smoothing loss path runs and is finite
    loss = plug._forward_fns(smooth)[0](model, to_device(batch, dev))
    assert torch.isfinite(loss) and loss.item() > 0
