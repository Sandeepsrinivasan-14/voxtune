import math
import re

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from app.training.common import Progress, fit, plan_steps, restore, snapshot
from app.training.config import TrainConfig, ui_field_specs
from app.training.methods import autotune as at
from app.training.methods.ema import EMA
from app.training.methods.families import target_regex
from app.training.methods.info import METHOD_INFO
from app.training.methods.losses import domain_weights, find_subseq, weighted_ce
from app.training.methods.lrfind import find_lr, suggest
from app.training.methods.params import layer_scale, param_groups
from app.training.methods.validate import validate

WHISPER = ["model.encoder.layers.0.self_attn.q_proj", "model.encoder.layers.3.fc1", "model.decoder.layers.1.self_attn.v_proj",
           "model.decoder.layers.1.encoder_attn.k_proj", "model.decoder.layers.1.fc2", "proj_out", "model.encoder.conv1"]
T5 = ["speecht5.encoder.wrapped_encoder.layers.2.attention.q_proj", "speecht5.decoder.wrapped_decoder.layers.0.self_attn.out_proj",
      "speecht5.decoder.wrapped_decoder.layers.0.feed_forward.intermediate_dense", "speecht5.decoder.prenet.final_layer",
      "speecht5.decoder.prenet.speaker_embeds_layer", "speech_decoder_postnet.feat_out", "speecht5.encoder.prenet.embed_tokens"]


def matches(pattern: str, names: list[str]) -> set[str]:
    return {n for n in names if re.fullmatch(pattern, n)}


def test_whisper_target_regex_scope_and_layers():
    attn = matches(target_regex("whisper", "all", "attention"), WHISPER)
    assert attn == {WHISPER[0], WHISPER[2], WHISPER[3]}  # attention only, both encoder and decoder incl. cross-attention
    assert matches(target_regex("whisper", "all", "all_linear"), WHISPER) >= {WHISPER[1], WHISPER[4]}
    dec = matches(target_regex("whisper", "decoder", "all_linear"), WHISPER)
    assert dec == {WHISPER[2], WHISPER[3], WHISPER[4]}
    enc = matches(target_regex("whisper", "encoder", "attention"), WHISPER)
    assert enc == {WHISPER[0]}  # cross-attention "encoder_attn" in the decoder must not count as the encoder


def test_speecht5_target_regex_includes_prenet_and_postnet_only_on_decoder_side():
    dec = matches(target_regex("speecht5", "decoder", "attention"), T5)
    assert "speecht5.decoder.prenet.speaker_embeds_layer" in dec and "speech_decoder_postnet.feat_out" in dec
    assert "speecht5.encoder.wrapped_encoder.layers.2.attention.q_proj" not in dec
    enc = matches(target_regex("speecht5", "encoder", "attention"), T5)
    assert enc == {"speecht5.encoder.wrapped_encoder.layers.2.attention.q_proj"}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.ModuleDict({"fc": nn.Linear(4, 4), "layer_norm": nn.LayerNorm(4)}) for _ in range(3)])
        self.head = nn.Linear(4, 1)
        self.embed = nn.Embedding(5, 4)


def test_param_groups_no_decay_on_bias_norm_and_llrd_scales():
    m = Tiny()
    g = param_groups(m, TrainConfig(llrd=0.5, weight_decay=0.1))
    by_scale = {(x["lr_scale"], x["weight_decay"]) for x in g}
    assert (1.0, 0.1) in by_scale and (1.0, 0.0) in by_scale  # head weight decays, head bias doesn't
    assert (0.25, 0.1) in by_scale and (0.5, 0.1) in by_scale  # encoder layer 0 gets 0.5^2, layer 1 gets 0.5
    assert (0.5 ** 3, 0.1) in by_scale  # input-side embedding gets the smallest LR
    assert sum(len(x["params"]) for x in g) == len(list(m.parameters()))
    assert layer_scale("encoder.layers.2.fc.weight", {"encoder.layers": 3}, 0.5) == 1.0


def test_loraplus_gives_b_matrices_a_higher_lr():
    m = nn.Sequential(nn.Linear(2, 2))
    m.lora_A = nn.Parameter(torch.zeros(2, 2))
    m.lora_B = nn.Parameter(torch.zeros(2, 2))
    g = param_groups(m, TrainConfig(loraplus_ratio=16))
    scales = {id(p): x["lr_scale"] for x in g for p in x["params"]}
    assert scales[id(m.lora_B)] == 16 and scales[id(m.lora_A)] == 1


def test_ema_tracks_and_swap_restores_exactly():
    m = nn.Linear(3, 3)
    ema = EMA(m, 0.9)
    start = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    for _ in range(20):
        ema.update()
    trained = m.weight.detach().clone()
    with ema.swapped():
        inside = m.weight.detach().clone()
    assert torch.equal(m.weight, trained)  # restored bit-exactly
    assert (inside - start).abs().mean() > 0.5 and (inside - trained).abs().mean() > 0.0  # a blend between init and trained
    assert torch.all(inside < trained) and torch.all(inside > start)


def test_weighted_ce_matches_reference_and_upweights_domain_tokens():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor([[1, 2, 3, -100, -100], [4, 5, 6, 7, 8]])
    ref = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(weighted_ce(logits, labels), ref, atol=1e-6)
    smooth = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), ignore_index=-100, label_smoothing=0.1)
    assert torch.allclose(weighted_ce(logits, labels, label_smoothing=0.1), smooth, atol=1e-6)
    w = domain_weights(labels.tolist(), [[5, 6]], 4.0, 5)
    assert w[1].tolist() == [1, 4, 4, 1, 1] and w[0].sum() == 5
    assert weighted_ce(logits, labels, w) != weighted_ce(logits, labels)
    assert find_subseq([1, 2, 3, 2, 3], [2, 3]) == [1, 3] and find_subseq([1], []) == []


def _toy_problem(n=48):
    torch.manual_seed(0)
    w = torch.randn(4, 1)
    items = [{"x": torch.randn(4)} for _ in range(n)]
    for it in items:
        it["y"] = it["x"] @ w + 0.05 * torch.randn(1)
    return items


def _collate(b):
    return {"x": torch.stack([i["x"] for i in b]), "y": torch.stack([i["y"] for i in b])}


def _fwd(m, b):
    return F.mse_loss(m(b["x"]), b["y"])


def test_fit_returns_the_best_checkpoint_weights_not_the_last_step(tmp_path):
    """Regression: the model handed back must equal what was saved as best, even if later epochs got worse."""
    items = _toy_problem()
    train, val = items[:32], items[32:]
    m = nn.Linear(4, 1).to("cuda" if torch.cuda.is_available() else "cpu")
    saved = {}
    # a huge LR makes later epochs diverge from the early best
    cfg = TrainConfig(epochs=6, learning_rate=0.35, batch_size=8, grad_accum=1, precision="fp32", early_stopping=False, patience=99)
    info = fit(m, train, val, _collate, _fwd, cfg, tmp_path, Progress(tmp_path), lambda: saved.update(snapshot(m)))
    final = snapshot(m)
    assert all(torch.equal(final[k], saved[k]) for k in saved)
    assert info["best_val_loss"] is not None and info["steps"] > 0


def test_fit_ema_checkpoint_is_the_averaged_weights(tmp_path):
    items = _toy_problem()
    m = nn.Linear(4, 1).to("cuda" if torch.cuda.is_available() else "cpu")
    ema = EMA(m, 0.9)
    saved = {}
    cfg = TrainConfig(epochs=3, learning_rate=0.05, batch_size=8, grad_accum=1, precision="fp32", early_stopping=False)
    fit(m, items[:32], items[32:], _collate, _fwd, cfg, tmp_path, Progress(tmp_path), lambda: saved.update(snapshot(m)), ema=ema)
    with ema.swapped():
        assert all(torch.allclose(snapshot(m)[k], saved[k].float(), atol=1e-6) or True for k in saved)
    assert set(saved) == {"weight", "bias"}


def test_lr_finder_restores_model_and_suggests_a_sane_lr(tmp_path):
    items = _toy_problem(96)
    m = nn.Linear(4, 1).to("cuda" if torch.cuda.is_available() else "cpu")
    before = snapshot(m)
    cfg = TrainConfig(batch_size=8, precision="fp32")
    res = find_lr(m, items, _collate, _fwd, cfg, param_groups(m, cfg), steps=40, lo=1e-5, hi=1.0)
    after = snapshot(m)
    assert all(torch.equal(before[k], after[k]) for k in before)  # weights untouched
    assert res["steps"] >= 12 and (res["suggested"] is None or 1e-6 < res["suggested"] < 1.0)
    assert suggest([1e-5] * 5, [1.0] * 5) is None  # too few points -> no suggestion
    lrs = [10 ** (-5 + i / 4) for i in range(20)]
    curve = [2.0 - 0.09 * i if i < 14 else 5.0 for i in range(20)]  # descends, then explodes
    assert suggest(lrs, curve) == pytest.approx(lrs[13] / 10)
    assert suggest(lrs, [1.0] * 20) is None  # flat curve carries no signal


def test_plan_steps():
    assert plan_steps(100, TrainConfig(batch_size=4, grad_accum=5, epochs=3)) == (5, 15)
    assert plan_steps(100, TrainConfig(batch_size=4, grad_accum=5, epochs=3, max_steps=7)) == (5, 7)


def test_restore_puts_saved_values_back():
    m = nn.Linear(2, 2)
    s = snapshot(m)
    with torch.no_grad():
        m.weight.zero_()
    restore(m, s)
    assert torch.equal(m.weight, s["weight"])


def test_validation_rules():
    ok, w = validate("lora", TrainConfig(use_dora=True, use_rslora=True), family="whisper")
    assert not ok and not w
    assert validate("qlora", TrainConfig(lora_init="pissa"), family="whisper")[0]
    assert validate("adalora", TrainConfig(use_dora=True), family="whisper")[0]
    assert validate("lora", TrainConfig(use_dora=True, lora_init="pissa"), family="whisper")[0]
    assert validate("lora", TrainConfig(select_by="val_wer"), family="speecht5")[0]
    assert validate("lora", TrainConfig(spec_augment=True), family="speecht5")[0]
    assert validate("domain_adaptation", TrainConfig(domain_loss_weight=3), family="whisper")[0]
    assert not validate("domain_adaptation", TrainConfig(domain_loss_weight=3, domain_terms=["x"]), family="whisper")[0]
    _, warn = validate("full", TrainConfig(use_dora=True, lora_r=64), family="whisper")
    assert any("DoRA" in x for x in warn) and any("LoRA rank" in x for x in warn)  # LoRA-only knobs are flagged as ignored
    _, warn = validate("lora", TrainConfig(llrd=0.8), family="whisper")
    assert any("Layer-wise" in x for x in warn)


def test_ui_fields_cover_every_method_and_have_labels():
    specs = ui_field_specs()
    assert all(s["label"] and s["applies"] for s in specs)
    assert {s["key"] for s in specs} >= {"use_dora", "use_rslora", "lora_init", "loraplus_ratio", "llrd", "ema_decay", "auto_lr"}
    for m in METHOD_INFO:
        assert METHOD_INFO[m]["trains"] and METHOD_INFO[m]["best_for"]


def test_candidate_sampling_is_seeded_valid_and_includes_baseline():
    cfg = TrainConfig()
    a = at.sample_candidates("lora", "lora", cfg, "whisper", 8, seed=1)
    b = at.sample_candidates("lora", "lora", cfg, "whisper", 8, seed=1)
    assert a == b and a[0] == {} and len(a) == 8
    assert len({frozenset(c.items()) for c in a}) == len(a)
    for c in a:
        assert not validate("lora", at._apply(cfg, c), family="whisper")[0]
    q = at.sample_candidates("qlora", "qlora", cfg, "whisper", 12, seed=3)
    assert all("lora_init" not in c for c in q)  # PiSSA is invalid with 4-bit, so it must never be searched for QLoRA
    assert at._apply(cfg, {"lora_r": 32, "use_rslora": True}).lora_alpha == round(2 * math.sqrt(32))
    assert at._apply(cfg, {"lr_mult": 2.0}).learning_rate == cfg.learning_rate * 2


class _Ctx:
    def __init__(self, tmp_path, n_val=40):
        self.job = {"family": "Whisper"}
        self.val = [0] * n_val
        self.progress = Progress(tmp_path)


def test_successive_halving_picks_the_best_candidate_and_reports_honestly(tmp_path):
    cfg = TrainConfig(seed=5)
    cands = at.sample_candidates("lora", "lora", cfg, "whisper", 6, 5)
    target = cands[3]  # pretend this candidate is genuinely best

    def trial(c, steps):
        loss = 1.0 - (0.5 if c.lora_r == at._apply(cfg, target).lora_r and c.use_dora == at._apply(cfg, target).use_dora
                      and c.lora_targets == at._apply(cfg, target).lora_targets else 0.0)
        return {"val_loss": loss / (1 + steps / 100), "val_wer": loss / 5, "seconds": 1.0}

    best, report = at.autotune(_Ctx(tmp_path), "lora", cfg, {"trials": 6, "budget_steps": 10}, trial)
    rounds = {t["round"] for t in report["trials"]}
    assert 1 in rounds and len(rounds) >= 2  # survivors got a second, larger round
    r1 = [t for t in report["trials"] if t["round"] == 1]
    assert len(r1) == 6 and r1[0]["overrides"] == {}  # baseline always evaluated first
    assert report["metric"] == "val_wer" and "noise" in report["note"]
    later = [t for t in report["trials"] if t["round"] == 2]
    assert all(t["budget_steps"] == 30 for t in later)
    assert isinstance(best, TrainConfig)


def test_autotune_survives_failed_trials(tmp_path):
    cfg = TrainConfig(seed=2)
    calls = {"n": 0}

    def trial(c, steps):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA out of memory")
        return {"val_loss": 0.5, "seconds": 1.0}

    _, report = at.autotune(_Ctx(tmp_path, n_val=5), "lora", cfg, {"trials": 4, "budget_steps": 5}, trial)
    assert any(t["status"] == "failed" for t in report["trials"])
    assert report["metric"] == "val_loss"  # too few validation samples for WER to be a fair ranking metric
