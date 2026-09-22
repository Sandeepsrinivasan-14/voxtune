"""Drive the running API end to end with REAL training: upload -> analyze -> prepare -> recommend -> train -> compare -> register -> export.

usage: python scripts/e2e.py stt whisper-tiny lora,adapter [--terms a,b] [--augment noise]
       python scripts/e2e.py tts speecht5-tts speaker_adaptation
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("model")
ap.add_argument("methods")
ap.add_argument("--terms", default="")
ap.add_argument("--augment", default="")
ap.add_argument("--goal", default="general")
ap.add_argument("--epochs", type=int, default=0)
ap.add_argument("--export", default="adapter")
a = ap.parse_args()
c = httpx.Client(base_url="http://127.0.0.1:8000/api", timeout=600)


def call(method, url, **kw):
    r = c.request(method, url, **kw)
    if r.status_code >= 400:
        sys.exit(f"{method} {url} -> {r.status_code} {r.text}")
    return r.json() if r.content else None


def wait_ds(ds_id, want):
    while True:
        d = call("GET", f"/datasets/{ds_id}")
        if d["status"] == want:
            return d
        if d["status"] == "failed":
            sys.exit(f"dataset failed: {d['error']}")
        time.sleep(2)


p = call("POST", "/projects", json={"name": f"e2e {a.task} {a.model}", "task": a.task})
zp = ROOT / "data" / "samples" / f"sample_{a.task}.zip"
with open(zp, "rb") as f:
    ds = call("POST", f"/projects/{p['id']}/datasets", data={"name": zp.stem, "language": "en"}, files={"file": (zp.name, f)})
call("POST", f"/datasets/{ds['id']}/analyze")
an = wait_ds(ds["id"], "analyzed")["analysis"]
print("ANALYSIS", json.dumps({k: an[k] for k in ("total_recordings", "total_hours", "speakers", "quality", "scores", "warnings")}, indent=1))
call("POST", f"/datasets/{ds['id']}/prepare", json={"augment": [x for x in a.augment.split(",") if x]})
prep = wait_ds(ds["id"], "prepared")["prepared"]
print("PREPARED", prep["counts"], prep["hours"])
terms = [t for t in a.terms.split(",") if t]
rec = call("POST", "/strategy/recommend", json={"model_id": a.model, "dataset_id": ds["id"], "goal": a.goal, "domain_terms": terms})
print("RECOMMEND", rec["method"], json.dumps(rec["config"]))
print("\n".join(" - " + x for x in rec["explanation"] + rec["warnings"]))

methods = a.methods.split(",")
cfg = {"domain_terms": terms}
if a.epochs:
    cfg["epochs"] = a.epochs
body = {"dataset_id": ds["id"], "model_id": a.model, "auto": True, "goal": a.goal, "config": cfg}
if len(methods) > 1:
    runs = call("POST", "/runs/batch", json={**body, "methods": methods})["runs"]
else:
    runs = [call("POST", "/runs", json={**body, "method": methods[0]})]
print("RUNS", [(r["id"], r["method"]) for r in runs])

seen = {r["id"]: 0 for r in runs}
pending = set(seen)
while pending:
    for rid in list(pending):
        pr = call("GET", f"/runs/{rid}/progress", params={"after": seen[rid]})
        seen[rid] = pr["next"]
        for e in pr["events"]:
            if e["type"] in ("start", "status", "base_eval", "tuned_eval", "eval", "oom_retry", "error"):
                print(f"[run {rid}]", {k: v for k, v in e.items() if k not in ("t", "examples")})
            elif e["type"] == "train" and e["step"] % 10 == 0:
                print(f"[run {rid}] step {e['step']}/{e['total_steps']} loss {e['loss']} gpu {e.get('gpu_used_gb')}GB eta {e['eta']}s")
        if pr["status"] in ("completed", "failed", "stopped"):
            pending.discard(rid)
    time.sleep(3)

done = [r for r in runs if call("GET", f"/runs/{r['id']}")["status"] == "completed"]
for r in runs:
    rr = call("GET", f"/runs/{r['id']}")
    print("RESULT", r["id"], rr["method"], rr["status"], rr["error"] or "")
if done:
    cmp = call("GET", "/experiments/compare", params={"run_ids": ",".join(str(r["id"]) for r in done)})
    print("BASE", cmp["base"])
    for row in cmp["rows"]:
        print("ROW", row["method"], row["metrics"], f"{row['train_seconds']}s", f"{row['peak_gpu_gb']}GB", "params", row["trainable_params"])
    print("LEADERS", cmp["leaders"])
    m = call("POST", "/registry", json={"run_id": done[0]["id"]})
    print("REGISTERED", m["name"], "v", m["version"])
    z = c.get(f"/registry/{m['id']}/export", params={"format": a.export})
    print("EXPORT", z.status_code, len(z.content), "bytes")
