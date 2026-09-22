import { useEffect, useState } from "react";
import { api, type Dataset, type ModelInfo, type Project, type Recommendation } from "./api";
import { Err, Section, useAsyncError, useMeta } from "./ui";

export default function TrainTab({ project, onStarted }: { project: Project; onStarted: () => void }) {
  const meta = useMeta();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [dsId, setDsId] = useState<number | "">("");
  const [modelId, setModelId] = useState("");
  const [goal, setGoal] = useState("general");
  const [terms, setTerms] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [method, setMethod] = useState("");
  const [cfg, setCfg] = useState<Record<string, any>>({});
  const [cmp, setCmp] = useState<string[]>([]);
  const [tune, setTune] = useState(false);
  const [tuneTrials, setTuneTrials] = useState(8);
  const [tuneSteps, setTuneSteps] = useState(24);
  const { err, wrap } = useAsyncError();
  const goals = meta.goals.filter((g) => g.tasks.includes(project.task));

  useEffect(() => {
    api.datasets(project.id).then((l) => setDatasets(l.filter((d) => d.prepared)));
    api.models(project.task).then((m) => { setModels(m); setModelId((cur) => cur || m[0]?.id || ""); });
  }, [project.id, project.task]);
  useEffect(() => { setRec(null); }, [dsId, modelId, goal, terms]);

  const model = models.find((m) => m.id === modelId);
  const termList = terms.split(",").map((t) => t.trim()).filter(Boolean);
  const supported = model ? Object.keys(model.methods).filter((m) => model.support[m]?.supported) : [];
  const ready = dsId !== "" && !!model;

  const recommend = wrap(async (force?: string) => {
    if (!ready) throw new Error("Pick a prepared dataset and a model first.");
    const r = await api.recommend({ model_id: modelId, dataset_id: dsId, goal, domain_terms: termList, method: force });
    setRec(r);
    if (r.method) {
      setMethod(r.method); setCfg(r.config ?? {});
      if (r.autotune) { setTuneTrials(r.autotune.trials); setTuneSteps(r.autotune.budget_steps); }
    }
  });

  const autotune = tune ? { trials: tuneTrials, budget_steps: tuneSteps } : undefined;
  const body = (extra: object) => ({ dataset_id: dsId, model_id: modelId, goal, autotune, ...extra });
  const start = wrap(async () => {
    if (!ready) throw new Error("Pick a prepared dataset and a model first.");
    if (advanced && rec) await api.startRun(body({ method, config: { ...cfg, domain_terms: termList } }));
    else await api.startRun(body({ auto: true, config: { domain_terms: termList } }));
    onStarted();
  });
  const compare = wrap(async () => {
    await api.startBatch(body({ auto: true, methods: cmp, config: { domain_terms: termList } }));
    onStarted();
  });

  // Advanced fields come from the backend's TrainConfig and are shown when they apply to the chosen method.
  const fields = meta.train_fields.filter((f) => f.applies.includes(method) && f.tasks.includes(project.task));
  const val = (f: { key: string; default: any }) => cfg[f.key] ?? f.default;
  const info = meta.method_info[method];

  return (
    <>
      <Err msg={err} />
      <Section title="What are we training?" aside={
        <div className="seg">
          <button className={!advanced ? "sel" : ""} onClick={() => setAdvanced(false)}>Beginner</button>
          <button className={advanced ? "sel" : ""} onClick={() => setAdvanced(true)}>Advanced</button>
        </div>}>
        <div className="form">
          <label>Dataset
            <select value={dsId} onChange={(e) => setDsId(e.target.value ? +e.target.value : "")}>
              <option value="">Select a prepared dataset…</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.prepared!.counts.train} train {project.task === "llm" ? "examples" : "clips"})</option>)}
            </select>
          </label>
          <label>Model{model?.gated && <span className="muted small"> — gated: accept the license on Hugging Face first</span>}
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.id} — {m.params_m}M params{m.local.cached ? "" : " (will download)"}</option>)}
            </select>
          </label>
          <label>Goal
            <select value={goal} onChange={(e) => setGoal(e.target.value)}>
              {goals.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
            </select>
          </label>
          {goal === "domain" && (
            <label>Domain terms (comma separated)
              <input value={terms} placeholder="gastroparesis, esophagitis, omeprazole" onChange={(e) => setTerms(e.target.value)} />
            </label>
          )}
        </div>
        {model && (
          <div className="methods-inline">
            {Object.keys(model.methods).map((m) => {
              const s = model.support[m];
              return (
                <span key={m} className={s?.supported ? "ok" : "no"} title={s?.reason}>
                  {meta.methods[m]}: {s?.supported ? (s.verified ? `verified · ${s.observed_gb} GB` : "available") : "unsupported"}
                </span>
              );
            })}
          </div>
        )}
      </Section>

      <Section title="Recommended strategy" aside={<button onClick={() => recommend()}>Get recommendation</button>}>
        {!rec && <p className="muted">The strategy engine picks a method and settings from your dataset size, this machine's GPU and your goal, and explains why. You can override it in Advanced mode.</p>}
        {rec && !rec.method && <div className="warnbox">{rec.explanation[0]}</div>}
        {rec?.method && (
          <>
            <h3>✓ {meta.methods[rec.method]}</h3>
            <ul className="why">{rec.explanation.map((x) => <li key={x}>{x}</li>)}</ul>
            {rec.warnings.map((w) => <div className="warnbox" key={w}>⚠ {w}</div>)}
            <p className="muted small">Evaluation metric: {rec.metric}</p>
            {info && (
              <p className="muted small">
                <b>{info.label}</b> trains: {info.trains} Best for: {info.best_for} Memory: {info.memory}. Output: {info.artifact}.
              </p>
            )}
          </>
        )}
      </Section>

      {advanced && rec?.method && (
        <Section title="Training configuration">
          <div className="form">
            <label>Method
              <select value={method} onChange={(e) => { setMethod(e.target.value); recommend(e.target.value); }}>
                {supported.map((m) => <option key={m} value={m}>{meta.methods[m]}</option>)}
              </select>
            </label>
            {fields.filter((f) => f.kind !== "boolean").map((f) => (
              <label key={f.key} title={f.help ?? undefined}>{f.label}{f.help ? ` (${f.help})` : ""}
                {f.kind === "select" ? (
                  <select value={val(f)} onChange={(e) => setCfg({ ...cfg, [f.key]: e.target.value })}>
                    {f.options!.map((o) => <option key={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" min={f.min ?? undefined} value={val(f)} onChange={(e) => setCfg({ ...cfg, [f.key]: +e.target.value })} />
                )}
              </label>
            ))}
            {fields.filter((f) => f.kind === "boolean").map((f) => (
              <label className="inline" key={f.key} title={f.help ?? undefined}>
                <input type="checkbox" checked={!!val(f)} onChange={(e) => setCfg({ ...cfg, [f.key]: e.target.checked })} /> {f.label}
              </label>
            ))}
          </div>
          <p className="muted small">Checkpoints are always saved (best by the selected metric). Incompatible combinations are rejected with the reason when you start.</p>
        </Section>
      )}

      <Section title="Auto-tune (optional)">
        <label className="inline">
          <input type="checkbox" checked={tune} onChange={(e) => setTune(e.target.checked)} /> Search for the best settings on my data before the final run
        </label>
        {rec?.autotune && <p className="muted small">{rec.autotune.reason}{rec.autotune.suggested ? "" : " Your dataset is small, so results would be noisy."}</p>}
        {tune && (
          <div className="form">
            <label>Candidates to try (2-32)
              <input type="number" min={2} max={32} value={tuneTrials} onChange={(e) => setTuneTrials(+e.target.value)} />
            </label>
            <label>Steps per candidate in round 1
              <input type="number" min={4} max={500} value={tuneSteps} onChange={(e) => setTuneSteps(+e.target.value)} />
            </label>
          </div>
        )}
      </Section>

      <div className="actions">
        <button className="primary big" onClick={start}>Start fine-tuning</button>
        {advanced && !rec && <span className="muted small">Advanced mode: get a recommendation first to edit its settings.</span>}
      </div>

      <Section title="Compare methods on the same dataset">
        <p className="muted small">Runs each selected method sequentially with settings tuned for that method, then compares accuracy, time and GPU memory. No method is called best universally.</p>
        <div className="checks">
          {supported.map((m) => (
            <label key={m}><input type="checkbox" checked={cmp.includes(m)} onChange={() => setCmp(cmp.includes(m) ? cmp.filter((x) => x !== m) : [...cmp, m])} /> {meta.methods[m]}</label>
          ))}
        </div>
        <button disabled={cmp.length < 2} onClick={compare}>Run comparison ({cmp.length})</button>
      </Section>
    </>
  );
}
