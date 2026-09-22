import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type AutotuneTrial, type Ev, type Project, type Run, type Task } from "./api";
import { Err, Section, Stat, dur, fmtMetric, useAsyncError, useMeta, usePoll } from "./ui";

const LIVE = ["queued", "running"];

function Monitor({ run, task, onChange }: { run: Run; task: Task; onChange: () => void }) {
  const [events, setEvents] = useState<Ev[]>([]);
  const [next, setNext] = useState(0);
  const [log, setLog] = useState("");
  const { err, wrap } = useAsyncError();

  useEffect(() => { setEvents([]); setNext(0); setLog(""); }, [run.id]);
  usePoll(async () => {
    const p = await api.progress(run.id, next);
    // place events by absolute position so overlapping/duplicate polls can't double-append
    if (p.events.length) setEvents((e) => [...e.slice(0, p.next - p.events.length), ...p.events]);
    setNext(p.next);
    if (!LIVE.includes(p.status) && LIVE.includes(run.status)) onChange();
    if (run.status === "failed" || LIVE.includes(run.status)) api.log(run.id).then((l) => setLog(l.log));
  }, 2000, true);

  const train = events.filter((e) => e.type === "train");
  const evals = events.filter((e) => e.type === "eval");
  const last = train[train.length - 1];
  const start = events.find((e) => e.type === "start");
  const status = [...events].reverse().find((e) => e.type === "status")?.message;
  const base = events.find((e) => e.type === "base_eval");
  const tuned = events.find((e) => e.type === "tuned_eval");
  const oom = events.filter((e) => e.type === "oom_retry");
  const lrFind = events.find((e) => e.type === "lr_find");
  const tuneEvents = events.filter((e) => e.type === "autotune_trial");
  const tuneDone = events.find((e) => e.type === "autotune_done");
  const tuneReport = run.metrics?.autotune;
  const trials: AutotuneTrial[] = tuneReport?.trials ?? (tuneEvents as unknown as AutotuneTrial[]);
  const info = run.metrics?.train;
  const chart = train.map((t) => {
    const v = evals.filter((e) => e.step <= t.step).pop();
    return { step: t.step, train: t.loss, val: v?.val_loss };
  });
  const m = run.metrics;
  const defs = useMeta().metrics[m?.kind ?? task];
  const b = m?.base ?? base, t = m?.tuned ?? tuned;

  return (
    <Section title={`Run #${String(run.id).padStart(4, "0")} — ${run.name}`} aside={
      LIVE.includes(run.status) ? <button onClick={wrap(async () => { await api.stop(run.id); onChange(); })}>Stop</button> : undefined}>
      <Err msg={err} />
      <div className="stats">
        <Stat label="Status" value={<span className={`badge ${run.status}`}>{run.status}</span>} sub={status} />
        <Stat label="Step" value={last ? `${last.step} / ${last.total_steps}` : start ? `0 / ${start.total_steps}` : "—"} sub={last ? `epoch ${last.epoch}` : undefined} />
        <Stat label="Training loss" value={last?.loss ?? "—"} />
        <Stat label="Validation loss" value={evals.length ? evals[evals.length - 1].val_loss : "—"} sub={evals.length ? `best ${evals[evals.length - 1].best}` : undefined} />
        <Stat label="GPU memory" value={last ? `${last.gpu_used_gb} / ${last.gpu_total_gb} GB` : "—"} sub={last ? `${last.gpu_util_pct}% util · peak ${last.gpu_peak_gb} GB` : undefined} />
        <Stat label="Elapsed" value={dur(last?.elapsed ?? run.train_seconds)} sub={LIVE.includes(run.status) && last ? `ETA ${dur(last.eta)}` : undefined} />
      </div>
      {oom.map((o, i) => <div className="warnbox" key={i}>Out of memory: batch reduced to {o.new_batch} (accumulation {o.grad_accum}) and restarted.</div>)}
      {(run.warnings ?? []).map((w) => <div className="warnbox" key={w}>⚠ {w}</div>)}
      {info && (
        <p className="muted small">
          {info.trainable_params != null && <>Trainable parameters: {Number(info.trainable_params).toLocaleString()} · </>}
          Best {info.monitor ?? "val_loss"}: {info.best_val_loss ?? "—"} · Scheduler: {info.scheduler ?? "—"} · Final learning rate: {info.learning_rate ?? "—"}
          {info.early_stopped ? " · stopped early" : ""}
        </p>
      )}
      {lrFind && (
        <>
          <h3>Learning-rate finder</h3>
          <p className="muted small">
            Default {lrFind.default} → chosen {lrFind.chosen}
            {lrFind.raw_suggestion != null && ` (curve suggested ${Number(lrFind.raw_suggestion).toExponential(2)}${lrFind.clamped ? ", limited to a safe range" : ""})`}.
            The weights were restored after the test.
          </p>
          <div className="chart">
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={((lrFind.curve as [number, number][] | undefined) ?? []).map(([lr, loss]) => ({ lr, loss }))}>
                <CartesianGrid stroke="var(--line)" />
                <XAxis dataKey="lr" scale="log" type="number" domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toExponential(0)} stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
                <Line type="monotone" dataKey="loss" name="Loss" stroke="#7aa2f7" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
      {trials.length > 0 && (
        <>
          <h3>Auto-tune {tuneDone ? "result" : "progress"}</h3>
          <table>
            <thead><tr><th>Round</th><th>Candidate</th><th>Settings changed</th><th>Val loss</th><th>Val WER</th><th></th></tr></thead>
            <tbody>
              {trials.map((t) => (
                <tr key={t.trial}>
                  <td>{t.round} ({t.budget_steps} steps)</td>
                  <td>{t.candidate === 0 ? "baseline" : `#${t.candidate}`}</td>
                  <td className="small">{Object.keys(t.overrides).length ? Object.entries(t.overrides).map(([k, v]) => `${k}=${v}`).join(", ") : "your settings"}</td>
                  <td className="num">{t.status === "ok" ? t.metrics.val_loss : "failed"}</td>
                  <td className="num">{t.metrics.val_wer ?? "—"}</td>
                  <td>{t.survived ? "kept" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {tuneReport && (
            <p className="muted small">
              {tuneReport.improved_over_baseline ? "A tuned candidate beat the baseline at equal budget." : "The baseline was best or tied; its settings were kept."} {tuneReport.note}
            </p>
          )}
        </>
      )}
      {chart.length > 0 && (
        <div className="chart">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chart}>
              <CartesianGrid stroke="var(--line)" />
              <XAxis dataKey="step" stroke="var(--muted)" />
              <YAxis stroke="var(--muted)" />
              <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
              <Legend />
              <Line type="monotone" dataKey="train" name="Training loss" stroke="#4cc2a5" dot={false} />
              <Line type="monotone" dataKey="val" name="Validation loss" stroke="#e0a84a" dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {run.error && <div className="error">{run.error}</div>}
      {b && t && (
        <>
          <h3>Base model vs fine-tuned</h3>
          <table>
            <thead><tr><th>Metric</th><th>Base</th><th>Fine-tuned</th></tr></thead>
            <tbody>
              {defs.filter((d) => b[d.key] != null || t[d.key] != null).map((d) => (
                <tr key={d.key}><td>{d.label} <span className="muted small">({d.higher_better ? "higher" : "lower"} is better)</span></td>
                  <td className="num">{fmtMetric(d, b[d.key])}</td>
                  <td className="num">{fmtMetric(d, t[d.key])}</td></tr>
              ))}
            </tbody>
          </table>
          {t.mos_note && <p className="muted small">MOS: {t.mos_note}</p>}
        </>
      )}
      {m?.kind === "llm" && m.tuned?.examples?.length > 0 && (
        <>
          <h3>Sample outputs on the held-out test split</h3>
          <table>
            <thead><tr><th>Prompt</th><th>Expected</th><th>Base model said</th><th>Fine-tuned said</th></tr></thead>
            <tbody>
              {m.tuned.examples.map((e: any, i: number) => (
                <tr key={i}>
                  <td className="small">{e.prompt}</td><td className="small">{e.ref}</td>
                  <td className="small">{m.base?.examples?.[i]?.hyp ?? "—"}</td><td className="small">{e.hyp}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Prompt format: {m.prompt_format === "chat" ? "the model's own chat template" : "plain instruction/response layout"}. The loss covers response tokens only.</p>
        </>
      )}
      {m?.samples_audio && m.samples_audio.length > 0 && (
        <>
          <h3>Listen</h3>
          {m.samples_audio.map((s: any) => (
            <div className="audio" key={s.base}>
              <p>“{s.text}”</p>
              <label>Base <audio controls src={`/api/runs/${run.id}/files/${s.base}`} /></label>
              <label>Fine-tuned <audio controls src={`/api/runs/${run.id}/files/${s.tuned}`} /></label>
            </div>
          ))}
        </>
      )}
      {run.status === "completed" && <RegisterBox run={run} />}
      {log && (run.status === "failed" || LIVE.includes(run.status)) && <pre className="log">{log}</pre>}
    </Section>
  );
}

function RegisterBox({ run }: { run: Run }) {
  const [name, setName] = useState("");
  const [done, setDone] = useState("");
  const { err, wrap } = useAsyncError();
  return (
    <div className="row">
      <Err msg={err} />
      <input placeholder={`Model name (default ${run.model_id}-${run.method})`} value={name} onChange={(e) => setName(e.target.value)} />
      <button className="primary" onClick={wrap(async () => { const m = await api.register(run.id, name || undefined); setDone(`Registered ${m.name} v${m.version}`); })}>
        Save to model registry
      </button>
      {done && <span className="ok-text">{done}</span>}
    </div>
  );
}

export default function RunsTab({ project, focus }: { project: Project; focus: number | null }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [sel, setSel] = useState<number | null>(focus);
  const primary = useMeta().metrics[project.task].find((d) => d.primary)!;
  const refresh = () => api.runs(project.id).then((r) => { setRuns(r); setSel((s) => s ?? r[0]?.id ?? null); });
  usePoll(refresh, 3000, true);
  const cur = runs.find((r) => r.id === sel);
  return (
    <>
      <Section title="Experiments">
        {runs.length === 0 && <p className="muted">No runs yet. Start one from the Train tab.</p>}
        <table>
          <thead><tr><th>#</th><th>Name</th><th>Status</th><th>Time</th><th>GPU peak</th><th>{primary.label}</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className={r.id === sel ? "sel" : ""} onClick={() => setSel(r.id)}>
                <td>{r.id}</td><td>{r.name}{r.group && <span className="muted"> · batch {r.group}</span>}</td>
                <td><span className={`badge ${r.status}`}>{r.status}</span></td>
                <td>{dur(r.train_seconds)}</td><td>{r.peak_gpu_gb ? `${r.peak_gpu_gb} GB` : "—"}</td>
                <td className="num">{fmtMetric(primary, r.metrics?.tuned[primary.key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
      {cur && <Monitor key={cur.id} run={cur} task={project.task} onChange={refresh} />}
    </>
  );
}
