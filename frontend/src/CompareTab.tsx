import { useState } from "react";
import { api, type Compare, type Project, type Run } from "./api";
import { Err, Section, dur, fmtMetric, useAsyncError, useMeta, usePoll } from "./ui";

export default function CompareTab({ project }: { project: Project }) {
  const meta = useMeta();
  const [runs, setRuns] = useState<Run[]>([]);
  const [ids, setIds] = useState<number[]>([]);
  const [cmp, setCmp] = useState<Compare | null>(null);
  const { err, wrap } = useAsyncError();
  usePoll(() => api.runs(project.id).then((r) => setRuns(r.filter((x) => x.status === "completed"))), 5000, true);

  const toggle = (id: number) => setIds(ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]);
  const lead = (col: string, id: number) => (cmp?.leaders[col] === id ? "leader" : "");
  const defs = cmp ? cmp.columns.map((c) => meta.metrics[cmp.kind].find((d) => d.key === c)!) : [];

  return (
    <>
      <Err msg={err} />
      <Section title="Pick completed runs to compare" aside={
        <button className="primary" disabled={ids.length < 1} onClick={wrap(async () => setCmp(await api.compare(ids)))}>Compare ({ids.length})</button>}>
        {runs.length === 0 && <p className="muted">No completed runs yet.</p>}
        <div className="checks">
          {runs.map((r) => (
            <label key={r.id}><input type="checkbox" checked={ids.includes(r.id)} onChange={() => toggle(r.id)} /> #{r.id} {r.name}</label>
          ))}
        </div>
      </Section>
      {cmp && (
        <Section title="Comparison">
          <table>
            <thead>
              <tr><th>Method</th>{defs.map((d) => <th key={d.key}>{d.label}</th>)}<th>Training time</th><th>GPU memory</th><th>Trainable params</th></tr>
            </thead>
            <tbody>
              {cmp.base && (
                <tr className="base"><td>Base model</td>{defs.map((d) => <td key={d.key} className="num">{fmtMetric(d, cmp.base![d.key])}</td>)}<td>—</td><td>—</td><td>—</td></tr>
              )}
              {cmp.rows.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.method} <span className="muted">#{r.run_id}</span>
                    {r.quantized_base && <span className="muted small" title="This run's own base evaluation used the 4-bit quantized base model"> · 4-bit base: {fmtMetric(defs[0], r.base_metrics[defs[0].key])}</span>}
                  </td>
                  {defs.map((d) => <td key={d.key} className={`num ${lead(d.key, r.run_id)}`}>{r.status === "completed" ? fmtMetric(d, r.metrics[d.key]) : r.status}</td>)}
                  <td className={`num ${lead("train_seconds", r.run_id)}`}>{dur(r.train_seconds)}</td>
                  <td className={`num ${lead("peak_gpu_gb", r.run_id)}`}>{r.peak_gpu_gb != null ? `${r.peak_gpu_gb} GB` : "—"}</td>
                  <td className="num">{r.trainable_params?.toLocaleString() ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Highlighted cells lead their own column only. {cmp.note}</p>
        </Section>
      )}
    </>
  );
}
