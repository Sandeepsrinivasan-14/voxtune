import { Fragment, useState } from "react";
import { api, type RegModel } from "./api";
import { Err, Section, dur, fmtMetric, useAsyncError, useMeta, usePoll } from "./ui";

export default function RegistryTab() {
  const meta = useMeta();
  const [models, setModels] = useState<RegModel[]>([]);
  const { err, wrap } = useAsyncError();
  const refresh = () => api.registry().then(setModels);
  usePoll(refresh, 5000, true);

  const groups = models.reduce<Record<string, RegModel[]>>((g, m) => ((g[m.base_model_id] ??= []).push(m), g), {});
  return (
    <>
      <Err msg={err} />
      {models.length === 0 && <Section title="Model registry"><p className="muted">No models yet. Finish a run and use “Save to model registry”.</p></Section>}
      {Object.entries(groups).map(([base, list]) => (
        <Section key={base} title={base}>
          <div className="grid">
            {list.map((m) => {
              const t = m.meta.tuned_metrics ?? {}, b = m.meta.base_metrics ?? {};
              const defs = (meta.metrics[m.meta.task as "stt" | "tts"] ?? []).filter((d) => d.compare && t[d.key] != null);
              return (
                <div className="card" key={m.id}>
                  <div className="card-head"><h3>{m.name}-v{m.version}</h3><span className="muted">{new Date(m.created_at).toLocaleDateString()}</span></div>
                  <dl>
                    <dt>Base model</dt><dd>{m.meta.base_hf_id}</dd>
                    <dt>Method</dt><dd>{meta.methods[m.method] ?? m.method}</dd>
                    <dt>Dataset</dt><dd>{m.dataset_name} {m.meta.dataset_hours?.train ? `(${m.meta.dataset_hours.train} h train)` : ""}</dd>
                    {defs.map((d) => (
                      <Fragment key={d.key}><dt>{d.label}</dt><dd>{fmtMetric(d, t[d.key])} <span className="muted">(base {fmtMetric(d, b[d.key])})</span></dd></Fragment>
                    ))}
                    <dt>Training</dt><dd>{dur(m.meta.train_seconds)} on {m.meta.gpu}</dd>
                  </dl>
                  <div className="row">
                    <a className="btn" href={`/api/registry/${m.id}/export?format=adapter`}>Export checkpoint</a>
                    {meta.merged_export_methods.includes(m.method) && (
                      <a className="btn" href={`/api/registry/${m.id}/export?format=merged`}>Export merged model</a>
                    )}
                    <button onClick={wrap(async () => { await api.deleteModel(m.id); await refresh(); })}>Delete</button>
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      ))}
    </>
  );
}
