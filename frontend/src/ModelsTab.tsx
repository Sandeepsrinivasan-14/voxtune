import { useEffect, useState } from "react";
import { api, type ModelInfo, type Project } from "./api";
import { Section, useMeta } from "./ui";

export default function ModelsTab({ project }: { project: Project }) {
  const meta = useMeta();
  const [models, setModels] = useState<ModelInfo[]>([]);
  useEffect(() => { api.models(project.task).then(setModels); }, [project.task]);
  return (
    <Section title="Model capability registry">
      <p className="muted small">
        Availability is checked live against this machine's GPU. GPU memory figures are conservative estimates until a real run
        finishes; after that the measured value is used instead and the option shows as verified.
      </p>
      <div className="grid">
        {models.map((m) => (
          <div className="card" key={m.id}>
            <div className="card-head"><h3>{m.id}</h3><span className="muted">{m.params_m} M params</span></div>
            <p className="muted">{m.family} · {m.license}{m.gated ? " · gated (license must be accepted on Hugging Face)" : ""}</p>
            <p className="small">
              {m.local.cached
                ? <span className="ok-text">Downloaded · {m.local.size_gb} GB · {m.local.where}</span>
                : <span className="muted">Not downloaded yet (fetched on first use)</span>}
            </p>
            <ul className="methods">
              {Object.keys(m.methods).map((meth) => {
                const s = m.support[meth];
                const obs = m.observed_gb[meth];
                return (
                  <li key={meth} className={s?.supported ? "ok" : "no"}>
                    <span title={meta.method_info[meth] ? `${meta.method_info[meth].trains} Best for: ${meta.method_info[meth].best_for}` : undefined}>
                      {meta.methods[meth] ?? meth}
                    </span>
                    <span className="need">{obs != null ? `measured ${obs} GB` : `~${m.methods[meth]} GB est.`}</span>
                    <span className="state">{s?.supported ? (s.verified ? "Verified" : "Available") : "Unsupported"}</span>
                    {!s?.supported && <em>{s?.reason}</em>}
                  </li>
                );
              })}
            </ul>
            {m.notes && <p className="muted small">{m.notes}</p>}
          </div>
        ))}
      </div>
    </Section>
  );
}
