import { useEffect, useState } from "react";
import { api, type Gpu, type Meta, type Project, type Task } from "./api";
import CompareTab from "./CompareTab";
import DashboardTab from "./DashboardTab";
import DataTab from "./DataTab";
import ModelsTab from "./ModelsTab";
import RegistryTab from "./RegistryTab";
import RunsTab from "./RunsTab";
import TrainTab from "./TrainTab";
import { Err, MetaContext, useAsyncError, usePoll } from "./ui";

const TABS = ["Dashboard", "Models", "Data", "Train", "Runs", "Compare", "Registry"] as const;
type Tab = (typeof TABS)[number];
const TASK_ICON: Record<string, string> = { stt: "🎙", tts: "🔊", llm: "✎" };

function GpuBadge() {
  const [gpu, setGpu] = useState<Gpu | null>(null);
  usePoll(() => api.gpu().then(setGpu), 4000, true);
  if (!gpu) return <span className="gpu">Detecting GPU…</span>;
  if (!gpu.available) return <span className="gpu warn">No CUDA GPU detected</span>;
  return <span className="gpu">{gpu.name} · {gpu.total_gb} GB ({gpu.free_gb} GB free) · {gpu.utilization_pct}%</span>;
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaErr, setMetaErr] = useState("");
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [tab, setTab] = useState<Tab>("Dashboard");
  const [name, setName] = useState("");
  const [task, setTask] = useState<Task>("stt");
  const { err, wrap } = useAsyncError();

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setMetaErr(`Cannot reach the VoxTune backend: ${e.message}`));
    api.projects().then((p) => { setProjects(p); setProject((cur) => cur ?? p[0] ?? null); }).catch(() => undefined);
  }, []);

  const create = wrap(async () => {
    const p = await api.createProject(name || `${task.toUpperCase()} project`, task);
    setProjects((l) => [p, ...l]); setProject(p); setName(""); setTab("Data");
  });

  if (metaErr) return <div className="shell"><Err msg={metaErr} /></div>;
  if (!meta) return <div className="shell"><p className="muted">Connecting to the backend…</p></div>;

  const needProject = tab !== "Dashboard" && tab !== "Registry";
  return (
    <MetaContext.Provider value={meta}>
      <div className="shell">
        <header>
          <h1>VoxTune <span className="muted">Fine-Tuning Studio</span></h1>
          <GpuBadge />
        </header>
        <Err msg={err} />

        <div className="projectbar">
          <label>Project
            <select value={project?.id ?? ""} onChange={(e) => setProject(projects.find((p) => p.id === +e.target.value) ?? null)}>
              {projects.length === 0 && <option value="">No projects yet</option>}
              {projects.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.task.toUpperCase()})</option>)}
            </select>
          </label>
          <div className="row">
            <input placeholder="New project name" value={name} onChange={(e) => setName(e.target.value)} />
            <div className="seg">
              {meta.tasks.map((t) => (
                <button key={t.id} className={task === t.id ? "sel" : ""} onClick={() => setTask(t.id)} title={t.families.join(", ")}>
                  {TASK_ICON[t.id]} {t.label}
                </button>
              ))}
            </div>
            <button className="primary" onClick={create}>Create</button>
          </div>
        </div>

        <nav>
          {TABS.map((t) => <button key={t} className={tab === t ? "sel" : ""} onClick={() => setTab(t)}>{t}</button>)}
        </nav>

        {tab === "Dashboard" && <DashboardTab project={project} />}
        {tab === "Registry" && <RegistryTab />}
        {needProject && !project && <p className="muted">Create a project to begin: choose {meta.tasks.map((t) => t.label).join(" or ")} above.</p>}
        {needProject && project && (
          <>
            {tab === "Models" && <ModelsTab project={project} />}
            {tab === "Data" && <DataTab project={project} />}
            {tab === "Train" && <TrainTab project={project} onStarted={() => setTab("Runs")} />}
            {tab === "Runs" && <RunsTab project={project} focus={null} />}
            {tab === "Compare" && <CompareTab project={project} />}
          </>
        )}
      </div>
    </MetaContext.Provider>
  );
}
