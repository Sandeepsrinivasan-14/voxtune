import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Dashboard, type LiveGpu, type Live, type Project, type SysInfo } from "./api";
import { Bar, Err, Section, Stat, dur, fmtMetric, toneFor, useAsyncError, useMeta, usePoll } from "./ui";

const HISTORY = 90; // samples kept in the live chart (2 s apart)
const STORAGE_LABELS: Record<string, string> = {
  uploads: "Uploads", datasets: "Datasets", runs: "Runs", registry: "Model registry", model_mirror: "Model mirror",
  database: "Database", hf_cache: "HF cache",
};

interface Sample { t: string; cpu: number; ram: number; gpu?: number; vram?: number }

function GpuCard({ g }: { g: LiveGpu }) {
  const vramPct = (g.mem_used_gb / g.mem_total_gb) * 100;
  return (
    <div className="card">
      <div className="card-head"><h3>GPU {g.index} · {g.name}</h3><span className="muted">{g.clock_mhz != null ? `${g.clock_mhz} MHz` : ""}</span></div>
      <Bar label="Utilization" value={(g.util_pct ?? 0) / 100} text={g.util_pct != null ? `${g.util_pct}%` : "n/a"} tone={toneFor(g.util_pct ?? 0)} />
      <Bar label="Memory" value={vramPct / 100} text={`${g.mem_used_gb} / ${g.mem_total_gb} GB`} tone={toneFor(vramPct)} />
      <Bar label="Temperature" value={(g.temp_c ?? 0) / 100} text={g.temp_c != null ? `${g.temp_c} °C` : "n/a"} tone={g.temp_c != null ? toneFor(g.temp_c) : undefined} />
      <Bar label="Power" value={g.power_w != null && g.power_limit_w ? g.power_w / g.power_limit_w : 0}
        text={g.power_w != null ? `${g.power_w} / ${g.power_limit_w ?? "?"} W` : "n/a"} />
    </div>
  );
}

export default function DashboardTab({ project }: { project: Project | null }) {
  const meta = useMeta();
  const [live, setLive] = useState<Live | null>(null);
  const [info, setInfo] = useState<SysInfo | null>(null);
  const [dash, setDash] = useState<Dashboard | null>(null);
  const [hist, setHist] = useState<Sample[]>([]);
  const [scoped, setScoped] = useState(false);
  const { err, wrap } = useAsyncError();
  const pid = scoped ? project?.id : undefined;

  usePoll(wrap(async () => {
    const l = await api.live();
    setLive(l);
    const g = l.gpus[0];
    setHist((h) => [...h.slice(-(HISTORY - 1)), {
      t: new Date(l.t * 1000).toLocaleTimeString(), cpu: l.cpu.percent, ram: l.ram.percent,
      gpu: g?.util_pct ?? undefined, vram: g ? Math.round((g.mem_used_gb / g.mem_total_gb) * 100) : undefined,
    }]);
  }), 2000);
  usePoll(wrap(async () => setDash(await api.dashboard(pid))), 3000, true);
  usePoll(wrap(async () => setInfo(await api.sysInfo())), 60000);

  const a = dash?.queue.active;
  const taskLabel = (id: string) => meta.tasks.find((t) => t.id === id)?.label ?? id;

  return (
    <>
      <Err msg={err} />
      <Section title="This machine (live)" aside={<span className="muted small">refreshes every 2 s</span>}>
        {!live ? <p className="muted">Reading hardware…</p> : (
          <>
            <div className="grid">
              {live.gpus.length === 0 && <div className="warnbox">No NVIDIA GPU detected (NVML unavailable). Training would fall back to CPU and most methods are marked unsupported.</div>}
              {live.gpus.map((g) => <GpuCard key={g.index} g={g} />)}
              <div className="card">
                <div className="card-head"><h3>CPU</h3><span className="muted">{live.cpu.freq_mhz ? `${live.cpu.freq_mhz} MHz` : ""}</span></div>
                <p className="muted small">{info ? `${info.cpu.name} · ${info.cpu.physical_cores} cores / ${info.cpu.logical_cores} threads` : "…"}</p>
                <Bar label="Usage" value={live.cpu.percent / 100} text={`${live.cpu.percent}%`} tone={toneFor(live.cpu.percent)} />
                <Bar label="RAM" value={live.ram.percent / 100} text={`${live.ram.used_gb} / ${live.ram.total_gb} GB`} tone={toneFor(live.ram.percent)} />
                <Bar label="Disk" value={live.disk.percent / 100} text={`${live.disk.free_gb} GB free`} tone={toneFor(live.disk.percent)} />
                <p className="muted small">Workspace: {live.disk.path}</p>
              </div>
            </div>
            <div className="chart">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={hist}>
                  <CartesianGrid stroke="var(--line)" />
                  <XAxis dataKey="t" stroke="var(--muted)" tick={false} />
                  <YAxis stroke="var(--muted)" domain={[0, 100]} unit="%" />
                  <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)" }} />
                  <Legend />
                  <Line dataKey="cpu" name="CPU" stroke="#7aa2f7" dot={false} isAnimationActive={false} />
                  <Line dataKey="ram" name="RAM" stroke="#bb9af7" dot={false} isAnimationActive={false} />
                  {live.gpus.length > 0 && <Line dataKey="gpu" name="GPU" stroke="#4cc2a5" dot={false} isAnimationActive={false} />}
                  {live.gpus.length > 0 && <Line dataKey="vram" name="VRAM" stroke="#e0a84a" dot={false} isAnimationActive={false} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </Section>

      <Section title="Training" aside={<span className="muted small">{dash ? `${dash.queue.queued} queued` : ""}</span>}>
        {!dash ? <p className="muted">Loading…</p> : !a ? <p className="muted">No training is running right now.</p> : (
          <>
            <h3 style={{ marginTop: 0 }}>Run #{a.id} · {a.name}</h3>
            {a.total_steps ? (
              <Bar label={`Step ${a.step} / ${a.total_steps}`} value={(a.step ?? 0) / a.total_steps} />
            ) : <p className="muted">{a.phase ?? "Starting…"}</p>}
            <div className="stats">
              <Stat label="Training loss" value={a.loss ?? "—"} />
              <Stat label="Validation loss" value={a.val_loss ?? "—"} />
              <Stat label="Elapsed" value={dur(a.elapsed)} sub={a.eta != null ? `ETA ${dur(a.eta)}` : undefined} />
              <Stat label="Phase" value={a.phase ?? "Training"} />
            </div>
          </>
        )}
      </Section>

      <Section title="Workspace" aside={
        <label className="inline muted small"><input type="checkbox" checked={scoped} disabled={!project} onChange={(e) => setScoped(e.target.checked)} /> only the selected project</label>}>
        {!dash ? <p className="muted">Loading…</p> : (
          <>
            <div className="stats">
              <Stat label="Projects" value={dash.counts.projects} />
              <Stat label="Datasets" value={dash.counts.datasets} sub={Object.entries(dash.datasets.by_status).map(([k, v]) => `${v} ${k}`).join(", ") || undefined} />
              <Stat label="Audio analyzed" value={`${dash.datasets.audio_hours.toFixed(2)} h`} sub={`${dash.datasets.recordings} recordings`} />
              <Stat label="Training data prepared" value={`${dash.datasets.train_hours.toFixed(2)} h`} />
              <Stat label="Runs" value={dash.counts.runs} sub={Object.entries(dash.runs.by_status).map(([k, v]) => `${v} ${k}`).join(", ") || undefined} />
              <Stat label="Time spent training" value={`${dash.runs.train_hours.toFixed(2)} h`} />
              <Stat label="Registered models" value={dash.counts.registered_models} />
            </div>
            {Object.keys(dash.runs.by_method).length > 0 && (
              <p className="muted small">Runs by method: {Object.entries(dash.runs.by_method).map(([k, v]) => `${k} × ${v}`).join(" · ")}</p>
            )}
          </>
        )}
      </Section>

      <Section title="Recent runs">
        {!dash || dash.recent_runs.length === 0 ? <p className="muted">No runs yet.</p> : (
          <table>
            <thead><tr><th>#</th><th>Run</th><th>Task</th><th>Status</th><th>Time</th><th>GPU peak</th><th>Headline metric</th></tr></thead>
            <tbody>
              {dash.recent_runs.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td><td>{r.name}</td><td>{taskLabel(r.task)}</td>
                  <td><span className={`badge ${r.status}`}>{r.status}</span></td>
                  <td>{dur(r.seconds)}</td><td>{r.peak_gpu_gb != null ? `${r.peak_gpu_gb} GB` : "—"}</td>
                  <td className="num">{r.metric.tuned == null ? "—" : `${r.metric.label}: ${fmtMetric(r.metric, r.metric.base)} → ${fmtMetric(r.metric, r.metric.tuned)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>

      <div className="two">
        <Section title="Disk usage by VoxTune">
          {!dash ? <p className="muted">Loading…</p> : Object.entries(dash.storage_gb).filter(([k]) => k !== "total").map(([k, v]) => (
            <Bar key={k} label={STORAGE_LABELS[k] ?? k} value={dash.storage_gb.total ? v / dash.storage_gb.total : 0} text={`${v} GB`} />
          ))}
          {dash && <p className="muted small">Total in workspace: {dash.storage_gb.total} GB (Hugging Face cache is separate)</p>}
        </Section>
        <Section title="Models on this machine">
          {!dash || dash.cached_models.length === 0 ? <p className="muted">No model weights downloaded yet. They download on first use.</p> : (
            <table>
              <tbody>
                {dash.cached_models.map((m) => (
                  <tr key={m.hf_id}><td>{m.label}</td><td className="muted">{m.where}</td><td className="num">{m.size_gb} GB</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>

      <Section title="GPU memory measured from real runs">
        {!dash || dash.observed_vram.length === 0 ? (
          <p className="muted">Nothing measured yet. Each finished run records the memory it really used on this GPU, and that replaces the registry's estimate.</p>
        ) : (
          <table>
            <thead><tr><th>Model</th><th>Method</th><th>Used (incl. CUDA context)</th></tr></thead>
            <tbody>{dash.observed_vram.map((o) => <tr key={o.model_id + o.method}><td>{o.model_id}</td><td>{o.method}</td><td className="num">{o.used_gb} GB</td></tr>)}</tbody>
          </table>
        )}
      </Section>

      <Section title="Environment">
        {!info ? <p className="muted">Loading…</p> : (
          <>
            <div className="stats">
              <Stat label="OS" value={info.os.split(" (")[0]} />
              <Stat label="Python" value={info.python} />
              <Stat label="PyTorch" value={info.torch.version ?? "not installed"} sub={info.torch.cuda_available ? `CUDA ${info.torch.cuda_runtime}` : "no CUDA"} />
              <Stat label="bf16 support" value={info.torch.bf16 ? "yes" : "no"} sub={info.torch.devices?.[0] ? `compute ${info.torch.devices[0].compute_capability}` : undefined} />
              <Stat label="System RAM" value={`${info.ram_total_gb} GB`} />
            </div>
            <p className="muted small">
              {Object.entries(info.packages).map(([k, v]) => `${k} ${v ?? "—"}`).join(" · ")}
            </p>
          </>
        )}
      </Section>
    </>
  );
}
