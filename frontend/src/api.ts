export type Task = "stt" | "tts" | "llm";

export interface Gpu { available: boolean; name: string; total_gb: number; free_gb: number; utilization_pct: number }
export interface Support { supported: boolean; reason: string; verified: boolean; observed_gb: number | null }
export interface LocalModel { cached: boolean; where: string | null; size_gb: number }
export interface ModelInfo {
  id: string; task: Task; family: string; hf_id: string; params_m: number;
  methods: Record<string, number>; license: string; notes: string; support: Record<string, Support>;
  observed_gb: Record<string, number>; local: LocalModel; gated?: boolean;
}
export interface Project { id: number; name: string; task: Task; created_at: string }
export interface Analysis {
  total_recordings: number; total_hours: number; avg_duration_sec: number; speakers: number;
  speaker_hours: Record<string, number>; scripts: string; sample_rates: Record<string, number>; vocabulary: number;
  quality: Record<string, number>; flag_counts: Record<string, number>; scores: Record<string, number>; warnings: string[];
  // text (LLM) datasets
  examples?: number; avg_prompt_chars?: number; avg_response_chars?: number; p95_chars?: number;
}
export interface Prepared { counts: Record<string, number>; hours: Record<string, number>; dropped: number; excluded_flags: string[]; augmentations: string[]; chars?: Record<string, number> }
export interface Dataset {
  id: number; project_id: number; name: string; task: Task; status: string; language: string;
  analysis: Analysis | null; prepared: Prepared | null; error: string | null;
}
export interface AutotuneSuggestion { suggested: boolean; trials: number; budget_steps: number; reason: string }
export interface Recommendation {
  method: string | null; config: Record<string, any> | null; explanation: string[]; warnings: string[];
  alternatives: Record<string, Support>; metric: string | null; autotune?: AutotuneSuggestion;
}
export type Metrics = Record<string, any>;
export interface AutotuneTrial {
  trial: number; candidate: number; round: number; budget_steps: number; overrides: Record<string, any>; status: string;
  metrics: Record<string, any>; survived?: boolean;
}
export interface AutotuneReport {
  metric: string; trials: AutotuneTrial[]; winner: { candidate: number; overrides: Record<string, any> };
  baseline_round1: Record<string, any>; winner_round1: Record<string, any>; improved_over_baseline: boolean; seconds: number; note: string;
}
export interface Run {
  id: number; dataset_id: number; model_id: string; method: string; name: string; group: string | null; status: string;
  config: Record<string, any> | null; warnings?: string[]; autotune?: Record<string, any> | null;
  metrics: { kind: Task; base: Metrics; tuned: Metrics; samples_audio?: any[]; autotune?: AutotuneReport; train?: Record<string, any>; prompt_format?: string } | null;
  train_seconds: number | null; peak_gpu_gb: number | null; error: string | null;
}
export interface Ev { type: string; [k: string]: any }
export interface Compare {
  kind: Task; columns: string[]; base: Metrics | null; leaders: Record<string, number>; note: string;
  rows: { run_id: number; name: string; method: string; status: string; metrics: Metrics; train_seconds: number | null; peak_gpu_gb: number | null; trainable_params: number | null; quantized_base: boolean; base_metrics: Metrics }[];
}
export interface RegModel {
  id: number; name: string; version: number; base_model_id: string; method: string; dataset_name: string; created_at: string; meta: Record<string, any>;
}

// ---- backend-defined UI vocabulary (GET /api/meta) ----
export interface MetricDef { key: string; label: string; fmt: "pct" | "num"; higher_better: boolean; compare: boolean; primary: boolean }
export interface FlagSpec { id: string; label: string; default_exclude: boolean; kind: string; threshold: string | null }
export interface FieldSpec { key: string; label: string; kind: "number" | "select" | "boolean"; options: string[] | null; help: string | null; default: any; min: number | null; applies: string[]; tasks: Task[] }
export interface MethodInfo { label: string; memory: string; artifact: string; trains: string; best_for: string }
export interface Meta {
  tasks: { id: Task; label: string; families: string[] }[];
  methods: Record<string, string>;
  method_info: Record<string, MethodInfo>;
  merged_export_methods: string[];
  goals: { id: string; label: string; tasks: Task[] }[];
  augmentations: { id: string; label: string }[];
  flags: Record<Task, FlagSpec[]>;
  quality_labels: Record<string, string>;
  score_labels: Record<string, string>;
  train_fields: FieldSpec[];
  metrics: Record<Task, MetricDef[]>;
}

// ---- real host / workspace stats ----
export interface LiveGpu {
  index: number; name: string; util_pct: number | null; mem_used_gb: number; mem_total_gb: number;
  temp_c: number | null; power_w: number | null; power_limit_w: number | null; clock_mhz: number | null;
}
export interface Live {
  t: number; cpu: { percent: number; freq_mhz: number | null };
  ram: { used_gb: number; total_gb: number; percent: number };
  disk: { path: string; used_gb: number; total_gb: number; free_gb: number; percent: number };
  gpus: LiveGpu[];
}
export interface SysInfo {
  os: string; python: string; cpu: { name: string; physical_cores: number; logical_cores: number }; ram_total_gb: number;
  torch: { installed: boolean; version?: string; cuda_runtime?: string; cuda_available?: boolean; bf16?: boolean; devices?: { index: number; name: string; compute_capability: string }[] };
  packages: Record<string, string | null>; data_dir: string;
}
export interface Dashboard {
  counts: { projects: number; datasets: number; runs: number; registered_models: number };
  datasets: { by_status: Record<string, number>; audio_hours: number; recordings: number; train_hours: number };
  runs: { by_status: Record<string, number>; train_hours: number; by_method: Record<string, number> };
  queue: { queued: number; active: null | { id: number; name: string; model_id: string; method: string; phase: string | null; step: number | null; total_steps: number | null; epoch: number | null; loss: number | null; val_loss: number | null; elapsed: number | null; eta: number | null } };
  recent_runs: { id: number; name: string; status: string; model_id: string; method: string; task: Task; seconds: number | null; peak_gpu_gb: number | null; created_at: string;
    metric: { key: string; label: string; fmt: "pct" | "num"; higher_better: boolean; base: number | null; tuned: number | null } }[];
  storage_gb: Record<string, number>;
  cached_models: { hf_id: string; label: string; where: string; size_gb: number }[];
  observed_vram: { model_id: string; method: string; used_gb: number }[];
}

const json = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = await r.text();
    try { const d = JSON.parse(msg).detail; msg = typeof d === "string" ? d : JSON.stringify(d); } catch { /* keep raw text */ }
    throw new Error(msg);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

export const api = {
  meta: () => req<Meta>("/api/meta"),
  gpu: () => req<Gpu>("/api/system/gpu"),
  live: () => req<Live>("/api/system/live"),
  sysInfo: () => req<SysInfo>("/api/system/info"),
  dashboard: (pid?: number) => req<Dashboard>(`/api/dashboard${pid ? `?project_id=${pid}` : ""}`),
  models: (task: Task) => req<ModelInfo[]>(`/api/models?task=${task}`),
  projects: () => req<Project[]>("/api/projects"),
  createProject: (name: string, task: Task) => req<Project>("/api/projects", json({ name, task })),
  datasets: (pid: number) => req<Dataset[]>(`/api/projects/${pid}/datasets`),
  upload: (pid: number, name: string, language: string, file: File) => {
    const f = new FormData();
    f.append("name", name); f.append("language", language); f.append("file", file);
    return req<Dataset>(`/api/projects/${pid}/datasets`, { method: "POST", body: f });
  },
  analyze: (id: number) => req<Dataset>(`/api/datasets/${id}/analyze`, { method: "POST" }),
  prepare: (id: number, body: object) => req<Dataset>(`/api/datasets/${id}/prepare`, json(body)),
  items: (id: number, flag: string) => req<{ total: number; items: any[] }>(`/api/datasets/${id}/items?flag=${flag}&limit=50`),
  recommend: (body: object) => req<Recommendation>("/api/strategy/recommend", json(body)),
  startRun: (body: object) => req<Run>("/api/runs", json(body)),
  startBatch: (body: object) => req<{ group: string; runs: Run[] }>("/api/runs/batch", json(body)),
  runs: (pid: number) => req<Run[]>(`/api/runs?project_id=${pid}`),
  progress: (id: number, after: number) => req<{ status: string; next: number; events: Ev[] }>(`/api/runs/${id}/progress?after=${after}`),
  log: (id: number) => req<{ log: string }>(`/api/runs/${id}/log`),
  stop: (id: number) => req<unknown>(`/api/runs/${id}/stop`, { method: "POST" }),
  compare: (ids: number[]) => req<Compare>(`/api/experiments/compare?run_ids=${ids.join(",")}`),
  register: (run_id: number, name?: string) => req<RegModel>("/api/registry", json({ run_id, name })),
  registry: () => req<RegModel[]>("/api/registry"),
  deleteModel: (id: number) => req<void>(`/api/registry/${id}`, { method: "DELETE" }),
};
