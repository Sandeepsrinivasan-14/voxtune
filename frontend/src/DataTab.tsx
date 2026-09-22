import { useState } from "react";
import { api, type Dataset, type Project } from "./api";
import { Bar, Err, Section, Stat, useAsyncError, useMeta, usePoll } from "./ui";

export default function DataTab({ project }: { project: Project }) {
  const meta = useMeta();
  const flags = meta.flags[project.task];
  const [list, setList] = useState<Dataset[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [lang, setLang] = useState("en");
  const [file, setFile] = useState<File | null>(null);
  const [excl, setExcl] = useState<string[]>(flags.filter((f) => f.default_exclude).map((f) => f.id));
  const [augs, setAugs] = useState<string[]>([]);
  const [val, setVal] = useState(10);
  const [test, setTest] = useState(10);
  const [flagged, setFlagged] = useState<{ flag: string; items: any[]; total: number } | null>(null);
  const { err, wrap } = useAsyncError();

  const refresh = () => api.datasets(project.id).then(setList);
  const cur = list.find((d) => d.id === sel) ?? null;
  const busy = list.some((d) => d.status === "analyzing" || d.status === "preparing");
  usePoll(refresh, 2000, true);

  const toggle = (arr: string[], set: (v: string[]) => void, k: string) =>
    set(arr.includes(k) ? arr.filter((x) => x !== k) : [...arr, k]);
  const flagLabel = (id: string) => flags.find((f) => f.id === id)?.label ?? id;
  const isText = project.task === "llm";

  const upload = wrap(async () => {
    if (!file) throw new Error(isText ? "Choose a .jsonl, .csv, .tsv or .zip file first." : "Choose a .zip file first.");
    const d = await api.upload(project.id, name || file.name.replace(/\.(zip|jsonl|csv|tsv)$/i, ""), lang, file);
    setSel(d.id); setFile(null); setName("");
    await refresh();
  });

  return (
    <>
      <Err msg={err} />
      <Section title="Upload dataset">
        {isText ? (
          <p className="muted small">
            A <code>.jsonl</code>, <code>.csv</code> or <code>.tsv</code> file (or a .zip containing one) with a prompt column
            (<code>prompt</code>, <code>instruction</code>, <code>question</code>…) and a response column (<code>response</code>, <code>output</code>,
            <code>answer</code>…). Alpaca-style <code>instruction</code> + <code>input</code> rows are joined into one prompt.
          </p>
        ) : (
          <p className="muted small">
            A .zip with audio files plus <code>metadata.csv</code> / <code>.jsonl</code> (columns: <code>audio,text</code>, optional <code>speaker</code>),
            or audio files with same-named <code>.txt</code> transcripts.
          </p>
        )}
        <div className="row">
          <input placeholder="Dataset name" value={name} onChange={(e) => setName(e.target.value)} />
          {!isText && <input className="narrow" placeholder="Language (en)" value={lang} onChange={(e) => setLang(e.target.value)} />}
          <input type="file" accept={isText ? ".zip,.jsonl,.csv,.tsv" : ".zip"} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <button className="primary" onClick={upload}>Upload</button>
        </div>
      </Section>

      <Section title="Datasets">
        {list.length === 0 && <p className="muted">No datasets yet.</p>}
        <div className="chips">
          {list.map((d) => (
            <button key={d.id} className={`chip ${sel === d.id ? "sel" : ""}`} onClick={() => { setSel(d.id); setFlagged(null); }}>
              {d.name} <span className={`badge ${d.status}`}>{d.status}</span>
            </button>
          ))}
        </div>
      </Section>

      {cur && (
        <>
          <Err msg={cur.error ?? ""} />
          <Section title={`Pipeline: ${cur.name}`} aside={
            <button className="primary" disabled={busy} onClick={wrap(async () => { await api.analyze(cur.id); await refresh(); })}>
              {cur.analysis ? "Re-analyze" : "Analyze"}
            </button>}>
            <p className="muted small">Upload → Validate → Analyze → Clean → Preprocess → Train/Val/Test split</p>
          </Section>

          {cur.analysis && (
            <Section title="Dataset analysis">
              {isText ? (
                <div className="stats">
                  <Stat label="Examples" value={cur.analysis.examples ?? cur.analysis.total_recordings} />
                  <Stat label="Avg prompt" value={`${cur.analysis.avg_prompt_chars} chars`} />
                  <Stat label="Avg response" value={`${cur.analysis.avg_response_chars} chars`} />
                  <Stat label="Longest 5%" value={`${cur.analysis.p95_chars}+ chars`} />
                  <Stat label="Vocabulary" value={cur.analysis.vocabulary} sub="unique words" />
                </div>
              ) : (
                <div className="stats">
                  <Stat label="Recordings" value={cur.analysis.total_recordings} />
                  <Stat label="Duration" value={`${cur.analysis.total_hours.toFixed(2)} h`} sub={`avg ${cur.analysis.avg_duration_sec}s`} />
                  <Stat label="Speakers" value={cur.analysis.speakers} />
                  <Stat label="Script" value={cur.analysis.scripts} sub={`${cur.analysis.vocabulary} unique words`} />
                  <Stat label="Sample rate" value={Object.keys(cur.analysis.sample_rates).map((s) => `${+s / 1000} kHz`).join(", ")} />
                </div>
              )}
              <div className="two">
                <div>
                  <h3>Quality</h3>
                  <table>
                    <tbody>
                      {Object.entries(cur.analysis.quality).map(([k, v]) => (
                        <tr key={k}><td>{meta.quality_labels[k] ?? k}</td><td className="num">{v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h3>Data quality score</h3>
                  {Object.entries(meta.score_labels).map(([k, label]) => (
                    cur.analysis!.scores[k] != null && <Bar key={k} label={label} value={cur.analysis!.scores[k]} />
                  ))}
                </div>
              </div>
              {cur.analysis.warnings.map((w) => <div className="warnbox" key={w}>⚠ {w}</div>)}
              <div className="chips">
                {Object.entries(cur.analysis.flag_counts).map(([f, n]) => (
                  <button key={f} className="chip" onClick={wrap(async () => { const r = await api.items(cur.id, f); setFlagged({ flag: f, ...r }); })}>
                    {flagLabel(f)} · {n}
                  </button>
                ))}
              </div>
              {flagged && (
                <table>
                  {isText ? (
                    <>
                      <thead><tr><th>{flagLabel(flagged.flag)} ({flagged.total})</th><th>Prompt</th><th>Response</th></tr></thead>
                      <tbody>
                        {flagged.items.map((i) => (
                          <tr key={i.id}><td>{i.id}</td><td>{String(i.prompt).slice(0, 160)}</td><td>{String(i.response).slice(0, 160)}</td></tr>
                        ))}
                      </tbody>
                    </>
                  ) : (
                    <>
                      <thead><tr><th>{flagLabel(flagged.flag)} ({flagged.total})</th><th>Speaker</th><th>Duration</th><th>SNR</th><th>Transcript</th></tr></thead>
                      <tbody>
                        {flagged.items.map((i) => (
                          <tr key={i.id}><td>{i.id}</td><td>{i.speaker}</td><td>{i.duration}s</td><td>{i.snr_db ?? "—"}</td><td>{i.text}</td></tr>
                        ))}
                      </tbody>
                    </>
                  )}
                </table>
              )}

              <h3>Clean, preprocess and split</h3>
              <p className="muted small">Excluded issues ({isText ? "examples" : "recordings"} with any checked flag are dropped):</p>
              <div className="checks">
                {flags.map((f) => (
                  <label key={f.id}><input type="checkbox" checked={excl.includes(f.id)} onChange={() => toggle(excl, setExcl, f.id)} />
                    {" "}{f.label}{f.threshold ? ` (${f.id === "too_short" ? "<" : ">"}${f.threshold})` : ""}</label>
                ))}
              </div>
              <div className="row">
                <label>Val % <input className="narrow" type="number" value={val} onChange={(e) => setVal(+e.target.value)} /></label>
                <label>Test % <input className="narrow" type="number" value={test} onChange={(e) => setTest(+e.target.value)} /></label>
              </div>
              {cur.task === "stt" && (
                <>
                  <p className="muted small">Augmentation (train split only; one extra copy per selected type):</p>
                  <div className="checks">
                    {meta.augmentations.map((a) => (
                      <label key={a.id}><input type="checkbox" checked={augs.includes(a.id)} onChange={() => toggle(augs, setAugs, a.id)} /> {a.label}</label>
                    ))}
                  </div>
                </>
              )}
              <button className="primary" disabled={busy} onClick={wrap(async () => {
                await api.prepare(cur.id, { exclude_flags: excl, augment: augs, val_frac: val / 100, test_frac: test / 100 });
                await refresh();
              })}>Prepare dataset</button>
            </Section>
          )}

          {cur.prepared && (
            <Section title="Prepared dataset">
              <div className="stats">
                {["train", "val", "test"].map((s) => (
                  <Stat key={s} label={s} value={cur.prepared!.counts[s]}
                    sub={isText ? `${(cur.prepared!.chars?.[s] ?? 0).toLocaleString()} chars` : `${cur.prepared!.hours[s]} h`} />
                ))}
                <Stat label="Dropped" value={cur.prepared.dropped} />
              </div>
              <p className="muted small">
                {isText ? "The same prompt never appears in more than one split." : "16 kHz mono. Identical sentences never appear in more than one split."} Ready to train.
              </p>
            </Section>
          )}
        </>
      )}
    </>
  );
}
