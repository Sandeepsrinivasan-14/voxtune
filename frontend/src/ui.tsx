import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { MetricDef, Meta } from "./api";

export const MetaContext = createContext<Meta | null>(null);
export function useMeta(): Meta {
  const m = useContext(MetaContext);
  if (!m) throw new Error("Meta not loaded");
  return m;
}

export function usePoll(fn: () => void, ms: number, active = true) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    ref.current();
    const t = setInterval(() => ref.current(), ms);
    return () => clearInterval(t);
  }, [ms, active]);
}

export function useAsyncError() {
  const [err, setErr] = useState("");
  const wrap = <A extends unknown[]>(f: (...a: A) => Promise<unknown>) => async (...a: A) => {
    setErr("");
    try { await f(...a); } catch (e) { setErr((e as Error).message); }
  };
  return { err, setErr, wrap };
}

export const Err = ({ msg }: { msg: string }) => (msg ? <div className="error">{msg}</div> : null);

export function Bar({ label, value, text, tone }: { label: string; value: number; text?: string; tone?: "warn" | "bad" }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div className="bar">
      <span>{label}</span>
      <div className="track"><div className={tone} style={{ width: `${Math.round(v * 100)}%` }} /></div>
      <b>{text ?? `${Math.round(v * 100)}%`}</b>
    </div>
  );
}

export const Section = ({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) => (
  <section>
    <div className="sec-head"><h2>{title}</h2>{aside}</div>
    {children}
  </section>
);

export const Stat = ({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) => (
  <div className="stat"><span>{label}</span><b>{value}</b>{sub && <em>{sub}</em>}</div>
);

export const pct = (v?: number | null, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const fmtMetric = (def: Pick<MetricDef, "fmt">, v: unknown) => (v == null ? "—" : def.fmt === "pct" ? pct(v as number) : String(v));
export const num = (v?: number | null, d = 1, unit = "") => (v == null ? "—" : `${v.toFixed(d)}${unit}`);
export const toneFor = (pctUsed: number): "warn" | "bad" | undefined => (pctUsed >= 90 ? "bad" : pctUsed >= 75 ? "warn" : undefined);
export const dur = (s?: number | null) => {
  if (s == null) return "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};
