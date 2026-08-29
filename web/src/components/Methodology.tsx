import { useEffect, useState } from "react";
import { api } from "../api";
import type { ModelCard } from "../types";

export function Methodology({ onClose }: { onClose: () => void }) {
  const [mc, setMc] = useState<ModelCard | null>(null);
  useEffect(() => {
    api.modelCard().then(setMc).catch(() => undefined);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const overall = (mc?.backtest as { overall?: Record<string, { MAE?: number }> } | undefined)?.overall;
  const events = (mc?.backtest as { events?: Record<string, { model?: { POD?: number; FAR?: number; CSI?: number } }> } | undefined)?.events;

  return (
    <div
      onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "flex-end", zIndex: 50 }}
    >
      <div
        className="fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(560px, 100%)",
          height: "100%",
          background: "var(--color-card)",
          borderLeft: "1px solid var(--color-border)",
          padding: "var(--space-6)",
          overflowY: "auto",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1>Methodology</h1>
          <button onClick={onClose} aria-label="Close">✕</button>
        </div>

        <p className="muted" style={{ fontSize: 13, lineHeight: 1.6 }}>
          VayuCast models the two-way meteorology–chemistry feedback that standard AQI
          forecasts ignore: temperature inversions and a shallow boundary layer trap PM2.5
          near the surface, while heavy aerosol loading blocks sunlight and suppresses
          boundary-layer growth further. The coupling is encoded as engineered features — an{" "}
          <strong>Inversion Strength Index</strong> and a{" "}
          <strong>stubble-plume transport vector</strong> — that a fast ML emulator learns
          from. A one-off offline WRF-Chem run (namelists + runbook in the repo) is the
          physics check against a historical stubble-burning spike.
        </p>

        <Section title="Model">
          <Row k="Engine" v={mc?.model === "lgbm" ? "LightGBM multi-horizon quantile emulator" : "awaiting first live refresh"} />
          <Row k="Horizon" v="72 hours, hourly, P10 / P50 / P90" />
        </Section>

        {overall && (
          <Section title="Backtest (walk-forward)">
            <Row k="MAE — model" v={`${overall.model?.MAE} µg/m³`} />
            <Row k="MAE — persistence" v={`${overall.persistence?.MAE} µg/m³`} />
            <Row k="MAE — climatology" v={`${overall.climatology?.MAE} µg/m³`} />
            {events?.very_poor?.model && (
              <Row
                k="Very Poor event (POD / FAR / CSI)"
                v={`${events.very_poor.model.POD} / ${events.very_poor.model.FAR} / ${events.very_poor.model.CSI}`}
              />
            )}
          </Section>
        )}

        <Section title="Data sources">
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
            {(mc?.data_sources ?? []).map((s) => (
              <li key={s} style={{ marginBottom: 4 }}>{s}</li>
            ))}
          </ul>
        </Section>

        <Section title="WRF-Chem validation">
          <p className="muted" style={{ fontSize: 13, margin: 0 }}>{mc?.wrfchem_validation}</p>
        </Section>

        <Section title="Limitations">
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
            {(mc?.limitations ?? []).map((s) => (
              <li key={s} style={{ marginBottom: 4 }} className="muted">{s}</li>
            ))}
          </ul>
        </Section>

        <p className="muted" style={{ fontSize: 11, marginTop: "var(--space-6)" }}>
          SIH 2026 · Problem Statement 26082 · Ministry of Earth Sciences → NCMRWF
        </p>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginTop: "var(--space-6)" }}>
      <h2>{title}</h2>
      {children}
    </div>
  );
}
function Row({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 13, padding: "4px 0", borderBottom: "1px solid var(--color-muted)" }}>
      <span className="muted">{k}</span>
      <span className="mono-num" style={{ textAlign: "right" }}>{v}</span>
    </div>
  );
}
