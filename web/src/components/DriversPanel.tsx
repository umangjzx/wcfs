import type { Station, StationDrivers } from "../types";

const GROUP_LABEL: Record<string, string> = {
  inversion_trapping: "Inversion trapping",
  stubble_transport: "Upwind stubble transport",
  local_emissions_persistence: "Local emissions / persistence",
  wind_ventilation: "Wind ventilation",
  other_meteorology: "Other meteorology",
  time_season: "Time & season",
};
const GROUP_COLOR: Record<string, string> = {
  inversion_trapping: "#a855f7",
  stubble_transport: "#fb923c",
  local_emissions_persistence: "#38bdf8",
  wind_ventilation: "#22c55e",
  other_meteorology: "#64748b",
  time_season: "#eab308",
};

interface Props {
  drivers: StationDrivers | null;
  station: Station | null;
  modelName?: string;
}

export function DriversPanel({ drivers, station, modelName }: Props) {
  if (!station) return null;
  if (!drivers || (modelName && modelName !== "lgbm")) {
    return (
      <div className="card">
        <h2>Why this forecast</h2>
        <p className="muted" style={{ fontSize: 12 }}>
          Driver attribution needs the trained ML emulator. Currently serving the baseline
          forecast — retrain to enable the SHAP breakdown.
        </p>
      </div>
    );
  }

  const groups = [...drivers.groups].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const maxAbs = Math.max(1, ...groups.map((g) => Math.abs(g.contribution)));
  const isi = drivers.isi ?? 0;
  const stubble = drivers.incoming_stubble_load ?? 0;
  const vent = drivers.ventilation_index ?? null;

  return (
    <div className="card fade-in" style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
      <h2>Why this forecast</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-2)" }}>
        <Meter label="Inversion Strength Index" value={isi} max={1} fmt={(v) => v.toFixed(2)} color="#a855f7" />
        <Meter
          label="Incoming stubble load"
          value={Math.min(stubble / 600, 1)}
          max={1}
          fmt={() => Math.round(stubble).toString()}
          color="#fb923c"
        />
      </div>
      <div className="muted mono-num" style={{ fontSize: 11, display: "flex", gap: 14, flexWrap: "wrap" }}>
        {drivers.plume_from_bearing_deg != null && (
          <span>plume from {compass(drivers.plume_from_bearing_deg)} ({Math.round(drivers.plume_from_bearing_deg)}°)</span>
        )}
        {vent != null && <span>ventilation index {Math.round(vent)} m²/s</span>}
      </div>

      <div>
        <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>
          Contribution to the forecast PM2.5 (µg/m³, SHAP)
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {groups.map((g) => (
            <div key={g.group} style={{ display: "grid", gridTemplateColumns: "150px 1fr 52px", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 12 }}>{GROUP_LABEL[g.group] ?? g.group}</span>
              <div style={{ position: "relative", height: 14, background: "var(--color-muted)", borderRadius: 4 }}>
                <div
                  style={{
                    position: "absolute",
                    left: g.contribution >= 0 ? "50%" : undefined,
                    right: g.contribution < 0 ? "50%" : undefined,
                    width: `${(Math.abs(g.contribution) / maxAbs) * 50}%`,
                    top: 0,
                    bottom: 0,
                    background: GROUP_COLOR[g.group] ?? "#64748b",
                    borderRadius: 4,
                  }}
                />
                <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--color-border)" }} />
              </div>
              <span className="mono-num" style={{ fontSize: 11, textAlign: "right", color: g.contribution >= 0 ? "#f97316" : "#22c55e" }}>
                {g.contribution >= 0 ? "+" : ""}
                {g.contribution.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Meter({ label, value, max, fmt, color }: { label: string; value: number; max: number; fmt: (v: number) => string; color: string }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div style={{ background: "var(--color-muted)", borderRadius: 8, padding: "8px 10px" }}>
      <div className="muted" style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.04em" }}>{label}</div>
      <div className="mono-num" style={{ fontSize: 18, fontWeight: 600 }}>{fmt(value)}</div>
      <div style={{ height: 5, background: "#0f172a", borderRadius: 4, marginTop: 4 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

function compass(deg: number): string {
  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return dirs[Math.round(deg / 22.5) % 16];
}
