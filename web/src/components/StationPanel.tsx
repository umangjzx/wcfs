import {
  Area,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { aqiBand } from "../lib/aqi";
import type { Station, StationForecast } from "../types";

interface Props {
  station: Station | null;
  forecast: StationForecast | null;
  horizon: number;
  onHover: (h: number) => void;
}

export function StationPanel({ station, forecast, horizon, onHover }: Props) {
  if (!station) return <div className="card">Select a station on the map.</div>;

  const points = forecast?.points ?? [];
  const cur = points.find((p) => p.horizon === Math.max(horizon, 1)) ?? points[0];
  const peak = points.reduce((a, b) => (b.pm25_p50 > (a?.pm25_p50 ?? -1) ? b : a), points[0]);
  const band = aqiBand(cur?.aqi ?? station.latest_aqi);
  const peakBand = aqiBand(peak?.aqi);

  const data = points.map((p) => ({
    h: p.horizon,
    p50: Math.round(p.pm25_p50),
    lo: Math.round(p.pm25_p10),
    hi: Math.round(p.pm25_p90),
    band: [Math.round(p.pm25_p10), Math.round(p.pm25_p90)],
    aqi: p.aqi,
  }));

  return (
    <div className="card fade-in" style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 15 }}>{station.name}</div>
          <div className="muted" style={{ fontSize: 12 }}>
            {station.city} · {station.agency} · {station.site_type}
          </div>
        </div>
        <span className="pill" style={{ background: band.color, color: "#fff" }}>
          {horizon === 0 ? "Now" : `+${horizon}h`} · {band.label}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "var(--space-2)" }}>
        <Stat label={horizon === 0 ? "AQI now" : `AQI +${horizon}h`} value={cur?.aqi ?? station.latest_aqi ?? "—"} color={band.color} />
        <Stat label={horizon === 0 ? "PM2.5 now" : "PM2.5 (P50)"} value={cur ? `${Math.round(cur.pm25_p50)}` : "—"} sub="µg/m³" />
        <Stat label={`Peak (72h)`} value={peak?.aqi ?? "—"} color={peakBand.color} sub={peak ? `+${peak.horizon}h` : ""} />
      </div>

      <div style={{ height: 180 }}>
        <ResponsiveContainer>
          <ComposedChart
            data={data}
            margin={{ top: 6, right: 8, bottom: 0, left: -18 }}
            onMouseMove={(s) => s?.activeLabel != null && onHover(Number(s.activeLabel))}
          >
            <XAxis dataKey="h" tick={{ fill: "#94a3b8", fontSize: 10 }} tickFormatter={(h) => (h === 0 ? "now" : `+${h}h`)} ticks={[1, 12, 24, 36, 48, 60, 72]} stroke="#334155" />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} stroke="#334155" width={40} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #334155", borderRadius: 8, fontFamily: "var(--font-mono)", fontSize: 12 }}
              labelFormatter={(h) => (h === 0 ? "now" : `+${h} h`)}
              formatter={(v: number, n: string) => [`${v} µg/m³`, n === "p50" ? "PM2.5" : n === "hi" ? "P90" : "P10"]}
            />
            <Area type="monotone" dataKey="band" stroke="none" fill="#16a34a" fillOpacity={0.16} isAnimationActive={false} />
            <Line type="monotone" dataKey="p50" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
            {horizon > 0 && <ReferenceLine x={horizon} stroke="#f8fafc" strokeOpacity={0.5} strokeDasharray="3 3" />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <p className="muted" style={{ fontSize: 12, margin: 0, borderLeft: `3px solid ${band.color}`, paddingLeft: 10 }}>
        {forecast?.advisory || band.advice}
      </p>
    </div>
  );
}

function Stat({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div style={{ background: "var(--color-muted)", borderRadius: 8, padding: "8px 10px" }}>
      <div className="muted" style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <div className="mono-num" style={{ fontSize: 20, fontWeight: 600, color: color ?? "var(--color-foreground)" }}>
        {value}
        {sub && <span className="muted" style={{ fontSize: 11, marginLeft: 4 }}>{sub}</span>}
      </div>
    </div>
  );
}
