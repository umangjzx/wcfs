import { fmtRelHour } from "../lib/aqi";

interface Props {
  horizon: number;
  onChange: (h: number) => void;
  validLabel: string;
  issued?: string;
}

const TICKS = [0, 12, 24, 36, 48, 60, 72];

export function TimeSlider({ horizon, onChange, validLabel, issued }: Props) {
  return (
    <div
      style={{
        borderTop: "1px solid var(--color-border)",
        padding: "10px var(--space-4) 12px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        background: "var(--color-card)",
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>
          {fmtRelHour(horizon)}
        </span>
        <span className="muted mono-num" style={{ fontSize: 12 }}>valid {validLabel} IST</span>
        <div style={{ flex: 1 }} />
        {issued && (
          <span className="muted mono-num" style={{ fontSize: 11 }}>
            issued {new Date(issued).toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour: "numeric", hour12: true })} IST
          </span>
        )}
      </div>
      <input
        type="range"
        min={0}
        max={72}
        step={1}
        value={horizon}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Forecast hour"
        style={{ width: "100%", accentColor: "var(--color-primary)" }}
      />
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        {TICKS.map((t) => (
          <button
            key={t}
            onClick={() => onChange(t)}
            aria-pressed={horizon === t}
            style={{ padding: "2px 8px", fontSize: 11 }}
            className="mono-num"
          >
            {t === 0 ? "Now" : `+${t}h`}
          </button>
        ))}
      </div>
    </div>
  );
}
