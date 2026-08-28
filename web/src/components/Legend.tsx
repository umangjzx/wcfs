import { AQI_BANDS } from "../lib/aqi";

export function Legend() {
  return (
    <div
      style={{
        position: "absolute",
        left: 10,
        bottom: 10,
        background: "rgba(17,24,39,0.9)",
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        padding: "8px 10px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
        backdropFilter: "blur(4px)",
      }}
    >
      <span className="muted" style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        AQI category
      </span>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", maxWidth: 260 }}>
        {AQI_BANDS.map((b) => (
          <span key={b.label} style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: b.color }} />
            {b.label}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 2, fontSize: 11 }} className="muted">
        <span>
          <span style={{ color: "#fb923c" }}>●</span> stubble fire
        </span>
        <span>
          <span style={{ color: "#f59e0b" }}>┈</span> plume transport
        </span>
      </div>
    </div>
  );
}
