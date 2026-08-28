import { useMemo } from "react";
import { categoryColor, fmtHour } from "../lib/aqi";
import type { Alert } from "../types";

export function AlertsBanner({ alerts, onPick }: { alerts: Alert[]; onPick: (id: string) => void }) {
  const soonest = useMemo(() => {
    const byLevel: Record<string, Alert> = {};
    for (const a of alerts) {
      if (!byLevel[a.level] || a.lead_hours < byLevel[a.level].lead_hours) byLevel[a.level] = a;
    }
    return ["Severe", "Very Poor"].map((l) => byLevel[l]).filter(Boolean) as Alert[];
  }, [alerts]);

  if (!soonest.length)
    return (
      <div style={bar("#14532d")}>
        <Dot color="#16a34a" />
        <span>No “Very Poor” or “Severe” air expected across NCR in the next 72 hours.</span>
      </div>
    );

  return (
    <div style={bar("#3f1d1d")}>
      {soonest.map((a) => (
        <button
          key={a.level}
          onClick={() => onPick(a.station_id)}
          style={{ background: "transparent", border: "none", padding: 0, color: "inherit", display: "inline-flex", gap: 8, alignItems: "center" }}
        >
          <Dot color={categoryColor(a.level)} />
          <span>
            <strong>{a.level}</strong> air at <strong>{a.name}</strong> in{" "}
            <span className="mono-num">{a.lead_hours} h</span> ({fmtHour(a.valid_ts)} IST, AQI {a.aqi})
          </span>
        </button>
      ))}
    </div>
  );
}

function bar(bg: string): React.CSSProperties {
  return {
    display: "flex",
    gap: 20,
    alignItems: "center",
    flexWrap: "wrap",
    padding: "7px var(--space-4)",
    background: bg,
    borderBottom: "1px solid var(--color-border)",
    fontSize: 13,
  };
}
function Dot({ color }: { color: string }) {
  return <span style={{ width: 8, height: 8, borderRadius: 999, background: color, display: "inline-block", flex: "none" }} />;
}
