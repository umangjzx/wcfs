export interface AqiBand {
  label: string;
  color: string;
  lo: number;
  hi: number;
  advice: string;
}

export const AQI_BANDS: AqiBand[] = [
  { label: "Good", color: "#16a34a", lo: 0, hi: 50, advice: "Air quality is good." },
  { label: "Satisfactory", color: "#84cc16", lo: 51, hi: 100, advice: "Minor discomfort for the highly sensitive." },
  { label: "Moderate", color: "#eab308", lo: 101, hi: 200, advice: "Sensitive groups should limit prolonged exertion." },
  { label: "Poor", color: "#f97316", lo: 201, hi: 300, advice: "Reduce outdoor activity; sensitive groups avoid it." },
  { label: "Very Poor", color: "#dc2626", lo: 301, hi: 400, advice: "Avoid outdoor activity; use N95 + purifiers." },
  { label: "Severe", color: "#7f1d1d", lo: 401, hi: 9999, advice: "Stay indoors; follow GRAP restrictions." },
];

export function aqiBand(aqi: number | null | undefined): AqiBand {
  if (aqi == null || Number.isNaN(aqi))
    return { label: "Unknown", color: "#475569", lo: 0, hi: 0, advice: "" };
  return AQI_BANDS.find((b) => aqi <= b.hi) ?? AQI_BANDS[AQI_BANDS.length - 1];
}

export function aqiColor(aqi: number | null | undefined): string {
  return aqiBand(aqi).color;
}

export function categoryColor(cat: string | null | undefined): string {
  const b = AQI_BANDS.find((x) => x.label === cat);
  return b ? b.color : "#475569";
}

/** MapLibre step expression stops for a continuous AQI fill. */
export const AQI_STEP_EXPR: unknown[] = [
  "step",
  ["get", "aqi"],
  "#16a34a",
  51, "#84cc16",
  101, "#eab308",
  201, "#f97316",
  301, "#dc2626",
  401, "#7f1d1d",
];

export function fmtHour(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    weekday: "short",
    hour: "numeric",
    hour12: true,
  });
}

export function fmtRelHour(h: number): string {
  if (h === 0) return "Now";
  return `+${h}h`;
}
