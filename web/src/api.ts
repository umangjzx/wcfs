import type {
  Alert,
  FiresOut,
  GridOut,
  Health,
  ModelCard,
  Station,
  StationDrivers,
  StationForecast,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  health: () => get<Health>("/api/health"),
  stations: () => get<Station[]>("/api/stations"),
  forecast: (id: string) => get<StationForecast>(`/api/forecast/${id}`),
  observations: (id: string, hours = 48) =>
    get<{ station_id: string; series: { pollutant: string; ts: string[]; value: (number | null)[] }[] }>(
      `/api/observations/${id}?hours=${hours}`,
    ),
  grid: (horizon: number) => get<GridOut>(`/api/grid?horizon=${horizon}`),
  drivers: (id: string) => get<StationDrivers>(`/api/drivers/${id}`),
  fires: () => get<FiresOut>("/api/fires"),
  alerts: () => get<{ as_of: string | null; alerts: Alert[] }>("/api/alerts"),
  modelCard: () => get<ModelCard>("/api/model-card"),
  refresh: () =>
    fetch(`${BASE}/api/refresh`, { method: "POST" }).then((r) => r.json()),
};
