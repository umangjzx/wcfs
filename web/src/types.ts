export interface Station {
  id: string;
  name: string;
  city: string;
  agency: string;
  lat: number;
  lon: number;
  site_type: string;
  latest_aqi: number | null;
  latest_category: string | null;
  latest_pm25: number | null;
  latest_ts: string | null;
}

export interface ForecastPoint {
  valid_ts: string;
  horizon: number;
  pm25_p10: number;
  pm25_p50: number;
  pm25_p90: number;
  aqi: number | null;
  category: string;
}

export interface StationForecast {
  station_id: string;
  name: string;
  issued_ts: string;
  dominant_pollutant: string;
  advisory: string;
  points: ForecastPoint[];
}

export interface GridCell {
  lat: number;
  lon: number;
  aqi: number;
}
export interface GridOut {
  valid_ts: string | null;
  horizon: number;
  bounds: number[];
  cells: GridCell[];
}

export interface DriverGroup {
  group: string;
  contribution: number;
}
export interface DriverFeature {
  feature: string;
  group: string;
  importance: number;
}
export interface StationDrivers {
  station_id: string;
  isi: number | null;
  isi_components: Record<string, number>;
  incoming_stubble_load: number | null;
  plume_from_bearing_deg: number | null;
  ventilation_index: number | null;
  groups: DriverGroup[];
  top_features: DriverFeature[];
}

export interface FireCluster {
  lat: number;
  lon: number;
  frp_sum: number;
  count: number;
  date: string;
}
export interface FiresOut {
  as_of: string | null;
  clusters: FireCluster[];
  plume_vector: { from_bearing_deg?: number; incoming_load?: number };
}

export interface Alert {
  station_id: string;
  name: string;
  level: string;
  lead_hours: number;
  valid_ts: string;
  aqi: number;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model_name: string;
  last_refresh: string | null;
  stale: boolean;
  sources: { source: string; ok: boolean; stale: boolean; message: string }[];
  stations_in_forecast: number;
}

export interface ModelCard {
  model: string;
  horizons: number[];
  backtest: Record<string, unknown>;
  data_sources: string[];
  limitations: string[];
  wrfchem_validation: string;
  region_bounds: number[];
}
