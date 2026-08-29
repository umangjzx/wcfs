// Minimal offline geographic context for the NCR map (no external basemap needed).

export const NCR_CITY_LABELS: { name: string; lat: number; lon: number }[] = [
  { name: "New Delhi", lat: 28.62, lon: 77.21 },
  { name: "Gurugram", lat: 28.44, lon: 77.03 },
  { name: "Faridabad", lat: 28.41, lon: 77.31 },
  { name: "Ghaziabad", lat: 28.67, lon: 77.43 },
  { name: "Noida", lat: 28.57, lon: 77.33 },
  { name: "Greater Noida", lat: 28.49, lon: 77.51 },
  { name: "Sonipat", lat: 28.99, lon: 77.02 },
  { name: "Bahadurgarh", lat: 28.69, lon: 76.93 },
];

// Rough Delhi NCT boundary (hand-simplified — for orientation only).
export const DELHI_OUTLINE: GeoJSON.Feature = {
  type: "Feature",
  properties: {},
  geometry: {
    type: "LineString",
    coordinates: [
      [77.10, 28.88], [77.22, 28.86], [77.28, 28.83], [77.34, 28.74],
      [77.35, 28.66], [77.31, 28.56], [77.28, 28.50], [77.20, 28.41],
      [77.12, 28.41], [77.03, 28.45], [76.93, 28.52], [76.90, 28.61],
      [76.94, 28.72], [77.00, 28.80], [77.05, 28.85], [77.10, 28.88],
    ],
  },
};

// A faint 0.25-degree graticule across the NCR view.
export function graticule(): GeoJSON.FeatureCollection {
  const feats: GeoJSON.Feature[] = [];
  for (let lat = 27.75; lat <= 29.5; lat += 0.25) {
    feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[76.2, lat], [78.4, lat]] } });
  }
  for (let lon = 76.25; lon <= 78.5; lon += 0.25) {
    feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[lon, 27.6], [lon, 29.6]] } });
  }
  return { type: "FeatureCollection", features: feats };
}

// Fallback when no basemap tiles are reachable (offline demo / firewalled venue).
export const BLANK_DARK_STYLE = {
  version: 8 as const,
  sources: {},
  layers: [{ id: "bg", type: "background" as const, paint: { "background-color": "#0b1220" } }],
};

// Primary basemap: Esri "Dark Gray Canvas" raster tiles — keyless, dark, reliable.
// Raster needs no glyphs/sprite, so it renders wherever the CDN is reachable; the
// dark background shows through if it is not (offline / firewalled venue).
export const BASEMAP_STYLE = {
  version: 8 as const,
  sources: {
    basemap: {
      type: "raster" as const,
      tiles: [
        "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      maxzoom: 16,
      attribution: "Esri, HERE, Garmin, © OpenStreetMap contributors",
    },
    "basemap-labels": {
      type: "raster" as const,
      tiles: [
        "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      maxzoom: 16,
    },
  },
  layers: [
    { id: "bg", type: "background" as const, paint: { "background-color": "#0b1220" } },
    { id: "basemap", type: "raster" as const, source: "basemap", paint: { "raster-opacity": 0.95 } },
    { id: "basemap-labels", type: "raster" as const, source: "basemap-labels", paint: { "raster-opacity": 0.9 } },
  ],
};
