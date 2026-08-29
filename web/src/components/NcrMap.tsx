import maplibregl, { GeoJSONSource, Map as MlMap, Marker } from "maplibre-gl";
import { useEffect, useRef } from "react";
import { AQI_STEP_EXPR, categoryColor } from "../lib/aqi";
import { BLANK_DARK_STYLE, DELHI_OUTLINE, NCR_CITY_LABELS, graticule } from "../lib/ncrGeo";
import type { FiresOut, GridOut, Station } from "../types";

const DELHI: [number, number] = [77.209, 28.6139];

interface Props {
  stations: Station[];
  grid: GridOut | null;
  fires: FiresOut | null;
  selected: string | null;
  onSelect: (id: string) => void;
}

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

// MapLibre is imperative and doesn't survive partial hot updates — force a clean full
// reload when this module (or anything it imports) changes in dev.
if (import.meta.hot) {
  import.meta.hot.accept(() => import.meta.hot?.invalidate());
}

export function NcrMap({ stations, grid, fires, selected, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const ready = useRef(false);
  const data = useRef<Props>({ stations, grid, fires, selected, onSelect });
  data.current = { stations, grid, fires, selected, onSelect };

  function redraw() {
    const m = map.current;
    if (!m || !ready.current) return;
    const { stations: st, grid: g, fires: fr, selected: sel } = data.current;

    (m.getSource("stations") as GeoJSONSource)?.setData({
      type: "FeatureCollection",
      features: st.map((s) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [s.lon, s.lat] },
        properties: {
          id: s.id, name: s.name, city: s.city,
          aqi: s.latest_aqi ?? "", category: s.latest_category ?? "",
          color: categoryColor(s.latest_category), sel: s.id === sel ? 1 : 0,
        },
      })),
    });

    (m.getSource("grid") as GeoJSONSource)?.setData(
      g
        ? {
            type: "FeatureCollection",
            features: g.cells.map((c) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [c.lon, c.lat] },
              properties: { aqi: c.aqi },
            })),
          }
        : EMPTY,
    );

    (m.getSource("fires") as GeoJSONSource)?.setData({
      type: "FeatureCollection",
      features: (fr?.clusters ?? []).map((c) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [c.lon, c.lat] },
        properties: { frp: c.frp_sum },
      })),
    });

    const b = fr?.plume_vector?.from_bearing_deg;
    const load = fr?.plume_vector?.incoming_load ?? 0;
    let line: GeoJSON.FeatureCollection = EMPTY;
    if (b != null && load > 1) {
      const distKm = 220;
      const rad = (b * Math.PI) / 180;
      const dLat = (Math.cos(rad) * distKm) / 111;
      const dLon = (Math.sin(rad) * distKm) / (111 * Math.cos((DELHI[1] * Math.PI) / 180));
      line = {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: { type: "LineString", coordinates: [[DELHI[0] + dLon, DELHI[1] + dLat], DELHI] },
            properties: {},
          },
        ],
      };
    }
    (m.getSource("plume") as GeoJSONSource)?.setData(line);
  }

  useEffect(() => {
    if (!box.current) return;
    // HMR / re-mount: drop a map whose container is no longer the live div
    if (map.current) {
      const c = map.current.getContainer?.();
      if (c === box.current && !("_removed" in map.current && (map.current as { _removed?: boolean })._removed)) {
        return;
      }
      try {
        map.current.remove();
      } catch {
        /* already gone */
      }
      map.current = null;
      ready.current = false;
    }
    box.current.querySelectorAll(".maplibregl-map").forEach((n) => n.remove()); // orphan HMR canvases
    const m = new maplibregl.Map({
      container: box.current,
      style: BLANK_DARK_STYLE as maplibregl.StyleSpecification,
      center: DELHI,
      zoom: 8.5,
      attributionControl: false,
      preserveDrawingBuffer: true,
      fadeDuration: 0,
    });
    map.current = m;
    (window as unknown as { __map?: MlMap }).__map = m;
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: "VayuCast · CPCB · NASA FIRMS · Open-Meteo",
      }),
    );

    m.on("load", () => {
      m.addSource("grat", { type: "geojson", data: graticule() });
      m.addLayer({ id: "grat", type: "line", source: "grat", paint: { "line-color": "#1e293b", "line-width": 1 } });

      m.addSource("outline", { type: "geojson", data: DELHI_OUTLINE });
      m.addLayer({
        id: "outline",
        type: "line",
        source: "outline",
        paint: { "line-color": "#475569", "line-width": 1.5, "line-dasharray": [3, 2] },
      });

      for (const id of ["grid", "plume", "fires", "stations"]) {
        m.addSource(id, { type: "geojson", data: EMPTY });
      }
      m.addLayer({
        id: "grid-blob",
        type: "circle",
        source: "grid",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 20, 10, 46],
          "circle-color": AQI_STEP_EXPR as unknown as maplibregl.ExpressionSpecification,
          "circle-blur": 1,
          "circle-opacity": 0.32,
        },
      });
      m.addLayer({
        id: "plume-line",
        type: "line",
        source: "plume",
        paint: { "line-color": "#f59e0b", "line-width": 3, "line-opacity": 0.8, "line-dasharray": [2, 1.5] },
      });
      m.addLayer({
        id: "fires-pt",
        type: "circle",
        source: "fires",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "frp"], 0, 2, 600, 8],
          "circle-color": "#fb923c",
          "circle-opacity": 0.85,
          "circle-stroke-color": "#7c2d12",
          "circle-stroke-width": 0.5,
        },
      });
      m.addLayer({
        id: "st-halo",
        type: "circle",
        source: "stations",
        paint: {
          "circle-radius": ["case", ["==", ["get", "sel"], 1], 13, 0],
          "circle-color": "rgba(248,250,252,0.12)",
          "circle-stroke-color": "#f8fafc",
          "circle-stroke-width": 2,
        },
      });
      m.addLayer({
        id: "st-pt",
        type: "circle",
        source: "stations",
        paint: {
          "circle-radius": 6,
          "circle-color": ["get", "color"],
          "circle-stroke-color": "#0b1220",
          "circle-stroke-width": 1.5,
        },
      });

      for (const c of NCR_CITY_LABELS) {
        const el = document.createElement("div");
        el.textContent = c.name;
        el.style.cssText =
          "font:600 11px var(--font-mono);color:#94a3b8;letter-spacing:.04em;text-shadow:0 1px 3px #000;pointer-events:none;white-space:nowrap";
        new Marker({ element: el, anchor: "left" }).setLngLat([c.lon, c.lat]).addTo(m);
      }

      const popup = new maplibregl.Popup({ closeButton: false, offset: 10 });
      m.on("mouseenter", "st-pt", (e) => {
        m.getCanvas().style.cursor = "pointer";
        const f = e.features?.[0];
        if (!f) return;
        const p = f.properties as Record<string, string>;
        popup
          .setLngLat((f.geometry as GeoJSON.Point).coordinates as [number, number])
          .setHTML(
            `<strong>${p.name}</strong><br/><span style="font-family:var(--font-mono)">AQI ${p.aqi || "—"} · ${p.category || "—"}</span><br/><span style="opacity:.7">${p.city}</span>`,
          )
          .addTo(m);
      });
      m.on("mouseleave", "st-pt", () => {
        m.getCanvas().style.cursor = "";
        popup.remove();
      });
      m.on("click", "st-pt", (e) => {
        const id = e.features?.[0]?.properties?.id as string | undefined;
        if (id) data.current.onSelect(id);
      });

      ready.current = true;
      redraw();
      requestAnimationFrame(() => m.resize());
    });

    const ro = new ResizeObserver(() => map.current?.resize());
    if (box.current) ro.observe(box.current);
    return () => {
      ro.disconnect();
      m.remove();
      map.current = null;
      ready.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(redraw, [stations, grid, fires, selected]);

  return <div ref={box} style={{ position: "absolute", inset: 0 }} aria-label="Delhi NCR air quality map" />;
}
