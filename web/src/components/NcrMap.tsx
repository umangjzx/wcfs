import maplibregl, { GeoJSONSource, Map as MlMap, Marker } from "maplibre-gl";
import { useEffect, useRef } from "react";
import { categoryColor } from "../lib/aqi";
import { BASEMAP_STYLE, BLANK_DARK_STYLE, DELHI_OUTLINE, NCR_CITY_LABELS, graticule } from "../lib/ncrGeo";
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
      style: BASEMAP_STYLE as maplibregl.StyleSpecification,
      center: DELHI,
      zoom: 8.7,
      attributionControl: false,
      preserveDrawingBuffer: true,
      fadeDuration: 0,
    });
    // A blocked tile CDN (offline / firewalled venue) must not kill the map — the dark
    // background + our own vector layers still give a usable picture.
    let fellBack = false;
    m.on("error", (e) => {
      const msg = String((e as { error?: { message?: string } })?.error?.message ?? "");
      if (!fellBack && /basemap|raster|tile|arcgis/i.test(msg)) {
        fellBack = true;
        try {
          m.setStyle(BLANK_DARK_STYLE as maplibregl.StyleSpecification);
        } catch {
          /* keep going */
        }
      }
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

    // Re-run on every style load (initial + any setStyle fallback) so our overlay
    // sources/layers survive a basemap swap. Idempotent — guarded by getSource/getLayer.
    function installOverlays() {
      const blank = fellBack || !m.getStyle().sources["basemap"];

      if (blank && !m.getSource("grat")) {
        m.addSource("grat", { type: "geojson", data: graticule() });
        m.addLayer({ id: "grat", type: "line", source: "grat", paint: { "line-color": "#1e293b", "line-width": 1 } });
      }

      if (!m.getSource("outline")) {
        // Real Delhi NCT admin boundary (Census 2011). Solid, thin — it's exact, not a sketch.
        m.addSource("outline", { type: "geojson", data: DELHI_OUTLINE });
        m.addLayer({
          id: "outline",
          type: "line",
          source: "outline",
          paint: {
            "line-color": blank ? "#64748b" : "#a9c7f5",
            "line-width": ["interpolate", ["linear"], ["zoom"], 7, 1, 11, 1.8],
            "line-opacity": blank ? 0.9 : 0.7,
          },
        });
      }

      for (const id of ["grid", "plume", "fires", "stations"]) {
        if (!m.getSource(id)) m.addSource(id, { type: "geojson", data: EMPTY });
      }
      if (!m.getLayer("grid-heat")) {
        // Smooth interpolated AQI field. Grid points are a ~0.03deg lattice, so the
        // heatmap radius has to track pixel spacing (which doubles every zoom level)
        // to keep the surface continuous instead of breaking into blobs when zoomed
        // in; intensity drops in step so the extra kernel overlap doesn't saturate.
        m.addLayer({
          id: "grid-heat",
          type: "heatmap",
          source: "grid",
          paint: {
            "heatmap-weight": [
              "interpolate", ["linear"], ["get", "aqi"],
              0, 0, 50, 0.15, 100, 0.3, 200, 0.55, 300, 0.75, 400, 0.95,
            ],
            "heatmap-intensity": [
              "interpolate", ["linear"], ["zoom"], 7, 1.0, 9, 1.0, 11, 0.95, 13, 0.8, 15, 0.7,
            ],
            "heatmap-radius": [
              "interpolate", ["linear"], ["zoom"], 7, 22, 9, 42, 11, 115, 13, 380, 15, 900,
            ],
            "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.5, 12, 0.44, 15, 0.4],
            "heatmap-color": [
              "interpolate", ["linear"], ["heatmap-density"],
              0, "rgba(11,18,32,0)",
              0.15, "rgba(34,197,94,0.5)",
              0.35, "rgba(163,193,59,0.55)",
              0.55, "rgba(234,179,8,0.6)",
              0.72, "rgba(249,115,22,0.65)",
              0.88, "rgba(220,38,38,0.7)",
              1, "rgba(127,29,29,0.8)",
            ],
          },
        });
      }
      if (!m.getLayer("plume-line")) {
        m.addLayer({
          id: "plume-line",
          type: "line",
          source: "plume",
          paint: { "line-color": "#f59e0b", "line-width": 3, "line-opacity": 0.8, "line-dasharray": [2, 1.5] },
        });
      }
      if (!m.getLayer("fires-pt")) {
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
      }
      if (!m.getLayer("st-halo")) {
        m.addLayer({
          id: "st-halo",
          type: "circle",
          source: "stations",
          paint: {
            "circle-radius": ["case", ["==", ["get", "sel"], 1], 15, 0],
            "circle-color": "rgba(248,250,252,0.12)",
            "circle-stroke-color": "#f8fafc",
            "circle-stroke-width": 2,
          },
        });
      }
      if (!m.getLayer("st-pt")) {
        m.addLayer({
          id: "st-pt",
          type: "circle",
          source: "stations",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 6, 10, 9, 13, 13],
            "circle-color": ["get", "color"],
            "circle-stroke-color": "#0b1220",
            "circle-stroke-width": 2,
          },
        });
      }

      ready.current = true;
      redraw();
      requestAnimationFrame(() => m.resize());
    }

    m.on("style.load", installOverlays);

    // City labels ride above the canvas as HTML markers — add once, independent of style.
    for (const c of NCR_CITY_LABELS) {
      const el = document.createElement("div");
      el.textContent = c.name;
      el.style.cssText =
        "font:600 11px var(--font-mono);color:#cbd5e1;letter-spacing:.04em;text-shadow:0 1px 4px #000,0 0 2px #000;pointer-events:none;white-space:nowrap";
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
