import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { AlertsBanner } from "./components/AlertsBanner";
import { DriversPanel } from "./components/DriversPanel";
import { Legend } from "./components/Legend";
import { Methodology } from "./components/Methodology";
import { NcrMap } from "./components/NcrMap";
import { StationPanel } from "./components/StationPanel";
import { TimeSlider } from "./components/TimeSlider";
import { fmtRelHour } from "./lib/aqi";
import type {
  Alert,
  FiresOut,
  GridOut,
  Health,
  Station,
  StationDrivers,
  StationForecast,
} from "./types";

export default function App() {
  const [stations, setStations] = useState<Station[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [horizon, setHorizon] = useState(0);
  const [grid, setGrid] = useState<GridOut | null>(null);
  const [forecast, setForecast] = useState<StationForecast | null>(null);
  const [drivers, setDrivers] = useState<StationDrivers | null>(null);
  const [fires, setFires] = useState<FiresOut | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [showMethod, setShowMethod] = useState(false);
  const [busy, setBusy] = useState(false);
  const gridReq = useRef(0);

  const loadCore = useCallback(async () => {
    const [st, al, fr, hz] = await Promise.allSettled([
      api.stations(),
      api.alerts(),
      api.fires(),
      api.health(),
    ]);
    if (st.status === "fulfilled") {
      setStations(st.value);
      setSelected((cur) => cur ?? worstStation(st.value));
    }
    if (al.status === "fulfilled") setAlerts(al.value.alerts);
    if (fr.status === "fulfilled") setFires(fr.value);
    if (hz.status === "fulfilled") setHealth(hz.value);
  }, []);

  useEffect(() => {
    loadCore();
    const t = setInterval(loadCore, 5 * 60_000);
    return () => clearInterval(t);
  }, [loadCore]);

  useEffect(() => {
    const id = ++gridReq.current;
    api
      .grid(horizon)
      .then((g) => id === gridReq.current && setGrid(g))
      .catch(() => undefined);
  }, [horizon]);

  useEffect(() => {
    if (!selected) return;
    setForecast(null);
    setDrivers(null);
    api.forecast(selected).then(setForecast).catch(() => undefined);
    api.drivers(selected).then(setDrivers).catch(() => setDrivers(null));
  }, [selected]);

  const doRefresh = async () => {
    setBusy(true);
    try {
      await api.refresh();
      await loadCore();
      setGrid(await api.grid(horizon));
      if (selected) setForecast(await api.forecast(selected));
    } finally {
      setBusy(false);
    }
  };

  const selectedStation = useMemo(
    () => stations.find((s) => s.id === selected) ?? null,
    [stations, selected],
  );

  const validLabel = useMemo(() => {
    if (horizon === 0) return "Now";
    const p = forecast?.points.find((x) => x.horizon === horizon);
    return p ? new Date(p.valid_ts).toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata", weekday: "short", hour: "numeric", hour12: true,
    }) : fmtRelHour(horizon);
  }, [horizon, forecast]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <TopBar health={health} busy={busy} onRefresh={doRefresh} onMethodology={() => setShowMethod(true)} />
      <AlertsBanner alerts={alerts} onPick={setSelected} />

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.55fr) minmax(340px, 1fr)",
          gap: "var(--space-3)",
          padding: "var(--space-3)",
        }}
        className="app-grid"
      >
        <section className="card" style={{ display: "flex", flexDirection: "column", minHeight: 0, padding: 0, overflow: "hidden" }}>
          <div style={{ position: "relative", flex: 1, minHeight: 260 }}>
            <NcrMap
              stations={stations}
              grid={grid}
              fires={fires}
              selected={selected}
              onSelect={setSelected}
            />
            <Legend />
          </div>
          <TimeSlider
            horizon={horizon}
            onChange={setHorizon}
            validLabel={validLabel}
            issued={forecast?.issued_ts}
          />
        </section>

        <section style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)", minHeight: 0, overflowY: "auto" }}>
          <StationPanel station={selectedStation} forecast={forecast} horizon={horizon} onHover={setHorizon} />
          <DriversPanel drivers={drivers} station={selectedStation} modelName={health?.model_name} />
        </section>
      </div>

      {showMethod && <Methodology onClose={() => setShowMethod(false)} />}
      <style>{`
        @media (max-width: 1000px) {
          .app-grid { grid-template-columns: 1fr !important; grid-auto-rows: minmax(340px, auto); }
        }
      `}</style>
    </div>
  );
}

function worstStation(st: Station[]): string | null {
  const withAqi = st.filter((s) => s.latest_aqi != null);
  if (!withAqi.length) return st[0]?.id ?? null;
  return withAqi.sort((a, b) => (b.latest_aqi ?? 0) - (a.latest_aqi ?? 0))[0].id;
}

function TopBar({
  health,
  busy,
  onRefresh,
  onMethodology,
}: {
  health: Health | null;
  busy: boolean;
  onRefresh: () => void;
  onMethodology: () => void;
}) {
  const refreshed = health?.last_refresh
    ? new Date(health.last_refresh).toLocaleString("en-IN", {
        timeZone: "Asia/Kolkata", hour: "numeric", minute: "2-digit", hour12: true,
      })
    : "—";
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-4)",
        padding: "10px var(--space-4)",
        borderBottom: "1px solid var(--color-border)",
        background: "var(--color-card)",
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <h1>
          Vayu<span style={{ color: "var(--color-primary)" }}>Cast</span>
        </h1>
        <span className="muted" style={{ fontSize: 12 }}>
          Delhi NCR · coupled 72-hour AQI forecast
        </span>
      </div>
      <div style={{ flex: 1 }} />
      <span className="pill" style={{ background: "var(--color-muted)", color: "var(--color-muted-foreground)" }}>
        <Dot ok={health ? !health.stale : false} />
        {health?.model_name === "lgbm" ? "ML emulator" : health?.model_name === "naive" ? "baseline" : "…"}
        {health?.stale ? " · stale" : ""}
      </span>
      <span className="muted mono-num" style={{ fontSize: 12 }}>updated {refreshed} IST</span>
      <button onClick={onMethodology}>Methodology</button>
      <button onClick={onRefresh} disabled={busy} aria-pressed={busy}>
        {busy ? "Refreshing…" : "Refresh"}
      </button>
    </header>
  );
}

function Dot({ ok }: { ok: boolean }) {
  return (
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: 999,
        background: ok ? "var(--color-primary)" : "var(--color-accent)",
        display: "inline-block",
      }}
    />
  );
}
