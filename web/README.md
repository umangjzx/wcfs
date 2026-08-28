# VayuCast web

React + Vite + TypeScript + MapLibre GL dashboard for the Delhi-NCR coupled AQI forecast.

```bash
npm install
npm run dev        # http://localhost:5173  (proxies /api -> VITE_API_TARGET)
npm run build      # type-check + production bundle to dist/
```

## Environment

| Var | Default | Purpose |
| --- | --- | --- |
| `VITE_API_TARGET` | `http://localhost:8000` | Dev-server proxy target for `/api` (see `vite.config.ts`) |
| `VITE_API_BASE` | *(empty)* | Production: absolute API origin; empty = same-origin `/api` |

Set them in a `.env` file (git-ignored) or the shell before `npm run dev`.

## Design

Tokens in `src/index.css` come from `design-system/vayucast/MASTER.md`
(ui-ux-pro-max): dark OLED operations theme, CPCB AQI category scale,
Fira Sans / Fira Code, subtle motion, `prefers-reduced-motion` respected.

## Structure

- `src/App.tsx` — layout + data orchestration
- `src/components/NcrMap.tsx` — MapLibre map: AQI choropleth (IDW grid), station markers, fire hotspots, plume-transport line
- `src/components/TimeSlider.tsx` — now → +72 h
- `src/components/StationPanel.tsx` — 72 h P50 line + P10–P90 band (Recharts), category, advisory
- `src/components/DriversPanel.tsx` — ISI + stubble meters, grouped SHAP contribution bars
- `src/components/AlertsBanner.tsx` — soonest Very Poor / Severe crossing
- `src/components/Methodology.tsx` — model card + backtest + WRF-Chem note
