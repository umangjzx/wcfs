import ReactDOM from "react-dom/client";
import App from "./App";
import "maplibre-gl/dist/maplibre-gl.css";
import "./index.css";

// StrictMode intentionally omitted: it double-invokes effects in dev, which churns
// the imperative MapLibre instance. Re-enable once the map lifecycle is hardened.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
