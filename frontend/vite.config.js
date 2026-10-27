import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api requests are forwarded to the FastAPI backend (no CORS trouble in development)
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true, proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } } },
});
