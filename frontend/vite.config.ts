import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the FastAPI backend through a dev proxy so that the
// app works with relative URLs (no localhost hardcoding in browser code).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Allow the dynamic preview host (e.g. 5173-<id>.e2b.app) plus local dev.
    allowedHosts: [".e2b.app", "localhost", "127.0.0.1"],
    proxy: {
      "/predict": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
      "/files": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
