import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // The API runs separately; proxying keeps the origin single so a read-only
  // service needs no CORS configuration. Applied to `preview` as well as
  // `dev`, so the production bundle can be exercised against the real API
  // before it is ever deployed.
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  preview: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: true },
});
