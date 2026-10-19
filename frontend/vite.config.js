import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API and WebSocket to a gateway on localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
