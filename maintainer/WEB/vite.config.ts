import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    fs: {
      allow: [here, fileURLToPath(new URL("../../assets", import.meta.url))],
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
