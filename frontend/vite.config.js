import { resolve } from "node:path";
import { defineConfig } from "vite";

const repoRoot = resolve(__dirname, "..");
const publicDir = resolve(repoRoot, "public");

export default defineConfig(({ command }) => {
  if (command === "serve") {
    // Local preview server: serves the already-built static site
    // (run `pixi run build-static` first) with live asset reloading.
    return {
      root: publicDir,
      publicDir: false,
    };
  }

  return {
    root: __dirname,
    publicDir: false,
    build: {
      outDir: resolve(publicDir, "static", "assets"),
      emptyOutDir: true,
      assetsDir: ".",
      rollupOptions: {
        input: {
          main: resolve(__dirname, "src", "main.js"),
        },
        output: {
          entryFileNames: "main.js",
          assetFileNames: "[name][extname]",
        },
      },
    },
  };
});
