import { defineConfig, devices } from "@playwright/test";

// Port sendiri, terpisah dari `npm run serve`, supaya tes selalu memakai servernya sendiri
const PORT = 4173;

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  // Tiap halaman meng-compile WASM 11 MB; terlalu banyak browser sekaligus membuatnya lambat
  workers: 2,
  timeout: 90_000,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // Kamera palsu bawaan Chromium, supaya alur kamera bisa diuji tanpa perangkat
          args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        },
      },
    },
    { name: "mobile", use: { ...devices["Pixel 7"] }, grepInvert: /kamera/ },
  ],
  webServer: {
    // http-server tahan terhadap banyak koneksi paralel; python -m http.server sesekali
    // membuat satu halaman menggantung saat memuat model
    command: `npx http-server web -p ${PORT} -c-1 --silent`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
  },
});
