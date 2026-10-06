import { defineConfig, devices } from "@playwright/test";

const PORT = 8000;

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
    command: `python -m http.server ${PORT} --directory web`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
  },
});
