import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const fixture = (name) => fileURLToPath(new URL(`./fixtures/${name}.jpg`, import.meta.url));

// Gambar dari test split Garbage Dataset v2 (lisensi MIT); model Python menjawab
// ketiganya dengan benar dan yakin, jadi situs harus menjawab sama.
const PHOTOS = [
  { name: "battery", bin: "B3 (berbahaya)", guess: "Baterai" },
  { name: "biological", bin: "Organik", guess: "Sisa organik" },
  { name: "plastic", bin: "Anorganik / daur ulang", guess: "Plastik" },
];

async function openReady(page) {
  await page.goto("/");
  await expect(page.locator("body")).toHaveAttribute("data-ready", "true", { timeout: 60_000 });
}

test("halaman memuat pengetahuan sampah dan model tanpa error", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));

  await openReady(page);

  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Sampah ini masuk ke mana?");
  await expect(page.locator(".bin")).toHaveCount(6);
  await expect(page.locator("#status")).toHaveText("Model siap. Nyalakan kamera atau pilih foto.");
  await expect(page.locator("#metrics")).toContainText("Akurasi pada data uji");
  expect(errors).toEqual([]);
});

for (const photo of PHOTOS) {
  test(`foto ${photo.name} diarahkan ke tempat sampah yang benar`, async ({ page }) => {
    await openReady(page);

    await page.locator("#pick-photo").setInputFiles(fixture(photo.name));

    const result = page.locator("#result");
    await expect(result).toBeVisible();
    await expect(result).toHaveAttribute("data-label", photo.name);
    await expect(result).toHaveAttribute("data-confident", "true");
    await expect(page.locator("#bin")).toHaveText(photo.bin);
    await expect(page.locator("#guess")).toContainText(photo.guess);
    await expect(page.locator("#guesses li")).toHaveCount(3);
  });
}

test("foto tidak pernah meninggalkan perangkat", async ({ page, baseURL }) => {
  const requests = [];
  page.on("request", (request) => requests.push(request));
  await openReady(page);
  const beforeUpload = requests.length;

  await page.locator("#pick-photo").setInputFiles(fixture("plastic"));
  await expect(page.locator("#result")).toBeVisible();
  await page.waitForLoadState("networkidle");

  const origin = new URL(baseURL).origin;
  for (const request of requests) {
    const url = new URL(request.url());
    // blob: adalah foto yang dibaca dari memori browser sendiri, bukan jaringan
    expect(url.protocol === "blob:" || url.origin === origin, request.url()).toBe(true);
    expect(request.method(), request.url()).toBe("GET");
    expect(request.postData(), request.url()).toBeNull();
  }
  const afterUpload = requests.slice(beforeUpload).filter((r) => !r.url().startsWith("blob:"));
  expect(afterUpload.map((request) => request.url())).toEqual([]);
});

test("berkas yang bukan gambar ditolak dengan pesan yang jelas", async ({ page }) => {
  await openReady(page);

  await page.locator("#pick-photo").setInputFiles({
    name: "catatan.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("ini bukan gambar"),
  });

  await expect(page.locator("#status")).toContainText("Foto ini tidak bisa dibaca");
  await expect(page.locator("#result")).toBeHidden();
});

test("kamera menghasilkan jawaban langsung", async ({ page }) => {
  await openReady(page);

  await page.getByRole("button", { name: "Nyalakan kamera" }).click();

  await expect(page.locator("#camera")).toBeVisible();
  await expect(page.locator("#result")).toBeVisible();
  await expect(page.locator("#guesses li")).toHaveCount(3);
});

test("halaman tidak bergeser ke samping", async ({ page }) => {
  await openReady(page);

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );

  expect(overflow).toBeLessThanOrEqual(0);
});
