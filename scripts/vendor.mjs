// Salin berkas pihak ketiga dari node_modules ke web/vendor, supaya situs tidak
// memuat apa pun dari CDN: ONNX Runtime Web (WASM, satu thread) dan dua font.
import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const COPIES = [
  {
    from: join("node_modules", "onnxruntime-web", "dist"),
    to: join("web", "vendor", "ort"),
    files: ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"],
  },
  {
    from: join("node_modules", "@fontsource-variable", "bricolage-grotesque", "files"),
    to: join("web", "vendor", "fonts"),
    files: ["bricolage-grotesque-latin-wght-normal.woff2"],
  },
  {
    from: join("node_modules", "@fontsource-variable", "instrument-sans", "files"),
    to: join("web", "vendor", "fonts"),
    files: ["instrument-sans-latin-wght-normal.woff2"],
  },
];

for (const { from, to, files } of COPIES) {
  mkdirSync(to, { recursive: true });
  for (const file of files) {
    copyFileSync(join(from, file), join(to, file));
    console.log(`disalin: ${join(to, file)}`);
  }
}
