// Pemetaan kelas model -> nama Indonesia, tempat sampah, dan saran singkat.
// Ini contoh umum; sesuaikan BIN_OF_CLASS dengan aturan daerah Anda.

export const BINS = Object.freeze({
  organik: { key: "organik", name: "Organik", color: "#2f8f4e" },
  anorganik: { key: "anorganik", name: "Anorganik / daur ulang", color: "#d9a300" },
  kertas: { key: "kertas", name: "Kertas", color: "#2b6cb0" },
  b3: { key: "b3", name: "B3 (berbahaya)", color: "#c8372d" },
  tekstil: { key: "tekstil", name: "Donasi / tekstil", color: "#7b4fb5" },
  residu: { key: "residu", name: "Residu", color: "#4b5157" },
});

const CLASSES = Object.freeze({
  battery: {
    name: "Baterai",
    bin: "b3",
    tip: "Jangan dibuang ke tempat sampah biasa. Kumpulkan, lalu serahkan ke titik pengumpulan limbah B3.",
  },
  biological: {
    name: "Sisa organik",
    bin: "organik",
    tip: "Pisahkan dari plastik pembungkusnya. Bisa dijadikan kompos.",
  },
  cardboard: {
    name: "Kardus",
    bin: "kertas",
    tip: "Lipat sampai pipih dan jaga tetap kering. Kardus berminyak masuk residu.",
  },
  clothes: {
    name: "Pakaian",
    bin: "tekstil",
    tip: "Kalau masih layak pakai, cuci lalu donasikan.",
  },
  glass: {
    name: "Kaca",
    bin: "anorganik",
    tip: "Bilas dulu. Kalau pecah, bungkus dengan kertas tebal supaya tidak melukai petugas.",
  },
  metal: {
    name: "Logam",
    bin: "anorganik",
    tip: "Bilas kaleng dan tekan sampai pipih supaya hemat tempat.",
  },
  paper: {
    name: "Kertas",
    bin: "kertas",
    tip: "Jaga tetap kering dan bersih. Tisu bekas dan kertas berminyak masuk residu.",
  },
  plastic: {
    name: "Plastik",
    bin: "anorganik",
    tip: "Kosongkan, bilas, lalu remas. Lepas label dan tutupnya kalau bisa.",
  },
  shoes: {
    name: "Sepatu",
    bin: "tekstil",
    tip: "Ikat sepasang supaya tidak terpisah. Kalau masih layak, donasikan.",
  },
  trash: {
    name: "Residu",
    bin: "residu",
    tip: "Tidak bisa didaur ulang. Bungkus rapat sebelum dibuang.",
  },
});

/** @returns {{name: string, tip: string, bin: {key: string, name: string, color: string}}} */
export function describeClass(label) {
  const info = CLASSES[label];
  if (!info) {
    throw new RangeError(`Kelas tidak dikenal: ${label}`);
  }
  return { name: info.name, tip: info.tip, bin: BINS[info.bin] };
}
