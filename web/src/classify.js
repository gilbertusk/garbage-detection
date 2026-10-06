// Menafsirkan keluaran model: softmax, tebakan teratas, dan keputusan "tidak yakin".

export const TOP_GUESSES = 3;

export function softmax(logits) {
  const values = Array.from(logits);
  const max = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - max));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

/**
 * @returns {{isConfident: boolean, top: {label: string, confidence: number}[]}}
 * top terurut dari yang paling yakin; isConfident false berarti app harus
 * menjawab "tidak yakin" alih-alih menebak.
 */
export function interpret(logits, labels, threshold, topCount = TOP_GUESSES) {
  if (logits.length !== labels.length) {
    throw new RangeError(
      `Jumlah keluaran model (${logits.length}) tidak sama dengan jumlah label (${labels.length})`,
    );
  }
  const top = softmax(logits)
    .map((confidence, index) => ({ label: labels[index], confidence }))
    .sort((first, second) => second.confidence - first.confidence)
    .slice(0, topCount);
  return { isConfident: top[0].confidence >= threshold, top };
}
