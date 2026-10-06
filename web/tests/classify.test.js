import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { describeClass } from "../src/bins.js";
import { interpret, softmax } from "../src/classify.js";

const LABELS = ["glass", "metal", "plastic", "trash"];

test("softmax sums to one and keeps the order of the logits", () => {
  const probabilities = softmax([2, 1, 0]);

  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
});

test("softmax survives very large logits", () => {
  assert.deepEqual(softmax([1000, 1000]), [0.5, 0.5]);
});

test("confident prediction returns the three best guesses in order", () => {
  const result = interpret([0, 5, 1, 2], LABELS, 0.6);

  assert.equal(result.isConfident, true);
  assert.deepEqual(result.top.map((guess) => guess.label), ["metal", "trash", "plastic"]);
});

test("prediction below the threshold is reported as not confident", () => {
  const result = interpret([1, 1.1, 1, 1], LABELS, 0.6);

  assert.equal(result.isConfident, false);
  assert.equal(result.top[0].label, "metal");
});

test("model output with the wrong number of classes is rejected", () => {
  assert.throws(() => interpret([1, 2], LABELS, 0.6), RangeError);
});

test("every class the exported model knows has a name and a bin", () => {
  const modelConfig = JSON.parse(
    readFileSync(new URL("../model/model.json", import.meta.url), "utf8"),
  );

  for (const label of modelConfig.labels) {
    const info = describeClass(label);
    assert.ok(info.name && info.bin.name, label);
  }
});

test("unknown class is rejected instead of shown without a bin", () => {
  assert.throws(() => describeClass("unicorn"), RangeError);
});
