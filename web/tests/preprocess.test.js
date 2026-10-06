import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { centerCropRect, toTensorData } from "../src/preprocess.js";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TOLERANCE = 1e-5;

test("tensor matches the torchvision fixture value for value", () => {
  // Fixture yang sama diperiksa terhadap torchvision di tests/test_export.py
  const fixture = JSON.parse(
    readFileSync(new URL("./fixtures/preprocess.json", import.meta.url), "utf8"),
  );

  const actual = toTensorData(Uint8ClampedArray.from(fixture.rgba), MEAN, STD);

  assert.equal(actual.length, fixture.expected.length);
  fixture.expected.forEach((expected, index) => {
    assert.ok(Math.abs(actual[index] - expected) < TOLERANCE, `indeks ${index}`);
  });
});

test("tensor is laid out channel-first and ignores alpha", () => {
  const twoPixels = Uint8ClampedArray.from([255, 0, 0, 7, 0, 255, 0, 9]);

  const actual = toTensorData(twoPixels, [0, 0, 0], [1, 1, 1]);

  assert.deepEqual(Array.from(actual), [1, 0, 0, 1, 0, 0]);
});

test("rgba data with a broken length is rejected", () => {
  assert.throws(() => toTensorData(new Uint8ClampedArray(5), MEAN, STD), RangeError);
});

test("crop is the centered square of a landscape frame", () => {
  assert.deepEqual(centerCropRect(1280, 720, 0.875), { sx: 325, sy: 45, size: 630 });
});

test("crop is the centered square of a portrait frame", () => {
  assert.deepEqual(centerCropRect(720, 1280, 0.875), { sx: 45, sy: 325, size: 630 });
});

test("crop rejects a frame without size", () => {
  assert.throws(() => centerCropRect(0, 0, 0.875), RangeError);
});
