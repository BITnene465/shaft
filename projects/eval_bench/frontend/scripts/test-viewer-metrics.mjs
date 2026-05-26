import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "viewerMetrics.ts");
const source = await readFile(sourcePath, "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
    importsNotUsedAsValues: ts.ImportsNotUsedAsValues.Remove,
    isolatedModules: true,
    strict: true
  },
  fileName: sourcePath
});

const tmpDir = await mkdtemp(path.join(os.tmpdir(), "eval-bench-viewer-metrics-"));
const modulePath = path.join(tmpDir, "viewerMetrics.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
const metrics = await import(modulePath);

const detail = {
  gt_instances: [
    { label: "arrow", bbox: [10, 20, 90, 120] },
    { label: "icon", bbox: [150, 80, 220, 160] },
    { label: "icon", bbox: [260, 90, 330, 170] }
  ],
  pred_instances: [
    { label: "arrow", bbox: [12, 22, 88, 118] },
    { label: "icon", bbox: [500, 500, 540, 540] }
  ],
  diagnostics: {
    matched_count: 1,
    false_negative_count: 2,
    false_positive_count: 1,
    mean_iou: 0.86,
    matches: [{ label: "arrow", gt_index: 0, pred_index: 0, iou: 0.86 }],
    false_negatives: [
      { index: 1, label: "icon", bbox: [150, 80, 220, 160] },
      { index: 2, label: "icon", bbox: [260, 90, 330, 170] }
    ],
    false_positives: [{ index: 1, label: "icon", bbox: [500, 500, 540, 540] }],
    labels: {
      arrow: {
        gt_count: 1,
        pred_count: 1,
        matched_count: 1,
        false_negative_count: 0,
        false_positive_count: 0,
        mean_iou: 0.86
      },
      icon: {
        gt_count: 2,
        pred_count: 1,
        matched_count: 0,
        false_negative_count: 2,
        false_positive_count: 1,
        mean_iou: 0
      }
    }
  }
};

const labels = new Set(["arrow", "icon"]);
assert.deepEqual(metrics.countInstancesByLabel(detail.gt_instances), { arrow: 1, icon: 2 });

const summary = metrics.visibleSampleMetrics(detail, labels);
assert.equal(summary.gtCount, 3);
assert.equal(summary.predCount, 2);
assert.equal(summary.matchedCount, 1);
assert.equal(summary.falsePositiveCount, 1);
assert.equal(summary.falseNegativeCount, 2);
assert.equal(summary.meanIou, 0.86);

const objectRows = metrics.buildObjectRows({
  gtInstances: detail.gt_instances,
  predInstances: detail.pred_instances,
  labels,
  diagnostics: detail.diagnostics
});
const gtArrow = objectRows.find((row) => row.id === "gt:0");
const predArrow = objectRows.find((row) => row.id === "pred:0");
const predIcon = objectRows.find((row) => row.id === "pred:1");
assert.equal(gtArrow.status, "match");
assert.equal(gtArrow.matchIndex, 0);
assert.equal(gtArrow.iou, 0.86);
assert.equal(predArrow.status, "match");
assert.equal(predArrow.matchIndex, 0);
assert.equal(predIcon.status, "fp");
assert.equal(metrics.objectStatusLabel("fn"), "FN");
assert.equal(metrics.objectMetricText(gtArrow, (value) => value.toFixed(2)), "pred #1 / IoU 0.86");
assert.equal(metrics.formatBbox([1, 2.25, 3, 4.5]), "bbox 1, 2.3, 3, 4.5");

await rm(tmpDir, { recursive: true, force: true });
console.log("viewer metric checks passed");
