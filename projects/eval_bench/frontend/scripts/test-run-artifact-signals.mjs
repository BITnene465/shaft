import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "runArtifactSignals.ts");
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

const tmpDir = await mkdtemp(path.join(os.tmpdir(), "eval-bench-run-artifacts-"));
const modulePath = path.join(tmpDir, "runArtifactSignals.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
const signals = await import(modulePath);

const baseRun = {
  run_id: "run",
  status: "evaluated",
  benchmark_id: "bench",
  tasks: ["detection"],
  spec_task: "detection",
  target_labels: ["arrow"],
  model_id: "model",
  model_path: "model",
  prompt_id: "prompt",
  prompt_path: null,
  prompt_hash: null,
  prompt_metadata: {},
  parser: "json",
  metric_profile: "detection_iou_v1",
  visualization_profile: "bbox",
  inference: {},
  created_at: "2026-05-26T10:00:00Z",
  prediction_count: 0,
  report_count: 0,
  manifest_path: "run.json",
  report_path: null,
  note: "",
  note_updated_at: null,
  note_max_length: 20_000,
  f1_iou50: null,
  precision_iou50: null,
  recall_iou50: null,
  mean_iou: null
};

const sorted = signals.recentRunsByCreatedAt(
  [
    { ...baseRun, run_id: "older", created_at: "2026-05-26T09:00:00Z" },
    { ...baseRun, run_id: "newer", created_at: "2026-05-26T11:00:00Z" },
    { ...baseRun, run_id: "unknown", created_at: null }
  ],
  2
);
assert.deepEqual(sorted.map((run) => run.run_id), ["newer", "older"]);

assert.deepEqual(signals.runArtifactReadiness(baseRun), {
  label: "draft",
  percent: 8,
  tone: "draft"
});
assert.deepEqual(
  signals.runArtifactReadiness({
    ...baseRun,
    prediction_count: 12,
    note: "repro clue"
  }),
  {
    label: "pred + note",
    percent: 60,
    tone: "ready"
  }
);
assert.deepEqual(
  signals.runArtifactReadiness({
    ...baseRun,
    prediction_count: 12,
    report_count: 1,
    report_path: "reports/metrics.json"
  }),
  {
    label: "report ready",
    percent: 88,
    tone: "complete"
  }
);

const now = Date.parse("2026-05-26T12:00:00Z");
assert.equal(signals.runAgeLabel(null, now), "-");
assert.equal(signals.runAgeLabel("not-a-date", now), "-");
assert.equal(signals.runAgeLabel("2026-05-26T11:59:40Z", now), "now");
assert.equal(signals.runAgeLabel("2026-05-26T11:45:00Z", now), "15m");
assert.equal(signals.runAgeLabel("2026-05-26T07:00:00Z", now), "5h");
assert.equal(signals.runAgeLabel("2026-05-23T12:00:00Z", now), "3d");

await rm(tmpDir, { recursive: true, force: true });
console.log("run artifact signal checks passed");
