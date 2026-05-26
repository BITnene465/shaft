import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "formatters.ts");
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

const tmpDir = await mkdtemp(path.join(os.tmpdir(), "eval-bench-formatters-"));
const modulePath = path.join(tmpDir, "formatters.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
const formatters = await import(modulePath);

const scoredRun = {
  run_id: "run-a",
  model_id: "model-a",
  precision_iou50: 0.75,
  recall_iou50: 0.6
};

assert.equal(formatters.formatMetric(formatters.f1Score(0.75, 0.6)), "0.667");
assert.equal(formatters.f1Score(null, 0.6), null);
assert.equal(formatters.f1Score(0, 0), null);
assert.equal(formatters.runF1Score(scoredRun).toFixed(3), "0.667");
assert.equal(formatters.formatRunOption(scoredRun), "run-a / model-a / F1 0.667");

await rm(tmpDir, { recursive: true, force: true });
console.log("formatter checks passed");
