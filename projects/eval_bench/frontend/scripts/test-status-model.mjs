import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "statusModel.ts");
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

const tmpDir = await mkdtemp(path.join(root, ".tmp-status-model-"));
const modulePath = path.join(tmpDir, "statusModel.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
const model = await import(modulePath);

assert.equal(model.statusInfo("running", "job").label, "运行中");
assert.equal(model.statusInfo("running", "job").live, true);
assert.equal(model.statusInfo("running", "service").label, "服务就绪");
assert.equal(model.statusInfo("imported", "run").label, "待评估");
assert.equal(model.statusClassName("running", "service"), "badge success live");

assert.equal(model.canCancelJob({ status: "queued" }), true);
assert.equal(model.canCancelJob({ status: "running" }), true);
assert.equal(model.canDeleteJob({ status: "running" }), false);
assert.equal(model.canDeleteJob({ status: "failed" }), true);

assert.equal(model.canEvaluateRun({ status: "imported", prediction_count: 3 }), true);
assert.equal(model.canEvaluateRun({ status: "imported", prediction_count: 0 }), false);
assert.equal(model.canArchiveRun({ status: "archived" }), false);
assert.equal(model.canDeleteRun({ status: "running" }), false);

assert.equal(model.canStartService({ kind: "local_vllm", status: "registered" }), true);
assert.equal(model.canStartService({ kind: "external_vllm", status: "registered" }), false);
assert.equal(model.canStopService({ status: "starting" }), true);
assert.equal(model.canDeleteService({ status: "running" }), false);
assert.equal(model.canDeleteService({ status: "failed" }), true);
assert.equal(
  model.jobProgress({
    status: "running",
    metadata: { progress_done: "3", progress_total: 10, progress_phase: "inference" }
  }).text,
  "3/10 (30%)"
);
assert.equal(model.jobProgress({ status: "succeeded", metadata: {} }).percent, 100);
assert.equal(model.progressPhaseText("starting_runtime"), "启动模型服务");

await rm(tmpDir, { recursive: true, force: true });
console.log("status model checks passed");
