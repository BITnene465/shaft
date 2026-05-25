import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "manifestTools.ts");
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

const tmpDir = await mkdtemp(path.join(root, ".tmp-manifest-tools-"));
const modulePath = path.join(tmpDir, "manifestTools.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
const tools = await import(modulePath);

const layoutManifest = {
  kind: "eval_job",
  eval: {
    task: "detection",
    prompt_id: "grounding_layout.latest",
    target_labels: ["icon", "image", "shape"]
  }
};

const arrowPrompt = {
  prompt_id: "grounding_arrow.latest",
  label: "Arrow Detection",
  task: "detection",
  system_prompt: "JSON only.",
  user_prompt: "Detect arrows.",
  parser: "raw_data_detection_v1",
  metric_profile: "detection_iou_v1",
  visualization_profile: "default",
  generation: { max_tokens: 4096 },
  data: { max_pixels: 1048576 },
  metadata: { target_labels: ["arrow"] }
};

const arrowManifest = tools.applyPromptTemplateToManifest(layoutManifest, arrowPrompt);
assert.deepEqual(arrowManifest.eval.target_labels, ["arrow"]);
assert.equal(arrowManifest.eval.target_labels_source, "prompt_metadata");
assert.equal(arrowManifest.eval.prompt_id, "grounding_arrow.latest");
assert.equal(arrowManifest.eval.prompt_text, "Detect arrows.");

const benchmarkDefaultManifest = tools.applyBenchmarkDefault(
  { kind: "eval_job", eval: { task: "keypoint", benchmark_id: "" } },
  [
    {
      benchmark_id: "banana_bench",
      tasks: ["detection"],
      labels: ["arrow"],
      layers: [],
      split: "val",
      sample_count: 400,
      root: "",
      manifest_path: "",
      created_at: null,
      source_manifest_path: null
    },
    {
      benchmark_id: "banana_point_arrow_bench",
      tasks: ["keypoint"],
      labels: ["arrow"],
      layers: [],
      split: "val",
      sample_count: 1966,
      root: "",
      manifest_path: "",
      created_at: null,
      source_manifest_path: null
    }
  ]
);
assert.equal(benchmarkDefaultManifest.eval.benchmark_id, "banana_point_arrow_bench");

const taskMismatchBenchmarkManifest = tools.applyBenchmarkDefault(
  { kind: "eval_job", eval: { task: "keypoint", benchmark_id: "banana_bench" } },
  [
    {
      benchmark_id: "banana_bench",
      tasks: ["detection"],
      labels: ["arrow"],
      layers: [],
      split: "val",
      sample_count: 400,
      root: "",
      manifest_path: "",
      created_at: null,
      source_manifest_path: null
    },
    {
      benchmark_id: "banana_point_arrow_bench",
      tasks: ["keypoint"],
      labels: ["arrow"],
      layers: [],
      split: "val",
      sample_count: 1966,
      root: "",
      manifest_path: "",
      created_at: null,
      source_manifest_path: null
    }
  ]
);
assert.equal(taskMismatchBenchmarkManifest.eval.benchmark_id, "banana_point_arrow_bench");

const customPromptWithoutLabels = {
  ...arrowPrompt,
  prompt_id: "custom.no-labels",
  metadata: {}
};
const customManifest = tools.applyPromptTemplateToManifest(layoutManifest, customPromptWithoutLabels);
assert.equal("target_labels" in customManifest.eval, false);
assert.equal("target_labels_source" in customManifest.eval, false);

const explicitLabelManifest = tools.updateManifestTargetLabels(layoutManifest, ["icon"]);
assert.deepEqual(explicitLabelManifest.eval.target_labels, ["icon"]);
assert.equal(explicitLabelManifest.eval.target_labels_source, "explicit");

const defaultPolicyManifest = tools.updateManifestTargetLabels(explicitLabelManifest, []);
assert.equal("target_labels" in defaultPolicyManifest.eval, false);
assert.equal("target_labels_source" in defaultPolicyManifest.eval, false);

await rm(tmpDir, { recursive: true, force: true });
console.log("manifest tools checks passed");
