import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "workspaceSettings.ts");
const source = await readFile(sourcePath, "utf8");
const tmpDir = await mkdtemp(path.join(root, ".tmp-workspace-settings-"));
await writeCompiledModule("workspaceSettingsSchema");
await writeCompiledModule("workspaceSettingsStorage");
await writeCompiledModule("workspaceSettings");
const modulePath = path.join(tmpDir, "workspaceSettings.mjs");
const settings = await import(modulePath);

const overlayNumberKeys = Object.keys(settings.DEFAULT_OVERLAY_STYLE)
  .filter((key) => key !== "predLineStyle")
  .sort();
const overlayControlKeys = settings.OVERLAY_STYLE_CONTROLS.map((control) => control.key).sort();
assert.deepEqual(overlayControlKeys, overlayNumberKeys);

const interactionKeys = Object.keys(settings.DEFAULT_INTERACTION_SETTINGS).sort();
const interactionControlKeys = settings.INTERACTION_SETTING_CONTROLS.map((control) => control.key).sort();
assert.deepEqual(interactionControlKeys, interactionKeys);

const allControls = [
  ...settings.OVERLAY_STYLE_CONTROLS,
  ...settings.INTERACTION_SETTING_CONTROLS
];
for (const control of allControls) {
  assert.equal(typeof control.label, "string");
  assert.ok(control.label.length > 0);
  assert.ok(control.min < control.max, `${control.key} min/max`);
  assert.ok(control.step > 0, `${control.key} step`);
  const midpoint = (control.min + control.max) / 2;
  const rendered = settings.settingControlValue(midpoint, control);
  const parsed = settings.settingValueFromControl(rendered, control);
  assert.ok(Math.abs(parsed - midpoint) <= control.step / 2 + 1e-12, `${control.key} control round-trip`);
}

const wheelControl = settings.INTERACTION_SETTING_CONTROLS.find(
  (control) => control.key === "wheelZoomSensitivity"
);
assert.equal(wheelControl.scale, 100000);
assert.equal(
  settings.settingControlValue(settings.DEFAULT_INTERACTION_SETTINGS.wheelZoomSensitivity, wheelControl),
  32
);

assert.deepEqual(
  settings.PRED_LINE_STYLE_OPTIONS.map((option) => option.value).sort(),
  ["dashed", "solid"]
);
assert.deepEqual(
  settings.INSTANCE_COLOR_ROLES.map((role) => role.key).sort(),
  ["fn", "fp", "gt", "pred"]
);
assert.equal(settings.labelColorKey(" Arrow "), "arrow");
assert.equal(settings.explicitLabelColor({ arrow: { gt: "#00ff00" } }, "ARROW", "gt"), "#00ff00");
assert.equal(settings.explicitLabelColor({ arrow: { gt: "#00ff00" } }, "ARROW", "pred"), undefined);
assert.ok(settings.SHORTCUT_ACTIONS.some((action) => action.id === "viewer.resetViewport"));
assert.ok(settings.SHORTCUT_ACTIONS.some((action) => action.id === "sample.previous"));
assert.ok(settings.SHORTCUT_ACTIONS.some((action) => action.id === "sample.next"));
assert.equal(settings.normalizeShortcutBinding("f"), "F");
assert.equal(settings.normalizeShortcutBinding("esc"), "Escape");
assert.equal(settings.normalizeShortcutBinding(" "), "Space");
assert.equal(settings.normalizeShortcutBinding("ctrl + shift + b"), "Ctrl+Shift+B");
assert.equal(
  settings.shortcutEventBinding({
    key: "b",
    ctrlKey: true,
    altKey: false,
    shiftKey: true,
    metaKey: false
  }),
  "Ctrl+Shift+B"
);
assert.equal(
  settings.shortcutEventBinding({
    key: "Shift",
    ctrlKey: false,
    altKey: false,
    shiftKey: true,
    metaKey: false
  }),
  ""
);
assert.deepEqual(
  settings.reconcileViewerLabelPreference({
    current: [],
    labels: ["arrow"],
    previousLabels: [],
    hasStoredPreference: false
  }),
  ["arrow"],
  "viewer should default to visible labels only before the user stores a preference"
);
assert.deepEqual(
  settings.reconcileViewerLabelPreference({
    current: ["arrow"],
    labels: ["icon"],
    previousLabels: ["arrow"],
    hasStoredPreference: true
  }),
  ["arrow", "icon"],
  "viewer must avoid an empty active label set when switching to a disjoint label scope"
);
assert.deepEqual(
  settings.reconcileViewerLabelPreference({
    current: ["arrow"],
    labels: ["arrow", "icon"],
    previousLabels: ["arrow"],
    hasStoredPreference: false
  }),
  ["arrow", "icon"],
  "viewer unstored all-label mode should pick up new labels without losing the previous preference"
);
assert.deepEqual(settings.visibleViewerLabels(["arrow"], ["icon"]), []);
assert.deepEqual(
  settings.applyViewerVisibleLabelSelection(["arrow"], ["icon"], (current) => [...current, "icon"]),
  ["arrow", "icon"],
  "viewer visible label changes must keep hidden label preferences"
);
assert.ok(source.includes("preferredLabels"));
assert.ok((await readFile(path.join(root, "src", "workspaceSettingsStorage.ts"), "utf8")).includes("hiddenPreference"));
assert.ok(source.includes("!nextPreference.some"));

await rm(tmpDir, { recursive: true, force: true });
console.log("workspace settings checks passed");

async function writeCompiledModule(moduleName) {
  const inputPath = path.join(root, "src", `${moduleName}.ts`);
  const input = await readFile(inputPath, "utf8");
  const compiled = ts.transpileModule(input, {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
      importsNotUsedAsValues: ts.ImportsNotUsedAsValues.Remove,
      isolatedModules: true,
      strict: true
    },
    fileName: inputPath
  });
  await writeFile(
    path.join(tmpDir, `${moduleName}.mjs`),
    compiled.outputText
      .replaceAll('from "./workspaceSettingsSchema"', 'from "./workspaceSettingsSchema.mjs"')
      .replaceAll('from "./workspaceSettingsStorage"', 'from "./workspaceSettingsStorage.mjs"'),
    "utf8"
  );
}
