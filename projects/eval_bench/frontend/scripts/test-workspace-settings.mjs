import { strict as assert } from "node:assert";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const sourcePath = path.join(root, "src", "workspaceSettings.ts");
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

const tmpDir = await mkdtemp(path.join(root, ".tmp-workspace-settings-"));
const modulePath = path.join(tmpDir, "workspaceSettings.mjs");
await writeFile(modulePath, compiled.outputText, "utf8");
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
  assert.ok(Math.abs(parsed - midpoint) < 1e-12, `${control.key} control round-trip`);
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

await rm(tmpDir, { recursive: true, force: true });
console.log("workspace settings checks passed");
