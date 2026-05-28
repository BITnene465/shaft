import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const shortcutSurfaceFiles = [
  "benchmarksPage.tsx",
  "runsPage.tsx",
  "sampleViewer.tsx",
  "viewerCanvas.tsx",
  "viewerPanels.tsx",
  "settingsPage.tsx",
  "settingsControls.tsx"
];
const shortcutSurfaceSource = (
  await Promise.all(
    shortcutSurfaceFiles.map((file) => readFile(path.join(root, "src", file), "utf8"))
  )
).join("\n");
const settingsSource = await readFile(path.join(root, "src", "workspaceSettings.ts"), "utf8");

const expectedActions = [
  "viewer.resetViewport",
  "sample.previous",
  "sample.next",
  "selection.clear",
  "layer.toggleGt",
  "layer.togglePred",
  "geometry.toggleBoxes",
  "geometry.toggleLines",
  "geometry.toggleKeypoints"
];

for (const action of expectedActions) {
  assert.ok(settingsSource.includes(`id: "${action}"`), `missing shortcut action ${action}`);
}

const hardCodedKeyChecks = [...shortcutSurfaceSource.matchAll(/event\.key\s*===\s*["']([^"']+)["']/g)].map(
  (match) => match[1]
);
const allowedLocalKeys = new Set(["Enter"]);
assert.deepEqual(
  hardCodedKeyChecks.filter((key) => !allowedLocalKeys.has(key)),
  [],
  "global viewer shortcuts must use actionForEvent instead of event.key"
);
assert.equal(
  countMatches(shortcutSurfaceSource, /window\.addEventListener\("keydown"/g),
  countMatches(shortcutSurfaceSource, /actionForEvent\(event\)/g),
  "every global keydown listener must resolve through actionForEvent"
);
assert.ok(
  countMatches(shortcutSurfaceSource, /useWorkspaceShortcuts\(\)/g) >= 5,
  "benchmark, run, canvas, viewer, and settings must consume shortcut settings"
);

const rawUrl = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/";
const shortcutTargets = await discoverShortcutTargets(new URL(rawUrl).origin);
const benchmarkId = process.env.EVAL_BENCH_BENCHMARK_ID ?? shortcutTargets.benchmarkId;
const runId = process.env.EVAL_BENCH_RUN_ID ?? shortcutTargets.runId;
if (!benchmarkId) {
  throw new Error("shortcut coverage requires a benchmark with at least two samples.");
}
if (!runId) {
  throw new Error("shortcut coverage requires an evaluated run with sample detail.");
}
const bindings = {
  "viewer.resetViewport": "R",
  "sample.previous": "M",
  "sample.next": "N",
  "selection.clear": "Q",
  "layer.toggleGt": "T",
  "layer.togglePred": "Y",
  "geometry.toggleBoxes": "U",
  "geometry.toggleLines": "I",
  "geometry.toggleKeypoints": "O"
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];

page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") {
    errors.push(message.text());
  }
});
await page.addInitScript((shortcutBindings) => {
  localStorage.setItem("eval_bench_shortcuts", JSON.stringify(shortcutBindings));
}, bindings);

await checkBenchmarkShortcuts(page);
await checkRunShortcuts(page);
await checkComparisonShortcuts(page);
await checkSettingsShortcutEditor(page);

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`shortcut coverage check passed ${appUrl("/")}`);
console.log(
  JSON.stringify(
    {
      static_actions_checked: expectedActions.length,
      benchmark_sample_navigation: true,
      run_sample_navigation: true,
      run_viewer_actions: true,
      comparison_viewer_actions: true,
      settings_keymap_editor: true,
      legacy_default_keys_bypassed_by_custom_bindings: true
    },
    null,
    2
  )
);

async function checkBenchmarkShortcuts(page) {
  await page.goto(appUrl(`/benchmarks/${benchmarkId}?sample=0`), { waitUntil: "networkidle" });
  await page.locator(".sample-row.selected").first().waitFor({ timeout: 10_000 });
  await expectSelectedSample(page, "1", "benchmark initial sample");
  await page.keyboard.press("]");
  await page.waitForTimeout(100);
  await expectSelectedSample(page, "1", "benchmark legacy next binding should be inactive");
  await page.keyboard.press(bindings["sample.next"]);
  await expectSelectedSample(page, "2", "benchmark sample.next binding");
  await page.keyboard.press(bindings["sample.previous"]);
  await expectSelectedSample(page, "1", "benchmark sample.previous binding");
}

async function discoverShortcutTargets(rootUrl) {
  try {
    const response = await fetch(`${rootUrl}/api/state`);
    if (!response.ok) {
      return { benchmarkId: "", runId: "" };
    }
    const state = await response.json();
    const benchmarks = Array.isArray(state.benchmarks) ? state.benchmarks : [];
    const benchmarkById = new Map(benchmarks.map((item) => [item.benchmark_id, item]));
    const benchmark = benchmarks.find((item) => Number(item.sample_count) > 1) ?? benchmarks[0];
    const runs = Array.isArray(state.runs) ? state.runs : [];
    const run =
      runs.find((item) => {
        if (!item?.run_id || !item?.report_path) {
          return false;
        }
        const runBenchmark = benchmarkById.get(item.benchmark_id);
        return !runBenchmark || Number(runBenchmark.sample_count) > 1;
      }) ??
      runs.find((item) => item?.run_id && item?.report_path);
    return {
      benchmarkId: benchmark?.benchmark_id ?? "",
      runId: run?.run_id ?? ""
    };
  } catch {
    return { benchmarkId: "", runId: "" };
  }
}

async function checkRunShortcuts(page) {
  await page.goto(appUrl(`/runs/${runId}?sample=0`), { waitUntil: "networkidle" });
  await page.locator(".viewer-canvas-layout .image-stage").first().waitFor({ timeout: 10_000 });
  await expectSelectedSample(page, "1", "run initial sample");
  await page.keyboard.press("]");
  await page.waitForTimeout(100);
  await expectSelectedSample(page, "1", "run legacy next binding should be inactive");
  await page.keyboard.press(bindings["sample.next"]);
  await expectSelectedSample(page, "2", "run sample.next binding");
  await page.keyboard.press(bindings["sample.previous"]);
  await expectSelectedSample(page, "1", "run sample.previous binding");

  await expectToggleActive(page, "真值", true, "GT initial");
  await page.keyboard.press("g");
  await page.waitForTimeout(100);
  await expectToggleActive(page, "真值", true, "legacy GT default key should not bypass custom binding");
  await page.keyboard.press(bindings["layer.toggleGt"]);
  await expectToggleActive(page, "真值", false, "custom GT toggle");
  await page.keyboard.press(bindings["layer.toggleGt"]);
  await expectToggleActive(page, "真值", true, "custom GT toggle restore");

  await expectToggleActive(page, "预测", true, "prediction initial");
  await page.keyboard.press(bindings["layer.togglePred"]);
  await expectToggleActive(page, "预测", false, "custom prediction toggle");
  await page.keyboard.press(bindings["layer.togglePred"]);
  await expectToggleActive(page, "预测", true, "custom prediction toggle restore");

  await expectToggleActive(page, "框", true, "boxes initial");
  await page.keyboard.press(bindings["geometry.toggleBoxes"]);
  await expectToggleActive(page, "框", false, "custom boxes toggle");
  await page.keyboard.press(bindings["geometry.toggleBoxes"]);
  await expectToggleActive(page, "框", true, "custom boxes toggle restore");

  await expectToggleActive(page, "线", true, "lines initial");
  await page.keyboard.press(bindings["geometry.toggleLines"]);
  await expectToggleActive(page, "线", false, "custom lines toggle");
  await page.keyboard.press(bindings["geometry.toggleLines"]);
  await expectToggleActive(page, "线", true, "custom lines toggle restore");

  await expectToggleActive(page, "点", true, "keypoints initial");
  await page.keyboard.press(bindings["geometry.toggleKeypoints"]);
  await expectToggleActive(page, "点", false, "custom keypoints toggle");
  await page.keyboard.press(bindings["geometry.toggleKeypoints"]);
  await expectToggleActive(page, "点", true, "custom keypoints toggle restore");

  const stage = page.locator(".viewer-canvas-layout .image-stage").first();
  const stageBox = await stage.boundingBox();
  if (!stageBox) {
    throw new Error("run image stage is not visible");
  }
  await page.mouse.move(stageBox.x + stageBox.width * 0.5, stageBox.y + stageBox.height * 0.5);
  await page.mouse.wheel(0, -500);
  await page.locator(".canvas-hud button").first().waitFor({ timeout: 3_000 });
  await page.keyboard.press(bindings["viewer.resetViewport"]);
  await page.waitForFunction(() => document.querySelectorAll(".canvas-hud button").length === 0);

  const firstObject = page.locator(".object-row").first();
  if ((await firstObject.count()) > 0) {
    await firstObject.click();
    await page.locator(".object-row.active").first().waitFor({ timeout: 3_000 });
    await page.mouse.move(8, 8);
    await page.keyboard.press(bindings["selection.clear"]);
    await page.waitForFunction(() => document.querySelectorAll(".object-row.active").length === 0);
  }
}

async function checkComparisonShortcuts(page) {
  await page.goto(appUrl(`/compare/${runId}/${runId}/0`), { waitUntil: "networkidle" });
  await page.locator(".comparison-run-panel .image-stage").first().waitFor({ timeout: 10_000 });
  const panelCount = await page.locator(".comparison-run-panel .image-stage").count();
  assert.ok(panelCount >= 2, "comparison sample page must render both viewer panels");
  await expectOverlayAtLeast(
    page,
    ".comparison-run-panel .overlay-instance.gt",
    1,
    "comparison GT overlay initial"
  );
  await page.keyboard.press(bindings["layer.toggleGt"]);
  await expectOverlayCount(
    page,
    ".comparison-run-panel .overlay-instance.gt",
    0,
    "comparison custom GT toggle"
  );
  await page.keyboard.press(bindings["layer.toggleGt"]);
  await expectOverlayAtLeast(
    page,
    ".comparison-run-panel .overlay-instance.gt",
    1,
    "comparison custom GT toggle restore"
  );
}

async function checkSettingsShortcutEditor(page) {
  await page.goto(appUrl("/settings"), { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /快捷键/ }).click();
  const rows = page.locator(".shortcut-map-row");
  await rows.first().waitFor({ timeout: 10_000 });
  assert.ok((await rows.count()) >= expectedActions.length, "settings shortcut table is incomplete");
  for (const action of ["viewer.resetViewport", "sample.next", "geometry.toggleKeypoints"]) {
    await page.locator(".shortcut-map-row").filter({ hasText: action }).first().waitFor({
      timeout: 3_000
    });
  }
  const sampleNextRow = page.locator(".shortcut-map-row").filter({ hasText: "sample.next" }).first();
  await sampleNextRow.locator(".shortcut-capture").focus();
  await page.keyboard.press("Control+Shift+J");
  await page.waitForFunction(() => {
    const raw = localStorage.getItem("eval_bench_shortcuts");
    return raw ? JSON.parse(raw)["sample.next"] === "Ctrl+Shift+J" : false;
  });
  await sampleNextRow.locator(".shortcut-capture", { hasText: "Ctrl+Shift+J" }).waitFor({
    timeout: 3_000
  });
}

async function expectSelectedSample(page, expected, context) {
  await page.waitForFunction(
    ({ expectedText }) =>
      document.querySelector(".sample-row.selected strong")?.textContent?.trim() === expectedText,
    { expectedText: expected },
    { timeout: 3_000 }
  ).catch(async () => {
    throw new Error(`${context}: expected selected sample ${expected}, got ${await selectedSampleText(page)}`);
  });
}

async function selectedSampleText(page) {
  return page.locator(".sample-row.selected strong").first().textContent().catch(() => "<missing>");
}

async function expectToggleActive(page, label, expected, context) {
  const toggle = page.locator(".layer-toggle-strip .control-check").filter({ hasText: label }).first();
  await toggle.waitFor({ timeout: 3_000 });
  await page.waitForFunction(
    ({ labelText, active }) => {
      const nodes = Array.from(document.querySelectorAll(".layer-toggle-strip .control-check"));
      const node = nodes.find((item) => item.textContent?.trim() === labelText);
      return Boolean(node?.classList.contains("active")) === active;
    },
    { labelText: label, active: expected },
    { timeout: 3_000 }
  ).catch(async () => {
    const className = await toggle.evaluate((node) => node.className);
    throw new Error(`${context}: expected active=${expected}, got class=${className}`);
  });
}

async function overlayCount(page, selector) {
  return page.locator(selector).count();
}

async function expectOverlayCount(page, selector, expected, context) {
  await page.waitForFunction(
    ({ cssSelector, expectedCount }) => document.querySelectorAll(cssSelector).length === expectedCount,
    { cssSelector: selector, expectedCount: expected },
    { timeout: 3_000 }
  ).catch(async () => {
    throw new Error(`${context}: expected ${selector} count ${expected}, got ${await overlayCount(page, selector)}`);
  });
}

async function expectOverlayAtLeast(page, selector, expected, context) {
  await page.waitForFunction(
    ({ cssSelector, minCount }) => document.querySelectorAll(cssSelector).length >= minCount,
    { cssSelector: selector, minCount: expected },
    { timeout: 3_000 }
  ).catch(async () => {
    throw new Error(`${context}: expected ${selector} count >= ${expected}, got ${await overlayCount(page, selector)}`);
  });
}

function appUrl(pathname) {
  const base = new URL(rawUrl);
  return new URL(pathname, base.origin).toString();
}

function countMatches(value, pattern) {
  return [...value.matchAll(pattern)].length;
}
