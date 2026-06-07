import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");
const loadingCases = [
  { path: "/benchmarks", delayedApi: "/api/benchmarks", label: "正在加载基准集" },
  { path: "/runs", delayedApi: "/api/runs", label: "正在加载评测记录" },
  { path: "/rank-board", delayedApi: "/api/rank-board", label: "正在加载排行榜" },
  { path: "/jobs", delayedApi: "/api/jobs", label: "正在加载队列状态" },
  { path: "/services", delayedApi: "/api/services", label: "正在加载服务" }
];

const browser = await chromium.launch({ headless: true });
const errors = [];

for (const loadingCase of loadingCases) {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        errors.push(text);
      }
    }
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === loadingCase.delayedApi) {
      await delay(1_000);
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await page.goto(`${baseUrl}${loadingCase.path}`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page
    .locator(".table-loading .table-refresh-indicator", { hasText: loadingCase.label })
    .waitFor({ timeout: 5_000 });
  assert.equal(
    await page.locator(".empty-panel", { hasText: loadingCase.label }).count(),
    0,
    `${loadingCase.path} must not use a full-page empty panel for first load`
  );
  await page.waitForTimeout(1_100);
  await page.close();
}

await assertReducedMotionLoadingState();
await assertDarkThemeLoadingState();
await assertDarkRankBoardTableControls();

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`loading state check passed on ${baseUrl}`);

async function assertReducedMotionLoadingState() {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  await page.emulateMedia({ reducedMotion: "reduce" });
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        errors.push(text);
      }
    }
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/runs") {
      await delay(1_000);
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await page.goto(`${baseUrl}/runs`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".table-loading .table-skeleton-line").first().waitFor({ timeout: 5_000 });
  const skeletonAnimationName = await page
    .locator(".table-loading .table-skeleton-line")
    .first()
    .evaluate((element) => getComputedStyle(element).animationName);
  const refreshAnimationName = await page
    .locator(".table-loading.table-shell")
    .first()
    .evaluate((element) => getComputedStyle(element, "::after").animationName);
  assert.equal(skeletonAnimationName, "none", "reduced-motion loading skeleton must not animate");
  assert.equal(refreshAnimationName, "none", "reduced-motion table refresh line must not animate");
  await page.waitForTimeout(1_100);
  await page.close();
}

async function assertDarkThemeLoadingState() {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        errors.push(text);
      }
    }
  });
  await page.addInitScript(() => {
    localStorage.setItem("eval_bench_theme_mode", "dark");
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/runs") {
      await delay(1_000);
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await page.goto(`${baseUrl}/runs`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".table-loading .table-skeleton-line").first().waitFor({ timeout: 5_000 });
  const snapshot = await page.locator(".table-loading").first().evaluate((shell) => {
    const skeleton = shell.querySelector(".table-skeleton-line");
    const indicator = shell.querySelector(".table-refresh-indicator");
    const shellStyle = getComputedStyle(shell);
    const skeletonStyle = skeleton ? getComputedStyle(skeleton) : null;
    const indicatorStyle = indicator ? getComputedStyle(indicator) : null;
    return {
      theme: document.documentElement.dataset.theme,
      shellBackground: shellStyle.backgroundColor,
      skeletonGlint: skeletonStyle?.getPropertyValue("--table-loading-glint").trim() ?? "",
      indicatorBackground: indicatorStyle?.backgroundColor ?? ""
    };
  });
  assert.equal(snapshot.theme, "dark", "dark loading smoke must run under the stored dark theme");
  assert(
    colorLuminance(snapshot.shellBackground) < 80,
    `dark loading table shell must not fall back to a bright surface: ${snapshot.shellBackground}`
  );
  assert(
    !snapshot.skeletonGlint.includes("255 255 255"),
    `dark loading skeleton glint must not use white shimmer: ${snapshot.skeletonGlint}`
  );
  assert(
    colorLuminance(snapshot.indicatorBackground) < 120,
    `dark loading indicator must not use a bright pill surface: ${snapshot.indicatorBackground}`
  );
  await page.locator(".run-note-preview").first().waitFor({ timeout: 5_000 });
  await page.locator(".run-note-preview.empty").first().waitFor({ timeout: 5_000 });
  const rowControlSnapshot = await page.evaluate(() => {
    const noted = document.querySelector(".run-note-preview:not(.empty)");
    const empty = document.querySelector(".run-note-preview.empty");
    const checkbox = document.querySelector(".row-select-checkbox");
    return {
      notedBackground: noted ? getComputedStyle(noted).backgroundColor : "",
      emptyBackground: empty ? getComputedStyle(empty).backgroundColor : "",
      checkboxBackground: checkbox ? getComputedStyle(checkbox).backgroundColor : "",
      checkboxBorder: checkbox ? getComputedStyle(checkbox).borderColor : ""
    };
  });
  assert(
    colorLuminance(rowControlSnapshot.notedBackground) < 120,
    `dark run note preview must not use a bright filled surface: ${rowControlSnapshot.notedBackground}`
  );
  assert(
    colorLuminance(rowControlSnapshot.emptyBackground) < 120,
    `dark empty run note preview must not use a bright empty surface: ${rowControlSnapshot.emptyBackground}`
  );
  assert(
    colorLuminance(rowControlSnapshot.checkboxBackground) < 120,
    `dark row selection checkbox must not use a bright native surface: ${rowControlSnapshot.checkboxBackground}`
  );
  assert(
    colorLuminance(rowControlSnapshot.checkboxBorder) < 150,
    `dark row selection checkbox border must stay within the dark table palette: ${rowControlSnapshot.checkboxBorder}`
  );
  await page.waitForTimeout(1_100);
  await page.close();
}

async function assertDarkRankBoardTableControls() {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        errors.push(text);
      }
    }
  });
  await page.addInitScript(() => {
    localStorage.setItem("eval_bench_theme_mode", "dark");
  });
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await page.goto(`${baseUrl}/rank-board`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".rank-primary-score").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-score-delta").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-sort-active-cell").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-facet-group").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-facet-button").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-facet-toggle").first().waitFor({ timeout: 5_000 });
  await page.locator(".rank-board-pager").first().waitFor({ timeout: 5_000 });
  const snapshot = await page.evaluate(() => {
    const modeSwitch = document.querySelector(".rank-mode-switch");
    const toolbar = document.querySelector(".rank-board-table-toolbar");
    const pager = document.querySelector(".rank-board-pager");
    const pagerText = pager?.querySelector("span");
    const summaryTitle = document.querySelector(".rank-board-summary strong");
    const summaryMeta = document.querySelector(".rank-board-summary span");
    const score = document.querySelector(".rank-primary-score");
    const delta = document.querySelector(".rank-score-delta");
    const activeCell = document.querySelector("td.rank-sort-active-cell");
    const activeHead = document.querySelector("th.rank-sort-active-cell");
    const facetGroup = document.querySelector(".rank-facet-group");
    const facetButton = document.querySelector(".rank-facet-button");
    const facetToggle = document.querySelector(".rank-facet-toggle");
    return {
      modeSwitchBackground: modeSwitch ? getComputedStyle(modeSwitch).backgroundColor : "",
      toolbarBorder: toolbar ? getComputedStyle(toolbar).borderBottomColor : "",
      pagerBackground: pager ? getComputedStyle(pager).backgroundColor : "",
      pagerTextColor: pagerText ? getComputedStyle(pagerText).color : "",
      summaryTitleColor: summaryTitle ? getComputedStyle(summaryTitle).color : "",
      summaryMetaColor: summaryMeta ? getComputedStyle(summaryMeta).color : "",
      scoreBackground: score ? getComputedStyle(score).backgroundColor : "",
      deltaBackground: delta ? getComputedStyle(delta).backgroundColor : "",
      activeCellBackground: activeCell ? getComputedStyle(activeCell).backgroundColor : "",
      activeHeadBackground: activeHead ? getComputedStyle(activeHead).backgroundColor : "",
      facetGroupBackground: facetGroup ? getComputedStyle(facetGroup).backgroundColor : "",
      facetButtonBackground: facetButton ? getComputedStyle(facetButton).backgroundColor : "",
      facetToggleBackground: facetToggle ? getComputedStyle(facetToggle).backgroundColor : ""
    };
  });
  assert(
    colorLuminance(snapshot.modeSwitchBackground) < 120,
    `dark rank mode switch must not use a bright switch surface: ${snapshot.modeSwitchBackground}`
  );
  assert(
    colorLuminance(snapshot.toolbarBorder) < 150,
    `dark rank toolbar divider must stay within the dark table palette: ${snapshot.toolbarBorder}`
  );
  assert(
    colorLuminance(snapshot.pagerBackground) < 120,
    `dark rank pager must not use a bright pagination surface: ${snapshot.pagerBackground}`
  );
  assert(
    colorLuminance(snapshot.pagerTextColor) > 95,
    `dark rank pager text must stay readable: ${snapshot.pagerTextColor}`
  );
  assert(
    colorLuminance(snapshot.summaryTitleColor) > 150,
    `dark rank summary title must not use light-theme dark ink: ${snapshot.summaryTitleColor}`
  );
  assert(
    colorLuminance(snapshot.summaryMetaColor) > 110,
    `dark rank summary meta must not use light-theme muted ink: ${snapshot.summaryMetaColor}`
  );
  assert(
    colorLuminance(snapshot.scoreBackground) < 120,
    `dark rank primary score must not use a bright score pill: ${snapshot.scoreBackground}`
  );
  assert(
    colorLuminance(snapshot.deltaBackground) < 120,
    `dark rank delta must not use a bright delta pill: ${snapshot.deltaBackground}`
  );
  assert(
    colorLuminance(snapshot.activeCellBackground) < 120,
    `dark rank active metric cell must not use a bright active background: ${snapshot.activeCellBackground}`
  );
  assert(
    colorLuminance(snapshot.activeHeadBackground) < 120,
    `dark rank active metric header must not use a bright active background: ${snapshot.activeHeadBackground}`
  );
  assert(
    colorLuminance(snapshot.facetGroupBackground) < 120,
    `dark rank facet group must not use a bright group surface: ${snapshot.facetGroupBackground}`
  );
  assert(
    colorLuminance(snapshot.facetButtonBackground) < 120,
    `dark rank facet chip must not use a bright chip surface: ${snapshot.facetButtonBackground}`
  );
  assert(
    colorLuminance(snapshot.facetToggleBackground) < 120,
    `dark rank facet toggle must not use a bright toggle surface: ${snapshot.facetToggleBackground}`
  );
  await page.close();
}

function colorLuminance(value) {
  const parsed = parseCssRgb(value);
  if (!parsed) {
    return 255;
  }
  return 0.2126 * parsed.red + 0.7152 * parsed.green + 0.0722 * parsed.blue;
}

function parseCssRgb(value) {
  const srgbMatch = value.match(/color\(srgb\s+([^)]+)\)/);
  if (srgbMatch) {
    const parts = srgbMatch[1]
      .split(/\s+/)
      .map((part) => Number(part))
      .filter((part) => Number.isFinite(part));
    if (parts.length >= 3) {
      return { red: parts[0] * 255, green: parts[1] * 255, blue: parts[2] * 255 };
    }
  }
  const match = value.match(/rgba?\(([^)]+)\)/);
  if (!match) {
    return null;
  }
  const parts = match[1]
    .split(/[,\s/]+/)
    .map((part) => Number(part))
    .filter((part) => Number.isFinite(part));
  if (parts.length < 3) {
    return null;
  }
  return { red: parts[0], green: parts[1], blue: parts[2] };
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function responseForApiPath(url) {
  if (url.pathname === "/api/state") {
    return {
      store_root: "/tmp/eval-bench",
      benchmark_count: 0,
      suite_count: 1,
      campaign_count: 0,
      run_count: 0,
      total_benchmark_samples: 0,
      prediction_count: 0,
      benchmarks: [],
      runs: []
    };
  }
  if (url.pathname === "/api/benchmarks") {
    return { benchmarks: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/runs") {
    return { runs: sampleRuns(), total: 2, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/rank-board") {
    return {
      offset: 0,
      limit: 80,
      total: 1,
      evaluated_count: 1,
      filters: {},
      primary_metric: "f1_iou50",
      primary_metric_label: "F1@.50",
      sort_by: "f1_iou50",
      sort_order: "desc",
      score_formula: "f1_iou50",
      facets: sampleRankFacets(),
      entries: sampleRankEntries()
    };
  }
  if (url.pathname === "/api/suite-rank-board") {
    return {
      offset: 0,
      limit: 80,
      total: 0,
      evaluated_count: 0,
      filters: {},
      primary_metric: "aggregate_score",
      sort_by: "aggregate_score",
      sort_order: "desc",
      facets: {},
      entries: []
    };
  }
  if (url.pathname === "/api/jobs") {
    return { jobs: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/scheduler/status") {
    return {
      enabled: true,
      loop_alive: false,
      max_concurrent_jobs: 1,
      active_worker_threads: [],
      reserved_cuda_devices: [],
      reserved_runtime_ports: [],
      live_running_jobs: []
    };
  }
  if (url.pathname === "/api/services") {
    return { services: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  return {};
}

function sampleRuns() {
  const baseRun = {
    status: "completed",
    benchmark_id: "layout_val",
    benchmark_split: "val",
    tasks: ["grounding_layout"],
    spec_task: "grounding_layout",
    target_labels: ["container", "text"],
    model_id: "qwen3vl-32b",
    model_path: "/models/qwen3vl-32b",
    prompt_id: "layout-default",
    prompt_path: null,
    prompt_hash: "abc123",
    prompt_metadata: {},
    parser: "layout-json",
    metric_profile: "iou50",
    visualization_profile: "layout",
    inference: {},
    created_at: "2026-06-08T06:00:00Z",
    prediction_count: 128,
    report_count: 1,
    manifest_path: "/tmp/eval-bench/manifest.json",
    report_path: "/tmp/eval-bench/report.json",
    benchmark_type: "suite",
    benchmark_official: true,
    integrity_status: "ok",
    integrity_reason: "",
    suite_ids: ["layout"],
    note_updated_at: "2026-06-08T06:05:00Z",
    note_max_length: 20_000,
    f1_iou50: 0.82,
    precision_iou50: 0.85,
    recall_iou50: 0.79,
    mean_iou: 0.73
  };
  return [
    {
      ...baseRun,
      run_id: "layout_val_qwen3vl_32b_note",
      note: "定位到 layout + arrow 组合视图下的稳定表现。"
    },
    {
      ...baseRun,
      run_id: "layout_val_qwen3vl_32b_empty_note",
      note: "",
      note_updated_at: null,
      report_count: 0,
      f1_iou50: 0.76,
      precision_iou50: 0.8,
      recall_iou50: 0.72
    }
  ];
}

function sampleRankEntries() {
  return [
    {
      rank: 1,
      f1_iou50: 0.82,
      run_id: "layout_val_qwen3vl_32b_note",
      score: 0.82,
      score_delta: 0,
      status: "completed",
      benchmark_id: "layout_val",
      benchmark_split: "val",
      suite_id: null,
      benchmark_type: "suite",
      task: "grounding_layout",
      target_labels: ["container", "text"],
      model_id: "qwen3vl-32b",
      prompt_id: "layout-default",
      metric_profile: "iou50",
      prediction_count: 128,
      precision_iou50: 0.85,
      recall_iou50: 0.79,
      mean_iou: 0.73,
      created_at: "2026-06-08T06:00:00Z",
      note: "定位到 layout + arrow 组合视图下的稳定表现。"
    }
  ];
}

function sampleRankFacets() {
  return {
    tasks: [
      { value: "grounding_layout", count: 1 },
      { value: "grounding_arrow", count: 1 },
      { value: "grounding_shape", count: 1 },
      { value: "grounding_icon_image", count: 1 },
      { value: "grounding_shape_arrow", count: 1 },
      { value: "point_arrow", count: 1 }
    ],
    benchmarks: [{ value: "layout_val", count: 1 }],
    splits: [{ value: "val", count: 1 }],
    statuses: [{ value: "completed", count: 1 }],
    labels: [
      { value: "container", count: 1 },
      { value: "text", count: 1 }
    ],
    models: [{ value: "qwen3vl-32b", count: 1 }],
    prompts: [{ value: "layout-default", count: 1 }],
    metric_profiles: [{ value: "iou50", count: 1 }]
  };
}
