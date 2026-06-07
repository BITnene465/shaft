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
    return { runs: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/rank-board") {
    return {
      offset: 0,
      limit: 80,
      total: 0,
      evaluated_count: 0,
      filters: {},
      primary_metric: "f1_iou50",
      primary_metric_label: "F1@.50",
      sort_by: "f1_iou50",
      sort_order: "desc",
      score_formula: "f1_iou50",
      facets: {},
      entries: []
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
