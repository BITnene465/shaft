import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");
const expectedPrefetches = [
  { label: "评测中心", pathPrefix: "/api/jobs", query: "limit=80" },
  { label: "评测中心", pathPrefix: "/api/scheduler/status" },
  { label: "结果库", pathPrefix: "/api/runs", query: "limit=80" },
  { label: "排行榜", pathPrefix: "/api/rank-board", query: "sort_by=f1_iou50" },
  { label: "模型服务", pathPrefix: "/api/services", query: "limit=80" },
  { label: "基准集", pathPrefix: "/api/benchmarks", query: "limit=80" },
  { label: "对比分析", pathPrefix: "/api/comparisons", query: "list=1" },
  { label: "工作台设置", pathPrefix: "/api/settings/preview-sample" }
];

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
const requestedApiUrls = [];
const errors = [];

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
  requestedApiUrls.push(`${url.pathname}${url.search}`);
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(responseForApiPath(url))
  });
});

await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });

for (const prefetch of expectedPrefetches) {
  await page.getByTitle(prefetch.label).hover();
  await waitForApiRequest(prefetch);
}

await page.getByTitle("结果库").hover();
await page.waitForTimeout(300);

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

const matchingRunsRequests = requestedApiUrls.filter(
  (value) => value.startsWith("/api/runs") && value.includes("limit=80")
);
assert.equal(
  matchingRunsRequests.length,
  2,
  "runs endpoint should be prefetched once for results and once for compare, not once per hover"
);

for (const prefetch of expectedPrefetches) {
  assert(
    hasMatchingRequest(prefetch),
    `${prefetch.label} did not prefetch ${prefetch.pathPrefix}`
  );
}

await assertPrefetchFailureStaysSilent();
await assertSaveDataSkipsPrefetch();
await browser.close();

console.log(`nav prefetch check passed on ${baseUrl}`);

async function assertPrefetchFailureStaysSilent() {
  const failurePage = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  const failureRequests = [];
  const failureErrors = [];
  let runsRequestCount = 0;
  failurePage.on("pageerror", (error) => failureErrors.push(error.message));
  failurePage.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        failureErrors.push(text);
      }
    }
  });
  await failurePage.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    failureRequests.push(`${url.pathname}${url.search}`);
    if (url.pathname === "/api/runs") {
      runsRequestCount += 1;
      if (runsRequestCount > 1) {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(responseForApiPath(url))
        });
        return;
      }
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "prefetch smoke outage" })
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await failurePage.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await failurePage.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await failurePage.getByTitle("结果库").hover();
  await waitForPrefetchUrl(failurePage, failureRequests, "/api/runs");
  await failurePage.waitForTimeout(400);
  assert.equal(
    await failurePage.locator(".toast-message").count(),
    0,
    "prefetch failure must not show toast"
  );
  await failurePage.getByTitle("结果库").hover();
  await waitForPrefetchCount(failurePage, failureRequests, "/api/runs", 2);
  assert.equal(runsRequestCount, 2, "failed route prefetch should retry on the next intent");
  if (failureErrors.length > 0) {
    throw new Error(`silent prefetch browser errors: ${failureErrors.join(" | ")}`);
  }
  await failurePage.close();
}

async function assertSaveDataSkipsPrefetch() {
  const saveDataPage = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  const saveDataRequests = [];
  const saveDataErrors = [];
  saveDataPage.on("pageerror", (error) => saveDataErrors.push(error.message));
  saveDataPage.on("console", (message) => {
    if (message.type() === "error") {
      const text = message.text();
      if (!text.includes("/api/") && !text.includes("Failed to load resource")) {
        saveDataErrors.push(text);
      }
    }
  });
  await saveDataPage.addInitScript(() => {
    Object.defineProperty(navigator, "connection", {
      configurable: true,
      value: { saveData: true }
    });
  });
  await saveDataPage.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    saveDataRequests.push(`${url.pathname}${url.search}`);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  await saveDataPage.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await saveDataPage.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  const requestsBeforeHover = saveDataRequests.length;
  await saveDataPage.getByTitle("排行榜").hover();
  await saveDataPage.getByTitle("结果库").hover();
  await saveDataPage.waitForTimeout(600);
  assert.deepEqual(
    saveDataRequests.slice(requestsBeforeHover),
    [],
    "nav prefetch must not fetch API data when navigator.connection.saveData is true"
  );
  if (saveDataErrors.length > 0) {
    throw new Error(`save-data prefetch browser errors: ${saveDataErrors.join(" | ")}`);
  }
  await saveDataPage.close();
}

async function waitForApiRequest(prefetch) {
  const deadline = Date.now() + 6_000;
  while (Date.now() < deadline) {
    if (hasMatchingRequest(prefetch)) {
      return;
    }
    await page.waitForTimeout(80);
  }
  throw new Error(`${prefetch.label} did not prefetch ${prefetch.pathPrefix}`);
}

async function waitForPrefetchUrl(targetPage, urls, pathPrefix) {
  const deadline = Date.now() + 6_000;
  while (Date.now() < deadline) {
    if (urls.some((value) => value.startsWith(pathPrefix))) {
      return;
    }
    await targetPage.waitForTimeout(80);
  }
  throw new Error(`did not prefetch ${pathPrefix}`);
}

async function waitForPrefetchCount(targetPage, urls, pathPrefix, expectedCount) {
  const deadline = Date.now() + 6_000;
  while (Date.now() < deadline) {
    const count = urls.filter((value) => value.startsWith(pathPrefix)).length;
    if (count >= expectedCount) {
      return;
    }
    await targetPage.waitForTimeout(80);
  }
  throw new Error(`did not prefetch ${pathPrefix} ${expectedCount} times`);
}

function hasMatchingRequest({ pathPrefix, query }) {
  return requestedApiUrls.some(
    (value) => value.startsWith(pathPrefix) && (!query || value.includes(query))
  );
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
  if (url.pathname === "/api/runs") {
    return { runs: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/services") {
    return { services: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/benchmarks") {
    return { benchmarks: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/comparisons") {
    return { comparisons: [], total: 0, offset: 0, limit: 50, filters: {} };
  }
  if (url.pathname === "/api/settings/preview-sample") {
    return {
      benchmark_id: "settings-preview",
      benchmark_split: "preview",
      task: "layout",
      target_labels: ["arrow"],
      sample: {
        index: 0,
        image_id: "settings-preview",
        image_path: "/static/settings_preview.svg",
        image_url: "/static/settings_preview.svg",
        image_width: 960,
        image_height: 600
      },
      gt_instances: []
    };
  }
  return {};
}
