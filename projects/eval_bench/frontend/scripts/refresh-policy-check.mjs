import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");

const browser = await chromium.launch({ headless: true });

try {
  await assertNormalOverviewRefreshes();
  await assertConstrainedOverviewDoesNotFastPoll();
  await assertConstrainedJobsDoesNotFastPoll();
} finally {
  await browser.close();
}

console.log(`refresh policy check passed on ${baseUrl}`);

async function assertNormalOverviewRefreshes() {
  const { page, requests, closeChecked } = await newApiCountPage();
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await waitForRequestCount(page, requests, "/api/jobs?limit=1", 1);
  const initialJobs = countRequests(requests, "/api/jobs?limit=1");
  await waitForRequestCount(page, requests, "/api/jobs?limit=1", initialJobs + 1, 7_000);
  await closeChecked();
}

async function assertConstrainedOverviewDoesNotFastPoll() {
  const { page, requests, closeChecked } = await newApiCountPage({
    connection: { effectiveType: "slow-2g" }
  });
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await waitForRequestCount(page, requests, "/api/jobs?limit=1", 1);
  await waitForRequestCount(page, requests, "/api/scheduler/status", 1);
  const initialJobs = countRequests(requests, "/api/jobs?limit=1");
  const initialScheduler = countRequests(requests, "/api/scheduler/status");
  await page.waitForTimeout(6_500);
  assert.equal(
    countRequests(requests, "/api/jobs?limit=1"),
    initialJobs,
    "constrained overview refresh must not keep polling jobs at the normal 5s cadence"
  );
  assert.equal(
    countRequests(requests, "/api/scheduler/status"),
    initialScheduler,
    "constrained overview refresh must not keep polling scheduler at the normal 5s cadence"
  );
  await closeChecked();
}

async function assertConstrainedJobsDoesNotFastPoll() {
  const { page, requests, closeChecked } = await newApiCountPage({
    connection: { saveData: true }
  });
  await page.goto(`${baseUrl}/jobs`, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await waitForRequestCount(page, requests, "/api/jobs?offset=0&limit=80", 1);
  await waitForRequestCount(page, requests, "/api/scheduler/status", 1);
  const initialJobs = countRequests(requests, "/api/jobs?offset=0&limit=80");
  const initialScheduler = countRequests(requests, "/api/scheduler/status");
  await page.waitForTimeout(5_500);
  assert.equal(
    countRequests(requests, "/api/jobs?offset=0&limit=80"),
    initialJobs,
    "constrained jobs page must not keep polling queue at the normal 4s cadence"
  );
  assert.equal(
    countRequests(requests, "/api/scheduler/status"),
    initialScheduler,
    "constrained jobs page must not keep polling scheduler at the normal 4s cadence"
  );
  await closeChecked();
}

async function newApiCountPage({ connection } = {}) {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  const requests = [];
  const errors = [];
  if (connection) {
    await page.addInitScript((networkConnection) => {
      Object.defineProperty(navigator, "connection", {
        configurable: true,
        value: networkConnection
      });
    }, connection);
  }
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
    requests.push(`${url.pathname}${url.search}`);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(responseForApiPath(url))
    });
  });
  async function closeChecked() {
    if (errors.length > 0) {
      throw new Error(`refresh policy browser errors: ${errors.join(" | ")}`);
    }
    await page.close();
  }
  return { page, requests, closeChecked };
}

async function waitForRequestCount(page, requests, pathPrefix, expectedCount, timeoutMs = 6_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (countRequests(requests, pathPrefix) >= expectedCount) {
      return;
    }
    await page.waitForTimeout(80);
  }
  throw new Error(`did not see ${expectedCount} requests for ${pathPrefix}`);
}

function countRequests(requests, pathPrefix) {
  return requests.filter((value) => value.startsWith(pathPrefix)).length;
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
  if (url.pathname === "/api/jobs") {
    return { jobs: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
  }
  if (url.pathname === "/api/services") {
    return { services: [], total: 0, offset: 0, limit: 80, filters: {}, facets: {} };
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
      score_label: "F1@.50",
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
  if (url.pathname === "/api/comparisons") {
    return { comparisons: [], total: 0, offset: 0, limit: 50, filters: {} };
  }
  return {};
}
