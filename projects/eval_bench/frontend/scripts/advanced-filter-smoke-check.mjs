import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");

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

await page.goto(`${baseUrl}/runs`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".run-id-link").first().waitFor({ timeout: 10_000 });

const initialRunsRequests = countRequests("/api/runs");
await page.getByRole("button", { name: /结果高级检索/ }).click();
const filterDialog = page.getByRole("dialog", { name: "结果高级检索 条件" });
await filterDialog.waitFor({ timeout: 5_000 });
const searchInput = page.getByRole("searchbox", { name: "全文检索" });
await searchInput.fill("layout smoke draft");
await page.waitForTimeout(60);
assert.equal(
  countRequests("/api/runs"),
  initialRunsRequests,
  "advanced filter draft typing must not refresh results before apply"
);
assert.equal(
  await page.getByRole("dialog", { name: "结果高级检索 条件" }).count(),
  1,
  "advanced filter popover must stay open while editing a draft"
);

await filterDialog.getByRole("button", { name: "收起", exact: true }).click();
await page.waitForTimeout(20);
await page.getByRole("button", { name: /结果高级检索/ }).click();
await filterDialog.waitFor({ timeout: 5_000 });
assert.equal(
  await searchInput.inputValue(),
  "layout smoke draft",
  "advanced filter draft must survive immediate close and reopen"
);
assert.equal(
  countRequests("/api/runs"),
  initialRunsRequests,
  "advanced filter draft persistence must not refresh results while reopening"
);

await page.getByRole("button", { name: "应用" }).click();
await waitForRequestCount("/api/runs", initialRunsRequests + 1);
assert(
  lastRequestFor("/api/runs")?.includes("query=layout+smoke+draft"),
  "advanced filter apply must refresh results with the applied draft query"
);
await page.locator(".advanced-filter-token", { hasText: "全文检索: layout smoke draft" }).waitFor({
  timeout: 5_000
});

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`advanced filter smoke passed on ${baseUrl}`);

async function waitForRequestCount(pathPrefix, expectedCount) {
  const deadline = Date.now() + 6_000;
  while (Date.now() < deadline) {
    if (countRequests(pathPrefix) >= expectedCount) {
      return;
    }
    await page.waitForTimeout(80);
  }
  throw new Error(`did not observe ${expectedCount} requests for ${pathPrefix}`);
}

function countRequests(pathPrefix) {
  return requestedApiUrls.filter((value) => value.startsWith(pathPrefix)).length;
}

function lastRequestFor(pathPrefix) {
  return requestedApiUrls.filter((value) => value.startsWith(pathPrefix)).at(-1);
}

function responseForApiPath(url) {
  if (url.pathname === "/api/state") {
    return {
      store_root: "/tmp/eval-bench",
      benchmark_count: 1,
      suite_count: 0,
      campaign_count: 0,
      run_count: 2,
      total_benchmark_samples: 128,
      prediction_count: 128,
      benchmarks: [],
      runs: sampleRuns()
    };
  }
  if (url.pathname === "/api/runs") {
    return {
      runs: sampleRuns(),
      total: 2,
      offset: 0,
      limit: 80,
      filters: {},
      facets: {
        statuses: [{ value: "completed", count: 2 }],
        tasks: [{ value: "grounding_layout", count: 2 }],
        benchmarks: [{ value: "layout_val", count: 2 }],
        splits: [{ value: "val", count: 2 }],
        models: [{ value: "qwen3vl-32b", count: 2 }],
        prompts: [{ value: "layout-default", count: 2 }],
        metric_profiles: [{ value: "iou50", count: 2 }]
      }
    };
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
      note: "稳定回归样本。"
    },
    {
      ...baseRun,
      run_id: "layout_val_qwen3vl_32b_empty_note",
      note: "",
      note_updated_at: null,
      report_count: 0
    }
  ];
}
