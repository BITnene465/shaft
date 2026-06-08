import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");
const expectedRouteChunks = [
  "benchmarksPage",
  "rankBoardPage",
  "runsPage",
  "jobsPage",
  "suiteReportPage",
  "comparePage",
  "comparisonSamplePage",
  "servicesPage",
  "settingsPage"
];

const browser = await chromium.launch({ headless: true });
const errors = [];

const warmupPage = await newCheckedPage();
await warmupPage.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await warmupPage.locator(".app-shell").first().waitFor({ timeout: 10_000 });
await warmupPage.waitForFunction(
  (expectedChunks) => {
    const resources = performance.getEntriesByType("resource").map((entry) => entry.name);
    return expectedChunks.every((chunkName) =>
      resources.some((resourceName) => resourceName.includes(chunkName))
    );
  },
  expectedRouteChunks,
  { timeout: 12_000 }
);
const warmedRouteChunks = await warmupPage.evaluate(loadedRouteChunks, expectedRouteChunks);

const saveDataPage = await newCheckedPage();
await saveDataPage.addInitScript(() => {
  Object.defineProperty(navigator, "connection", {
    configurable: true,
    value: { saveData: true }
  });
});
await saveDataPage.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await saveDataPage.locator(".app-shell").first().waitFor({ timeout: 10_000 });
await saveDataPage.waitForTimeout(3_200);
const saveDataRouteChunks = await saveDataPage.evaluate(loadedRouteChunks, expectedRouteChunks);

const constrainedNetworkPage = await newCheckedPage();
await constrainedNetworkPage.addInitScript(() => {
  Object.defineProperty(navigator, "connection", {
    configurable: true,
    value: { effectiveType: "2g" }
  });
});
await constrainedNetworkPage.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await constrainedNetworkPage.locator(".app-shell").first().waitFor({ timeout: 10_000 });
await constrainedNetworkPage.waitForTimeout(3_200);
const constrainedNetworkRouteChunks = await constrainedNetworkPage.evaluate(
  loadedRouteChunks,
  expectedRouteChunks
);

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

assert.deepEqual(warmedRouteChunks, expectedRouteChunks, "core route chunks must warm up after idle");
assert.deepEqual(
  saveDataRouteChunks,
  [],
  "route warmup must not fetch core chunks when navigator.connection.saveData is true"
);
assert.deepEqual(
  constrainedNetworkRouteChunks,
  [],
  "route warmup must not fetch core chunks on constrained effective network types"
);

console.log(`route warmup check passed on ${baseUrl}`);

async function newCheckedPage() {
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
  return page;
}

function loadedRouteChunks(expectedChunks) {
  const resources = performance.getEntriesByType("resource").map((entry) => entry.name);
  return expectedChunks.filter((chunkName) =>
    resources.some((resourceName) => resourceName.includes(chunkName))
  );
}
