import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
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

await page.addInitScript(() => {
  window.__EVAL_BENCH_TOAST_AUTO_DISMISS_MS__ = 600;
});
await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });

await dispatchApiError(page, "500 Internal Server Error: scheduler unavailable");
await dispatchApiError(page, "500 Internal Server Error: scheduler unavailable");
await dispatchApiError(page, "500 Internal Server Error: scheduler unavailable");
await page.locator(".toast-message").first().waitFor({ timeout: 5_000 });
await page.waitForFunction(() => document.querySelectorAll(".toast-message").length === 1);
let snapshot = await toastSnapshot(page);
assert.equal(snapshot.count, 1, "duplicate API errors should coalesce into one toast");
assert.equal(snapshot.firstTitle, "操作失败 x3", "duplicate API errors should increment toast count");
assert.equal(
  snapshot.firstBody,
  "500 Internal Server Error: scheduler unavailable",
  "toast body should keep the API error message"
);

await page.waitForTimeout(350);
await dispatchApiError(page, "500 Internal Server Error: scheduler unavailable");
await page.waitForTimeout(450);
snapshot = await toastSnapshot(page);
assert.equal(snapshot.count, 1, "duplicate API error should refresh the auto-dismiss timer");
assert.equal(snapshot.firstTitle, "操作失败 x4", "refreshed duplicate error should keep incrementing count");
await page.waitForFunction(() => document.querySelectorAll(".toast-message").length === 0, {
  timeout: 1_500
});

await dispatchApiError(page, "404 Not Found: run missing");
await dispatchApiError(page, "403 Forbidden: service locked");
await page.waitForFunction(() => document.querySelectorAll(".toast-message").length === 2);
snapshot = await toastSnapshot(page);
assert.equal(snapshot.count, 2, "different API errors should remain separately visible");

await page.locator(".toast-message").first().getByRole("button", { name: "关闭提醒" }).click();
await page.waitForFunction(() => document.querySelectorAll(".toast-message").length === 1);
snapshot = await toastSnapshot(page);
assert.equal(snapshot.count, 1, "manual close should remove only the selected toast");
assert.equal(snapshot.firstBody, "403 Forbidden: service locked", "remaining toast should preserve its message");

await dispatchApiError(page, "400 Bad Request: filter invalid");
await dispatchApiError(page, "401 Unauthorized: token expired");
await dispatchApiError(page, "409 Conflict: job already queued");
await dispatchApiError(page, "422 Unprocessable Entity: manifest invalid");
await page.waitForFunction(() => document.querySelectorAll(".toast-message").length === 3);
snapshot = await toastSnapshot(page);
assert.deepEqual(
  snapshot.bodies,
  [
    "401 Unauthorized: token expired",
    "409 Conflict: job already queued",
    "422 Unprocessable Entity: manifest invalid"
  ],
  "toast stack should keep only the latest three distinct API errors"
);

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`toast smoke passed on ${baseUrl}`);

async function dispatchApiError(page, message) {
  await page.evaluate((payload) => {
    window.dispatchEvent(
      new CustomEvent("eval-bench-api-error", {
        detail: { message: payload }
      })
    );
  }, message);
}

async function toastSnapshot(page) {
  return page.evaluate(() => {
    const toasts = Array.from(document.querySelectorAll(".toast-message"));
    const first = toasts[0];
    return {
      count: toasts.length,
      firstTitle: first?.querySelector("strong")?.textContent?.trim() ?? "",
      firstBody: first?.querySelector("span")?.textContent?.trim() ?? "",
      bodies: toasts.map((toast) => toast.querySelector("span")?.textContent?.trim() ?? "")
    };
  });
}
