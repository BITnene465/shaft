import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const url = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/";
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
  if (!sessionStorage.getItem("eval_bench_theme_test_cleared")) {
    localStorage.removeItem("eval_bench_theme_mode");
    sessionStorage.setItem("eval_bench_theme_test_cleared", "1");
  }
});

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });

const initial = await themeSnapshot(page);
assert.equal(initial.theme, "light", "theme must default to light");
assert.equal(initial.colorScheme, "light", "light theme must set color-scheme");
assert(initial.topbarBackground !== initial.sidebarBackground, "shell must retain B-side rail contrast");

await page.getByRole("button", { name: "切换到夜间主题" }).click();
await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
const dark = await themeSnapshot(page);
assert.equal(dark.theme, "dark", "theme button must switch to dark");
assert.equal(dark.storage, "dark", "dark theme must persist to localStorage");
assert.equal(dark.colorScheme, "dark", "dark theme must set color-scheme");
assert.notEqual(dark.topbarBackground, initial.topbarBackground, "topbar token must react to dark theme");
await assertDarkRouteSurface(page, "/rank-board", [".rank-board-page .workspace-card", ".rank-board-table-card"]);
await assertDarkRouteSurface(page, "/settings", [".settings-workbench-shell", ".settings-command-bar"]);

await page.getByRole("button", { name: "切换到日间主题" }).click();
await page.waitForFunction(() => document.documentElement.dataset.theme === "light");
const light = await themeSnapshot(page);
assert.equal(light.theme, "light", "theme button must switch back to light");
assert.equal(light.storage, "light", "light theme must persist to localStorage");

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`theme toggle check passed ${url}`);

async function themeSnapshot(page) {
  return page.evaluate(() => {
    const root = document.documentElement;
    const topbar = document.querySelector(".topbar");
    const sidebar = document.querySelector(".sidebar");
    return {
      theme: root.dataset.theme,
      storage: localStorage.getItem("eval_bench_theme_mode"),
      colorScheme: root.style.colorScheme,
      topbarBackground: topbar ? getComputedStyle(topbar).backgroundColor : "",
      sidebarBackground: sidebar ? getComputedStyle(sidebar).backgroundImage : ""
    };
  });
}

async function assertDarkRouteSurface(page, pathname, selectors) {
  const nextUrl = new URL(pathname, url).toString();
  await page.goto(nextUrl, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
  const snapshot = await page.evaluate((surfaceSelectors) => {
    for (const selector of surfaceSelectors) {
      const element = document.querySelector(selector);
      if (!element) {
        continue;
      }
      return {
        selector,
        background: getComputedStyle(element).backgroundColor,
        color: getComputedStyle(element).color
      };
    }
    return null;
  }, selectors);
  assert(snapshot, `${pathname} must expose a themed surface`);
  assert.notEqual(snapshot.background, "rgb(255, 255, 255)", `${pathname} dark surface is still pure white`);
  assert.notEqual(snapshot.color, "rgb(17, 24, 39)", `${pathname} dark text is still light-theme ink`);
}
