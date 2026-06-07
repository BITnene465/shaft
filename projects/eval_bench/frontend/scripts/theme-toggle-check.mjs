import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const url = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/";
const darkSurfaceRoutes = [
  "/",
  "/benchmarks",
  "/services",
  "/jobs",
  "/runs",
  "/rank-board",
  "/suite-report",
  "/compare",
  "/settings"
];
const darkSurfaceCandidateSelector = [
  '[class*="card"]',
  '[class*="panel"]',
  '[class*="pane"]',
  '[class*="rail"]',
  '[class*="shell"]',
  '[class*="strip"]',
  '[class*="bar"]',
  '[class*="table"]',
  '[class*="filter"]',
  '[class*="drawer"]',
  '[class*="dock"]',
  '[class*="stage"]',
  '[class*="workspace"]'
].join(",");
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
await assertDarkRouteSurface(page, "/", [".overview-home-v18", ".overview-v18-card", ".overview-v18-console"]);
await assertDarkRouteSurface(page, "/rank-board", [".rank-board-page .workspace-card", ".rank-board-table-card"]);
await assertDarkSelectPopover(page, "/rank-board");
await assertDarkRouteSurface(page, "/jobs", [".scheduler-strip"]);
await assertDarkRouteSurface(page, "/compare", [".compare-run-rail", ".compare-report-pane", ".compare-context-pane"]);
await assertDarkRouteSurface(page, "/suite-report", [
  ".composite-report-shell",
  ".composite-composer-dock",
  ".composite-stage-region"
]);
await assertDarkRouteSurface(page, "/settings", [
  ".settings-workbench-shell",
  ".settings-command-bar",
  ".settings-preference-drawer",
  ".settings-drawer-head"
]);
await assertNoBrightDarkSurfaces(page);
await assertDarkMicroInteractions(page);

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
  const snapshots = await page.evaluate((surfaceSelectors) => {
    const items = [];
    for (const selector of surfaceSelectors) {
      const element = document.querySelector(selector);
      if (!element) {
        continue;
      }
      items.push({
        selector,
        background: getComputedStyle(element).backgroundColor,
        color: getComputedStyle(element).color
      });
    }
    return items;
  }, selectors);
  assert(snapshots.length > 0, `${pathname} must expose a themed surface`);
  for (const snapshot of snapshots) {
    assert.notEqual(
      snapshot.background,
      "rgb(255, 255, 255)",
      `${pathname} ${snapshot.selector} dark surface is still pure white`
    );
    assert.notEqual(
      snapshot.color,
      "rgb(17, 24, 39)",
      `${pathname} ${snapshot.selector} dark text is still light-theme ink`
    );
  }
}

async function assertDarkSelectPopover(page, pathname) {
  const nextUrl = new URL(pathname, url).toString();
  await page.goto(nextUrl, { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await page.locator(".advanced-filter-head").first().click();
  await page.locator(".advanced-filter-controls .select-popover-trigger").first().click();
  await page.locator('[data-select-popover-menu="true"]').first().waitFor({ timeout: 5_000 });
  const snapshot = await page.locator('[data-select-popover-menu="true"]').first().evaluate((element) => ({
    background: getComputedStyle(element).backgroundColor,
    color: getComputedStyle(element).color
  }));
  assert.notEqual(snapshot.background, "rgb(255, 255, 255)", `${pathname} dark select menu is still pure white`);
  assert.notEqual(snapshot.color, "rgb(17, 24, 39)", `${pathname} dark select menu text is still light-theme ink`);
}

async function assertNoBrightDarkSurfaces(page) {
  const findings = [];
  for (const pathname of darkSurfaceRoutes) {
    const nextUrl = new URL(pathname, url).toString();
    await page.goto(nextUrl, { waitUntil: "domcontentloaded", timeout: 15_000 });
    await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
    await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
    const routeFindings = await page.evaluate(
      ({ selector, pathname }) => {
        const items = [];
        for (const element of document.querySelectorAll(selector)) {
          const rect = element.getBoundingClientRect();
          if (rect.width < 24 || rect.height < 18) {
            continue;
          }
          const className = String(element.className ?? "").trim().replace(/\s+/g, ".");
          if (className.includes("sidebar-toggle")) {
            continue;
          }
          const background = getComputedStyle(element).backgroundColor;
          const parsed = parseCssRgb(background);
          if (!parsed || parsed.alpha <= 0.2 || parsed.lightness <= 225) {
            continue;
          }
          items.push({
            pathname,
            tag: element.tagName.toLowerCase(),
            className,
            background,
            width: Math.round(rect.width),
            height: Math.round(rect.height)
          });
        }
        return items;
        function parseCssRgb(value) {
          const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([0-9.]+))?/);
          if (!match) {
            return null;
          }
          const red = Number(match[1]);
          const green = Number(match[2]);
          const blue = Number(match[3]);
          const alpha = match[4] === undefined ? 1 : Number(match[4]);
          return { alpha, lightness: (red + green + blue) / 3 };
        }
      },
      { selector: darkSurfaceCandidateSelector, pathname }
    );
    findings.push(...routeFindings);
  }
  assert.equal(
    findings.length,
    0,
    `dark theme still exposes bright surfaces: ${JSON.stringify(findings.slice(0, 20))}`
  );
}

async function assertDarkMicroInteractions(page) {
  await page.goto(new URL("/", url).toString(), { waitUntil: "domcontentloaded", timeout: 15_000 });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
  await page.locator(".theme-toggle").focus();
  const snapshot = await page.evaluate(() => {
    const rootStyle = getComputedStyle(document.documentElement);
    const status = document.querySelector(".status-pill");
    const focused = document.querySelector(".theme-toggle");
    return {
      scrollbarColor: rootStyle.scrollbarColor,
      statusBackground: status ? getComputedStyle(status).backgroundColor : "",
      statusColor: status ? getComputedStyle(status).color : "",
      focusOutlineColor: focused ? getComputedStyle(focused).outlineColor : "",
      focusBoxShadow: focused ? getComputedStyle(focused).boxShadow : ""
    };
  });
  assert.notEqual(snapshot.scrollbarColor, "auto", "dark theme must define non-native scrollbar colors");
  assert(!isBrightRgb(snapshot.statusBackground), `dark status pill is too bright: ${snapshot.statusBackground}`);
  assert.notEqual(snapshot.statusColor, "rgb(21, 84, 60)", "dark status pill still uses light success ink");
  assert.notEqual(snapshot.focusOutlineColor, "rgba(99, 127, 149, 0.12)", "dark focus ring is still light-theme neutral");
  assert.notEqual(snapshot.focusBoxShadow, "none", "dark focus state must expose a visible focus ring");
}

function isBrightRgb(value) {
  const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([0-9.]+))?/);
  if (!match) {
    return false;
  }
  const alpha = match[4] === undefined ? 1 : Number(match[4]);
  if (alpha <= 0.2) {
    return false;
  }
  return (Number(match[1]) + Number(match[2]) + Number(match[3])) / 3 > 190;
}
