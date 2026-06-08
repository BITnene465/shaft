import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const url = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/";
const TOPBAR_MAX_HEIGHT = 56;
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
  '[class*="pager"]',
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
assert.equal(initial.themeTogglePressed, "false", "light theme toggle must expose aria-pressed=false");
assert(initial.topbarBackground !== initial.sidebarBackground, "shell must retain B-side rail contrast");

await page.getByRole("button", { name: "切换到夜间主题" }).click();
await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
const dark = await themeSnapshot(page);
assert.equal(dark.theme, "dark", "theme button must switch to dark");
assert.equal(dark.storage, "dark", "dark theme must persist to localStorage");
assert.equal(dark.colorScheme, "dark", "dark theme must set color-scheme");
assert.equal(dark.themeTogglePressed, "true", "dark theme toggle must expose aria-pressed=true");
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
await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
const reloadedDark = await themeSnapshot(page);
assert.equal(reloadedDark.theme, "dark", "stored dark theme must survive reload");
assert.equal(reloadedDark.storage, "dark", "stored dark theme must remain in localStorage after reload");
assert.equal(reloadedDark.themeTogglePressed, "true", "stored dark theme toggle state must remain pressed");
await assertNoBrightDarkSurfaces(page);
await assertTopbarDensity(page);
await assertDarkMicroInteractions(page);

await page.getByRole("button", { name: "切换到日间主题" }).click();
await page.waitForFunction(() => document.documentElement.dataset.theme === "light");
const light = await themeSnapshot(page);
assert.equal(light.theme, "light", "theme button must switch back to light");
assert.equal(light.storage, "light", "light theme must persist to localStorage");
await page.reload({ waitUntil: "domcontentloaded", timeout: 15_000 });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
const reloadedLight = await themeSnapshot(page);
assert.equal(reloadedLight.theme, "light", "stored light theme must survive reload");
assert.equal(reloadedLight.storage, "light", "stored light theme must remain in localStorage after reload");

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
    const themeToggle = document.querySelector(".theme-toggle");
    return {
      theme: root.dataset.theme,
      storage: localStorage.getItem("eval_bench_theme_mode"),
      colorScheme: root.style.colorScheme,
      themeTogglePressed: themeToggle?.getAttribute("aria-pressed") ?? "",
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
  await page.locator(selectors.join(",")).first().waitFor({ timeout: 10_000 });
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
    assert(
      !isNearWhite(snapshot.background),
      `${pathname} ${snapshot.selector} dark surface is still pure white`
    );
    assert(
      !isLightThemeInk(snapshot.color),
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
  assert(!isNearWhite(snapshot.background), `${pathname} dark select menu is still pure white`);
  assert(!isLightThemeInk(snapshot.color), `${pathname} dark select menu text is still light-theme ink`);
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
      },
      { selector: darkSurfaceCandidateSelector, pathname }
    );
    findings.push(...routeFindings.filter(isBrightSurfaceFinding));
  }
  assert.equal(
    findings.length,
    0,
    `dark theme still exposes bright surfaces: ${JSON.stringify(findings.slice(0, 20))}`
  );
}

function isBrightSurfaceFinding(finding) {
  const parsed = parseCssColor(finding.background);
  if (!parsed || parsed.alpha <= 0.2) {
    return false;
  }
  return (parsed.red + parsed.green + parsed.blue) / 3 > 225;
}

async function assertTopbarDensity(page) {
  const findings = [];
  for (const pathname of darkSurfaceRoutes) {
    const nextUrl = new URL(pathname, url).toString();
    await page.goto(nextUrl, { waitUntil: "domcontentloaded", timeout: 15_000 });
    await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
    await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
    const snapshot = await page.evaluate(() => {
      const topbar = document.querySelector(".topbar");
      const title = topbar?.querySelector("h1");
      const actions = topbar?.querySelector(".topbar-actions");
      const rect = topbar?.getBoundingClientRect();
      const titleRect = title?.getBoundingClientRect();
      return {
        height: rect ? Math.round(rect.height) : 0,
        titleHeight: titleRect ? Math.round(titleRect.height) : 0,
        hasProfileChip: Boolean(document.querySelector(".user-profile-chip")),
        hasThemeToggle: Boolean(topbar?.querySelector(".theme-toggle")),
        hasStatusPill: Boolean(topbar?.querySelector(".status-pill")),
        actionCount: actions?.children.length ?? 0
      };
    });
    if (
      snapshot.height === 0 ||
      snapshot.height > TOPBAR_MAX_HEIGHT ||
      snapshot.titleHeight > 24 ||
      snapshot.hasProfileChip ||
      !snapshot.hasThemeToggle ||
      !snapshot.hasStatusPill ||
      snapshot.actionCount !== 2
    ) {
      findings.push({ pathname, ...snapshot });
    }
  }
  assert.equal(
    findings.length,
    0,
    `topbar density regressed: ${JSON.stringify(findings)}`
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
  assert(!isLightSuccessInk(snapshot.statusColor), "dark status pill still uses light success ink");
  assert(!isLightNeutralFocus(snapshot.focusOutlineColor), "dark focus ring is still light-theme neutral");
  assert.notEqual(snapshot.focusBoxShadow, "none", "dark focus state must expose a visible focus ring");
}

function isBrightRgb(value) {
  const parsed = parseCssColor(value);
  if (!parsed || parsed.alpha <= 0.2) {
    return false;
  }
  return (parsed.red + parsed.green + parsed.blue) / 3 > 190;
}

function isNearWhite(value) {
  const parsed = parseCssColor(value);
  if (!parsed || parsed.alpha <= 0.2) {
    return false;
  }
  return parsed.red > 245 && parsed.green > 245 && parsed.blue > 245;
}

function isLightThemeInk(value) {
  return isNearCssColor(value, { red: 17, green: 24, blue: 39 }, 2);
}

function isLightSuccessInk(value) {
  return isNearCssColor(value, { red: 21, green: 84, blue: 60 }, 2);
}

function isLightNeutralFocus(value) {
  return isNearCssColor(value, { red: 99, green: 127, blue: 149 }, 2, 0.12);
}

function isNearCssColor(value, expected, channelTolerance, expectedAlpha = 1) {
  const parsed = parseCssColor(value);
  if (!parsed) {
    return false;
  }
  return (
    Math.abs(parsed.red - expected.red) <= channelTolerance &&
    Math.abs(parsed.green - expected.green) <= channelTolerance &&
    Math.abs(parsed.blue - expected.blue) <= channelTolerance &&
    Math.abs(parsed.alpha - expectedAlpha) <= 0.02
  );
}

function parseCssColor(value) {
  const srgbMatch = value.match(/color\(srgb\s+([^)]+)\)/);
  if (srgbMatch) {
    const parts = srgbMatch[1]
      .split(/\s+/)
      .map((part) => Number(part))
      .filter((part) => Number.isFinite(part));
    if (parts.length >= 3) {
      return {
        red: parts[0] * 255,
        green: parts[1] * 255,
        blue: parts[2] * 255,
        alpha: parts[3] === undefined ? 1 : parts[3]
      };
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
  return {
    red: parts[0],
    green: parts[1],
    blue: parts[2],
    alpha: parts[3] === undefined ? 1 : parts[3]
  };
}
