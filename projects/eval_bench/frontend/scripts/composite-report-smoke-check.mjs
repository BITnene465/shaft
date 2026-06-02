import { strict as assert } from "node:assert";
import { chromium } from "playwright";

const baseUrl = process.env.EVAL_BENCH_URL || "http://127.0.0.1:8765/";
const url = new URL("/suite-report", baseUrl).toString();
const viewports = [
  { name: "wide", width: 1440, height: 900 },
  { name: "desktop-narrow", width: 1180, height: 760 },
  { name: "short-console", width: 980, height: 720 },
];
const browser = await chromium.launch({ headless: true });
const consoleErrors = [];

try {
  for (const viewport of viewports) {
    await assertCompositeReportViewport(viewport);
  }
  assert.deepEqual(consoleErrors, [], "composite report must not log console errors");
  console.log(
    `composite report smoke checks passed (${viewports.map((item) => item.name).join(", ")})`,
  );
} finally {
  await browser.close();
}

async function assertCompositeReportViewport(viewport) {
  const page = await browser.newPage({ viewport });
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(`[${viewport.name}] ${message.text()}`);
    }
  });

  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForTimeout(1200);

    assert.equal(
      await page.locator(".composite-report-shell.sidebar-collapsed").count(),
      1,
      "composite report must start with collapsed composer sidebar",
    );
    assert.equal(
      await page.locator(".composite-composer-dock.collapsed").count(),
      1,
      "composer dock must expose the collapsed rail",
    );
    assert.equal(
      await page.locator(".composite-sidebar-drawer").count(),
      0,
      "composer drawer must not occupy the initial visual workspace",
    );
    assert.equal(
      await page.locator(".composite-image-navigator").count(),
      1,
      "image navigator must render for composite reports",
    );
    assert(
      (await page.locator(".composite-workbench-canvas").count()) >= 1,
      "visual canvas must render at least one workbench canvas",
    );

    const openButton = page.getByTitle("展开报告编排器").first();
    assert.equal(await openButton.count(), 1, "collapsed dock must provide an open action");
    await openButton.evaluate((element) => element.click());
    await page.waitForTimeout(250);

    assert.equal(
      await page.locator(".composite-report-shell.sidebar-open").count(),
      1,
      "composer dock open action must open the drawer",
    );
    assert.equal(
      await page.locator(".composite-sidebar-drawer").count(),
      1,
      "composer drawer must be mounted after opening",
    );

    await page.locator(".composite-sidebar-backdrop").first().evaluate((element) => element.click());
    await page.waitForTimeout(250);
    assert.equal(
      await page.locator(".composite-report-shell.sidebar-collapsed").count(),
      1,
      "composer backdrop must close the drawer and return workspace focus",
    );

    const searchInput = page.locator(".image-navigator-search input").first();
    assert.equal(await searchInput.count(), 1, "image navigator search input must be present");
    await searchInput.fill("1", { timeout: 2500 });
    await page.waitForTimeout(350);
    assert.equal(
      await page.locator(".image-jump-popover").count(),
      1,
      "typing in image search must open the jump popover",
    );
    assert(
      (await page.locator(".image-jump-result").count()) > 0,
      "image search popover must expose jump results",
    );

    await page.keyboard.press("Escape").catch(() => {});
    await page.waitForTimeout(120);

    const canvas = page.locator(".composite-workbench-canvas").first();
    const box = await canvas.boundingBox();
    assert(box && box.width > 0 && box.height > 0, "workbench canvas must have a visible box");
    await canvas.dispatchEvent("pointermove", {
      clientX: box.x + Math.min(120, box.width / 2),
      clientY: box.y + Math.min(120, box.height / 2),
      pointerType: "mouse",
      bubbles: true,
    });
    await page.waitForTimeout(250);

    assert.equal(
      await page.locator('.composite-workbench-canvas[data-pointer-reticle="active"]').count(),
      1,
      "canvas pointer movement must activate the pointer reticle state",
    );
    assert.equal(
      await page.locator(".composite-canvas-pointer-reticle").count(),
      1,
      "canvas pointer movement must render the reticle overlay",
    );
    assert(
      (await page.locator(".composite-canvas-gesture-hud").count()) >= 1,
      "canvas must expose gesture HUD feedback",
    );
  } finally {
    await page.close();
  }
}
