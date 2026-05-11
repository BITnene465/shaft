import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const url = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/";
const screenshotPath =
  process.env.SCREENSHOT_PATH ??
  path.resolve(process.cwd(), "../../..", "temp", "eval_bench_dashboard.png");

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];

page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") {
    errors.push(message.text());
  }
});

await page.goto(url, { waitUntil: "networkidle" });
await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
await page.locator(".content").first().waitFor({ timeout: 10_000 });
await assertPragmaticDefaults(page);
if (process.env.INTERACTION_SMOKE === "1") {
  await exerciseChips(page);
  await exerciseSelects(page);
  await exerciseCheckboxes(page);
  const stage = page.locator(".image-stage").first();
  if ((await stage.count()) > 0) {
    await stage.waitFor({ timeout: 10_000 });
    const box = await stage.boundingBox();
    if (!box) {
      throw new Error("image stage was not visible for interaction smoke");
    }
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.wheel(0, -500);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.58, box.y + box.height * 0.56, { steps: 8 });
    await page.mouse.up();
    await exerciseButtons(page);
    await exerciseDetailsPanels(page);
    await exerciseOverlayStyleControls(page);
    const objectRow = page.locator(".object-row").first();
    if ((await objectRow.count()) > 0) {
      await objectRow.hover();
      await objectRow.click();
      await page.locator(".object-row.active").first().waitFor({ timeout: 3_000 });
    }
    const labelMetricCard = page.locator(".label-metric-card").first();
    if ((await labelMetricCard.count()) > 0) {
      await labelMetricCard.waitFor({ timeout: 3_000 });
    }
    const objectMetric = page.locator(".object-match").first();
    if ((await objectMetric.count()) > 0) {
      await objectMetric.waitFor({ timeout: 3_000 });
    }
    await exerciseKeyboardShortcuts(page);
  }
  await page.waitForTimeout(150);
}
await mkdir(path.dirname(screenshotPath), { recursive: true });
await page.screenshot({ path: screenshotPath, fullPage: true });
await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`rendered ${url}`);
console.log(`screenshot ${screenshotPath}`);

async function assertPragmaticDefaults(page) {
  const scroll = await page.evaluate(() => ({
    body: document.body.scrollHeight,
    document: document.documentElement.scrollHeight,
    viewport: window.innerHeight
  }));
  if (Math.max(scroll.body, scroll.document) > scroll.viewport + 2) {
    throw new Error(
      `dashboard should not use global page scroll: body=${scroll.body}, document=${scroll.document}, viewport=${scroll.viewport}`
    );
  }
  const actionPanels = page.locator("details.action-panel");
  const actionPanelCount = await actionPanels.count();
  for (let index = 0; index < actionPanelCount; index += 1) {
    if (await actionPanels.nth(index).evaluate((node) => node.hasAttribute("open"))) {
      throw new Error("low-frequency action panel should be collapsed by default");
    }
  }
  const runConfig = page.locator("details.run-config-panel").first();
  if ((await runConfig.count()) > 0 && await runConfig.evaluate((node) => node.hasAttribute("open"))) {
    throw new Error("run config should be collapsed by default");
  }
  const labelMetric = page.locator("details.label-metric-card").first();
  if (
    (await labelMetric.count()) > 0 &&
    await labelMetric.evaluate((node) => node.hasAttribute("open"))
  ) {
    throw new Error("label metric details should be collapsed by default");
  }
  const layout = page.locator(".viewer-canvas-layout").first();
  if ((await layout.count()) > 0) {
    const stageBox = await page.locator(".image-stage").first().boundingBox();
    const sideBox = await page.locator(".viewer-side-panel").first().boundingBox();
    if (!stageBox || !sideBox) {
      throw new Error("run inspector canvas layout is not visible");
    }
    if (stageBox.width <= sideBox.width * 2.4) {
      throw new Error(
        `run inspector canvas is too narrow: canvas=${stageBox.width}, side=${sideBox.width}`
      );
    }
    const fittedImage = page.locator(".image-zoom-layer:not(.static)").first();
    const imageBox = await fittedImage.boundingBox();
    if (!imageBox) {
      throw new Error("run inspector fitted image layer is not visible");
    }
    if (imageBox.width > stageBox.width + 2 || imageBox.height > stageBox.height + 2) {
      throw new Error(
        `image should be contained in stage: image=${imageBox.width}x${imageBox.height}, stage=${stageBox.width}x${stageBox.height}`
      );
    }
    const overlayState = await page.evaluate(() => {
      const stage = document.querySelector(".image-stage");
      const image = document.querySelector(".image-zoom-layer img");
      const overlay = document.querySelector(".overlay-svg");
      const rect = document.querySelector(".overlay-instance rect");
      const line = document.querySelector(".overlay-instance polyline");
      const stageRect = stage?.getBoundingClientRect();
      const imageRect = image?.getBoundingClientRect();
      const overlayRect = overlay?.getBoundingClientRect();
      const rectStroke = rect ? getComputedStyle(rect).stroke : "";
      const lineStroke = line ? getComputedStyle(line).stroke : "";
      return {
        rectCount: document.querySelectorAll(".overlay-instance rect").length,
        lineCount: document.querySelectorAll(".overlay-instance polyline").length,
        imageInside:
          !!stageRect &&
          !!imageRect &&
          imageRect.left >= stageRect.left - 1 &&
          imageRect.top >= stageRect.top - 1 &&
          imageRect.right <= stageRect.right + 1 &&
          imageRect.bottom <= stageRect.bottom + 1,
        overlayMatchesImage:
          !!imageRect &&
          !!overlayRect &&
          Math.abs(imageRect.width - overlayRect.width) < 1 &&
          Math.abs(imageRect.height - overlayRect.height) < 1,
        visibleStroke: [rectStroke, lineStroke].some(
          (stroke) => stroke && stroke !== "none" && stroke !== "rgba(0, 0, 0, 0)"
        )
      };
    });
    if (!overlayState.imageInside) {
      throw new Error("image is clipped outside its stage");
    }
    if (!overlayState.overlayMatchesImage) {
      throw new Error("overlay size does not match fitted image size");
    }
    if (overlayState.rectCount + overlayState.lineCount > 0 && !overlayState.visibleStroke) {
      throw new Error("overlay geometry exists but has no visible stroke");
    }
  }
}

async function exerciseChips(page) {
  const count = Math.min(await page.locator(".query-chip").count(), 6);
  for (let index = 0; index < count; index += 1) {
    const chips = page.locator(".query-chip");
    if ((await chips.count()) <= index) {
      break;
    }
    const chip = chips.nth(index);
    if (!(await chip.isEnabled())) {
      continue;
    }
    await chip.click();
    await page.waitForTimeout(40);
  }
}

async function exerciseKeyboardShortcuts(page) {
  for (const key of ["g", "p", "b", "l", "k", "f", "Escape", "]", "["]) {
    await page.keyboard.press(key);
    await page.waitForTimeout(25);
  }
}

async function exerciseSelects(page) {
  const selects = page.locator("select");
  const count = await selects.count();
  for (let index = 0; index < count; index += 1) {
    const select = selects.nth(index);
    if (!(await select.isEnabled()) || !(await select.isVisible())) {
      continue;
    }
    const options = await select.locator("option").evaluateAll((nodes) =>
      nodes.map((node) => node.value).filter(Boolean)
    );
    const current = await select.inputValue();
    const next = options.find((value) => value !== current);
    if (next) {
      await select.selectOption(next);
      await page.waitForTimeout(50);
      if (options.includes(current)) {
        await select.selectOption(current);
      }
    }
  }
}

async function exerciseButtons(page) {
  const reset = page.locator(".canvas-hud button").first();
  if ((await reset.count()) > 0 && (await reset.isEnabled())) {
    await reset.click({ force: true });
    await page.waitForTimeout(80);
  }
}

async function exerciseDetailsPanels(page) {
  const runConfig = page.locator("details.run-config-panel").first();
  if ((await runConfig.count()) > 0) {
    const runConfigSummary = runConfig.locator(":scope > summary");
    await runConfigSummary.click();
    await page.locator(".run-config-grid").first().waitFor({ timeout: 3_000 });
    const panelBox = await runConfig.boundingBox();
    if (!panelBox) {
      throw new Error("run config details did not expand visibly");
    }
    const viewportHeight = page.viewportSize()?.height ?? 0;
    const panelBottom = panelBox.y + panelBox.height;
    if (panelBottom > viewportHeight + 2) {
      throw new Error(`run config details is clipped below viewport: bottom=${panelBottom}`);
    }
    await runConfigSummary.click();
  }

  const labelMetric = page.locator("details.label-metric-card").first();
  if ((await labelMetric.count()) > 0) {
    const labelMetricSummary = labelMetric.locator(":scope > summary");
    await labelMetricSummary.click();
    await page.locator(".label-metric-table").first().waitFor({ timeout: 3_000 });
    await labelMetricSummary.click();
  }
}

async function exerciseOverlayStyleControls(page) {
  const stylePanel = page.locator("details.control-popover").filter({ hasText: "样式" }).first();
  if ((await stylePanel.count()) === 0) {
    return;
  }
  const summary = stylePanel.locator(":scope > summary");
  if (!(await stylePanel.evaluate((node) => node.hasAttribute("open")))) {
    await summary.click();
  }
  const firstRange = stylePanel.locator('input[type="range"]').first();
  await firstRange.waitFor({ timeout: 3_000 });
  await firstRange.fill("5");
  const lineStyle = stylePanel.locator("select").first();
  if ((await lineStyle.count()) > 0) {
    await lineStyle.selectOption("solid");
    await lineStyle.selectOption("dashed");
  }
  await summary.click();

  const colorPanel = page.locator("details.control-popover").filter({ hasText: "颜色" }).first();
  if ((await colorPanel.count()) === 0) {
    return;
  }
  const colorSummary = colorPanel.locator(":scope > summary");
  if (!(await colorPanel.evaluate((node) => node.hasAttribute("open")))) {
    await colorSummary.click();
  }
  const firstColor = colorPanel.locator('input[type="color"]').first();
  await firstColor.waitFor({ timeout: 3_000 });
  await firstColor.fill("#23c483");
  await colorSummary.click();
}

async function exerciseCheckboxes(page) {
  const checkboxes = page.locator('input[type="checkbox"]');
  const count = Math.min(await checkboxes.count(), 8);
  for (let index = 0; index < count; index += 1) {
    const checkbox = checkboxes.nth(index);
    if (!(await checkbox.isEnabled()) || !(await checkbox.isVisible())) {
      continue;
    }
    await checkbox.click();
    await page.waitForTimeout(30);
    await checkbox.click();
  }
}
