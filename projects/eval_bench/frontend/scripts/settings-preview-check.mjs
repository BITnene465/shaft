import { chromium } from "@playwright/test";

const rawUrl = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/settings";
const url = new URL(rawUrl);
if (!url.pathname.endsWith("/settings")) {
  url.pathname = "/settings";
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
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
  localStorage.removeItem("eval_bench_overlay_style");
  localStorage.removeItem("eval_bench_label_colors");
  localStorage.removeItem("eval_bench_interaction_settings");
});

await page.goto(url.toString(), { waitUntil: "networkidle" });
await page.locator(".settings-preview-stage .image-stage").first().waitFor({ timeout: 10_000 });
await page.locator(".settings-preview-stage .overlay-instance").first().waitFor({ timeout: 10_000 });

const firstLabel = await page
  .locator(".settings-preview-stage .overlay-instance text")
  .first()
  .textContent();
if (!firstLabel) {
  throw new Error("settings preview has no label text");
}

await expectPreviewColor(page, "rgb(35, 196, 131)", "default GT role color is not visible");

await page.getByRole("button", { name: /标签颜色/ }).click();
await page.locator(".label-color-add-row input:not([type='color'])").fill(firstLabel.toUpperCase());
await selectLabelColorRole(page, "pred");
await page.locator(".label-color-add-row input[type='color']").fill("#0000ff");
await page.getByRole("button", { name: "添加" }).click();
await expectPreviewColor(
  page,
  "rgb(35, 196, 131)",
  "pred label color should not override GT preview"
);

await page.locator(".label-color-add-row input:not([type='color'])").fill(firstLabel.toUpperCase());
await selectLabelColorRole(page, "gt");
await page.locator(".label-color-add-row input[type='color']").fill("#00ff00");
await page.getByRole("button", { name: "添加" }).click();
await expectPreviewColor(
  page,
  "rgb(0, 255, 0)",
  "case-insensitive GT label color did not update preview"
);

await page.getByRole("button", { name: /外观/ }).click();
await page.locator(".settings-number-grid input[type='number']").first().fill("9");
await expectPreviewStrokeWidth(page, 9, "box stroke width did not update preview");

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`settings preview check passed ${url.toString()}`);
console.log(
  JSON.stringify(
    {
      label_tested: firstLabel,
      default_role_color_visible: true,
      label_color_case_insensitive: true,
      label_role_cartesian_product: true,
      stroke_width_realtime: true
    },
    null,
    2
  )
);

async function expectPreviewColor(page, expected, message) {
  try {
    await page.waitForFunction(
      ({ expectedColor }) => {
        const node =
          document.querySelector(".settings-preview-stage .overlay-instance .overlay-box") ??
          document.querySelector(".settings-preview-stage .overlay-instance polyline") ??
          document.querySelector(".settings-preview-stage .overlay-instance circle");
        if (!node) {
          return false;
        }
        const style = getComputedStyle(node);
        return style.stroke === expectedColor || style.fill === expectedColor;
      },
      { expectedColor: expected },
      { timeout: 3_000 }
    );
  } catch {
    throw new Error(`${message}; expected ${expected}, got ${await previewStrokeSnapshot(page)}`);
  }
}

async function selectLabelColorRole(page, value) {
  await page.locator(".label-color-add-row .select-popover-trigger").click();
  await page
    .locator(`[data-select-popover-menu="true"] [data-select-value="${value}"]`)
    .click();
}

async function previewStrokeSnapshot(page) {
  return page.evaluate(() => {
    const node =
      document.querySelector(".settings-preview-stage .overlay-instance .overlay-box") ??
      document.querySelector(".settings-preview-stage .overlay-instance polyline") ??
      document.querySelector(".settings-preview-stage .overlay-instance circle");
    if (!node) {
      return "<missing>";
    }
    const style = getComputedStyle(node);
    return `stroke=${style.stroke}, fill=${style.fill}`;
  });
}

async function expectPreviewStrokeWidth(page, expected, message) {
  try {
    await page.waitForFunction(
      ({ expectedWidth }) => {
        const node = document.querySelector(".settings-preview-stage .overlay-instance .overlay-box");
        if (!node) {
          return false;
        }
        return Math.abs(parseFloat(getComputedStyle(node).strokeWidth) - expectedWidth) < 0.01;
      },
      { expectedWidth: expected },
      { timeout: 3_000 }
    );
  } catch {
    throw new Error(`${message}; expected ${expected}, got ${await previewStrokeWidthSnapshot(page)}`);
  }
}

async function previewStrokeWidthSnapshot(page) {
  return page.evaluate(() => {
    const node = document.querySelector(".settings-preview-stage .overlay-instance .overlay-box");
    return node ? getComputedStyle(node).strokeWidth : "<missing>";
  });
}
