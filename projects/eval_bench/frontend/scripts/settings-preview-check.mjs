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
  if (sessionStorage.getItem("eval_bench_settings_preview_test_cleared")) {
    return;
  }
  localStorage.removeItem("eval_bench_overlay_style");
  localStorage.removeItem("eval_bench_label_colors");
  localStorage.removeItem("eval_bench_interaction_settings");
  localStorage.removeItem("eval_bench_typography_settings");
  localStorage.removeItem("eval_bench_typography_settings_version");
  sessionStorage.setItem("eval_bench_settings_preview_test_cleared", "1");
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

await page.getByRole("button", { name: /字体/ }).click();
await page.getByLabel("基础字号").fill("13.5");
await page.getByLabel("界面字体族").fill('"EvalBenchText", sans-serif');
await page.getByLabel("等宽字体族").fill('"EvalBenchMono", monospace');
await page.getByLabel("字体 CSS URL").fill("/fonts/eval-bench-font.css");
await page.getByLabel("自定义字体名称").fill("EvalBenchCustom");
await page.getByLabel("字体文件 URL").fill("/fonts/EvalBenchCustom.woff2");
await expectTypographySettings(page);
await expectTypographySettingsOnAppShell(page);

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
      stroke_width_realtime: true,
      typography_settings_realtime: true,
      typography_app_shell_sync: true
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

async function expectTypographySettings(page) {
  try {
    await page.waitForFunction(
      () => {
        const root = document.documentElement;
        const shell = document.querySelector(".settings-workbench-shell");
        const preview = document.querySelector(".typography-preview-strip");
        const link = document.getElementById("eval-bench-custom-font");
        const fontFace = document.getElementById("eval-bench-custom-font-face");
        const storedRaw = localStorage.getItem("eval_bench_typography_settings");
        const storedVersion = localStorage.getItem("eval_bench_typography_settings_version");
        if (!shell || !preview || !storedRaw || !link || !fontFace) {
          return false;
        }
        const stored = JSON.parse(storedRaw);
        const rootStyle = getComputedStyle(root);
        const shellStyle = getComputedStyle(shell);
        const previewStyle = getComputedStyle(preview);
        return (
          rootStyle.getPropertyValue("--app-base-font-size").trim() === "13.5px" &&
          rootStyle.getPropertyValue("--app-font-family").includes("EvalBenchCustom") &&
          rootStyle.getPropertyValue("--mono-font").includes("EvalBenchMono") &&
          shellStyle.getPropertyValue("--app-base-font-size").trim() === "13.5px" &&
          previewStyle.fontFamily.includes("EvalBenchCustom") &&
          link.getAttribute("href")?.endsWith("/fonts/eval-bench-font.css") &&
          fontFace.textContent.includes('font-family:"EvalBenchCustom"') &&
          fontFace.textContent.includes('/fonts/EvalBenchCustom.woff2') &&
          fontFace.textContent.includes("unicode-range:U+0000-024F") &&
          stored.baseFontSize === 13.5 &&
          stored.fontFamily.includes("EvalBenchText") &&
          stored.monoFontFamily.includes("EvalBenchMono") &&
          stored.fontCssUrl === "/fonts/eval-bench-font.css" &&
          stored.customFontName === "EvalBenchCustom" &&
          stored.customFontFileUrl === "/fonts/EvalBenchCustom.woff2" &&
          storedVersion === "12px-default"
        );
      },
      { timeout: 3_000 }
    );
  } catch {
    throw new Error(`typography settings did not update DOM/storage; got ${await typographySnapshot(page)}`);
  }
}

async function typographySnapshot(page) {
  return page.evaluate(() => {
    const rootStyle = getComputedStyle(document.documentElement);
    const shell = document.querySelector(".settings-workbench-shell");
    const preview = document.querySelector(".typography-preview-strip");
    return JSON.stringify({
      rootBase: rootStyle.getPropertyValue("--app-base-font-size").trim(),
      rootFont: rootStyle.getPropertyValue("--app-font-family"),
      rootMono: rootStyle.getPropertyValue("--mono-font"),
      shellBase: shell ? getComputedStyle(shell).getPropertyValue("--app-base-font-size").trim() : "",
      previewFont: preview ? getComputedStyle(preview).fontFamily : "",
      linkHref: document.getElementById("eval-bench-custom-font")?.getAttribute("href") ?? "",
      fontFace: document.getElementById("eval-bench-custom-font-face")?.textContent ?? "",
      stored: localStorage.getItem("eval_bench_typography_settings"),
      storedVersion: localStorage.getItem("eval_bench_typography_settings_version")
    });
  });
}

async function expectTypographySettingsOnAppShell(page) {
  const overviewUrl = new URL("/", url);
  await page.goto(overviewUrl.toString(), { waitUntil: "domcontentloaded" });
  await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
  try {
    await page.waitForFunction(
      () => {
        const root = document.documentElement;
        const topbarTitle = document.querySelector(".topbar h1");
        const storeChip = document.querySelector(".store-chip strong");
        const rootStyle = getComputedStyle(root);
        const titleStyle = topbarTitle ? getComputedStyle(topbarTitle) : null;
        const storeStyle = storeChip ? getComputedStyle(storeChip) : null;
        return (
          rootStyle.getPropertyValue("--app-base-font-size").trim() === "13.5px" &&
          rootStyle.getPropertyValue("--app-font-family").includes("EvalBenchCustom") &&
          rootStyle.getPropertyValue("--mono-font").includes("EvalBenchMono") &&
          Boolean(titleStyle?.fontFamily.includes("EvalBenchCustom")) &&
          Boolean(storeStyle?.fontFamily.includes("EvalBenchCustom")) &&
          Boolean(document.getElementById("eval-bench-custom-font")) &&
          Boolean(document.getElementById("eval-bench-custom-font-face"))
        );
      },
      { timeout: 3_000 }
    );
  } catch {
    throw new Error(`typography settings did not sync to AppShell; got ${await appShellTypographySnapshot(page)}`);
  }
}

async function appShellTypographySnapshot(page) {
  return page.evaluate(() => {
    const rootStyle = getComputedStyle(document.documentElement);
    const topbarTitle = document.querySelector(".topbar h1");
    const storeChip = document.querySelector(".store-chip strong");
    return JSON.stringify({
      rootBase: rootStyle.getPropertyValue("--app-base-font-size").trim(),
      rootFont: rootStyle.getPropertyValue("--app-font-family"),
      rootMono: rootStyle.getPropertyValue("--mono-font"),
      titleFont: topbarTitle ? getComputedStyle(topbarTitle).fontFamily : "",
      storeFont: storeChip ? getComputedStyle(storeChip).fontFamily : "",
      hasLink: Boolean(document.getElementById("eval-bench-custom-font")),
      hasFontFace: Boolean(document.getElementById("eval-bench-custom-font-face"))
    });
  });
}
