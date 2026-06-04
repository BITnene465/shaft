import { chromium } from "@playwright/test";

const rawUrl = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/";
const url = withPerfFlag(await resolveViewerPerformanceUrl(rawUrl));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
let imageRequests = 0;

page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") {
    errors.push(message.text());
  }
});
page.on("request", (request) => {
  const requestUrl = request.url();
  if (/\/api\/(?:runs|benchmarks)\/.+\/samples\/\d+\/image(?:\/preview)?(?:\?|$)/.test(requestUrl)) {
    imageRequests += 1;
  }
});

await page.goto(url, { waitUntil: "networkidle" });
const stage = page.locator(".image-stage").first();
await stage.waitFor({ timeout: 10_000 });
const pointerSurface = page.locator(".viewer-pointer-surface").first();
await pointerSurface.waitFor({ timeout: 10_000 });
await page.locator(".image-zoom-layer").first().waitFor({ timeout: 10_000 });
await page.locator(".overlay-svg").first().waitFor({ timeout: 10_000 });

const before = await renderMetrics(page);
const beforeTransform = await transformStyle(page);
const box = await stage.boundingBox();
if (!box) {
  throw new Error("image stage is not visible");
}
const x = box.x + box.width / 2;
const y = box.y + box.height / 2;

await page.mouse.move(x, y);
await page.waitForTimeout(80);
if ((await page.locator('.viewer-pointer-surface[data-pointer-reticle="active"]').count()) !== 1) {
  throw new Error("ordinary viewer pointer movement did not activate the reticle state");
}
if ((await page.locator(".viewer-pointer-surface .composite-canvas-pointer-reticle").count()) !== 1) {
  throw new Error("ordinary viewer did not render the shared canvas pointer reticle");
}
const coordinateLabel = await page
  .locator(".viewer-pointer-surface .composite-canvas-coordinate-tag")
  .first()
  .textContent();
if (!coordinateLabel || !/\d+\s*\/\s*\d+/.test(coordinateLabel)) {
  throw new Error(`ordinary viewer coordinate status did not update: ${coordinateLabel ?? "<empty>"}`);
}

for (let index = 0; index < 36; index += 1) {
  await page.mouse.move(x, y);
  await page.mouse.wheel(0, -80);
}
await page.waitForTimeout(120);
const afterWheelTransform = await transformStyle(page);
if (afterWheelTransform === beforeTransform) {
  throw new Error("wheel interaction did not update image transform");
}

await page.mouse.move(x, y);
await page.mouse.down();
for (let index = 0; index < 80; index += 1) {
  await page.mouse.move(x + index * 2, y + index, { steps: 1 });
}
await page.mouse.up();
await page.waitForTimeout(160);

const after = await renderMetrics(page);
const canvasDelta = (after.canvasStage ?? 0) - (before.canvasStage ?? 0);
const gtLayerDelta = (after["instanceLayer:gt"] ?? 0) - (before["instanceLayer:gt"] ?? 0);
const predLayerDelta = (after["instanceLayer:pred"] ?? 0) - (before["instanceLayer:pred"] ?? 0);

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}
if (canvasDelta > 5) {
  throw new Error(`pan/zoom caused too many CanvasStage renders: ${canvasDelta}`);
}
if (gtLayerDelta + predLayerDelta > 1) {
  throw new Error(
    `pan/zoom should not rerender heavy overlay layers: gt=${gtLayerDelta}, pred=${predLayerDelta}`
  );
}
if (imageRequests > 3) {
  throw new Error(`viewer opened too many image requests during initial inspection: ${imageRequests}`);
}

console.log(`viewer performance check passed ${url}`);
console.log(
  JSON.stringify(
    {
      canvas_renders_during_pan_zoom: canvasDelta,
      gt_layer_renders_during_pan_zoom: gtLayerDelta,
      pred_layer_renders_during_pan_zoom: predLayerDelta,
      pointer_reticle: "active",
      image_requests_during_initial_inspection: imageRequests
    },
    null,
    2
  )
);

function withPerfFlag(value) {
  const parsed = new URL(value);
  parsed.searchParams.set("perf", "1");
  return parsed.toString();
}

async function resolveViewerPerformanceUrl(value) {
  const parsed = new URL(value);
  if (/^\/runs\/[^/]+/.test(parsed.pathname)) {
    return parsed.toString();
  }
  try {
    const response = await fetch(new URL("/api/state", parsed.origin));
    if (!response.ok) {
      throw new Error(`state request failed with ${response.status}`);
    }
    const state = await response.json();
    const run = Array.isArray(state.runs)
      ? state.runs.find((item) => item?.run_id && item?.report_path) ?? state.runs.find((item) => item?.run_id)
      : null;
    if (run?.run_id) {
      return new URL(`/runs/${encodeURIComponent(run.run_id)}`, parsed.origin).toString();
    }
  } catch (error) {
    throw new Error(`viewer performance check could not discover a run from ${parsed.origin}: ${error}`);
  }
  throw new Error("viewer performance check requires EVAL_BENCH_URL=/runs/<run_id> or a store with runs.");
}

async function renderMetrics(page) {
  return page.evaluate(() => {
    const target = window;
    return { ...(target.__evalBenchRenderMetrics ?? {}) };
  });
}

async function transformStyle(page) {
  return page.locator(".image-zoom-layer").first().evaluate((node) => node.style.transform);
}
