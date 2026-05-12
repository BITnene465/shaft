import { chromium } from "@playwright/test";

const rawUrl = process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/runs/config_smoke_prompt_params";
const url = withPerfFlag(rawUrl);

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
await page.locator(".image-zoom-layer").first().waitFor({ timeout: 10_000 });
await page.locator(".overlay-svg").first().waitFor({ timeout: 10_000 });

const before = await renderMetrics(page);
const beforeTransform = await transformStyle(page);
const box = await stage.boundingBox();
if (!box) {
  throw new Error("image stage is not visible");
}
const x = box.x + 18;
const y = box.y + 18;

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

async function renderMetrics(page) {
  return page.evaluate(() => {
    const target = window;
    return { ...(target.__evalBenchRenderMetrics ?? {}) };
  });
}

async function transformStyle(page) {
  return page.locator(".image-zoom-layer").first().evaluate((node) => node.style.transform);
}
