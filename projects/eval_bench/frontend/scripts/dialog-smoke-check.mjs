import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/").replace(/\/+$/, "");

const cases = [
  { path: "/jobs", button: "新建评测", form: ".manifest-job-form" },
  { path: "/benchmarks", button: "创建副本", form: ".benchmark-form" },
  { path: "/runs", button: "导入预测", form: ".import-form" },
  { path: "/services", button: "登记服务", form: ".service-form" }
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];

page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") {
    errors.push(message.text());
  }
});

for (const item of cases) {
  await page.goto(`${baseUrl}${item.path}`, { waitUntil: "networkidle" });
  await page.locator(".content").first().waitFor({ timeout: 10_000 });
  await page.getByRole("button", { name: item.button }).click();
  await page.locator(".workspace-dialog").first().waitFor({ timeout: 5_000 });
  await page.locator(item.form).first().waitFor({ timeout: 5_000 });
  await page.keyboard.press("Escape");
  await page.locator(".workspace-dialog").waitFor({ state: "hidden", timeout: 5_000 });
}

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`dialog smoke passed on ${baseUrl}`);
