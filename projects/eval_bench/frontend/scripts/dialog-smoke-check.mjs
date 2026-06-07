import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/").replace(/\/+$/, "");

const cases = [
  { path: "/jobs", button: "新建评测", form: ".manifest-job-form" },
  { path: "/benchmarks", button: "创建副本", form: ".benchmark-form" },
  { path: "/runs", button: "导入预测", form: ".import-form" },
  { path: "/services", button: "登记服务", form: ".service-form" }
];

const dangerCases = [
  { path: "/runs", buttonTitle: "删除 run", confirmLabel: "移入回收站" },
  { path: "/jobs", buttonTitle: "删除任务记录", confirmLabel: "删除记录" },
  { path: "/services", buttonTitle: "删除服务记录", confirmLabel: "删除服务" }
];

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

for (const item of cases) {
  await page.goto(`${baseUrl}${item.path}`, { waitUntil: "networkidle" });
  await page.locator(".content").first().waitFor({ timeout: 10_000 });
  await page.getByRole("button", { name: item.button }).click();
  await page.locator(".workspace-dialog").first().waitFor({ timeout: 5_000 });
  await page.locator(item.form).first().waitFor({ timeout: 5_000 });
  await assertDialogInteraction(page, `dialog:${item.path}`);
  for (let index = 0; index < 8; index += 1) {
    await page.keyboard.press("Tab");
    await assertDialogInteraction(page, `dialog:${item.path}:tab-${index}`);
  }
  await page.keyboard.press("Escape");
  await page.locator(".workspace-dialog").waitFor({ state: "hidden", timeout: 5_000 });
  await assertBodyScrollRestored(page, `dialog:${item.path}`);
}

for (const item of dangerCases) {
  await page.goto(`${baseUrl}${item.path}`, { waitUntil: "networkidle" });
  await page.locator(".content").first().waitFor({ timeout: 10_000 });
  const trigger = page.locator(`button[title="${item.buttonTitle}"]:not([disabled])`).first();
  if ((await trigger.count()) === 0) {
    continue;
  }
  await trigger.click();
  await page.locator(".workspace-dialog").first().waitFor({ timeout: 5_000 });
  await page
    .locator(".workspace-dialog")
    .getByRole("button", { name: item.confirmLabel, exact: true })
    .waitFor({ timeout: 5_000 });
  await assertDialogInteraction(page, `danger-dialog:${item.path}`);
  await page.keyboard.press("Escape");
  await page.locator(".workspace-dialog").waitFor({ state: "hidden", timeout: 5_000 });
  await assertBodyScrollRestored(page, `danger-dialog:${item.path}`);
}

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`dialog smoke passed on ${baseUrl}`);

async function assertDialogInteraction(page, scope) {
  const state = await page.evaluate(() => {
    const dialog = document.querySelector(".workspace-dialog");
    return {
      exists: Boolean(dialog),
      activeInside: dialog ? dialog.contains(document.activeElement) : false,
      bodyOverflow: document.body.style.overflow,
      tabIndex: dialog?.getAttribute("tabindex") ?? "",
      describedBy: dialog?.getAttribute("aria-describedby") ?? ""
    };
  });
  if (!state.exists) {
    throw new Error(`${scope}: dialog is missing`);
  }
  if (!state.activeInside) {
    throw new Error(`${scope}: focus escaped workspace dialog`);
  }
  if (state.bodyOverflow !== "hidden") {
    throw new Error(`${scope}: body scroll is not locked while dialog is open`);
  }
  if (state.tabIndex !== "-1" || !state.describedBy) {
    throw new Error(`${scope}: dialog accessibility attributes are incomplete`);
  }
}

async function assertBodyScrollRestored(page, scope) {
  const overflow = await page.evaluate(() => document.body.style.overflow);
  if (overflow === "hidden") {
    throw new Error(`${scope}: body scroll lock was not restored after dialog closed`);
  }
}
