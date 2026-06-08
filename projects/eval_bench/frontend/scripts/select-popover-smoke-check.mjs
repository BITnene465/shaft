import { strict as assert } from "node:assert";
import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/").replace(/\/+$/, "");
const candidateRoutes = ["/rank-board", "/runs", "/jobs", "/benchmarks", "/services", "/compare"];
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
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith("eval_bench_advanced_filter_open:")) {
      localStorage.removeItem(key);
    }
  }
});

const route = await openFirstAdvancedFilterWithSelect(page);
assert(route, "at least one eval-bench page must expose an advanced filter select popover");

const trigger = page.locator(".advanced-filter-controls .select-popover-trigger").first();
await trigger.click();
const menu = page.locator('[data-select-popover-menu="true"]').first();
await menu.waitFor({ timeout: 5_000 });

await assertSelectStructure(page, "opened");
await assertSearchClearKeepsInputFocus(page);
await assertKeyboardActiveDescendant(page);
await assertEscapeRestoresTriggerFocus(page);

await browser.close();

if (errors.length > 0) {
  throw new Error(`browser console/page errors: ${errors.join(" | ")}`);
}

console.log(`select popover smoke passed on ${baseUrl}${route}`);

async function openFirstAdvancedFilterWithSelect(page) {
  for (const pathname of candidateRoutes) {
    await page.goto(`${baseUrl}${pathname}`, { waitUntil: "domcontentloaded", timeout: 15_000 });
    await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
    const head = page.locator(".advanced-filter-head").first();
    if ((await head.count()) === 0) {
      continue;
    }
    if ((await head.getAttribute("aria-expanded")) !== "true") {
      await head.click();
    }
    const popover = page.locator(".advanced-filter-popover").first();
    await popover.waitFor({ timeout: 5_000 });
    if ((await page.locator(".advanced-filter-controls .select-popover-trigger").count()) > 0) {
      return pathname;
    }
  }
  return "";
}

async function assertSelectStructure(page, scope) {
  const snapshot = await page.evaluate(() => {
    const trigger = document.querySelector(".advanced-filter-controls .select-popover-trigger");
    const menu = document.querySelector('[data-select-popover-menu="true"]');
    const listbox = menu?.querySelector('[role="listbox"]');
    const options = Array.from(menu?.querySelectorAll('[role="option"]') ?? []);
    const searchInput = menu?.querySelector('input[type="search"]');
    return {
      triggerExpanded: trigger?.getAttribute("aria-expanded") ?? "",
      triggerControls: trigger?.getAttribute("aria-controls") ?? "",
      menuPlacement: menu?.getAttribute("data-placement") ?? "",
      listboxId: listbox?.id ?? "",
      listboxRole: listbox?.getAttribute("role") ?? "",
      visibleLimit: listbox?.getAttribute("data-select-visible-limit") ?? "",
      optionCount: options.length,
      firstOptionPos: options[0]?.getAttribute("aria-posinset") ?? "",
      firstOptionSetSize: options[0]?.getAttribute("aria-setsize") ?? "",
      searchControls: searchInput?.getAttribute("aria-controls") ?? "",
      searchAutocomplete: searchInput?.getAttribute("aria-autocomplete") ?? ""
    };
  });
  assert.equal(snapshot.triggerExpanded, "true", `${scope}: trigger must expose open state`);
  assert(snapshot.triggerControls, `${scope}: trigger must point at the listbox`);
  assert.equal(snapshot.triggerControls, snapshot.listboxId, `${scope}: trigger controls must match listbox id`);
  assert.equal(snapshot.listboxRole, "listbox", `${scope}: menu must contain a listbox`);
  assert(snapshot.optionCount > 0, `${scope}: select popover must render options`);
  assert(snapshot.visibleLimit, `${scope}: listbox must expose its virtual render limit`);
  assert.equal(Number(snapshot.visibleLimit), 36, `${scope}: virtual render limit must stay compact`);
  assert(
    snapshot.optionCount <= Number(snapshot.visibleLimit),
    `${scope}: select popover must not render beyond its virtual window`
  );
  assert.equal(snapshot.firstOptionPos, "1", `${scope}: options must expose aria-posinset`);
  assert(snapshot.firstOptionSetSize, `${scope}: options must expose aria-setsize`);
  assert(
    snapshot.menuPlacement === "top" || snapshot.menuPlacement === "bottom",
    `${scope}: menu must expose adaptive placement`
  );
  if (snapshot.searchControls) {
    assert.equal(
      snapshot.searchControls,
      snapshot.listboxId,
      `${scope}: search input must control the same listbox`
    );
    assert.equal(
      snapshot.searchAutocomplete,
      "list",
      `${scope}: searchable select must declare list autocomplete`
    );
  }
}

async function assertSearchClearKeepsInputFocus(page) {
  const searchInput = page.locator('[data-select-popover-menu="true"] input[type="search"]').first();
  if ((await searchInput.count()) === 0) {
    return;
  }
  const firstOptionLabel = (await page.locator('[data-select-popover-menu="true"] [role="option"]').first().innerText())
    .trim()
    .slice(0, 1);
  await searchInput.fill(firstOptionLabel || "a");
  await assertSearchResultsStayVirtualized(page);
  await page.getByRole("button", { name: "清空搜索" }).click();
  await page.waitForFunction(() => {
    const input = document.querySelector('[data-select-popover-menu="true"] input[type="search"]');
    return document.activeElement === input && input?.value === "";
  });
}

async function assertSearchResultsStayVirtualized(page) {
  const snapshot = await page.evaluate(() => {
    const listbox = document.querySelector('[data-select-popover-menu="true"] [role="listbox"]');
    const options = Array.from(listbox?.querySelectorAll('[role="option"]') ?? []);
    return {
      optionCount: options.length,
      visibleLimit: listbox?.getAttribute("data-select-visible-limit") ?? ""
    };
  });
  assert(snapshot.visibleLimit, "searched select listbox must expose its virtual render limit");
  assert(
    snapshot.optionCount <= Number(snapshot.visibleLimit),
    "searched select popover must stay inside its virtual render window"
  );
}

async function assertKeyboardActiveDescendant(page) {
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("PageDown");
  const snapshot = await page.evaluate(() => {
    const menu = document.querySelector('[data-select-popover-menu="true"]');
    const listbox = menu?.querySelector('[role="listbox"]');
    const searchInput = menu?.querySelector('input[type="search"]');
    const activeId =
      searchInput?.getAttribute("aria-activedescendant") ??
      listbox?.getAttribute("aria-activedescendant") ??
      "";
    const activeNode = activeId ? document.getElementById(activeId) : null;
    return {
      activeId,
      activeNodeRole: activeNode?.getAttribute("role") ?? "",
      activeNodeInsideMenu: activeNode ? Boolean(menu?.contains(activeNode)) : false,
      activeNodeWindowIndex: activeNode?.getAttribute("data-select-window-index") ?? "",
      listWindowStart: listbox?.getAttribute("data-select-window-start") ?? ""
    };
  });
  assert(snapshot.activeId, "keyboard navigation must publish aria-activedescendant");
  assert.equal(snapshot.activeNodeRole, "option", "aria-activedescendant must point at an option");
  assert(snapshot.activeNodeInsideMenu, "active descendant must exist inside the open menu");
  assert(snapshot.activeNodeWindowIndex, "active option must expose its absolute window index");
  assert(snapshot.listWindowStart !== "", "listbox must expose current render window start");
}

async function assertEscapeRestoresTriggerFocus(page) {
  await page.keyboard.press("Escape");
  await page.locator('[data-select-popover-menu="true"]').waitFor({ state: "hidden", timeout: 5_000 });
  const snapshot = await page.evaluate(() => ({
    activeIsTrigger: document.activeElement?.classList.contains("select-popover-trigger") ?? false,
    triggerExpanded:
      document
        .querySelector(".advanced-filter-controls .select-popover-trigger")
        ?.getAttribute("aria-expanded") ?? ""
  }));
  assert(snapshot.activeIsTrigger, "Escape must restore focus to the select trigger");
  assert.equal(snapshot.triggerExpanded, "false", "Escape must close the select popover");
}
