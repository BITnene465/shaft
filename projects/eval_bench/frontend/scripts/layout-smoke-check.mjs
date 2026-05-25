import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/").replace(/\/+$/, "");

const viewports = [
  { name: "desktop", width: 1440, height: 960 },
  { name: "compact", width: 1024, height: 760 },
  { name: "narrow", width: 760, height: 860 }
];

const staticRoutes = [
  {
    name: "overview",
    path: "/",
    selectors: [
      ".dashboard-home",
      ".overview-console",
      ".overview-rhythm-strip",
      ".overview-grid.refined",
      ".overview-mini-chart"
    ]
  },
  {
    name: "rank-board",
    path: "/rank-board",
    selectors: [".rank-board-page", ".advanced-filter-bar", ".rank-facet-rail", ".table-shell"],
    requireRankChunk: true
  },
  {
    name: "runs",
    path: "/runs",
    selectors: [".run-table-stack", ".advanced-filter-bar", ".table-shell"]
  },
  {
    name: "benchmarks",
    path: "/benchmarks",
    selectors: [".advanced-filter-bar", ".table-shell"]
  },
  {
    name: "jobs",
    path: "/jobs",
    selectors: [".queue-stack", ".advanced-filter-bar"]
  },
  {
    name: "services",
    path: "/services",
    selectors: [".advanced-filter-bar"]
  },
  {
    name: "settings",
    path: "/settings",
    selectors: [".settings-workbench-shell", ".settings-preview-stage", ".settings-drawer-scroll"]
  },
  {
    name: "compare",
    path: "/compare",
    selectors: [".compare-page", ".compare-workspace", ".compare-context-pane"],
    forbiddenSelectors: [".compare-leaderboard-pane"],
    requireCompareChunk: true
  }
];

const routes = [...staticRoutes, ...(await discoverInspectorRoutes(baseUrl))];

const dialogCases = [
  { path: "/jobs", button: "新建评测", form: ".manifest-job-form" },
  { path: "/benchmarks", button: "创建副本", form: ".benchmark-form" },
  { path: "/runs", button: "导入预测", form: ".import-form" },
  { path: "/services", button: "登记服务", form: ".service-form" }
];

const browser = await chromium.launch();
const errors = [];

try {
  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport });
    attachErrorListeners(page, errors, viewport.name);
    for (const route of routes) {
      await page.goto(`${baseUrl}${route.path}`, { waitUntil: "networkidle" });
      await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
      await page.locator(".content").first().waitFor({ timeout: 10_000 });
      for (const selector of route.selectors) {
        await page.locator(selector).first().waitFor({ timeout: 10_000 });
      }
      await assertShellLayout(page, `${viewport.name}:${route.name}`);
      await assertTopbarStatus(page, `${viewport.name}:${route.name}`);
      await assertPageStack(page, `${viewport.name}:${route.name}`);
      await assertTablesCanScroll(page, `${viewport.name}:${route.name}`);
      await assertAdvancedFilters(page, `${viewport.name}:${route.name}`);
      await assertForbiddenSelectors(page, `${viewport.name}:${route.name}`, route.forbiddenSelectors ?? []);
      if (route.name === "overview") {
        await assertOverviewDensity(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireInspectorFilters) {
        await assertInspectorFilters(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunInspectorCounts) {
        await assertRunInspectorCountStrip(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRankChunk) {
        await assertRankBoardChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireCompareChunk) {
        await assertComparePageChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
    }
    await page.close();
  }

  const dialogPage = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  attachErrorListeners(dialogPage, errors, "dialogs");
  for (const item of dialogCases) {
    await dialogPage.goto(`${baseUrl}${item.path}`, { waitUntil: "networkidle" });
    await dialogPage.locator(".content").first().waitFor({ timeout: 10_000 });
    await dialogPage.getByRole("button", { name: item.button }).click();
    await dialogPage.locator(".workspace-dialog").first().waitFor({ timeout: 5_000 });
    await dialogPage.locator(item.form).first().waitFor({ timeout: 5_000 });
    await assertDialogLayout(dialogPage, `dialog:${item.path}`);
    await dialogPage.keyboard.press("Escape");
    await dialogPage.locator(".workspace-dialog").waitFor({ state: "hidden", timeout: 5_000 });
  }
  await dialogPage.close();
} finally {
  await browser.close();
}

if (errors.length > 0) {
  throw new Error(`layout smoke failed:\n${errors.join("\n")}`);
}

console.log(`layout smoke passed on ${baseUrl}`);
console.log(
  JSON.stringify(
    {
      routes: routes.map((route) => route.path),
      viewports: viewports.map((viewport) => `${viewport.name}:${viewport.width}x${viewport.height}`),
      dialogs: dialogCases.map((item) => item.path)
    },
    null,
    2
  )
);

function attachErrorListeners(page, errors, scope) {
  page.on("pageerror", (error) => errors.push(`${scope}: page error: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`${scope}: console error: ${message.text()}`);
    }
  });
}

async function discoverInspectorRoutes(rootUrl) {
  try {
    const response = await fetch(`${rootUrl}/api/state`);
    if (!response.ok) {
      return [];
    }
    const state = await response.json();
    const routes = [];
    const benchmark = Array.isArray(state.benchmarks)
      ? state.benchmarks.find((item) => item.sample_count > 0)
      : null;
    const run = Array.isArray(state.runs)
      ? state.runs.find((item) => item.benchmark_id && item.report_path)
      : null;
    if (benchmark?.benchmark_id) {
      routes.push({
        name: "benchmark-inspector",
        path: `/benchmarks/${encodeURIComponent(benchmark.benchmark_id)}`,
        selectors: [
          ".visual-inspector-page",
          ".inspector-sidebar",
          ".advanced-filter-bar",
          ".sample-list",
          ".viewer-panel"
        ],
        requireInspectorFilters: true
      });
    }
    if (run?.run_id) {
      routes.push({
        name: "run-inspector",
        path: `/runs/${encodeURIComponent(run.run_id)}`,
        selectors: [
          ".visual-inspector-page",
          ".inspector-sidebar",
          ".advanced-filter-bar",
          ".sample-list",
          ".viewer-panel"
        ],
        requireInspectorFilters: true,
        requireRunInspectorCounts: true
      });
    }
    return routes;
  } catch {
    return [];
  }
}

async function assertShellLayout(page, scope) {
  const state = await page.evaluate(() => {
    const app = document.querySelector(".app-shell")?.getBoundingClientRect();
    const content = document.querySelector(".content")?.getBoundingClientRect();
    return {
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      bodyScrollWidth: document.body.scrollWidth,
      docScrollWidth: document.documentElement.scrollWidth,
      bodyScrollHeight: document.body.scrollHeight,
      docScrollHeight: document.documentElement.scrollHeight,
      fatalVisible: !!document.querySelector(".fatal-panel"),
      app,
      content
    };
  });
  if (state.fatalVisible) {
    throw new Error(`${scope}: fatal panel is visible`);
  }
  if (Math.max(state.bodyScrollWidth, state.docScrollWidth) > state.viewportWidth + 2) {
    throw new Error(
      `${scope}: global horizontal overflow ${state.bodyScrollWidth}/${state.docScrollWidth} > ${state.viewportWidth}`
    );
  }
  if (Math.max(state.bodyScrollHeight, state.docScrollHeight) > state.viewportHeight + 2) {
    throw new Error(
      `${scope}: global vertical overflow ${state.bodyScrollHeight}/${state.docScrollHeight} > ${state.viewportHeight}`
    );
  }
  for (const [name, rect] of [
    ["app-shell", state.app],
    ["content", state.content]
  ]) {
    if (!rect) {
      throw new Error(`${scope}: ${name} is missing`);
    }
    if (rect.left < -2 || rect.top < -2 || rect.right > state.viewportWidth + 2 || rect.bottom > state.viewportHeight + 2) {
      throw new Error(`${scope}: ${name} is clipped outside viewport ${formatRect(rect)}`);
    }
  }
}

async function assertPageStack(page, scope) {
  const state = await page.evaluate(() => {
    const stack = document.querySelector(".page-stack");
    const content = document.querySelector(".content");
    if (!stack || !content) {
      return null;
    }
    const stackRect = stack.getBoundingClientRect();
    const contentRect = content.getBoundingClientRect();
    const style = getComputedStyle(stack);
    return {
      stackRect,
      contentRect,
      scrollHeight: stack.scrollHeight,
      clientHeight: stack.clientHeight,
      scrollWidth: stack.scrollWidth,
      clientWidth: stack.clientWidth,
      overflowX: style.overflowX,
      overflowY: style.overflowY
    };
  });
  if (!state) {
    throw new Error(`${scope}: page stack is missing`);
  }
  if (state.stackRect.bottom > state.contentRect.bottom + 2) {
    throw new Error(
      `${scope}: page stack extends beyond clipped content: stack=${formatRect(
        state.stackRect
      )}, content=${formatRect(state.contentRect)}`
    );
  }
  if (state.scrollHeight > state.clientHeight + 2 && !allowsScroll(state.overflowY)) {
    throw new Error(
      `${scope}: page stack needs vertical scroll but overflow-y=${state.overflowY}`
    );
  }
  if (state.scrollWidth > state.clientWidth + 2 && !allowsScroll(state.overflowX)) {
    throw new Error(`${scope}: page stack needs horizontal scroll but overflow-x=${state.overflowX}`);
  }
}

async function assertTopbarStatus(page, scope) {
  const state = await page.evaluate(() => {
    const actions = document.querySelector(".topbar-actions");
    const status = document.querySelector(".topbar .status-pill");
    const actionStyle = actions ? getComputedStyle(actions) : null;
    const statusStyle = status ? getComputedStyle(status) : null;
    return {
      action: actionStyle
        ? {
            padding: actionStyle.padding,
            backgroundColor: actionStyle.backgroundColor,
            borderTopWidth: actionStyle.borderTopWidth,
            boxShadow: actionStyle.boxShadow
          }
        : null,
      status: statusStyle
        ? {
            borderRadius: statusStyle.borderRadius,
            minHeight: statusStyle.minHeight,
            overflow: statusStyle.overflow
          }
        : null
    };
  });
  if (!state.action) {
    throw new Error(`${scope}: topbar actions are missing`);
  }
  if (
    state.action.padding !== "0px" ||
    state.action.borderTopWidth !== "0px" ||
    state.action.boxShadow !== "none" ||
    state.action.backgroundColor !== "rgba(0, 0, 0, 0)"
  ) {
    throw new Error(
      `${scope}: topbar actions still look like an outer rounded container ${JSON.stringify(
        state.action
      )}`
    );
  }
  if (!state.status) {
    throw new Error(`${scope}: status pill is missing`);
  }
  if (!state.status.borderRadius.includes("999") || state.status.overflow !== "hidden") {
    throw new Error(`${scope}: status pill is not a rounded capsule ${JSON.stringify(state.status)}`);
  }
}

async function assertOverviewDensity(page, scope) {
  const state = await page.evaluate(() => {
    const recentRows = Array.from(document.querySelectorAll(".overview-run-list a")).map((node) => {
      const rect = node.getBoundingClientRect();
      return Math.round(rect.height);
    });
    const chartRects = Array.from(document.querySelectorAll(".overview-mini-chart")).map((node) => {
      const rect = node.getBoundingClientRect();
      return { width: Math.round(rect.width), height: Math.round(rect.height) };
    });
    const rhythm = document.querySelector(".overview-rhythm-strip");
    const rhythmStyle = rhythm ? getComputedStyle(rhythm) : null;
    return {
      rhythmBars: document.querySelectorAll(".overview-rhythm-bars span").length,
      miniCharts: chartRects.length,
      chartRects,
      recentRows,
      oldTimelinePanels: document.querySelectorAll(
        ".overview-timeline-panel, .overview-sparkline, .overview-timeline-labels"
      ).length,
      rhythmHeight: rhythm ? Math.round(rhythm.getBoundingClientRect().height) : 0,
      rhythmOverflowX: rhythmStyle?.overflowX ?? "",
      rhythmOverflowY: rhythmStyle?.overflowY ?? ""
    };
  });
  if (state.rhythmBars !== 12) {
    throw new Error(`${scope}: overview rhythm should use 12 compact bars, got ${state.rhythmBars}`);
  }
  if (state.miniCharts < 4) {
    throw new Error(`${scope}: overview should expose at least four mini charts, got ${state.miniCharts}`);
  }
  if (state.oldTimelinePanels > 0) {
    throw new Error(`${scope}: old oversized overview timeline markup is still present`);
  }
  if (state.recentRows.some((height) => height > 72)) {
    throw new Error(`${scope}: recent run rows are stretched ${state.recentRows.join(",")}`);
  }
  if (state.rhythmHeight > 90 || state.rhythmOverflowX === "visible" || state.rhythmOverflowY === "visible") {
    throw new Error(
      `${scope}: rhythm strip is not compact ${JSON.stringify({
        height: state.rhythmHeight,
        overflowX: state.rhythmOverflowX,
        overflowY: state.rhythmOverflowY
      })}`
    );
  }
  for (const [index, rect] of state.chartRects.entries()) {
    if (rect.width <= 0 || rect.height <= 0) {
      throw new Error(`${scope}: mini chart ${index} is not visible ${JSON.stringify(rect)}`);
    }
  }
}

async function assertTablesCanScroll(page, scope) {
  const tables = await page.locator(".table-shell").evaluateAll((nodes) =>
    nodes
      .filter((node) => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((node, index) => {
        const style = getComputedStyle(node);
        return {
          index,
          scrollWidth: node.scrollWidth,
          clientWidth: node.clientWidth,
          scrollHeight: node.scrollHeight,
          clientHeight: node.clientHeight,
          overflowX: style.overflowX,
          overflowY: style.overflowY
        };
      })
  );
  for (const table of tables) {
    if (table.scrollWidth > table.clientWidth + 2 && !allowsScroll(table.overflowX)) {
      throw new Error(`${scope}: table ${table.index} needs horizontal scroll`);
    }
    if (table.scrollHeight > table.clientHeight + 2 && !allowsScroll(table.overflowY)) {
      throw new Error(`${scope}: table ${table.index} needs vertical scroll`);
    }
  }
}

async function assertAdvancedFilters(page, scope) {
  const filters = await page.locator(".advanced-filter-bar").evaluateAll((nodes) =>
    nodes
      .filter((node) => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((node, index) => {
        const style = getComputedStyle(node);
        return {
          index,
          scrollWidth: node.scrollWidth,
          clientWidth: node.clientWidth,
          scrollHeight: node.scrollHeight,
          clientHeight: node.clientHeight,
          overflowX: style.overflowX,
          overflowY: style.overflowY
        };
      })
  );
  for (const filter of filters) {
    if (filter.scrollWidth > filter.clientWidth + 2 && !allowsScroll(filter.overflowX)) {
      throw new Error(`${scope}: advanced filter ${filter.index} clips horizontally`);
    }
    if (filter.scrollHeight > filter.clientHeight + 2 && !allowsScroll(filter.overflowY)) {
      throw new Error(`${scope}: advanced filter ${filter.index} clips vertically`);
    }
  }
}

async function assertInspectorFilters(page, scope) {
  const state = await page.evaluate(() => {
    const sidebar = document.querySelector(".inspector-sidebar");
    const filter = sidebar?.querySelector(".advanced-filter-bar");
    const oldFilters = sidebar?.querySelectorAll(".sample-filters").length ?? 0;
    const filterRect = filter?.getBoundingClientRect();
    const sidebarRect = sidebar?.getBoundingClientRect();
    const controlsVisible = Boolean(filter?.querySelector(".advanced-filter-controls"));
    const button = filter?.querySelector(".advanced-filter-head");
    const buttonRect = button?.getBoundingClientRect();
    return {
      hasFilter: Boolean(filter),
      oldFilters,
      controlsVisible,
      filterRect,
      sidebarRect,
      buttonRect,
      expanded: button?.getAttribute("aria-expanded") ?? null
    };
  });
  if (!state.hasFilter) {
    throw new Error(`${scope}: inspector sample filters are not using AdvancedFilterBar`);
  }
  if (state.oldFilters > 0) {
    throw new Error(`${scope}: legacy sample-filters container is still present`);
  }
  if (state.controlsVisible || state.expanded !== "false") {
    throw new Error(`${scope}: inspector advanced filters should be collapsed by default`);
  }
  if (!state.filterRect || !state.sidebarRect || !state.buttonRect) {
    throw new Error(`${scope}: inspector filter geometry is missing`);
  }
  if (state.filterRect.right > state.sidebarRect.right + 2) {
    throw new Error(
      `${scope}: inspector filter overflows sidebar filter=${formatRect(
        state.filterRect
      )}, sidebar=${formatRect(state.sidebarRect)}`
    );
  }
  if (state.buttonRect.height > 44) {
    throw new Error(`${scope}: inspector collapsed filter head is too tall ${formatRect(state.buttonRect)}`);
  }
}

async function assertRunInspectorCountStrip(page, scope) {
  const state = await page.evaluate(() => {
    const strip = document.querySelector(".viewer-side-panel .diagnostic-strip");
    return {
      text: strip?.textContent?.replace(/\s+/g, " ").trim() ?? "",
      chipCount: strip?.querySelectorAll(":scope > span").length ?? 0,
      compact: strip?.classList.contains("compact-counts") ?? false
    };
  });
  if (!state.compact || state.chipCount !== 2) {
    throw new Error(`${scope}: run inspector count strip should expose exactly two compact chips`);
  }
  if (!state.text.includes("真实") || !state.text.includes("预测")) {
    throw new Error(`${scope}: run inspector count strip is missing GT/pred counts: ${state.text}`);
  }
  if (/\b(TP|FP|FN)\b|IoU|平均/.test(state.text)) {
    throw new Error(`${scope}: run inspector count strip exposes fine metrics: ${state.text}`);
  }
}

async function assertDialogLayout(page, scope) {
  const state = await page.evaluate(() => {
    const dialog = document.querySelector(".workspace-dialog");
    const body = document.querySelector(".workspace-dialog-body");
    if (!dialog || !body) {
      return null;
    }
    const dialogRect = dialog.getBoundingClientRect();
    const bodyStyle = getComputedStyle(body);
    return {
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      dialogRect,
      bodyScrollHeight: body.scrollHeight,
      bodyClientHeight: body.clientHeight,
      bodyScrollWidth: body.scrollWidth,
      bodyClientWidth: body.clientWidth,
      bodyOverflowX: bodyStyle.overflowX,
      bodyOverflowY: bodyStyle.overflowY
    };
  });
  if (!state) {
    throw new Error(`${scope}: dialog is missing`);
  }
  if (
    state.dialogRect.left < -2 ||
    state.dialogRect.top < -2 ||
    state.dialogRect.right > state.viewportWidth + 2 ||
    state.dialogRect.bottom > state.viewportHeight + 2
  ) {
    throw new Error(`${scope}: dialog escapes viewport ${formatRect(state.dialogRect)}`);
  }
  if (state.bodyScrollHeight > state.bodyClientHeight + 2 && !allowsScroll(state.bodyOverflowY)) {
    throw new Error(`${scope}: dialog body needs vertical scroll but overflow-y=${state.bodyOverflowY}`);
  }
  if (state.bodyScrollWidth > state.bodyClientWidth + 2 && !allowsScroll(state.bodyOverflowX)) {
    throw new Error(`${scope}: dialog body needs horizontal scroll but overflow-x=${state.bodyOverflowX}`);
  }
}

async function assertForbiddenSelectors(page, scope, selectors) {
  for (const selector of selectors) {
    if ((await page.locator(selector).count()) > 0) {
      throw new Error(`${scope}: forbidden selector still exists: ${selector}`);
    }
  }
}

async function assertRankBoardChunkLoaded(page, scope) {
  const loaded = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .filter((name) => name.includes("rankBoardPage"))
  );
  if (loaded.length === 0) {
    throw new Error(`${scope}: rank-board route did not load the independent rankBoardPage chunk`);
  }
}

async function assertComparePageChunkLoaded(page, scope) {
  const loaded = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .filter((name) => name.includes("comparePage"))
  );
  if (loaded.length === 0) {
    throw new Error(`${scope}: compare route did not load the independent comparePage chunk`);
  }
}

function allowsScroll(value) {
  return value === "auto" || value === "scroll";
}

function formatRect(rect) {
  return `x=${Math.round(rect.left)},y=${Math.round(rect.top)},w=${Math.round(
    rect.width
  )},h=${Math.round(rect.height)},bottom=${Math.round(rect.bottom)},right=${Math.round(rect.right)}`;
}
