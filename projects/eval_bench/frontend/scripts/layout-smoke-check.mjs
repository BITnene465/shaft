import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/").replace(/\/+$/, "");
const expectedConsoleErrors = new WeakMap();

const viewports = [
  { name: "wide", width: 1920, height: 1080 },
  { name: "desktop", width: 1440, height: 960 },
  { name: "laptop", width: 1280, height: 720 },
  { name: "compact", width: 1024, height: 760 },
  { name: "short", width: 900, height: 600 },
  { name: "landscape", width: 900, height: 430 },
  { name: "narrow", width: 760, height: 860 }
];

const staticRoutes = [
  {
    name: "overview",
    path: "/",
    selectors: [
      ".dashboard-home",
      ".overview-home-v18",
      ".overview-v18-grid",
      ".overview-v18-primary",
      ".overview-v18-queue",
      ".overview-v18-recent",
      ".overview-v18-resources",
      ".overview-v18-console",
      ".overview-v18-signal-map",
      ".overview-v18-flow-item"
    ]
  },
  {
    name: "rank-board",
    path: "/rank-board",
    selectors: [
      ".rank-board-page",
      ".rank-board-summary",
      ".rank-board-table-card",
      ".advanced-filter-bar",
      ".rank-board-table-card"
    ],
    requireRankChunk: true
  },
  {
    name: "suite-report",
    path: "/suite-report",
    selectors: [
      ".composite-report-page",
      ".composite-report-shell",
      ".composite-composer-dock",
      ".composite-stage-region",
      ".composite-report-stage-card"
    ],
    requireCompositeReport: true
  },
  {
    name: "runs",
    path: "/runs",
    selectors: [".run-table-stack", ".advanced-filter-bar", ".table-shell", ".run-list-pager"],
    requireRunsChunk: true
  },
  {
    name: "benchmarks",
    path: "/benchmarks",
    selectors: [".advanced-filter-bar", ".table-shell", ".benchmark-list-pager"],
    requireBenchmarksChunk: true
  },
  {
    name: "jobs",
    path: "/jobs",
    selectors: [".queue-stack", ".advanced-filter-bar", ".job-list-pager"]
  },
  {
    name: "services",
    path: "/services",
    selectors: [".advanced-filter-bar", ".service-list-pager"]
  },
  {
    name: "settings",
    path: "/settings",
    selectors: [".settings-workbench-shell", ".settings-preview-stage", ".settings-drawer-scroll"]
  },
  {
    name: "compare",
    path: "/compare",
    selectors: [".compare-page", ".compare-workspace", ".compare-context-pane", ".compare-run-pager"],
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
      await page.goto(`${baseUrl}${route.path}`, { waitUntil: "domcontentloaded" });
      await page.locator(".app-shell").first().waitFor({ timeout: 10_000 });
      await page.locator(".content").first().waitFor({ timeout: 10_000 });
      for (const selector of route.selectors) {
        await page.locator(selector).first().waitFor({ timeout: 10_000 });
      }
      await assertShellLayout(page, `${viewport.name}:${route.name}`);
      await assertTopbarStatus(page, `${viewport.name}:${route.name}`);
      await assertPageStack(page, `${viewport.name}:${route.name}`);
      await assertNoClippedPanels(page, `${viewport.name}:${route.name}`);
      await assertTablesCanScroll(page, `${viewport.name}:${route.name}`);
      await assertPagersFit(page, `${viewport.name}:${route.name}`);
      await assertTableActionCellsFit(page, `${viewport.name}:${route.name}`);
      await assertFormControlsFit(page, `${viewport.name}:${route.name}`);
      await assertAdvancedFilters(page, `${viewport.name}:${route.name}`);
      await assertAdvancedFilterKeyboardFlow(page, `${viewport.name}:${route.name}`);
      if (route.name === "runs" || route.name === "rank-board") {
        await assertAdvancedFilterClear(page, `${viewport.name}:${route.name}`);
      }
      await assertForbiddenSelectors(page, `${viewport.name}:${route.name}`, route.forbiddenSelectors ?? []);
      if (route.name === "overview") {
        await assertOverviewDensity(page, `${viewport.name}:${route.name}`);
      }
      if (route.name === "jobs") {
        await assertJobsDetailSidebar(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireInspectorFilters) {
        await assertInspectorFilters(page, `${viewport.name}:${route.name}`);
        await assertInspectorSampleList(page, `${viewport.name}:${route.name}`);
        await assertInspectorCanvasPane(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunInspectorCounts) {
        await assertRunInspectorCountStrip(page, `${viewport.name}:${route.name}`);
        await assertRunViewerLayerPreferencesPersist(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunNoteTemplates) {
        await assertRunNoteTemplates(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireInspectorFilters) {
        await assertInspectorFilteredEmptyState(page, `${viewport.name}:${route.name}`, route.name);
      }
      if (route.requireRankChunk) {
        await assertRankBoardChunkLoaded(page, `${viewport.name}:${route.name}`);
        await assertRankHeaderSorting(page, `${viewport.name}:${route.name}`);
        await assertRankFacetRail(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireCompositeReport) {
        await assertCompositeReportWorkbench(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireBenchmarksChunk) {
        await assertBenchmarksPageChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunsChunk) {
        await assertRunsPageChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireCompareChunk) {
        await assertComparePageChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireComparisonSampleChunk) {
        await assertComparisonSamplePageChunkLoaded(page, `${viewport.name}:${route.name}`);
      }
    }
    await page.close();
  }

  const dialogPage = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  attachErrorListeners(dialogPage, errors, "dialogs");
  for (const item of dialogCases) {
    await dialogPage.goto(`${baseUrl}${item.path}`, { waitUntil: "domcontentloaded" });
    await dialogPage.locator(".content").first().waitFor({ timeout: 10_000 });
    await dialogPage.getByRole("button", { name: item.button }).click();
    await dialogPage.locator(".workspace-dialog").first().waitFor({ timeout: 5_000 });
    await dialogPage.locator(item.form).first().waitFor({ timeout: 5_000 });
    await assertDialogLayout(dialogPage, `dialog:${item.path}`);
    await assertFormControlsFit(dialogPage, `dialog:${item.path}`);
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
      const text = message.text();
      const allowedCount = expectedConsoleErrors.get(page) ?? 0;
      if (allowedCount > 0 && text.includes("400 (Bad Request)")) {
        expectedConsoleErrors.set(page, allowedCount - 1);
        return;
      }
      errors.push(`${scope}: console error: ${text}`);
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
        requireRunNoteTemplates: true,
        requireRunInspectorCounts: true,
        requireRunsChunk: true
      });
    }
    const comparisonsResponse = await fetch(`${rootUrl}/api/comparisons?limit=1`);
    if (comparisonsResponse.ok) {
      const comparisonsState = await comparisonsResponse.json();
      const comparison = Array.isArray(comparisonsState.comparisons)
        ? comparisonsState.comparisons[0]
        : null;
      if (comparison?.baseline_run_id && comparison?.candidate_run_id) {
        routes.push({
          name: "comparison-sample",
          path: `/compare/${encodeURIComponent(comparison.baseline_run_id)}/${encodeURIComponent(
            comparison.candidate_run_id
          )}/0`,
          selectors: [
            ".comparison-sample-page",
            ".comparison-sample-detail",
            ".comparison-run-panel",
            ".viewer-stack"
          ],
          requireComparisonSampleChunk: true
        });
      }
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

async function assertNoClippedPanels(page, scope) {
  const panels = await page.evaluate(() => {
    const selectors = [
      ".workspace-card",
      ".queue-stack",
      ".manifest-card",
      ".manifest-editor-pane",
      ".manifest-result-pane",
      ".visual-inspector-page",
      ".inspector-sidebar",
      ".sample-list",
      ".viewer-panel",
      ".viewer-side-panel",
      ".compare-workspace",
      ".compare-report-pane",
      ".compare-context-pane",
      ".comparison-sample-detail",
      ".comparison-run-panel",
      ".settings-workbench-shell",
      ".settings-drawer-scroll",
      ".settings-preview-stage"
    ];
    return selectors.flatMap((selector) =>
      Array.from(document.querySelectorAll(selector))
        .filter((node) => {
          const rect = node.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        })
        .map((node, index) => {
          const style = getComputedStyle(node);
          const rect = node.getBoundingClientRect();
          return {
            selector,
            index,
            rect,
            scrollWidth: node.scrollWidth,
            clientWidth: node.clientWidth,
            scrollHeight: node.scrollHeight,
            clientHeight: node.clientHeight,
            overflowX: style.overflowX,
            overflowY: style.overflowY
          };
        })
    );
  });
  for (const panel of panels) {
    if (panel.scrollWidth > panel.clientWidth + 2 && clipsOverflow(panel.overflowX)) {
      throw new Error(
        `${scope}: ${panel.selector}[${panel.index}] clips horizontal content ${JSON.stringify({
          scrollWidth: panel.scrollWidth,
          clientWidth: panel.clientWidth,
          overflowX: panel.overflowX,
          rect: formatRect(panel.rect)
        })}`
      );
    }
    if (panel.scrollHeight > panel.clientHeight + 2 && clipsOverflow(panel.overflowY)) {
      throw new Error(
        `${scope}: ${panel.selector}[${panel.index}] clips vertical content ${JSON.stringify({
          scrollHeight: panel.scrollHeight,
          clientHeight: panel.clientHeight,
          overflowY: panel.overflowY,
          rect: formatRect(panel.rect)
        })}`
      );
    }
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
  const overviewState = await page.evaluate(() => {
    const root = document.querySelector(".overview-home-v18");
    const grid = document.querySelector(".overview-v18-grid");
    const primary = document.querySelector(".overview-v18-primary");
    const queue = document.querySelector(".overview-v18-queue");
    const recent = document.querySelector(".overview-v18-recent");
    const resources = document.querySelector(".overview-v18-resources");
    const primaryRect = primary?.getBoundingClientRect();
    const queueRect = queue?.getBoundingClientRect();
    const recentRect = recent?.getBoundingClientRect();
    const resourceRect = resources?.getBoundingClientRect();
    const runRows = Array.from(document.querySelectorAll(".overview-v18-run-list a")).map((node) => {
      const rect = node.getBoundingClientRect();
      return {
        height: Math.round(rect.height),
        text: node.textContent ?? ""
      };
    });
    return {
      text: root?.textContent ?? "",
      cardCount: document.querySelectorAll(".overview-v18-card").length,
      flowItems: document.querySelectorAll(".overview-v18-flow-item").length,
      consoleCount: document.querySelectorAll(".overview-v18-console").length,
      consoleTabs: document.querySelectorAll(".overview-v18-surface-tab").length,
      signalMaps: document.querySelectorAll(".overview-v18-signal-map").length,
      signalNodes: document.querySelectorAll(".overview-v18-signal-node").length,
      signalInspectors: document.querySelectorAll(".overview-v18-signal-inspector").length,
      primaryCount: document.querySelectorAll(".overview-v18-primary").length,
      queueCount: document.querySelectorAll(".overview-v18-queue").length,
      recentCount: document.querySelectorAll(".overview-v18-recent").length,
      resourceCount: document.querySelectorAll(".overview-v18-resources").length,
      runRows,
      runArtifactRails: document.querySelectorAll(".overview-v18-run-artifacts i b").length,
      legacyPanels: document.querySelectorAll(
        ".overview-ops-board, .overview-rank-console, .overview-decision-metric, .overview-telemetry-trace, .overview-resource-chips, .overview-state-strip, .overview-score-dial, .overview-run-focus, .overview-ops-signal, .overview-flow-node, .overview-mini-chart, .overview-chart-matrix, .overview-activity-matrix"
      ).length,
      gridColumns: grid ? getComputedStyle(grid).gridTemplateColumns.split(" ").length : 0,
      primary: primaryRect
        ? { width: Math.round(primaryRect.width), height: Math.round(primaryRect.height) }
        : null,
      queue: queueRect
        ? { width: Math.round(queueRect.width), height: Math.round(queueRect.height) }
        : null,
      recent: recentRect
        ? { width: Math.round(recentRect.width), height: Math.round(recentRect.height) }
        : null,
      resources: resourceRect
        ? { width: Math.round(resourceRect.width), height: Math.round(resourceRect.height) }
        : null
    };
  });
  if (
    overviewState.cardCount !== 4 ||
    overviewState.primaryCount !== 1 ||
    overviewState.queueCount !== 1 ||
    overviewState.recentCount !== 1 ||
    overviewState.resourceCount !== 1 ||
    overviewState.flowItems !== 3 ||
    overviewState.consoleCount !== 1 ||
    overviewState.consoleTabs !== 3 ||
    overviewState.signalMaps !== 1 ||
    overviewState.signalNodes !== 3 ||
    overviewState.signalInspectors !== 1
  ) {
    throw new Error(`${scope}: overview v18 should keep four focused modules ${JSON.stringify(overviewState)}`);
  }
  if (overviewState.legacyPanels > 0) {
    throw new Error(`${scope}: deprecated overview panels are still rendered`);
  }
  if (
    !scope.startsWith("narrow") &&
    !scope.startsWith("short") &&
    !scope.startsWith("landscape") &&
    overviewState.gridColumns < 2
  ) {
    throw new Error(`${scope}: overview should use two or three columns on wide screens`);
  }
  const recentMinimumHeight = overviewState.runRows.length > 0 ? 120 : 80;
  if (
    !overviewState.primary ||
    !overviewState.queue ||
    !overviewState.recent ||
    !overviewState.resources ||
    overviewState.primary.height < 170 ||
    overviewState.recent.height < recentMinimumHeight
  ) {
    throw new Error(`${scope}: overview panels collapsed ${JSON.stringify(overviewState)}`);
  }
  if (
    !scope.startsWith("narrow") &&
    !scope.startsWith("short") &&
    !scope.startsWith("landscape") &&
    overviewState.primary.width <= overviewState.queue.width
  ) {
    throw new Error(`${scope}: primary overview module should occupy the larger column span`);
  }
  if (overviewState.runRows.some((row) => row.height > 84)) {
    throw new Error(
      `${scope}: recent run rows are stretched ${overviewState.runRows.map((row) => row.height).join(",")}`
    );
  }
  if (overviewState.runRows.some((row) => /\b(P@|R@|precision|recall|iou|miou)\b/i.test(row.text))) {
    throw new Error(`${scope}: recent run stream exposes fine metrics`);
  }
  if (overviewState.runRows.length > 0 && overviewState.runArtifactRails !== overviewState.runRows.length) {
    throw new Error(`${scope}: recent run rows should expose artifact readiness rails`);
  }
  if (
    /只保留系统运行态、数据规模和近期写入节奏|精细指标进入排行榜与对比页|首页只保留|可以看排行|可以进入排行|查看排行榜|等待报告进入排行|主指标 F1 可排行|从样本到排行|rankable|F1 ready|先处理阻塞|补齐评估闭环|队列正在推进/.test(
      overviewState.text
    )
  ) {
    throw new Error(`${scope}: overview should not use slogan-style implementation copy`);
  }
  if (/\b(precision|recall|iou|miou)\b/i.test(overviewState.text)) {
    throw new Error(`${scope}: overview exposes fine-grained eval metric text`);
  }
  return;

  const state = await page.evaluate(() => {
    const recentRows = Array.from(document.querySelectorAll(".overview-run-list a")).map((node) => {
      const rect = node.getBoundingClientRect();
      return {
        height: Math.round(rect.height),
        text: node.textContent ?? ""
      };
    });
    const decisionTiles = Array.from(document.querySelectorAll(".overview-decision-metric")).map((node) => {
      const rect = node.getBoundingClientRect();
      return { width: Math.round(rect.width), height: Math.round(rect.height) };
    });
    const stateStripText = document.querySelector(".overview-state-strip")?.textContent ?? "";
    const commandShell = document.querySelector(".overview-ops-board");
    const nowPanel = document.querySelector(".overview-ops-board");
    const livePanel = document.querySelector(".overview-rank-console");
    const recentCard = document.querySelector(".overview-recent-card");
    const opsSignal = document.querySelector(".overview-ops-signal");
    const decisionMetric = document.querySelector(".overview-decision-metric");
    const scoreDial = document.querySelector(".overview-score-dial");
    const telemetryTrace = document.querySelector(".overview-telemetry-trace");
    const decisionMetricRails = Array.from(
      document.querySelectorAll(".overview-decision-metric > i > b")
    ).map((node) => Math.round(node.getBoundingClientRect().width));
    const telemetryRails = Array.from(
      document.querySelectorAll(".overview-telemetry-bar i b")
    ).map((node) => Math.round(node.getBoundingClientRect().width));
    const panelHeights = Array.from(
      document.querySelectorAll(
        ".overview-ops-board, .overview-recent-card"
      )
    ).map((node) => Math.round(node.getBoundingClientRect().height));
    const commandShellStyle = commandShell ? getComputedStyle(commandShell) : null;
    const nowPanelStyle = nowPanel ? getComputedStyle(nowPanel) : null;
    const livePanelStyle = livePanel ? getComputedStyle(livePanel) : null;
    const recentCardStyle = recentCard ? getComputedStyle(recentCard) : null;
    const opsSignalStyle = opsSignal ? getComputedStyle(opsSignal) : null;
    const decisionMetricStyle = decisionMetric ? getComputedStyle(decisionMetric) : null;
    const scoreDialStyle = scoreDial ? getComputedStyle(scoreDial) : null;
    const telemetryStyle = telemetryTrace ? getComputedStyle(telemetryTrace) : null;
    const bodyText = document.querySelector(".dashboard-home")?.textContent ?? "";
    return {
      commandShellDisplay: commandShellStyle?.display ?? "",
      flowNodes: document.querySelectorAll(".overview-flow-node").length,
      scoreDials: document.querySelectorAll(".overview-score-dial").length,
      runFocusCards: document.querySelectorAll(".overview-run-focus").length,
      opsSignals: document.querySelectorAll(".overview-ops-signal").length,
      decisionMetrics: document.querySelectorAll(".overview-decision-metric").length,
      decisionMetricRails,
      decisionTiles,
      decisionIcons: document.querySelectorAll(".overview-decision-icon").length,
      telemetryTraces: document.querySelectorAll(".overview-telemetry-trace").length,
      telemetryBars: document.querySelectorAll(".overview-telemetry-bar").length,
      telemetryRails,
      resourceChips: document.querySelectorAll(".overview-resource-chips span").length,
      stateStripItems: document.querySelectorAll(".overview-state-strip span").length,
      stateStripText,
      runArtifactRails: document.querySelectorAll(".overview-run-artifacts i b").length,
      runStates: document.querySelectorAll(".overview-run-state .badge").length,
      panelHeights,
      recentRows,
      recentCards: document.querySelectorAll(".overview-recent-card").length,
      nowPanels: document.querySelectorAll(".overview-ops-board").length,
      livePanels: document.querySelectorAll(".overview-rank-console").length,
      loopPanels: document.querySelectorAll(".overview-loop-panel").length,
      workflowRows: document.querySelectorAll(".overview-evidence-row").length,
      miniCharts: document.querySelectorAll(".overview-mini-chart").length,
      chartMatrix: document.querySelectorAll(".overview-chart-matrix").length,
      legacyActivityMatrix: document.querySelectorAll(".overview-activity-matrix").length,
      oldTimelinePanels: document.querySelectorAll(
        ".overview-timeline-panel, .overview-sparkline, .overview-timeline-labels"
      ).length,
      bodyText,
      nowPanelHeight: nowPanel ? Math.round(nowPanel.getBoundingClientRect().height) : 0,
      livePanelHeight: livePanel ? Math.round(livePanel.getBoundingClientRect().height) : 0,
      recentCardHeight: recentCard ? Math.round(recentCard.getBoundingClientRect().height) : 0,
      nowPanelDisplay: nowPanelStyle?.display ?? "",
      livePanelDisplay: livePanelStyle?.display ?? "",
      recentCardDisplay: recentCardStyle?.display ?? "",
      opsSignalTransition: opsSignalStyle?.transitionDuration ?? "",
      decisionMetricTransition: decisionMetricStyle?.transitionDuration ?? "",
      scoreDialTransition: scoreDialStyle?.transitionDuration ?? "",
      telemetryDisplay: telemetryStyle?.display ?? "",
      opsBoardRect: commandShell?.getBoundingClientRect() ?? null,
      telemetryRect: telemetryTrace?.getBoundingClientRect() ?? null
    };
  });
  if (state.legacyActivityMatrix !== 0) {
    throw new Error(`${scope}: overview should not render the old activity matrix`);
  }
  if (state.miniCharts !== 0 || state.chartMatrix !== 0) {
    throw new Error(
      `${scope}: overview should not expose the old low-value mini chart wall ${JSON.stringify({
        miniCharts: state.miniCharts,
        chartMatrix: state.chartMatrix
      })}`
    );
  }
  if (
    state.commandShellDisplay !== "flex" ||
    state.nowPanelDisplay !== "flex" ||
    state.livePanelDisplay !== "flex" ||
    state.recentCardDisplay !== "block"
  ) {
    throw new Error(
      `${scope}: overview should use a value-first operations desk ${JSON.stringify({
        commandShellDisplay: state.commandShellDisplay,
        nowPanelDisplay: state.nowPanelDisplay,
        livePanelDisplay: state.livePanelDisplay,
        recentCardDisplay: state.recentCardDisplay
      })}`
    );
  }
  if (
    state.flowNodes !== 4 ||
    state.scoreDials !== 1 ||
    state.runFocusCards !== 1 ||
    state.opsSignals !== 1 ||
    state.decisionMetrics !== 4 ||
    state.decisionIcons !== 4 ||
    state.telemetryTraces !== 1 ||
    state.telemetryBars !== 4 ||
    state.resourceChips !== 4
  ) {
    throw new Error(
      `${scope}: overview should expose score focus, flow nodes, one next action, telemetry, and decision tiles ${JSON.stringify({
        flowNodes: state.flowNodes,
        scoreDials: state.scoreDials,
        runFocusCards: state.runFocusCards,
        opsSignals: state.opsSignals,
        decisionMetrics: state.decisionMetrics,
        decisionIcons: state.decisionIcons,
        telemetryTraces: state.telemetryTraces,
        telemetryBars: state.telemetryBars,
        resourceChips: state.resourceChips
      })}`
    );
  }
  if (
    state.nowPanels !== 1 ||
    state.livePanels !== 1 ||
    state.loopPanels !== 0 ||
    state.recentCards !== 1 ||
    state.workflowRows !== 0
  ) {
    throw new Error(
      `${scope}: overview should keep one mission board, score cluster, and recent panel without the old duplicate loop row ${JSON.stringify({
        nowPanels: state.nowPanels,
        livePanels: state.livePanels,
        loopPanels: state.loopPanels,
        recentCards: state.recentCards,
        workflowRows: state.workflowRows
      })}`
    );
  }
  if (
    state.nowPanelHeight < 180 ||
    state.livePanelHeight < 120 ||
    state.recentCardHeight < 120
  ) {
    throw new Error(
      `${scope}: overview control panels collapsed ${JSON.stringify({
        nowPanelHeight: state.nowPanelHeight,
        livePanelHeight: state.livePanelHeight,
        recentCardHeight: state.recentCardHeight
      })}`
    );
  }
  if (!state.opsSignalTransition || state.opsSignalTransition === "0s") {
    throw new Error(`${scope}: overview ops signal is missing interaction transition`);
  }
  if (!state.decisionMetricTransition || state.decisionMetricTransition === "0s") {
    throw new Error(`${scope}: overview decision metrics are missing interaction transition`);
  }
  if (!state.scoreDialTransition || state.scoreDialTransition === "0s") {
    throw new Error(`${scope}: overview score dial is missing interaction transition`);
  }
  if (state.decisionMetricRails.length !== 4 || state.decisionMetricRails.some((width) => width < 0)) {
    throw new Error(`${scope}: overview decision metrics should expose progress rails`);
  }
  if (
    state.telemetryDisplay !== "flex" ||
    state.telemetryRails.length !== 4 ||
    !state.telemetryRails.some((width) => width > 0)
  ) {
    throw new Error(
      `${scope}: overview telemetry trace is missing live resource rails ${JSON.stringify({
        telemetryDisplay: state.telemetryDisplay,
        telemetryRails: state.telemetryRails
      })}`
    );
  }
  if (
    state.opsBoardRect &&
    state.telemetryRect &&
    state.telemetryRect.bottom > state.opsBoardRect.bottom - 8
  ) {
    throw new Error(
      `${scope}: overview telemetry trace is visually clipped ${JSON.stringify({
        opsBoard: formatRect(state.opsBoardRect),
        telemetry: formatRect(state.telemetryRect)
      })}`
    );
  }
  if (state.oldTimelinePanels > 0) {
    throw new Error(`${scope}: old oversized overview timeline markup is still present`);
  }
  if (
    state.stateStripItems !== 4 ||
    /只保留系统运行态、数据规模和近期写入节奏|精细指标进入排行榜与对比页|首页只保留|可以看排行|可以进入排行|查看排行榜|等待报告进入排行|主指标 F1 可排行|从样本到排行|rankable|F1 ready|先处理阻塞|补齐评估闭环|队列正在推进/.test(
      state.bodyText
    )
  ) {
    throw new Error(
      `${scope}: overview should use numeric state strip instead of slogan hero ${JSON.stringify({
        items: state.stateStripItems,
        text: state.stateStripText
      })}`
    );
  }
  if (/\b(precision|recall|iou|miou)\b/i.test(state.bodyText)) {
    throw new Error(`${scope}: overview exposes fine-grained eval metric text`);
  }
  if (
    /Notes|Tasks|Label footprint|样本\/label|模型分布|Job 日历|Scheduler 资源|Benchmark 任务|Run 日历/.test(
      state.bodyText
    )
  ) {
    throw new Error(`${scope}: overview exposes low-value diagnostic panels`);
  }
  if (state.recentRows.some((row) => row.height > 76)) {
    throw new Error(
      `${scope}: recent run rows are stretched ${state.recentRows.map((row) => row.height).join(",")}`
    );
  }
  if (state.recentRows.some((row) => /\b(P@|R@|precision|recall|iou|miou)\b/i.test(row.text))) {
    throw new Error(`${scope}: recent run stream exposes fine metrics`);
  }
  if (state.runArtifactRails !== state.recentRows.length || state.runStates !== state.recentRows.length) {
    throw new Error(
      `${scope}: recent run stream should expose artifact rails and status capsules ${JSON.stringify({
        rows: state.recentRows.length,
        rails: state.runArtifactRails,
        states: state.runStates
      })}`
    );
  }
  if (state.panelHeights.some((height) => height <= 0)) {
    throw new Error(`${scope}: overview panels are not visible ${state.panelHeights.join(",")}`);
  }
  if (state.decisionTiles.length !== 4) {
    throw new Error(
      `${scope}: overview decision rail is missing ${JSON.stringify({
        decisionTiles: state.decisionTiles.length,
        rails: state.decisionMetricRails
      })}`
    );
  }
  if (!state.decisionMetricRails.some((width) => width > 0)) {
    throw new Error(`${scope}: overview decision rail does not show live progress`);
  }
  for (const [index, rect] of state.decisionTiles.entries()) {
    if (rect.width <= 0 || rect.height <= 0) {
      throw new Error(`${scope}: decision tile ${index} is not visible ${JSON.stringify(rect)}`);
    }
    if (!scope.startsWith("narrow") && rect.width < 160) {
      throw new Error(`${scope}: decision tile ${index} is too compressed ${JSON.stringify(rect)}`);
    }
  }
  await assertOverviewPointerField(page, scope);
}

async function assertOverviewPointerField(page, scope) {
  const before = await page.locator(".overview-home-v17").evaluate((node) => ({
    x: node.style.getPropertyValue("--overview-pointer-x"),
    y: node.style.getPropertyValue("--overview-pointer-y")
  }));
  await page.locator(".overview-home-v17").evaluate((node) => {
    const rect = node.getBoundingClientRect();
    node.dispatchEvent(
      new PointerEvent("pointermove", {
        bubbles: true,
        clientX: rect.left + rect.width * 0.23,
        clientY: rect.top + rect.height * 0.67
      })
    );
  });
  const after = await page.locator(".overview-home-v17").evaluate((node) => ({
    x: node.style.getPropertyValue("--overview-pointer-x"),
    y: node.style.getPropertyValue("--overview-pointer-y")
  }));
  if (!after.x.endsWith("%") || !after.y.endsWith("%") || (after.x === before.x && after.y === before.y)) {
    throw new Error(
      `${scope}: overview pointer field did not update ${JSON.stringify({ before, after })}`
    );
  }
}

async function assertJobsDetailSidebar(page, scope) {
  const before = await page.evaluate(() => ({
    recentCards: document.querySelectorAll(".recent-run-card").length,
    sidebarCount: document.querySelectorAll(".job-detail-sidebar").length,
    mainWidth: Math.round(document.querySelector(".job-queue-main")?.getBoundingClientRect().width ?? 0),
    rowCount: document.querySelectorAll(".queue-stack tbody tr").length
  }));
  if (before.recentCards !== 0) {
    throw new Error(`${scope}: jobs page must not render recent result cards`);
  }
  if (before.sidebarCount !== 0) {
    throw new Error(`${scope}: jobs detail sidebar should be hidden before selecting a job`);
  }
  if (before.rowCount === 0) {
    return;
  }
  await page.locator(".queue-stack tbody tr").first().click();
  await page.waitForTimeout(220);
  const after = await page.evaluate(() => {
    const sidebar = document.querySelector(".job-detail-sidebar");
    const main = document.querySelector(".job-queue-main");
    const workbench = document.querySelector(".job-queue-workbench");
    const log = document.querySelector(".job-log-tail, .job-log-empty");
    const sidebarRect = sidebar?.getBoundingClientRect();
    const mainRect = main?.getBoundingClientRect();
    const logRect = log?.getBoundingClientRect();
    return {
      sidebarCount: document.querySelectorAll(".job-detail-sidebar").length,
      workbenchClass: workbench?.className ?? "",
      mainWidth: Math.round(mainRect?.width ?? 0),
      sidebarWidth: Math.round(sidebarRect?.width ?? 0),
      sidebarLeft: Math.round(sidebarRect?.left ?? 0),
      mainRight: Math.round(mainRect?.right ?? 0),
      logHeight: Math.round(logRect?.height ?? 0)
    };
  });
  if (after.sidebarCount !== 1 || !after.workbenchClass.includes("detail-open")) {
    throw new Error(`${scope}: selecting a job should open the right detail sidebar ${JSON.stringify(after)}`);
  }
  if (!scope.startsWith("narrow") && after.sidebarWidth < 320) {
    throw new Error(`${scope}: jobs detail sidebar is too narrow ${JSON.stringify(after)}`);
  }
  if (!scope.startsWith("narrow") && after.mainWidth >= before.mainWidth - 24) {
    throw new Error(`${scope}: jobs main panel did not shrink for the detail sidebar ${JSON.stringify({ before, after })}`);
  }
  if (!scope.startsWith("narrow") && after.sidebarLeft < after.mainRight - 2) {
    throw new Error(`${scope}: jobs detail sidebar is not placed to the right ${JSON.stringify(after)}`);
  }
  if (after.logHeight < 32) {
    throw new Error(`${scope}: jobs detail sidebar log area is not visible ${JSON.stringify(after)}`);
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

async function assertPagersFit(page, scope) {
  const pagers = await page.locator(".rank-board-pager").evaluateAll((nodes) =>
    nodes
      .filter((node) => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((node, index) => {
        const style = getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        const label = node.querySelector("span");
        const actionGroup = node.querySelector("div");
        const labelRect = label?.getBoundingClientRect();
        const actionRect = actionGroup?.getBoundingClientRect();
        return {
          index,
          rect,
          labelRect,
          actionRect,
          scrollWidth: node.scrollWidth,
          clientWidth: node.clientWidth,
          scrollHeight: node.scrollHeight,
          clientHeight: node.clientHeight,
          overflowX: style.overflowX,
          overflowY: style.overflowY,
          flexWrap: style.flexWrap
        };
      })
  );
  for (const pager of pagers) {
    if (pager.scrollWidth > pager.clientWidth + 2 && !allowsScroll(pager.overflowX)) {
      throw new Error(
        `${scope}: pager ${pager.index} clips horizontally ${JSON.stringify({
          scrollWidth: pager.scrollWidth,
          clientWidth: pager.clientWidth,
          overflowX: pager.overflowX,
          rect: formatRect(pager.rect)
        })}`
      );
    }
    if (pager.scrollHeight > pager.clientHeight + 2 && !allowsScroll(pager.overflowY)) {
      throw new Error(
        `${scope}: pager ${pager.index} clips vertically ${JSON.stringify({
          scrollHeight: pager.scrollHeight,
          clientHeight: pager.clientHeight,
          overflowY: pager.overflowY,
          rect: formatRect(pager.rect)
        })}`
      );
    }
    for (const [name, rect] of [
      ["label", pager.labelRect],
      ["actions", pager.actionRect]
    ]) {
      if (!rect) {
        continue;
      }
      if (
        (rect.left < pager.rect.left - 2 || rect.right > pager.rect.right + 2) &&
        !allowsScroll(pager.overflowX)
      ) {
        throw new Error(
          `${scope}: pager ${pager.index} ${name} leaks outside pager ${JSON.stringify({
            child: formatRect(rect),
            pager: formatRect(pager.rect)
          })}`
        );
      }
    }
  }
}

async function assertTableActionCellsFit(page, scope) {
  const actionCells = await page.locator(".table-shell td .row-actions").evaluateAll((nodes) =>
    nodes
      .filter((node) => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      })
      .map((node, index) => {
        const cell = node.closest("td");
        const shell = node.closest(".table-shell");
        const rect = node.getBoundingClientRect();
        const cellRect = cell?.getBoundingClientRect();
        const shellRect = shell?.getBoundingClientRect();
        const shellStyle = shell ? getComputedStyle(shell) : null;
        const style = getComputedStyle(node);
        return {
          index,
          flexWrap: style.flexWrap,
          scrollWidth: node.scrollWidth,
          clientWidth: node.clientWidth,
          rect,
          cellRect,
          shellRect,
          shellScrollWidth: shell?.scrollWidth ?? 0,
          shellClientWidth: shell?.clientWidth ?? 0,
          shellOverflowX: shellStyle?.overflowX ?? ""
        };
      })
  );
  for (const cell of actionCells) {
    if (cell.flexWrap !== "wrap") {
      throw new Error(`${scope}: table action cell ${cell.index} must wrap actions`);
    }
    if (cell.scrollWidth > cell.clientWidth + 2) {
      throw new Error(
        `${scope}: table action cell ${cell.index} action group overflows ${JSON.stringify({
          scrollWidth: cell.scrollWidth,
          clientWidth: cell.clientWidth,
          rect: formatRect(cell.rect),
          cell: formatRect(cell.cellRect)
        })}`
      );
    }
    const shellHasHorizontalScroll = cell.shellScrollWidth > cell.shellClientWidth + 2;
    if (
      cell.shellRect &&
      !shellHasHorizontalScroll &&
      !allowsScroll(cell.shellOverflowX) &&
      cell.rect.right > cell.shellRect.right + 2
    ) {
      throw new Error(
        `${scope}: table action cell ${cell.index} leaks outside table shell ${JSON.stringify({
          rect: formatRect(cell.rect),
          shell: formatRect(cell.shellRect)
        })}`
      );
    }
  }
}

async function assertFormControlsFit(page, scope) {
  const controls = await page
    .locator(
      [
        ".job-form input",
        ".job-form .select-popover-trigger",
        ".job-form textarea",
        ".manifest-job-form input",
        ".manifest-job-form .select-popover-trigger",
        ".manifest-job-form textarea",
        ".service-form input",
        ".service-form .select-popover-trigger",
        ".service-form textarea",
        ".import-form input",
        ".import-form .select-popover-trigger",
        ".import-form textarea",
        ".benchmark-form input",
        ".benchmark-form .select-popover-trigger",
        ".benchmark-form textarea",
        ".comparison-controls input",
        ".comparison-controls .select-popover-trigger",
        ".comparison-controls textarea",
        ".advanced-filter-controls input",
        ".advanced-filter-controls .select-popover-trigger",
        ".advanced-filter-controls textarea"
      ].join(", ")
    )
    .evaluateAll((nodes) =>
      nodes
        .filter((node) => {
          const rect = node.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        })
        .map((node, index) => {
          const container =
            node.closest(
              "label, .advanced-filter-search-control, .advanced-filter-number-control, .advanced-filter-text-control, .compact-select, .filter-select"
            ) ?? node.parentElement;
          const rect = node.getBoundingClientRect();
          const containerRect = container?.getBoundingClientRect();
          const style = getComputedStyle(node);
          const containerStyle = container ? getComputedStyle(container) : null;
          return {
            index,
            tag: node.tagName,
            type: node instanceof HTMLInputElement ? node.type : "",
            className: node instanceof HTMLElement ? node.className : "",
            rect,
            containerRect,
            width: style.width,
            minWidth: style.minWidth,
            containerOverflowX: containerStyle?.overflowX ?? ""
          };
        })
    );
  for (const control of controls) {
    if (!control.containerRect) {
      continue;
    }
    if (control.rect.width > control.containerRect.width + 3 && !allowsScroll(control.containerOverflowX)) {
      throw new Error(
        `${scope}: form control ${control.index} is wider than its field ${JSON.stringify({
          tag: control.tag,
          type: control.type,
          width: control.width,
          minWidth: control.minWidth,
          rect: formatRect(control.rect),
          container: formatRect(control.containerRect)
        })}`
      );
    }
    if (
      (control.rect.left < control.containerRect.left - 2 ||
        control.rect.right > control.containerRect.right + 2) &&
      !allowsScroll(control.containerOverflowX)
    ) {
      throw new Error(
        `${scope}: form control ${control.index} leaks outside its field ${JSON.stringify({
          tag: control.tag,
          type: control.type,
          rect: formatRect(control.rect),
          container: formatRect(control.containerRect)
        })}`
      );
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

async function assertAdvancedFilterClear(page, scope) {
  const filter = page.locator(".advanced-filter-bar").first();
  const searchInput = await openAdvancedFilter(filter);
  await searchInput.fill("layout-smoke-token-reset");
  const dirtyState = await filter.evaluate((node) => {
    const applyButton = node.querySelector(".advanced-filter-apply");
    return {
      dirty: node.classList.contains("dirty"),
      keyshortcuts: applyButton?.getAttribute("aria-keyshortcuts") ?? "",
      disabled: applyButton instanceof HTMLButtonElement ? applyButton.disabled : null
    };
  });
  if (!dirtyState.dirty || dirtyState.keyshortcuts !== "Enter" || dirtyState.disabled) {
    throw new Error(`${scope}: advanced filter draft state is not visible ${JSON.stringify(dirtyState)}`);
  }
  await searchInput.press("Enter");
  await filter.locator(".advanced-filter-token").first().waitFor({ timeout: 5_000 });
  await filter.locator(".advanced-filter-token").first().click();
  const tokenState = await filter.evaluate((node) => {
    const input = node.querySelector(".advanced-filter-controls input");
    const summary = node.querySelector(".advanced-filter-head div span");
    return {
      inputValue: input instanceof HTMLInputElement ? input.value : "",
      summary: summary?.textContent?.trim() ?? "",
      tokenVisible: Boolean(node.querySelector(".advanced-filter-token"))
    };
  });
  if (tokenState.inputValue !== "" || tokenState.tokenVisible || tokenState.summary !== "未设条件") {
    throw new Error(`${scope}: advanced filter token clear did not reset filter ${JSON.stringify(tokenState)}`);
  }
  await page.waitForTimeout(120);
  await assertAdvancedFilterPreservesDraftAfterTokenClear(filter, scope);
  const resetSearchInput = await openAdvancedFilter(filter);
  await resetSearchInput.fill("layout-smoke-filter-reset");
  await filter.locator(".advanced-filter-apply").click({ force: true });
  await filter.locator(".advanced-filter-clear").waitFor({ timeout: 5_000 });
  await filter.locator(".advanced-filter-clear").click();
  const state = await filter.evaluate((node) => {
    const input = node.querySelector(".advanced-filter-controls input");
    const summary = node.querySelector(".advanced-filter-head div span");
    return {
      inputValue: input instanceof HTMLInputElement ? input.value : "",
      summary: summary?.textContent?.trim() ?? "",
      clearVisible: Boolean(node.querySelector(".advanced-filter-clear"))
    };
  });
  if (state.inputValue !== "" || state.clearVisible || state.summary !== "未设条件") {
    throw new Error(`${scope}: advanced filter clear did not reset filters ${JSON.stringify(state)}`);
  }
}

async function assertAdvancedFilterPreservesDraftAfterTokenClear(filter, scope) {
  await openAdvancedFilter(filter);
  const select = filter.locator(".advanced-filter-controls .select-popover-control").first();
  if ((await select.count()) === 0) {
    return;
  }
  await select.locator(".select-popover-trigger").click();
  const targetOption = select
    .locator(".select-popover-option")
    .filter({ hasNotText: "全部" })
    .first();
  if ((await targetOption.count()) === 0) {
    return;
  }
  const targetValue = await targetOption.evaluate((node) => {
    if (!(node instanceof HTMLElement)) {
      return "";
    }
    return node.dataset.selectValue ?? "";
  });
  if (!targetValue) {
    return;
  }
  const selectLabel = await select.evaluate((node) => {
    return node.querySelector(".select-popover-label")?.textContent?.trim() ?? "";
  });
  await targetOption.click();
  await filter.locator(".advanced-filter-apply").click({ force: true });
  const appliedToken = selectLabel
    ? filter.locator(".advanced-filter-token", { hasText: `${selectLabel}:` }).first()
    : filter.locator(".advanced-filter-token").first();
  await appliedToken.waitFor({ timeout: 5_000 });
  const searchInput = filter.locator('.advanced-filter-controls input[type="search"]').first();
  if ((await searchInput.count()) === 0) {
    return;
  }
  await searchInput.fill("layout-smoke-preserved-draft");
  await filter.evaluate(
    (node, expectedValue) =>
      new Promise((resolve, reject) => {
        let attempts = 0;
        const tick = () => {
          const input = node.querySelector('.advanced-filter-controls input[type="search"]');
          if (input instanceof HTMLInputElement && input.value === expectedValue) {
            resolve(true);
            return;
          }
          attempts += 1;
          if (attempts > 20) {
            reject(new Error("search draft value was not committed before token clear"));
            return;
          }
          window.setTimeout(tick, 25);
        };
        tick();
      }),
    "layout-smoke-preserved-draft"
  );
  await appliedToken.click();
  const state = await filter.evaluate((node) => {
    const input = node.querySelector('.advanced-filter-controls input[type="search"]');
    return {
      inputValue: input instanceof HTMLInputElement ? input.value : "",
      dirty: node.classList.contains("dirty")
    };
  });
  if (state.inputValue !== "layout-smoke-preserved-draft" || !state.dirty) {
    throw new Error(
      `${scope}: clearing an applied token should preserve unrelated draft filters ${JSON.stringify(state)}`
    );
  }
}

async function assertAdvancedFilterKeyboardFlow(page, scope) {
  const filter = page.locator(".advanced-filter-bar").first();
  if ((await filter.count()) === 0) {
    return;
  }
  const head = filter.locator(".advanced-filter-head");
  await head.focus();
  await head.click();
  const popover = filter.locator(".advanced-filter-popover").first();
  await popover.waitFor({ timeout: 5_000 });
  await page.waitForTimeout(50);
  const openState = await filter.evaluate((node) => {
    const popoverNode = node.querySelector(".advanced-filter-popover");
    const controls = Array.from(
      node.querySelectorAll(
        ".advanced-filter-controls input:not([disabled]), .advanced-filter-controls textarea:not([disabled]), .advanced-filter-controls button:not([disabled])"
      )
    );
    const active = document.activeElement;
    return {
      open: Boolean(popoverNode),
      focusInsidePopover: Boolean(popoverNode?.contains(active)),
      focusInsideControls: controls.some((control) => control === active),
      activeTag: active?.tagName ?? "",
      activeClass: active instanceof HTMLElement ? active.className : ""
    };
  });
  if (!openState.open || !openState.focusInsidePopover || !openState.focusInsideControls) {
    throw new Error(`${scope}: advanced filter should focus first filter control ${JSON.stringify(openState)}`);
  }

  for (let index = 0; index < 20; index += 1) {
    await page.keyboard.press("Tab");
    const focusEscaped = await filter.evaluate((node) => {
      const popoverNode = node.querySelector(".advanced-filter-popover");
      return Boolean(popoverNode) && !popoverNode.contains(document.activeElement);
    });
    if (focusEscaped) {
      throw new Error(`${scope}: advanced filter Tab focus escaped popover at step ${index + 1}`);
    }
  }

  await page.keyboard.press("Escape");
  await popover.waitFor({ state: "hidden", timeout: 5_000 });
  await filter.locator(".advanced-filter-head").evaluate((node) =>
    new Promise((resolve, reject) => {
      let attempts = 0;
      const tick = () => {
        if (document.activeElement === node) {
          resolve(true);
          return;
        }
        attempts += 1;
        if (attempts > 20) {
          reject(new Error("advanced filter trigger focus was not restored"));
          return;
        }
        window.setTimeout(tick, 25);
      };
      tick();
    })
  );
  const closedState = await filter.evaluate((node) => {
    const headNode = node.querySelector(".advanced-filter-head");
    return {
      popoverVisible: Boolean(node.querySelector(".advanced-filter-popover")),
      focusReturned: document.activeElement === headNode
    };
  });
  if (closedState.popoverVisible || !closedState.focusReturned) {
    throw new Error(
      `${scope}: advanced filter Escape should close and restore trigger focus ${JSON.stringify(closedState)}`
    );
  }
}

async function openAdvancedFilter(filter) {
  const input = filter.locator(".advanced-filter-controls input").first();
  if (await input.isVisible().catch(() => false)) {
    return input;
  }
  const head = filter.locator(".advanced-filter-head");
  await head.click();
  if (!(await input.isVisible().catch(() => false))) {
    await head.click();
  }
  await input.waitFor({ timeout: 5_000 });
  return input;
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

async function assertInspectorSampleList(page, scope) {
  const state = await page.evaluate(() => {
    const sidebar = document.querySelector(".inspector-sidebar");
    const list = sidebar?.querySelector(".sample-list");
    const rows = Array.from(list?.querySelectorAll(".sample-row") ?? []).map((row) => {
      const rect = row.getBoundingClientRect();
      return {
        height: Math.round(rect.height),
        width: Math.round(rect.width),
        text: row.textContent?.replace(/\s+/g, " ").trim() ?? ""
      };
    });
    const listRect = list?.getBoundingClientRect();
    const sidebarRect = sidebar?.getBoundingClientRect();
    const style = list ? getComputedStyle(list) : null;
    return {
      exists: Boolean(list),
      listRect,
      sidebarRect,
      display: style?.display ?? "",
      overflowX: style?.overflowX ?? "",
      overflowY: style?.overflowY ?? "",
      scrollWidth: list?.scrollWidth ?? 0,
      clientWidth: list?.clientWidth ?? 0,
      scrollHeight: list?.scrollHeight ?? 0,
      clientHeight: list?.clientHeight ?? 0,
      rows,
      empty: list?.classList.contains("empty") ?? false
    };
  });
  if (!state.exists || !state.listRect || !state.sidebarRect) {
    throw new Error(`${scope}: inspector sample list is missing`);
  }
  if (state.display !== "flex") {
    throw new Error(`${scope}: inspector sample list should stay a flex scroll pane`);
  }
  if (state.listRect.right > state.sidebarRect.right + 2) {
    throw new Error(
      `${scope}: inspector sample list overflows sidebar ${JSON.stringify({
        list: formatRect(state.listRect),
        sidebar: formatRect(state.sidebarRect)
      })}`
    );
  }
  if (state.scrollHeight > state.clientHeight + 2 && !allowsScroll(state.overflowY)) {
    throw new Error(
      `${scope}: inspector sample list needs vertical scroll but overflow-y=${state.overflowY}`
    );
  }
  if (state.scrollWidth > state.clientWidth + 2 && clipsOverflow(state.overflowX)) {
    throw new Error(
      `${scope}: inspector sample list clips horizontal content ${JSON.stringify({
        scrollWidth: state.scrollWidth,
        clientWidth: state.clientWidth,
        overflowX: state.overflowX
      })}`
    );
  }
  if (!state.empty && state.rows.length === 0) {
    throw new Error(`${scope}: inspector sample list has no rows and no empty state`);
  }
  if (state.rows.some((row) => row.height > 86)) {
    throw new Error(
      `${scope}: inspector sample rows are stretched ${state.rows
        .map((row) => row.height)
        .join(",")}`
    );
  }
}

async function assertRunInspectorCountStrip(page, scope) {
  const state = await page.evaluate(() => {
    const strip = document.querySelector(".viewer-side-panel .diagnostic-strip");
    const labelMetricCards = document.querySelectorAll(".viewer-side-panel .label-metric-card").length;
    const tableHeaders = Array.from(document.querySelectorAll(".viewer-side-panel th")).map(
      (node) => node.textContent?.trim() ?? ""
    );
    return {
      text: strip?.textContent?.replace(/\s+/g, " ").trim() ?? "",
      chipCount: strip?.querySelectorAll(":scope > span").length ?? 0,
      compact: strip?.classList.contains("compact-counts") ?? false,
      labelMetricCards,
      tableHeaders
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
  if (state.labelMetricCards !== 0) {
    throw new Error(`${scope}: run inspector should not render resident label metric cards`);
  }
  if (state.tableHeaders.some((header) => /^TP$|^FP$|^FN$|^P@|^R@|IoU|平均/.test(header))) {
    throw new Error(
      `${scope}: run inspector side panel exposes fine metric table headers ${state.tableHeaders.join(", ")}`
    );
  }
}

async function assertRunViewerLayerPreferencesPersist(page, scope) {
  const pagerState = await page.evaluate(() => {
    const pager = document.querySelector(".run-inspector-page .sample-pager");
    const nextButton = Array.from(pager?.querySelectorAll("button") ?? []).find((button) =>
      (button.textContent ?? "").includes("下一页")
    );
    return {
      hasPager: Boolean(pager),
      pagerText: pager?.textContent?.replace(/\s+/g, " ").trim() ?? "",
      nextDisabled: nextButton instanceof HTMLButtonElement ? nextButton.disabled : true
    };
  });
  if (!pagerState.hasPager || pagerState.nextDisabled) {
    return;
  }
  const predictionToggle = page
    .locator(".run-inspector-page .layer-toggle-strip .control-check")
    .filter({ hasText: "预测" })
    .first();
  await predictionToggle.waitFor({ timeout: 5_000 });
  if ((await viewerPredictionLayerState(page)).predictionToggleActive) {
    await predictionToggle.click();
  }
  await page.waitForFunction(() => {
    const label = Array.from(
      document.querySelectorAll(".run-inspector-page .layer-toggle-strip .control-check")
    ).find((node) => (node.textContent ?? "").includes("预测"));
    return Boolean(label) && !label.classList.contains("active");
  });

  const previousPagerText = pagerState.pagerText;
  await page.locator(".run-inspector-page .sample-pager button", { hasText: "下一页" }).click();
  await page.waitForFunction((oldText) => {
    const text =
      document
        .querySelector(".run-inspector-page .sample-pager")
        ?.textContent?.replace(/\s+/g, " ")
        .trim() ?? "";
    return text !== oldText;
  }, previousPagerText);
  await page
    .locator(".run-inspector-page .viewer-canvas-layout .image-stage")
    .first()
    .waitFor({
      timeout: 10_000
    });
  const afterNext = await viewerPredictionLayerState(page);
  if (afterNext.predictionToggleActive || afterNext.predictionOverlayCount !== 0) {
    throw new Error(
      `${scope}: prediction layer preference reset after sample pager next ${JSON.stringify(afterNext)}`
    );
  }

  await page.locator(".run-inspector-page .sample-pager button", { hasText: "上一页" }).click();
  await page.waitForFunction((oldText) => {
    const text =
      document
        .querySelector(".run-inspector-page .sample-pager")
        ?.textContent?.replace(/\s+/g, " ")
        .trim() ?? "";
    return text !== oldText;
  }, afterNext.pagerText);
  const afterPrevious = await viewerPredictionLayerState(page);
  if (afterPrevious.predictionToggleActive || afterPrevious.predictionOverlayCount !== 0) {
    throw new Error(
      `${scope}: prediction layer preference reset after sample pager previous ${JSON.stringify(
        afterPrevious
      )}`
    );
  }
  if (!(await viewerPredictionLayerState(page)).predictionToggleActive) {
    await predictionToggle.click();
  }
}

async function viewerPredictionLayerState(page) {
  return page.evaluate(() => {
    const predictionToggle = Array.from(
      document.querySelectorAll(".run-inspector-page .layer-toggle-strip .control-check")
    ).find((node) => (node.textContent ?? "").includes("预测"));
    return {
      pagerText:
        document
          .querySelector(".run-inspector-page .sample-pager")
          ?.textContent?.replace(/\s+/g, " ")
          .trim() ?? "",
      predictionToggleActive: predictionToggle?.classList.contains("active") ?? false,
      predictionOverlayCount: document.querySelectorAll(
        ".run-inspector-page .overlay-instance.pred"
      ).length
    };
  });
}

async function assertRunNoteTemplates(page, scope) {
  await page.locator(".run-config-panel summary").first().click();
  const state = await page.evaluate(() => {
    const bar = document.querySelector(".run-note-template-bar");
    const buttons = Array.from(bar?.querySelectorAll("button") ?? []);
    const appendPanel = document.querySelector(".run-note-append-panel");
    const appendTextarea = appendPanel?.querySelector("textarea");
    const appendAction = Array.from(appendPanel?.querySelectorAll("button") ?? []).find((button) =>
      (button.textContent ?? "").includes("追加线索")
    );
    const editor = document.querySelector(".run-note-editor");
    const editorTextarea = editor?.querySelector(":scope > textarea");
    const actionRect = appendAction?.getBoundingClientRect();
    const appendRect = appendPanel?.getBoundingClientRect();
    const editorStyle = editor ? getComputedStyle(editor) : null;
    return {
      hasBar: Boolean(bar),
      hasAppendPanel: Boolean(appendPanel),
      hasAppendTextarea: Boolean(appendTextarea),
      hasAppendAction: Boolean(appendAction),
      buttonCount: buttons.length,
      labels: buttons.map((button) => button.textContent?.trim() ?? ""),
      editorHeight: editor ? Math.round(editor.getBoundingClientRect().height) : 0,
      editorOverflowY: editorStyle?.overflowY ?? "",
      editorTextareaHeight: editorTextarea
        ? Math.round(editorTextarea.getBoundingClientRect().height)
        : 0,
      appendTextareaHeight: appendTextarea
        ? Math.round(appendTextarea.getBoundingClientRect().height)
        : 0,
      appendActionVisible: Boolean(
        actionRect &&
          appendRect &&
          actionRect.width > 0 &&
          actionRect.height > 0 &&
          actionRect.bottom <= appendRect.bottom + 2
      )
    };
  });
  if (!state.hasBar || !state.hasAppendPanel || !state.hasAppendTextarea || !state.hasAppendAction || state.buttonCount < 4) {
    throw new Error(`${scope}: run note template bar is missing templates ${JSON.stringify(state)}`);
  }
  if (
    state.editorHeight < 240 ||
    state.editorTextareaHeight < 100 ||
    state.appendTextareaHeight < 76 ||
    !state.appendActionVisible
  ) {
    throw new Error(`${scope}: run note editor collapsed or clipped ${JSON.stringify(state)}`);
  }
  if (!allowsScroll(state.editorOverflowY)) {
    throw new Error(`${scope}: run note editor must own overflow when content grows ${JSON.stringify(state)}`);
  }
  for (const label of ["复现", "Idea", "异常", "Next"]) {
    if (!state.labels.includes(label)) {
      throw new Error(`${scope}: run note template ${label} is missing ${JSON.stringify(state)}`);
    }
  }
  await page.locator(".run-config-panel summary").first().click();
}

async function assertInspectorCanvasPane(page, scope) {
  const state = await page.evaluate(() => {
    const pageNode = document.querySelector(".visual-inspector-page");
    const stage = document.querySelector(".viewer-panel .image-stage");
    const sidePanel = document.querySelector(".viewer-panel .viewer-side-panel");
    const pageStyle = pageNode ? getComputedStyle(pageNode) : null;
    const sideStyle = sidePanel ? getComputedStyle(sidePanel) : null;
    const stageRect = stage?.getBoundingClientRect();
    return {
      viewportHeight: window.innerHeight,
      page: pageNode
        ? {
            scrollHeight: pageNode.scrollHeight,
            clientHeight: pageNode.clientHeight,
            overflowY: pageStyle?.overflowY ?? ""
          }
        : null,
      stage: stageRect
        ? {
            width: Math.round(stageRect.width),
            height: Math.round(stageRect.height)
          }
        : null,
      side: sidePanel
        ? {
            scrollHeight: sidePanel.scrollHeight,
            clientHeight: sidePanel.clientHeight,
            overflowY: sideStyle?.overflowY ?? ""
          }
        : null
    };
  });
  if (!state.stage) {
    throw new Error(`${scope}: inspector image stage is missing`);
  }
  const minimumStageHeight = state.viewportHeight <= 460 ? 140 : 180;
  if (state.stage.height < minimumStageHeight) {
    throw new Error(
      `${scope}: inspector image stage collapsed ${JSON.stringify({
        stage: state.stage,
        viewportHeight: state.viewportHeight,
        minimumStageHeight
      })}`
    );
  }
  if (
    state.page &&
    state.page.scrollHeight > state.page.clientHeight + 2 &&
    !allowsScroll(state.page.overflowY)
  ) {
    throw new Error(`${scope}: inspector page clips stacked panes without scroll ${JSON.stringify(state.page)}`);
  }
  if (
    state.side &&
    state.side.scrollHeight > state.side.clientHeight + 2 &&
    !allowsScroll(state.side.overflowY)
  ) {
    throw new Error(`${scope}: inspector side panel clips controls without scroll ${JSON.stringify(state.side)}`);
  }
}

async function assertInspectorFilteredEmptyState(page, scope, routeName) {
  const pattern =
    routeName === "benchmark-inspector"
      ? "**/api/benchmarks/*/samples?**"
      : routeName === "run-inspector"
        ? "**/api/runs/*/samples?**"
        : "";
  if (!pattern) {
    return;
  }
  const filterHead = page.locator(".inspector-sidebar .advanced-filter-head").first();
  await filterHead.click();
  const selects = page.locator(".inspector-sidebar .advanced-filter-controls .select-popover-control");
  const selectCount = await selects.count();
  const select =
    routeName === "benchmark-inspector" && selectCount > 1 ? selects.nth(1) : selects.first();
  await select.waitFor({ timeout: 5_000 });
  await select.locator(".select-popover-trigger").click();
  const options = select.locator(".select-popover-option");
  const optionValues = await options.evaluateAll((nodes) =>
    nodes
      .map((option) => (option instanceof HTMLElement ? option.dataset.selectValue ?? "" : ""))
      .filter((value) => value && value !== "all")
  );
  if (optionValues.length === 0) {
    await filterHead.click();
    return;
  }
  const handler = async (networkRoute) => {
    const url = new URL(networkRoute.request().url());
    if (!url.searchParams.has("label") && !url.searchParams.has("error_filter")) {
      await networkRoute.continue();
      return;
    }
    await networkRoute.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        offset: 0,
        limit: 80,
        total: 0,
        labels: ["layout-smoke-empty"],
        samples: []
      })
    });
  };
  await page.route(pattern, handler);
  try {
    await options.filter({ hasNotText: "全部" }).first().click();
    await page.locator(".inspector-sidebar .advanced-filter-apply").click({ force: true });
    await page
      .locator(".viewer-panel .empty-panel", { hasText: "没有符合过滤条件的样本" })
      .waitFor({ timeout: 10_000 });
    const state = await page.evaluate(() => {
      const sidebar = document.querySelector(".inspector-sidebar");
      const filter = sidebar?.querySelector(".advanced-filter-bar");
      const sampleList = sidebar?.querySelector(".sample-list.empty");
      const viewerEmpty = document.querySelector(".viewer-panel .empty-panel");
      return {
        hasSidebar: Boolean(sidebar),
        hasFilter: Boolean(filter),
        sampleListText: sampleList?.textContent?.replace(/\s+/g, " ").trim() ?? "",
        viewerText: viewerEmpty?.textContent?.replace(/\s+/g, " ").trim() ?? ""
      };
    });
    if (!state.hasSidebar || !state.hasFilter) {
      throw new Error(`${scope}: filtered-empty state removed inspector controls`);
    }
    if (
      !state.sampleListText.includes("没有符合过滤条件") ||
      !state.viewerText.includes("没有符合过滤条件")
    ) {
      throw new Error(`${scope}: filtered-empty state is not recoverable ${JSON.stringify(state)}`);
    }
  } finally {
    await page.unroute(pattern, handler);
  }
}

async function assertCompositeReportWorkbench(page, scope) {
  const collapsedState = await page.evaluate(() => {
    const shell = document.querySelector(".composite-report-shell");
    const dock = document.querySelector(".composite-composer-dock");
    const drawer = document.querySelector(".composite-sidebar-drawer");
    const stageRegion = document.querySelector(".composite-stage-region");
    const search = document.querySelector(".image-navigator-search input");
    const canvas = document.querySelector(".image-stage");
    const overlay = document.querySelector(".overlay-svg");
    const focus = document.querySelector(".composite-report-focus");
    const inspector = document.querySelector(".composite-inspector-panel");
    const shellRect = shell?.getBoundingClientRect();
    const dockRect = dock?.getBoundingClientRect();
    const stageRect = stageRegion?.getBoundingClientRect();
    return {
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      sidebarOpen: shell?.classList.contains("sidebar-open") ?? false,
      hasDrawer: Boolean(drawer),
      shellWidth: shellRect?.width ?? 0,
      shellHeight: shellRect?.height ?? 0,
      dockWidth: dockRect?.width ?? 0,
      dockHeight: dockRect?.height ?? 0,
      stageWidth: stageRect?.width ?? 0,
      stageHeight: stageRect?.height ?? 0,
      nearbyItems: document.querySelectorAll(".image-nearby-card").length,
      hasNearbyRail: Boolean(document.querySelector(".image-nearby-rail")),
      hasIndexMeter: Boolean(document.querySelector(".image-index-meter")),
      hasSearch: Boolean(search),
      hasCanvas: Boolean(canvas),
      hasOverlay: Boolean(overlay),
      hasFocus: Boolean(focus),
      hasInspector: Boolean(inspector)
    };
  });
  if (collapsedState.sidebarOpen || collapsedState.hasDrawer) {
    throw new Error(`${scope}: composite report composer should default collapsed ${JSON.stringify(collapsedState)}`);
  }
  if (collapsedState.viewportWidth > 940 && collapsedState.dockWidth > 70) {
    throw new Error(`${scope}: collapsed composer dock is too wide ${JSON.stringify(collapsedState)}`);
  }
  if (collapsedState.viewportWidth <= 940 && collapsedState.dockHeight > 72) {
    throw new Error(`${scope}: compact composer dock is too tall ${JSON.stringify(collapsedState)}`);
  }
  if (!collapsedState.hasIndexMeter) {
    return;
  }
  if (collapsedState.hasNearbyRail || collapsedState.nearbyItems > 0 || !collapsedState.hasIndexMeter || !collapsedState.hasSearch) {
    throw new Error(`${scope}: composite report should use compact image controls without a bottom nearby image list ${JSON.stringify(collapsedState)}`);
  }
  if (!collapsedState.hasCanvas || !collapsedState.hasOverlay || !collapsedState.hasFocus || !collapsedState.hasInspector) {
    throw new Error(`${scope}: composite report visual stage did not render ${JSON.stringify(collapsedState)}`);
  }
  await assertCompositeObjectInteraction(page, scope);
  await assertCompositeInteractionPalette(page, scope);
  await page.locator(".composite-composer-dock .icon-button").first().click();
  await page.locator(".composite-sidebar-drawer").waitFor({ timeout: 3_000 });
  const drawerState = await page.evaluate(() => {
    const resultPool = document.querySelector(".report-run-pool");
    const layerPlan = document.querySelector(".report-layer-plan");
    const drawer = document.querySelector(".composite-sidebar-drawer");
    const drawerRect = drawer?.getBoundingClientRect();
    return {
      drawerWidth: drawerRect?.width ?? 0,
      drawerHeight: drawerRect?.height ?? 0,
      resultCards: resultPool?.querySelectorAll(".report-run-card").length ?? 0,
      layerRows: layerPlan?.querySelectorAll(".report-layer-row").length ?? 0
    };
  });
  if (drawerState.resultCards < 1 || drawerState.layerRows < 2 || drawerState.drawerWidth < 260 || drawerState.drawerHeight < 120) {
    throw new Error(`${scope}: composer drawer did not expose result pool and layered report plan ${JSON.stringify(drawerState)}`);
  }
  await page.locator(".composite-sidebar-head .icon-button").first().click();
  await page.locator(".composite-sidebar-drawer").waitFor({ state: "hidden", timeout: 3_000 });
  await page.locator(".image-navigator-search input").fill("1");
  await page.locator(".image-jump-popover").waitFor({ timeout: 3000 });
  const searchState = await page.evaluate(() => ({
    popover: Boolean(document.querySelector(".image-jump-popover")),
    atlas: Boolean(document.querySelector(".image-jump-atlas")),
    activePreview: document.querySelector(".image-jump-active-preview")?.textContent?.trim() ?? "",
    mapBins: document.querySelectorAll(".image-map-bin").length,
    activeMapBins: document.querySelectorAll(".image-map-bin.active").length,
    results: document.querySelectorAll(".image-jump-result").length,
    nearbyItems: document.querySelectorAll(".image-nearby-card").length,
    hasNearbyRail: Boolean(document.querySelector(".image-nearby-rail"))
  }));
  if (
    !searchState.popover ||
    !searchState.atlas ||
    searchState.mapBins < 8 ||
    searchState.activeMapBins !== 1 ||
    !searchState.activePreview ||
    searchState.results < 1 ||
    searchState.results > 48
  ) {
    throw new Error(`${scope}: composite image search popover is not bounded ${JSON.stringify(searchState)}`);
  }
  await assertCompositeImageSearchPreview(page, scope);
  if (searchState.hasNearbyRail || searchState.nearbyItems > 0) {
    throw new Error(`${scope}: composite image search should not restore the removed nearby image list ${JSON.stringify(searchState)}`);
  }
  await page.locator(".image-navigator-search input").fill("");
  await page.keyboard.press("/");
  const shortcutSearchState = await page.evaluate(() => ({
    focusedSearch: document.activeElement?.matches(".image-navigator-search input") ?? false,
    popover: Boolean(document.querySelector(".image-jump-popover"))
  }));
  if (!shortcutSearchState.focusedSearch || !shortcutSearchState.popover) {
    throw new Error(`${scope}: composite / shortcut did not open image search ${JSON.stringify(shortcutSearchState)}`);
  }
  await page.keyboard.press("Escape");
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });
  const beforeKeyboardIndex = await compositeActiveImageIndex(page);
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction((previous) => {
    const label = document.querySelector(".image-index-meter")?.getAttribute("aria-label") ?? "";
    const match = label.match(/(\d+)\s*\/\s*(\d+)/);
    return Number(match?.[1] ?? 0) !== previous;
  }, beforeKeyboardIndex);
  const afterKeyboardIndex = await compositeActiveImageIndex(page);
  if (afterKeyboardIndex !== beforeKeyboardIndex + 1) {
    throw new Error(
      `${scope}: composite ArrowRight shortcut did not step images ${JSON.stringify({
        beforeKeyboardIndex,
        afterKeyboardIndex
      })}`
    );
  }
}

async function assertCompositeImageSearchPreview(page, scope) {
  const secondResult = page.locator(".image-jump-result").nth(1);
  if ((await secondResult.count()) === 0) {
    return;
  }
  const beforePreview = await page.evaluate(
    () => document.querySelector(".image-jump-active-preview")?.textContent?.trim() ?? ""
  );
  await secondResult.hover();
  await page.waitForFunction((previous) => {
    const preview = document.querySelector(".image-jump-active-preview")?.textContent?.trim() ?? "";
    return Boolean(preview) && preview !== previous;
  }, beforePreview);
  const previewState = await page.evaluate(() => {
    const preview = document.querySelector(".image-jump-active-preview");
    const active = document.querySelector(".image-jump-result.active");
    return {
      preview: preview?.textContent?.trim() ?? "",
      activeResult: active?.textContent?.trim() ?? "",
      previewTitle: preview?.getAttribute("title") ?? "",
      activeTitle: active?.getAttribute("title") ?? ""
    };
  });
  if (!previewState.preview || !previewState.activeResult || previewState.previewTitle !== previewState.activeTitle) {
    throw new Error(`${scope}: composite image search active preview did not follow hovered result ${JSON.stringify(previewState)}`);
  }
  const beforeWheelPreview = previewState.previewTitle;
  await page.locator(".image-jump-results").hover();
  await page.mouse.wheel(0, 120);
  await page.waitForFunction((previous) => {
    const preview = document.querySelector(".image-jump-active-preview");
    return Boolean(preview?.getAttribute("title")) && preview?.getAttribute("title") !== previous;
  }, beforeWheelPreview);
  const wheelState = await page.evaluate(() => {
    const preview = document.querySelector(".image-jump-active-preview");
    const active = document.querySelector(".image-jump-result.active");
    return {
      preview: preview?.textContent?.trim() ?? "",
      activeResult: active?.textContent?.trim() ?? "",
      previewTitle: preview?.getAttribute("title") ?? "",
      activeTitle: active?.getAttribute("title") ?? ""
    };
  });
  if (!wheelState.preview || !wheelState.activeResult || wheelState.previewTitle !== wheelState.activeTitle) {
    throw new Error(`${scope}: composite image search wheel navigation did not drive active preview ${JSON.stringify(wheelState)}`);
  }
}

async function assertCompositeInteractionPalette(page, scope) {
  const paletteState = await page.evaluate(() => ({
    toolbar: Boolean(document.querySelector(".composite-interaction-palette[role='toolbar']")),
    previous: Boolean(document.querySelector(".interaction-palette-tool[data-tool='previous']")),
    next: Boolean(document.querySelector(".interaction-palette-tool[data-tool='next']")),
    search: Boolean(document.querySelector(".interaction-palette-tool[data-tool='search']")),
    reset: Boolean(document.querySelector(".interaction-palette-tool[data-tool='reset']"))
  }));
  if (!paletteState.toolbar || !paletteState.previous || !paletteState.next || !paletteState.search || !paletteState.reset) {
    throw new Error(`${scope}: composite interaction palette is not an actionable toolbar ${JSON.stringify(paletteState)}`);
  }
  await page.locator(".interaction-palette-tool[data-tool='search']").click();
  const searchState = await page.evaluate(() => ({
    focusedSearch: document.activeElement?.matches(".image-navigator-search input") ?? false,
    popover: Boolean(document.querySelector(".image-jump-popover"))
  }));
  if (!searchState.focusedSearch || !searchState.popover) {
    throw new Error(`${scope}: composite interaction palette search button did not open search ${JSON.stringify(searchState)}`);
  }
  await page.keyboard.press("Escape");
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });
  const beforeIndex = await compositeActiveImageIndex(page);
  const nextButton = page.locator(".interaction-palette-tool[data-tool='next']");
  if (!(await nextButton.isDisabled())) {
    await nextButton.click();
    await page.waitForFunction((previous) => {
      const label = document.querySelector(".image-index-meter")?.getAttribute("aria-label") ?? "";
      const match = label.match(/(\d+)\s*\/\s*(\d+)/);
      return Number(match?.[1] ?? 0) === previous + 1;
    }, beforeIndex);
    const previousButton = page.locator(".interaction-palette-tool[data-tool='previous']");
    await previousButton.click();
    await page.waitForFunction((previous) => {
      const label = document.querySelector(".image-index-meter")?.getAttribute("aria-label") ?? "";
      const match = label.match(/(\d+)\s*\/\s*(\d+)/);
      return Number(match?.[1] ?? 0) === previous;
    }, beforeIndex);
  }
}

async function assertCompositeScrubPreview(page, scope) {
  const scrubTrack = page.locator(".image-scrub-track").first();
  await scrubTrack.scrollIntoViewIfNeeded();
  const rect = await scrubTrack.boundingBox();
  if (!rect || rect.width <= 8 || rect.height <= 4) {
    throw new Error(`${scope}: composite scrub track is not measurable ${JSON.stringify(rect)}`);
  }
  await page.mouse.move(Math.max(0, rect.x - 24), Math.max(0, rect.y - 24));
  await page.mouse.move(rect.x + rect.width * 0.72, rect.y + rect.height / 2);
  try {
    await page.waitForFunction(() => {
      const preview = document.querySelector(".image-scrub-preview");
      return Boolean(preview?.textContent?.trim());
    }, null, { timeout: 3_000 });
  } catch {
    const hitState = await page.evaluate(() => {
      const track = document.querySelector(".image-scrub-track");
      const rect = track?.getBoundingClientRect();
      if (!track || !rect) {
        return { hasTrack: false };
      }
      const hit = document.elementFromPoint(rect.left + rect.width * 0.72, rect.top + rect.height / 2);
      return {
        hasTrack: true,
        hitClass: hit?.className ?? "",
        hitTag: hit?.tagName ?? "",
        isTrackHit: hit === track || track.contains(hit)
      };
    });
    throw new Error(`${scope}: composite scrub hover preview did not render after hover ${JSON.stringify(hitState)}`);
  }
  const hoverState = await page.evaluate(() => {
    const preview = document.querySelector(".image-scrub-preview");
    return {
      visible: Boolean(preview),
      text: preview?.textContent?.trim() ?? "",
      rect: preview?.getBoundingClientRect()
    };
  });
  if (!hoverState.visible || !hoverState.text) {
    throw new Error(`${scope}: composite scrub hover preview did not render ${JSON.stringify(hoverState)}`);
  }
  await page.mouse.move(rect.x + rect.width + 40, rect.y + rect.height + 40);
  await page.waitForFunction(() => !document.querySelector(".image-scrub-preview"));
}

async function assertCompositeObjectInteraction(page, scope) {
  const objectChip = page.locator(".layer-object-chip").first();
  if ((await objectChip.count()) === 0) {
    return;
  }
  await objectChip.hover();
  const hoverState = await page.evaluate(() => ({
    activeChips: document.querySelectorAll(".layer-object-chip.active").length,
    activeOverlays: document.querySelectorAll(".overlay-instance.active").length
  }));
  if (hoverState.activeChips < 1 || hoverState.activeOverlays < 1) {
    throw new Error(`${scope}: composite object hover did not highlight canvas ${JSON.stringify(hoverState)}`);
  }
  await objectChip.click();
  const lockedState = await page.evaluate(() => ({
    activeChips: document.querySelectorAll(".layer-object-chip.active").length,
    lockedChips: document.querySelectorAll(".layer-object-chip.locked").length,
    activeOverlays: document.querySelectorAll(".overlay-instance.active").length
  }));
  if (lockedState.lockedChips < 1 || lockedState.activeOverlays < 1) {
    throw new Error(`${scope}: composite object click did not lock canvas highlight ${JSON.stringify(lockedState)}`);
  }
  await assertCompositePanAfterObjectLock(page, scope);
  await objectChip.hover();
  const activeOverlay = page.locator(".overlay-instance.active").first();
  if ((await activeOverlay.count()) === 0) {
    throw new Error(`${scope}: composite object hover did not expose an inspectable overlay`);
  }
  await activeOverlay.click({ button: "right", force: true });
  const menuState = await page.evaluate(() => ({
    activeOverlays: document.querySelectorAll(".overlay-instance.active").length,
    menu: Boolean(document.querySelector(".composite-object-context-menu"))
  }));
  if (menuState.activeOverlays < 1 || !menuState.menu) {
    throw new Error(`${scope}: composite object context menu did not open from canvas ${JSON.stringify(menuState)}`);
  }
  await page.keyboard.press("Escape");
}

async function assertCompositeObjectWheelNavigation(page, scope) {
  const stripIndex = await page.evaluate(() =>
    Array.from(document.querySelectorAll(".layer-object-strip")).findIndex(
      (strip) => strip.querySelectorAll(".layer-object-chip").length > 1
    )
  );
  if (stripIndex < 0) {
    return;
  }
  const strip = page.locator(".layer-object-strip").nth(stripIndex);
  await strip.locator(".layer-object-chip").first().click();
  const beforeWheel = await page.evaluate(() => ({
    lockedTitle: document.querySelector(".layer-object-chip.locked")?.getAttribute("title") ?? "",
    hud: document.querySelector(".composite-object-hud")?.textContent ?? ""
  }));
  await strip.hover();
  await page.mouse.wheel(0, 120);
  await page.waitForFunction((previous) => {
    const lockedTitle = document.querySelector(".layer-object-chip.locked")?.getAttribute("title") ?? "";
    return Boolean(lockedTitle) && lockedTitle !== previous;
  }, beforeWheel.lockedTitle);
  const wheelState = await page.evaluate(() => ({
    beforeTitle: document.querySelector(".layer-object-chip.locked")?.getAttribute("title") ?? "",
    lockedChips: document.querySelectorAll(".layer-object-chip.locked").length,
    activeOverlays: document.querySelectorAll(".overlay-instance.active").length,
    hud: document.querySelector(".composite-object-hud")?.textContent ?? ""
  }));
  if (
    wheelState.lockedChips !== 1 ||
    wheelState.activeOverlays < 1 ||
    !wheelState.hud.includes("Locked Object") ||
    wheelState.beforeTitle === beforeWheel.lockedTitle
  ) {
    throw new Error(`${scope}: composite object wheel navigation did not cycle locked object ${JSON.stringify({
      beforeWheel,
      wheelState
    })}`);
  }
}

async function assertCompositePanAfterObjectLock(page, scope) {
  const dragPoint = await page.evaluate(() => {
    const stage = document.querySelector(".composite-report-focus .image-stage");
    const activeOverlay = document.querySelector(".composite-report-focus .overlay-instance.active");
    const hud = document.querySelector(".composite-report-focus .canvas-hud");
    if (!stage) {
      return null;
    }
    const stageRect = stage.getBoundingClientRect();
    const activeRect = activeOverlay?.getBoundingClientRect();
    const hudRect = hud?.getBoundingClientRect();
    const candidates = [
      { x: stageRect.left + 26, y: stageRect.top + 26 },
      { x: stageRect.right - 52, y: stageRect.top + 30 },
      { x: stageRect.left + 32, y: stageRect.bottom - 58 },
      { x: stageRect.left + stageRect.width * 0.5, y: stageRect.top + stageRect.height * 0.5 }
    ];
    const outside = (point, rect) =>
      !rect ||
      point.x < rect.left - 4 ||
      point.x > rect.right + 4 ||
      point.y < rect.top - 4 ||
      point.y > rect.bottom + 4;
    return candidates.find((point) => outside(point, activeRect) && outside(point, hudRect)) ?? null;
  });
  if (!dragPoint) {
    throw new Error(`${scope}: composite canvas did not expose a safe blank point for pan test`);
  }
  await page.mouse.move(dragPoint.x, dragPoint.y);
  await page.mouse.wheel(0, -520);
  await page.waitForFunction(() => Boolean(document.querySelector(".composite-report-focus .canvas-reset-button")));
  const beforeTransform = await page.evaluate(() => {
    const layer = document.querySelector(".composite-report-focus .image-zoom-layer");
    return layer ? getComputedStyle(layer).transform : "";
  });
  await page.mouse.down();
  await page.mouse.move(dragPoint.x + 72, dragPoint.y + 38, { steps: 8 });
  await page.mouse.up();
  await page.waitForFunction((previous) => {
    const layer = document.querySelector(".composite-report-focus .image-zoom-layer");
    return layer && getComputedStyle(layer).transform !== previous;
  }, beforeTransform);
  const panState = await page.evaluate((previous) => {
    const layer = document.querySelector(".composite-report-focus .image-zoom-layer");
    const reset = document.querySelector(".composite-report-focus .canvas-reset-button");
    return {
      beforeTransform: previous,
      afterTransform: layer ? getComputedStyle(layer).transform : "",
      hasReset: Boolean(reset),
      activeOverlayCount: document.querySelectorAll(".composite-report-focus .overlay-instance.active").length
    };
  }, beforeTransform);
  if (panState.beforeTransform === panState.afterTransform || !panState.hasReset || panState.activeOverlayCount < 1) {
    throw new Error(`${scope}: composite canvas blank-area pan is blocked after object lock ${JSON.stringify(panState)}`);
  }
  await page.locator(".interaction-palette-tool[data-tool='reset']").click();
  await page.waitForFunction(() => !document.querySelector(".composite-report-focus .canvas-reset-button"));
  const resetState = await page.evaluate(() => ({
    hasReset: Boolean(document.querySelector(".composite-report-focus .canvas-reset-button")),
    activeOverlayCount: document.querySelectorAll(".composite-report-focus .overlay-instance.active").length
  }));
  if (resetState.hasReset || resetState.activeOverlayCount < 1) {
    throw new Error(`${scope}: composite interaction palette reset did not reset viewport while preserving object selection ${JSON.stringify(resetState)}`);
  }
  await page.mouse.move(dragPoint.x, dragPoint.y);
  await page.mouse.wheel(0, -520);
  await page.waitForFunction(() => Boolean(document.querySelector(".composite-report-focus .canvas-reset-button")));
  await page.mouse.dblclick(dragPoint.x, dragPoint.y);
  await page.waitForFunction(() => !document.querySelector(".composite-report-focus .canvas-reset-button"));
  const doubleClickResetState = await page.evaluate(() => ({
    hasReset: Boolean(document.querySelector(".composite-report-focus .canvas-reset-button")),
    activeOverlayCount: document.querySelectorAll(".composite-report-focus .overlay-instance.active").length,
    lockedChips: document.querySelectorAll(".layer-object-chip.locked").length
  }));
  if (
    doubleClickResetState.hasReset ||
    doubleClickResetState.activeOverlayCount < 1 ||
    doubleClickResetState.lockedChips < 1
  ) {
    throw new Error(`${scope}: composite canvas blank double click did not reset viewport while preserving object lock ${JSON.stringify(doubleClickResetState)}`);
  }
}

async function compositeActiveImageIndex(page) {
  return page.evaluate(() => {
    const label = document.querySelector(".image-index-meter")?.getAttribute("aria-label") ?? "";
    const match = label.match(/(\d+)\s*\/\s*(\d+)/);
    return Number(match?.[1] ?? 0);
  });
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
      activeInside: dialog.contains(document.activeElement),
      documentBodyOverflow: document.body.style.overflow,
      dialogTabIndex: dialog.getAttribute("tabindex") ?? "",
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
  if (!state.activeInside || state.dialogTabIndex !== "-1") {
    throw new Error(
      `${scope}: dialog focus is not managed ${JSON.stringify({
        activeInside: state.activeInside,
        tabIndex: state.dialogTabIndex
      })}`
    );
  }
  if (state.documentBodyOverflow !== "hidden") {
    throw new Error(`${scope}: dialog does not lock body scroll`);
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

async function assertRankFacetRail(page, scope) {
  const state = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll(".rank-facet-button"));
    const groups = Array.from(document.querySelectorAll(".rank-facet-group")).map((group) => ({
      title: group.querySelector(":scope > span")?.textContent?.trim() ?? "",
      buttonCount: group.querySelectorAll(".rank-facet-button").length
    }));
    return {
      groups,
      buttonCount: buttons.length,
      staticCountNodes: document.querySelectorAll(".rank-facet-group em strong").length
    };
  });
  const expectedGroups = [
    "Tasks",
    "Benchmarks",
    "Splits",
    "Status",
    "Labels",
    "Models",
    "Prompts",
    "Metrics"
  ];
  if (state.groups.length === 0) {
    return;
  }
  const actualGroups = state.groups.map((group) => group.title);
  const unknownGroups = actualGroups.filter((title) => !expectedGroups.includes(title));
  const emptyGroups = state.groups.filter((group) => group.buttonCount === 0).map((group) => group.title);
  if (unknownGroups.length > 0 || emptyGroups.length > 0) {
    throw new Error(
      `${scope}: rank facet rail should only render non-empty backend groups ${JSON.stringify({
        ...state,
        unknownGroups,
        emptyGroups
      })}`
    );
  }
  if (state.buttonCount === 0) {
    return;
  }
  if (state.staticCountNodes > 0) {
    throw new Error(`${scope}: rank facet rail regressed to static count chips ${JSON.stringify(state)}`);
  }
  await assertRankFacetExpansion(page, scope);
  await page.locator(".rank-facet-button").first().click();
  await page.locator(".rank-facet-button.active").first().waitFor({ timeout: 10_000 });
  await page.locator(".rank-board-table-card").first().waitFor({ timeout: 10_000 });
}

async function assertRankFacetExpansion(page, scope) {
  const toggle = page.locator(".rank-facet-toggle").first();
  if ((await toggle.count()) === 0) {
    return;
  }
  await toggle.click();
  const state = await page.evaluate(() => {
    const group = document.querySelector(".rank-facet-group.expanded");
    const chipPane = group?.querySelector(":scope > div");
    const style = chipPane ? getComputedStyle(chipPane) : null;
    const rect = group?.getBoundingClientRect();
    const paneRect = chipPane?.getBoundingClientRect();
    return {
      expanded: Boolean(group),
      rect,
      paneRect,
      flexWrap: style?.flexWrap ?? "",
      overflowX: style?.overflowX ?? "",
      overflowY: style?.overflowY ?? "",
      scrollWidth: chipPane?.scrollWidth ?? 0,
      clientWidth: chipPane?.clientWidth ?? 0,
      scrollHeight: chipPane?.scrollHeight ?? 0,
      clientHeight: chipPane?.clientHeight ?? 0
    };
  });
  if (!state.expanded || !state.rect || !state.paneRect) {
    throw new Error(`${scope}: rank facet toggle did not expand a facet group`);
  }
  if (state.flexWrap !== "wrap") {
    throw new Error(`${scope}: expanded rank facet chips should wrap, got ${state.flexWrap}`);
  }
  if (state.rect.height > 182) {
    throw new Error(
      `${scope}: expanded rank facet group is too tall ${JSON.stringify({
        rect: formatRect(state.rect),
        pane: formatRect(state.paneRect)
      })}`
    );
  }
  if (state.scrollHeight > state.clientHeight + 2 && !allowsScroll(state.overflowY)) {
    throw new Error(
      `${scope}: expanded rank facet needs vertical scroll but overflow-y=${state.overflowY}`
    );
  }
  if (state.scrollWidth > state.clientWidth + 2 && clipsOverflow(state.overflowX)) {
    throw new Error(
      `${scope}: expanded rank facet clips horizontal content ${JSON.stringify({
        scrollWidth: state.scrollWidth,
        clientWidth: state.clientWidth,
        overflowX: state.overflowX
      })}`
    );
  }
  await toggle.click();
}

async function assertRankHeaderSorting(page, scope) {
  const state = await page.evaluate(() => {
    const summary = document.querySelector(".rank-board-summary");
    const tableCard = document.querySelector(".rank-board-table-card");
    const sortHeaders = Array.from(document.querySelectorAll(".rank-sort-header"));
    const activeSortHeaders = sortHeaders.filter((header) => header.classList.contains("active"));
    const headerLabels = sortHeaders.map((header) => header.textContent ?? "");
    const activeCells = Array.from(document.querySelectorAll(".rank-sort-active-cell"));
    const primarySelect = document.querySelector(".rank-sort-section.primary select");
    const sortChips = Array.from(document.querySelectorAll(".rank-sort-chip, .rank-order-chip"));
    const leading = document.querySelector(".rank-board-leading");
    const topPanel = document.querySelector(".rank-top-panel");
    const spreadPanel = document.querySelector(".rank-spread-panel");
    const metricStrip = document.querySelector(".rank-metric-strip");
    const advancedSortControls = Array.from(document.querySelectorAll(".advanced-filter-bar [id]"))
      .map((node) => node.id)
      .filter((id) => id.includes("rank-sort"));
    const summaryRect = summary?.getBoundingClientRect();
    const tableRect = tableCard?.getBoundingClientRect();
    return {
      hasSummary: Boolean(summary),
      hasTableCard: Boolean(tableCard),
      hasPrimarySelect: Boolean(primarySelect),
      sortHeaderCount: sortHeaders.length,
      activeSortHeaderCount: activeSortHeaders.length,
      headerLabels,
      activeCellCount: activeCells.length,
      sortChipCount: sortChips.length,
      hasLeading: Boolean(leading),
      hasTopPanel: Boolean(topPanel),
      hasSpreadPanel: Boolean(spreadPanel),
      hasMetricStrip: Boolean(metricStrip),
      advancedSortControls,
      summaryHeight: summaryRect ? Math.round(summaryRect.height) : 0,
      summaryInsideTableCard: Boolean(
        summaryRect &&
          tableRect &&
          summaryRect.top >= tableRect.top &&
          summaryRect.bottom <= tableRect.bottom
      )
    };
  });
  if (
    !state.hasSummary ||
    !state.hasTableCard ||
    state.hasPrimarySelect ||
    state.sortChipCount !== 0 ||
    state.hasLeading ||
    state.hasTopPanel ||
    state.hasSpreadPanel ||
    state.hasMetricStrip ||
    state.summaryHeight > 42 ||
    !state.summaryInsideTableCard ||
    state.advancedSortControls.length > 0
  ) {
    throw new Error(`${scope}: rank header sorting contract failed ${JSON.stringify(state)}`);
  }
  if (state.sortHeaderCount === 0) {
    return;
  }
  if (
    state.sortHeaderCount < 7 ||
    state.activeSortHeaderCount !== 1 ||
    !state.headerLabels.some((text) => text.includes("F1@.50")) ||
    !state.headerLabels.some((text) => text.includes("P@.50")) ||
    !state.headerLabels.some((text) => text.includes("R@.50")) ||
    !state.headerLabels.some((text) => text.includes("mIoU")) ||
    !state.headerLabels.some((text) => text.includes("预测数")) ||
    !state.headerLabels.some((text) => text.includes("Run")) ||
    !state.headerLabels.some((text) => text.includes("创建时间")) ||
    state.activeCellCount === 0
  ) {
    throw new Error(`${scope}: rank table header sorting contract failed ${JSON.stringify(state)}`);
  }
  await page.keyboard.press("Escape");
  await page.locator(".rank-sort-header", { hasText: "mIoU" }).click();
  await page.locator(".rank-primary-score").first().waitFor({ timeout: 10_000 });
  await page.locator(".rank-board-table-card").first().waitFor({ timeout: 10_000 });
  await page.locator(".rank-sort-header", { hasText: "Run" }).click();
  await page.locator(".rank-sort-header", { hasText: "F1@.50" }).click();
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

async function assertBenchmarksPageChunkLoaded(page, scope) {
  const loaded = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .filter((name) => name.includes("benchmarksPage"))
  );
  if (loaded.length === 0) {
    throw new Error(`${scope}: benchmarks route did not load the independent benchmarksPage chunk`);
  }
}

async function assertRunsPageChunkLoaded(page, scope) {
  const loaded = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .filter((name) => name.includes("runsPage"))
  );
  if (loaded.length === 0) {
    throw new Error(`${scope}: runs route did not load the independent runsPage chunk`);
  }
}

async function assertComparisonSamplePageChunkLoaded(page, scope) {
  const loaded = await page.evaluate(() =>
    performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .filter((name) => name.includes("comparisonSamplePage"))
  );
  if (loaded.length === 0) {
    throw new Error(
      `${scope}: comparison sample route did not load the independent comparisonSamplePage chunk`
    );
  }
}

function allowsScroll(value) {
  return value === "auto" || value === "scroll";
}

function clipsOverflow(value) {
  return value === "hidden" || value === "clip";
}

function formatRect(rect) {
  return `x=${Math.round(rect.left)},y=${Math.round(rect.top)},w=${Math.round(
    rect.width
  )},h=${Math.round(rect.height)},bottom=${Math.round(rect.bottom)},right=${Math.round(rect.right)}`;
}
