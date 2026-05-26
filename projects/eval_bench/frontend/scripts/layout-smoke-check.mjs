import { chromium } from "@playwright/test";

const baseUrl = (process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:8765/").replace(/\/+$/, "");
const expectedConsoleErrors = new WeakMap();

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
      ".overview-home-v16",
      ".overview-mission-board",
      ".overview-score-cluster",
      ".overview-workflow-row",
      ".overview-decision-metrics",
      ".overview-decision-metric",
      ".overview-run-focus",
      ".overview-loop-panel",
      ".overview-flow-spine",
      ".overview-flow-node",
      ".overview-score-dial",
      ".overview-next-action",
      ".overview-signal-stack",
      ".overview-signal-card",
      ".overview-recent-card"
    ]
  },
  {
    name: "rank-board",
    path: "/rank-board",
    selectors: [
      ".rank-board-page",
      ".rank-decision-panel",
      ".rank-sort-chip",
      ".rank-top-panel",
      ".rank-spread-panel",
      ".advanced-filter-bar",
      ".rank-scheme-panel",
      ".rank-facet-rail",
      ".table-shell"
    ],
    requireRankChunk: true
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
      await page.goto(`${baseUrl}${route.path}`, { waitUntil: "networkidle" });
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
      await assertAdvancedFilters(page, `${viewport.name}:${route.name}`);
      if (route.name === "runs" || route.name === "rank-board") {
        await assertAdvancedFilterClear(page, `${viewport.name}:${route.name}`);
      }
      await assertForbiddenSelectors(page, `${viewport.name}:${route.name}`, route.forbiddenSelectors ?? []);
      if (route.name === "overview") {
        await assertOverviewDensity(page, `${viewport.name}:${route.name}`);
      }
      if (route.name === "jobs") {
        await assertJobsRecentRuns(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireInspectorFilters) {
        await assertInspectorFilters(page, `${viewport.name}:${route.name}`);
        await assertInspectorCanvasPane(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunInspectorCounts) {
        await assertRunInspectorCountStrip(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireRunNoteTemplates) {
        await assertRunNoteTemplates(page, `${viewport.name}:${route.name}`);
      }
      if (route.requireInspectorFilters) {
        await assertInspectorFilteredEmptyState(page, `${viewport.name}:${route.name}`, route.name);
      }
      if (route.requireRankChunk) {
        await assertRankBoardChunkLoaded(page, `${viewport.name}:${route.name}`);
        await assertRankDecisionPanel(page, `${viewport.name}:${route.name}`);
        await assertRankFacetRail(page, `${viewport.name}:${route.name}`);
        await assertRankSchemePanel(page, `${viewport.name}:${route.name}`);
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
  const state = await page.evaluate(() => {
    const recentRows = Array.from(document.querySelectorAll(".overview-run-list a")).map((node) => {
      const rect = node.getBoundingClientRect();
      return {
        height: Math.round(rect.height),
        text: node.textContent ?? ""
      };
    });
    const signalCards = Array.from(document.querySelectorAll(".overview-signal-card")).map((node) => {
      const rect = node.getBoundingClientRect();
      return { width: Math.round(rect.width), height: Math.round(rect.height) };
    });
    const commandShell = document.querySelector(".overview-mission-board");
    const nowPanel = document.querySelector(".overview-mission-board");
    const livePanel = document.querySelector(".overview-score-cluster");
    const loopPanel = document.querySelector(".overview-loop-panel");
    const recentCard = document.querySelector(".overview-recent-card");
    const workflowRow = document.querySelector(".overview-workflow-row");
    const nextAction = document.querySelector(".overview-next-action");
    const decisionMetric = document.querySelector(".overview-decision-metric");
    const scoreDial = document.querySelector(".overview-score-dial");
    const decisionMetricRails = Array.from(
      document.querySelectorAll(".overview-decision-metric > i > b")
    ).map((node) => Math.round(node.getBoundingClientRect().width));
    const signalRails = Array.from(document.querySelectorAll(".overview-signal-card > i > b")).map((node) =>
      Math.round(node.getBoundingClientRect().width)
    );
    const panelHeights = Array.from(
      document.querySelectorAll(
        ".overview-mission-board, .overview-loop-panel, .overview-recent-card"
      )
    ).map((node) => Math.round(node.getBoundingClientRect().height));
    const commandShellStyle = commandShell ? getComputedStyle(commandShell) : null;
    const nowPanelStyle = nowPanel ? getComputedStyle(nowPanel) : null;
    const livePanelStyle = livePanel ? getComputedStyle(livePanel) : null;
    const loopPanelStyle = loopPanel ? getComputedStyle(loopPanel) : null;
    const recentCardStyle = recentCard ? getComputedStyle(recentCard) : null;
    const workflowRowStyle = workflowRow ? getComputedStyle(workflowRow) : null;
    const nextActionStyle = nextAction ? getComputedStyle(nextAction) : null;
    const decisionMetricStyle = decisionMetric ? getComputedStyle(decisionMetric) : null;
    const scoreDialStyle = scoreDial ? getComputedStyle(scoreDial) : null;
    const bodyText = document.querySelector(".dashboard-home")?.textContent ?? "";
    return {
      commandShellDisplay: commandShellStyle?.display ?? "",
      flowNodes: document.querySelectorAll(".overview-flow-node").length,
      scoreDials: document.querySelectorAll(".overview-score-dial").length,
      runFocusCards: document.querySelectorAll(".overview-run-focus").length,
      nextActions: document.querySelectorAll(".overview-next-action").length,
      decisionMetrics: document.querySelectorAll(".overview-decision-metric").length,
      decisionMetricRails,
      signalCards,
      signalRails,
      runArtifactRails: document.querySelectorAll(".overview-run-artifacts i b").length,
      runStates: document.querySelectorAll(".overview-run-state .badge").length,
      panelHeights,
      recentRows,
      recentCards: document.querySelectorAll(".overview-recent-card").length,
      nowPanels: document.querySelectorAll(".overview-mission-board").length,
      livePanels: document.querySelectorAll(".overview-score-cluster").length,
      loopPanels: document.querySelectorAll(".overview-loop-panel").length,
      workflowRows: document.querySelectorAll(".overview-workflow-row").length,
      miniCharts: document.querySelectorAll(".overview-mini-chart").length,
      chartMatrix: document.querySelectorAll(".overview-chart-matrix").length,
      legacyActivityMatrix: document.querySelectorAll(".overview-activity-matrix").length,
      oldTimelinePanels: document.querySelectorAll(
        ".overview-timeline-panel, .overview-sparkline, .overview-timeline-labels"
      ).length,
      bodyText,
      nowPanelHeight: nowPanel ? Math.round(nowPanel.getBoundingClientRect().height) : 0,
      livePanelHeight: livePanel ? Math.round(livePanel.getBoundingClientRect().height) : 0,
      loopPanelHeight: loopPanel ? Math.round(loopPanel.getBoundingClientRect().height) : 0,
      recentCardHeight: recentCard ? Math.round(recentCard.getBoundingClientRect().height) : 0,
      nowPanelDisplay: nowPanelStyle?.display ?? "",
      livePanelDisplay: livePanelStyle?.display ?? "",
      loopPanelDisplay: loopPanelStyle?.display ?? "",
      recentCardDisplay: recentCardStyle?.display ?? "",
      workflowRowDisplay: workflowRowStyle?.display ?? "",
      nextActionTransition: nextActionStyle?.transitionDuration ?? "",
      decisionMetricTransition: decisionMetricStyle?.transitionDuration ?? "",
      scoreDialTransition: scoreDialStyle?.transitionDuration ?? ""
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
    state.loopPanelDisplay !== "block" ||
    state.recentCardDisplay !== "block" ||
    state.workflowRowDisplay !== "flex"
  ) {
    throw new Error(
      `${scope}: overview should use a value-first operations desk ${JSON.stringify({
        commandShellDisplay: state.commandShellDisplay,
        nowPanelDisplay: state.nowPanelDisplay,
        livePanelDisplay: state.livePanelDisplay,
        loopPanelDisplay: state.loopPanelDisplay,
        recentCardDisplay: state.recentCardDisplay,
        workflowRowDisplay: state.workflowRowDisplay
      })}`
    );
  }
  if (
    state.flowNodes !== 4 ||
    state.scoreDials !== 1 ||
    state.runFocusCards !== 1 ||
    state.nextActions !== 1 ||
    state.decisionMetrics !== 3 ||
    state.signalCards.length !== 4
  ) {
    throw new Error(
      `${scope}: overview should expose score focus, flow nodes, one next action, and pulse cards ${JSON.stringify({
        flowNodes: state.flowNodes,
        scoreDials: state.scoreDials,
        runFocusCards: state.runFocusCards,
        nextActions: state.nextActions,
        decisionMetrics: state.decisionMetrics,
        signalCards: state.signalCards.length
      })}`
    );
  }
  if (
    state.nowPanels !== 1 ||
    state.livePanels !== 1 ||
    state.loopPanels !== 1 ||
    state.recentCards !== 1 ||
    state.workflowRows !== 1
  ) {
    throw new Error(
      `${scope}: overview should keep one mission board, score cluster, loop panel, workflow row, and recent panel ${JSON.stringify({
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
    state.loopPanelHeight < 120 ||
    state.recentCardHeight < 120
  ) {
    throw new Error(
      `${scope}: overview control panels collapsed ${JSON.stringify({
        nowPanelHeight: state.nowPanelHeight,
        livePanelHeight: state.livePanelHeight,
        loopPanelHeight: state.loopPanelHeight,
        recentCardHeight: state.recentCardHeight
      })}`
    );
  }
  if (!state.nextActionTransition || state.nextActionTransition === "0s") {
    throw new Error(`${scope}: overview next action is missing interaction transition`);
  }
  if (!state.decisionMetricTransition || state.decisionMetricTransition === "0s") {
    throw new Error(`${scope}: overview decision metrics are missing interaction transition`);
  }
  if (!state.scoreDialTransition || state.scoreDialTransition === "0s") {
    throw new Error(`${scope}: overview score dial is missing interaction transition`);
  }
  if (state.decisionMetricRails.length !== 3 || state.decisionMetricRails.some((width) => width < 0)) {
    throw new Error(`${scope}: overview decision metrics should expose progress rails`);
  }
  if (state.oldTimelinePanels > 0) {
    throw new Error(`${scope}: old oversized overview timeline markup is still present`);
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
  if (
    state.signalCards.length !== 4 ||
    state.signalRails.length !== 4 ||
    !state.signalRails.some((width) => width > 0)
  ) {
    throw new Error(
      `${scope}: overview signal visualization is missing ${JSON.stringify({
        cards: state.signalCards.length,
        rails: state.signalRails
      })}`
    );
  }
  for (const [index, rect] of state.signalCards.entries()) {
    if (rect.width <= 0 || rect.height <= 0) {
      throw new Error(`${scope}: signal card ${index} is not visible ${JSON.stringify(rect)}`);
    }
    if (!scope.startsWith("narrow") && rect.width < 140) {
      throw new Error(`${scope}: signal card ${index} is too compressed ${JSON.stringify(rect)}`);
    }
  }
}

async function assertJobsRecentRuns(page, scope) {
  const state = await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll(".recent-run-card")).map((node) => {
      const rect = node.getBoundingClientRect();
      return {
        height: Math.round(rect.height),
        text: node.textContent ?? "",
        artifactRails: node.querySelectorAll(".recent-run-artifacts i b").length,
        statusBadges: node.querySelectorAll(".badge").length
      };
    });
    return {
      cards,
      metricBlocks: document.querySelectorAll(".recent-run-metrics").length
    };
  });
  if (state.metricBlocks !== 0) {
    throw new Error(`${scope}: jobs recent results should not render fine metric blocks`);
  }
  for (const [index, card] of state.cards.entries()) {
    if (card.height > 112) {
      throw new Error(`${scope}: jobs recent result card ${index} is too tall ${card.height}`);
    }
    if (/\b(P@|R@|precision|recall|iou|miou)\b/i.test(card.text)) {
      throw new Error(`${scope}: jobs recent result card ${index} exposes fine metrics`);
    }
    if (card.artifactRails !== 1 || card.statusBadges !== 1) {
      throw new Error(
        `${scope}: jobs recent result card ${index} is missing artifact rail or status badge ${JSON.stringify(card)}`
      );
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

async function assertAdvancedFilterClear(page, scope) {
  const filter = page.locator(".advanced-filter-bar").first();
  const searchInput = await openAdvancedFilter(filter);
  await searchInput.fill("layout-smoke-token-reset");
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
  const resetSearchInput = await openAdvancedFilter(filter);
  await resetSearchInput.fill("layout-smoke-filter-reset");
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
    return {
      hasBar: Boolean(bar),
      hasAppendPanel: Boolean(appendPanel),
      hasAppendTextarea: Boolean(appendTextarea),
      hasAppendAction: Boolean(appendAction),
      buttonCount: buttons.length,
      labels: buttons.map((button) => button.textContent?.trim() ?? "")
    };
  });
  if (!state.hasBar || !state.hasAppendPanel || !state.hasAppendTextarea || !state.hasAppendAction || state.buttonCount < 4) {
    throw new Error(`${scope}: run note template bar is missing templates ${JSON.stringify(state)}`);
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
  if (state.stage.height < 180) {
    throw new Error(`${scope}: inspector image stage collapsed ${JSON.stringify(state.stage)}`);
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
  const select = page.locator(".inspector-sidebar .advanced-filter-controls select").first();
  await select.waitFor({ timeout: 5_000 });
  const optionValues = await select.locator("option").evaluateAll((options) =>
    options
      .map((option) => option.value)
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
    await select.selectOption(optionValues[0]);
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
  const expectedGroups = ["Tasks", "Benchmarks", "Status", "Labels", "Models", "Prompts", "Metrics"];
  const actualGroups = state.groups.map((group) => group.title);
  const missingGroups = expectedGroups.filter((title) => !actualGroups.includes(title));
  if (missingGroups.length > 0 || state.groups.length < expectedGroups.length) {
    throw new Error(
      `${scope}: rank facet rail should expose all backend groups ${JSON.stringify({
        ...state,
        missingGroups
      })}`
    );
  }
  if (state.buttonCount === 0) {
    return;
  }
  if (state.staticCountNodes > 0) {
    throw new Error(`${scope}: rank facet rail regressed to static count chips ${JSON.stringify(state)}`);
  }
  await page.locator(".rank-facet-button").first().click();
  await page.locator(".rank-facet-button.active").first().waitFor({ timeout: 10_000 });
  await page.locator(".rank-board-page .table-shell").first().waitFor({ timeout: 10_000 });
}

async function assertRankDecisionPanel(page, scope) {
  const state = await page.evaluate(() => {
    const panel = document.querySelector(".rank-decision-panel");
    const sortChips = Array.from(document.querySelectorAll(".rank-sort-chip"));
    const primarySortChips = Array.from(document.querySelectorAll(".rank-sort-chip.primary"));
    const auxiliarySortChips = Array.from(document.querySelectorAll(".rank-sort-chip.auxiliary"));
    const sortSections = Array.from(document.querySelectorAll(".rank-sort-section")).map(
      (node) => node.textContent ?? ""
    );
    const orderChips = Array.from(document.querySelectorAll(".rank-order-chip"));
    const activeSortChips = sortChips.filter((chip) => chip.classList.contains("active"));
    const topRows = document.querySelectorAll(".rank-top-row");
    const spreadBars = document.querySelectorAll(".rank-spread-bars span");
    const advancedSortControls = Array.from(document.querySelectorAll(".advanced-filter-bar [id]"))
      .map((node) => node.id)
      .filter((id) => id.includes("rank-sort"));
    return {
      hasPanel: Boolean(panel),
      sortChipCount: sortChips.length,
      primarySortChipCount: primarySortChips.length,
      auxiliarySortChipCount: auxiliarySortChips.length,
      sortSections,
      orderChipCount: orderChips.length,
      activeSortChipCount: activeSortChips.length,
      topRowCount: topRows.length,
      spreadBarCount: spreadBars.length,
      advancedSortControls
    };
  });
  if (
    !state.hasPanel ||
    state.sortChipCount !== 7 ||
    state.primarySortChipCount !== 5 ||
    state.auxiliarySortChipCount !== 2 ||
    !state.sortSections.some((text) => text.includes("主指标")) ||
    !state.sortSections.some((text) => text.includes("辅助排序")) ||
    state.orderChipCount !== 2 ||
    state.activeSortChipCount !== 1 ||
    state.spreadBarCount !== 5 ||
    state.advancedSortControls.length > 0
  ) {
    throw new Error(`${scope}: rank decision panel contract failed ${JSON.stringify(state)}`);
  }
  await page.locator(".rank-sort-chip", { hasText: "mIoU" }).first().click();
  await page.locator(".rank-sort-chip.active", { hasText: "mIoU" }).first().waitFor({ timeout: 10_000 });
  await page.locator(".rank-board-page .table-shell").first().waitFor({ timeout: 10_000 });
}

async function assertRankSchemePanel(page, scope) {
  const state = await page.evaluate(() => {
    const panel = document.querySelector(".rank-scheme-panel");
    const textarea = panel?.querySelector("textarea");
    const summary = panel?.querySelector("summary");
    return {
      hasPanel: Boolean(panel),
      open: panel?.hasAttribute("open") ?? false,
      hasTextarea: Boolean(textarea),
      summaryHeight: summary ? Math.round(summary.getBoundingClientRect().height) : 0
    };
  });
  if (!state.hasPanel || !state.hasTextarea) {
    throw new Error(`${scope}: rank scheme panel is missing`);
  }
  if (state.open) {
    throw new Error(`${scope}: rank scheme panel should be collapsed by default`);
  }
  if (state.summaryHeight > 54) {
    throw new Error(`${scope}: rank scheme summary is too tall ${state.summaryHeight}`);
  }
  const benchmarkId = await page.evaluate(async () => {
    const response = await fetch("/api/state");
    if (!response.ok) {
      return "";
    }
    const state = await response.json();
    const run = Array.isArray(state.runs)
      ? state.runs.find((item) => item.benchmark_id && item.report_path)
      : null;
    return run?.benchmark_id ?? "";
  });
  if (!benchmarkId) {
    return;
  }
  const scheme = JSON.stringify(
    {
      name: "layout_smoke_weighted",
      terms: [
        { benchmark_id: benchmarkId, metric: "f1_iou50", weight: 0.7, missing: "drop" },
        { benchmark_id: benchmarkId, metric: "mean_iou", weight: 0.3, missing: "zero" }
      ]
    },
    null,
    2
  );
  await page.locator(".rank-scheme-panel summary").click();
  await page.locator(".rank-scheme-panel textarea").fill(scheme);
  await page.locator(".rank-scheme-body .control-check input").check();
  await page.locator(".rank-formula-chip.weighted").waitFor({ timeout: 10_000 });
  const weightedState = await page.evaluate(() => ({
    chipText: document.querySelector(".rank-formula-chip")?.textContent?.trim() ?? "",
    headers: Array.from(document.querySelectorAll(".table-shell th")).map(
      (node) => node.textContent?.trim() ?? ""
    ),
    primaryScores: document.querySelectorAll(".rank-primary-score").length
  }));
  if (
    !weightedState.chipText.includes("Weighted") ||
    !weightedState.headers.includes("layout_smoke_weighted") ||
    !weightedState.headers.includes("Components") ||
    weightedState.primaryScores < 1
  ) {
    throw new Error(`${scope}: weighted rank scheme did not apply ${JSON.stringify(weightedState)}`);
  }
  const invalidScheme = JSON.stringify(
    {
      name: "layout_smoke_invalid",
      terms: [
        { benchmark_id: benchmarkId, metric: "unsupported_metric", weight: 1, missing: "drop" }
      ]
    },
    null,
    2
  );
  expectedConsoleErrors.set(page, (expectedConsoleErrors.get(page) ?? 0) + 2);
  await page.locator(".rank-scheme-panel textarea").fill(invalidScheme);
  await page.locator(".rank-scheme-status.error").waitFor({ timeout: 10_000 });
  const invalidState = await page.evaluate(() => ({
    statusText: document.querySelector(".rank-scheme-status")?.textContent?.trim() ?? "",
    hasTable: Boolean(document.querySelector(".table-shell")),
    bodyText: document.body.textContent ?? ""
  }));
  if (!invalidState.statusText.includes("metric is not supported") || !invalidState.hasTable) {
    throw new Error(
      `${scope}: invalid weighted scheme should stay inline ${JSON.stringify({
        statusText: invalidState.statusText,
        hasTable: invalidState.hasTable
      })}`
    );
  }
  if (invalidState.bodyText.includes("排行榜加载失败")) {
    throw new Error(`${scope}: invalid weighted scheme collapsed the whole rank-board page`);
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
