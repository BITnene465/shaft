import { strict as assert } from "node:assert";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const srcRoot = path.join(root, "src");
const sourceFiles = await collectSourceFiles(srcRoot);

for (const filePath of sourceFiles) {
  const source = await readFile(filePath, "utf8");
  const relativePath = path.relative(root, filePath);
  assertNoBlockingBrowserDialogs(source, relativePath);
  assertNoBusinessDialogShell(source, relativePath);
  assertNoLegacySampleFilters(source, relativePath);
}

const jobsPage = await readSource("src/jobsPage.tsx");
const uiSource = await readSource("src/ui.tsx");
const filterControls = await readSource("src/filterControls.tsx");
assert(
  filterControls.includes('import { ActionButton } from "./ui";') &&
    filterControls.includes("function resetAdvancedFilters()") &&
    filterControls.includes("function defaultFilterValue(") &&
    filterControls.includes('className="advanced-filter-clear"'),
  "advanced filter reset action must be centralized in AdvancedFilterBar",
);
assert(
  uiSource.includes("export function SelectableRowButton("),
  "sample row selection must be centralized in SelectableRowButton",
);
assert(
  uiSource.includes("export function OptionChipButton("),
  "query chip selection must be centralized in OptionChipButton",
);
assert(
  jobsPage.includes('<ActionButton variant="mini" onClick={() => onChange(labelOptions)}>'),
  "label subtask select-all action must use ActionButton",
);
assert(
  jobsPage.includes('<ActionButton variant="mini" onClick={() => onChange([])}>'),
  "label subtask default-policy action must use ActionButton",
);
assert(
  jobsPage.includes('<ActionButton variant="mini" type="submit">'),
  "label subtask custom-label submit must use ActionButton",
);
assert(
  !jobsPage.includes('<button type="submit">添加</button>'),
  "label subtask submit regressed to a raw button",
);
assert(
  jobsPage.includes("import { CompactSelectControl } from \"./controlPrimitives\";"),
  "manifest toolbar selects must use CompactSelectControl",
);
assert(
  (jobsPage.match(/<CompactSelectControl/g) ?? []).length >= 2,
  "manifest toolbar must render template and prompt through CompactSelectControl",
);
assert(
  !jobsPage.includes('className="filter-select compact"'),
  "jobs page must not create ad hoc compact filter selects outside filterControls",
);
assertNoLegacyFormSubmitClass(jobsPage, "jobsPage.tsx");
assert(
  jobsPage.includes('OptionChipButton,'),
  "label subtask chips must import OptionChipButton",
);
assert(
  jobsPage.includes("<OptionChipButton") &&
    !jobsPage.includes('className={selectedSet.has(label) ? "query-chip active" : "query-chip"}'),
  "label subtask chips must use OptionChipButton instead of raw query-chip buttons",
);
assert(
  jobsPage.includes('if (task !== "detection")') &&
    jobsPage.includes("return null;") &&
    jobsPage.includes("<LabelSubtaskPanel"),
  "label subtask panel must stay detection-only; keypoint jobs must not expose label subset UI",
);

const settingsControls = await readSource("src/settingsControls.tsx");
assert(
  settingsControls.includes('<ActionButton variant="mini" onClick={() => onReset(action.id)}>'),
  "shortcut reset action must use ActionButton",
);
assert(
  settingsControls.includes(
    '<ActionButton variant="secondary" className="settings-inline-action" onClick={onResetAll}>',
  ),
  "shortcut reset-all action must use ActionButton",
);
assertNoRawSelectElement(settingsControls, "settingsControls.tsx");
assert(
  settingsControls.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    settingsControls.includes('className="inline-select-control"') &&
    settingsControls.includes("hideLabel"),
  "settings inline label color role select must use FormSelectControl",
);

const settingsPage = await readSource("src/settingsPage.tsx");
assert(
  settingsPage.includes('import { CompactSelectControl, NumberSettingControl } from "./controlPrimitives";'),
  "settings page selects must use CompactSelectControl",
);
assert(
  /<CompactSelectControl\s+dense\s+label="预测线型"/.test(settingsPage),
  "settings prediction line style select must use CompactSelectControl",
);
assert(
  settingsPage.includes('className="settings-search-clear"'),
  "settings search clear action must use IconActionButton",
);
assert(
  settingsPage.includes("SelectableCardButton") &&
    !/<button[\s\S]{0,220}settings-section-button/.test(settingsPage),
  "settings section navigation must use SelectableCardButton instead of raw section buttons",
);
assert(
  !settingsPage.includes('className="compact-select dense"'),
  "settings page must not create ad hoc compact select shells",
);
assert(
  !/<button[^>]+className="settings-inline-action"/.test(settingsPage),
  "settings inline standard actions must use ActionButton",
);
assert(
  !/<button[^>]+removeLabelColor/.test(settingsPage),
  "settings label clear action must use ActionButton",
);
const overviewPage = await readSource("src/overviewPage.tsx");
assert(
  overviewPage.includes("export function OverviewPage()"),
  "overview page module must export OverviewPage",
);
assert(
  overviewPage.includes("overview-command-deck") &&
    overviewPage.includes("overview-focus-panel") &&
    overviewPage.includes("OverviewTrackGroup") &&
    overviewPage.includes("OverviewActivityMatrix") &&
    overviewPage.includes("OverviewRecentRunsPanel") &&
    !overviewPage.includes("OverviewMiniChartPanel") &&
    !overviewPage.includes("overviewCharts") &&
    !overviewPage.includes("overview-chart-matrix") &&
    !/Notes|Tasks|Label footprint|样本\/label|模型分布|Job 日历|Scheduler 资源|Benchmark 任务|Run 日历/.test(
      overviewPage,
    ),
  "overview must stay a curated high-value command deck instead of a low-value panel wall",
);
const mainEntry = await readSource("src/main.tsx");
assert(
  mainEntry.includes('import { OverviewPage } from "./overviewPage";'),
  "main.tsx must route to the extracted OverviewPage module",
);
assert(
  mainEntry.includes('className="sidebar-toggle"') &&
    mainEntry.includes("<IconActionButton") &&
    !/<button[\s\S]{0,180}className="sidebar-toggle"/.test(mainEntry),
  "sidebar collapse control must use IconActionButton instead of a raw button",
);
const benchmarksPage = await readSource("src/benchmarksPage.tsx");
assert(
  benchmarksPage.includes("export function BenchmarksPage()") &&
    benchmarksPage.includes("export function BenchmarkDetailPage()"),
  "benchmarks page module must export list and detail pages",
);
assert(
  benchmarksPage.includes("SelectableRowButton") &&
    !benchmarksPage.includes('className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}'),
  "benchmark sample list rows must use SelectableRowButton",
);
assertNoLegacyFormSubmitClass(benchmarksPage, "benchmarksPage.tsx");
assert(
  mainEntry.includes('lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarksPage")') &&
    mainEntry.includes('lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarkDetailPage")'),
  "main.tsx must lazy-route to the extracted benchmarks page module",
);
const runsPage = await readSource("src/runsPage.tsx");
assert(
  runsPage.includes("export function RunsPage()") &&
    runsPage.includes("export function RunDetailPage()"),
  "runs page module must export list and detail pages",
);
assert(
  runsPage.includes("SelectableRowButton") &&
    !runsPage.includes('className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}'),
  "run sample list rows must use SelectableRowButton",
);
assertNoLegacyFormSubmitClass(runsPage, "runsPage.tsx");
assertNoRawSelectElement(runsPage, "runsPage.tsx");
assert(
  runsPage.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    (runsPage.match(/<FormSelectControl/g) ?? []).length >= 2,
  "runs import dialog selects must use FormSelectControl",
);
const servicesPage = await readSource("src/servicesPage.tsx");
assertNoLegacyFormSubmitClass(servicesPage, "servicesPage.tsx");
assertNoRawSelectElement(servicesPage, "servicesPage.tsx");
assert(
  servicesPage.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    (servicesPage.match(/<FormSelectControl/g) ?? []).length >= 1,
  "service registration dialog selects must use FormSelectControl",
);
const comparePage = await readSource("src/comparePage.tsx");
assertNoRawSelectElement(comparePage, "comparePage.tsx");
assert(
  comparePage.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    (comparePage.match(/<FormSelectControl/g) ?? []).length >= 1,
  "compare run rail selects must use FormSelectControl",
);
assert(
  comparePage.includes("SelectableCardButton") &&
    (comparePage.match(/<SelectableCardButton/g) ?? []).length >= 2 &&
    !/<button[\s\S]{0,240}label-delta-card/.test(comparePage),
  "compare label delta cards must use SelectableCardButton instead of raw buttons",
);
const rankBoardPage = await readSource("src/rankBoardPage.tsx");
assert(
  rankBoardPage.includes("const RANK_PAGE_SIZE = 80;") &&
    rankBoardPage.includes("function RankBoardPager(") &&
    rankBoardPage.includes("offset: pageOffset") &&
    rankBoardPage.includes("limit: RANK_PAGE_SIZE") &&
    !rankBoardPage.includes("limit: 200"),
  "rank board page must use paged API requests instead of a fixed 200-row slice",
);
const sampleViewer = await readSource("src/sampleViewer.tsx");
assert(
  sampleViewer.includes("export function SampleViewer("),
  "sample viewer module must export the shared SampleViewer",
);
assert(
  sampleViewer.includes("OptionChipButton") && !sampleViewer.includes('className="query-chip"'),
  "sample viewer utility chips must use OptionChipButton",
);
const viewerPanels = await readSource("src/viewerPanels.tsx");
assert(
  viewerPanels.includes('import { CompactSelectControl, ToggleButton } from "./controlPrimitives";'),
  "viewer layer preset select must use CompactSelectControl",
);
assert(
  viewerPanels.includes("OptionChipButton"),
  "viewer label chips must import OptionChipButton",
);
assert(
  viewerPanels.includes("SelectableCardButton") &&
    !/<button[\s\S]{0,220}object-row/.test(viewerPanels),
  "viewer object rows must use SelectableCardButton instead of raw object-row buttons",
);
assert(
  viewerPanels.includes("<CompactSelectControl") &&
    !viewerPanels.includes('<label className="compact-select">'),
  "viewer controls must not hand-roll compact select markup",
);
assert(
  viewerPanels.includes("<OptionChipButton") &&
    !/<button[\s\S]{0,240}label-select/.test(viewerPanels),
  "viewer label chips must use OptionChipButton instead of raw label-select buttons",
);
const viewerCanvas = await readSource("src/viewerCanvas.tsx");
assert(
  viewerCanvas.includes('import { ActionButton } from "./ui";') &&
    viewerCanvas.includes('className="canvas-reset-button"') &&
    !/<button[\s\S]{0,120}resetViewport/.test(viewerCanvas),
  "viewer canvas reset control must use ActionButton instead of a raw button",
);
assert(
  mainEntry.includes('lazyRouteComponent(() => import("./runsPage"), "RunsPage")') &&
    mainEntry.includes('lazyRouteComponent(() => import("./runsPage"), "RunDetailPage")'),
  "main.tsx must lazy-route to the extracted runs page module",
);
const comparisonSamplePage = await readSource("src/comparisonSamplePage.tsx");
assert(
  comparisonSamplePage.includes("export function ComparisonSamplePage()"),
  "comparison sample page module must export ComparisonSamplePage",
);
assert(
  comparisonSamplePage.includes('import { SampleViewer } from "./sampleViewer";'),
  "comparison sample page must reuse the shared SampleViewer",
);
assert(
  mainEntry.includes(
    'lazyRouteComponent(() => import("./comparisonSamplePage"), "ComparisonSamplePage")',
  ),
  "main.tsx must lazy-route to the extracted comparison sample page module",
);
assert(
  !mainEntryHasOverviewImplementation(mainEntry),
  "main.tsx should only route to OverviewPage, not implement the overview workbench",
);
assert(
  !mainEntryHasBenchmarksImplementation(mainEntry),
  "main.tsx should only route to BenchmarksPage, not implement benchmark workbenches",
);
assert(
  !mainEntryHasRunsImplementation(mainEntry),
  "main.tsx should only route to RunsPage, not implement run workbenches",
);
assert(
  !mainEntryHasComparisonSampleImplementation(mainEntry),
  "main.tsx should only route to ComparisonSamplePage, not implement comparison sample workbenches",
);
assert(
  !mainEntryHasSettingsImplementation(mainEntry),
  "main.tsx should only route to SettingsPage, not implement the settings workbench",
);

console.log("ui contract checks passed");

async function collectSourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectSourceFiles(entryPath)));
    } else if (/\.(ts|tsx)$/.test(entry.name)) {
      files.push(entryPath);
    }
  }
  return files;
}

async function readSource(relativePath) {
  return readFile(path.join(root, relativePath), "utf8");
}

function assertNoBlockingBrowserDialogs(source, relativePath) {
  const match = source.match(/\b(?:window\.)?(confirm|alert|prompt)\s*\(/);
  assert(!match, `${relativePath}: blocking browser dialog '${match?.[1]}' is not allowed`);
}

function assertNoBusinessDialogShell(source, relativePath) {
  if (relativePath === "src/ui.tsx") {
    return;
  }
  assert(
    !/className=\{?["'`][^"'`]*workspace-dialog/.test(source),
    `${relativePath}: dialog shell classes belong in WorkspaceDialog`,
  );
}

function assertNoLegacySampleFilters(source, relativePath) {
  assert(!source.includes("sample-filters"), `${relativePath}: legacy sample-filters are not allowed`);
}

function assertNoLegacyFormSubmitClass(source, relativePath) {
  assert(
    !source.includes("form-submit-button"),
    `${relativePath}: form submit actions must use ActionButton without legacy form-submit-button class`,
  );
}

function assertNoRawSelectElement(source, relativePath) {
  assert(
    !/<select\b/.test(source),
    `${relativePath}: local selects must use controlPrimitives instead of raw <select>`,
  );
}

function mainEntryHasSettingsImplementation(source) {
  return /function\s+SettingsPage\s*\(/.test(source) || source.includes("settings-workbench-shell");
}

function mainEntryHasOverviewImplementation(source) {
  return (
    /function\s+OverviewPage\s*\(/.test(source) ||
    source.includes("overview-console") ||
    source.includes("overview-chart-matrix")
  );
}

function mainEntryHasBenchmarksImplementation(source) {
  return (
    /function\s+BenchmarksPage\s*\(/.test(source) ||
    /function\s+BenchmarkDetailPage\s*\(/.test(source) ||
    source.includes("benchmark-form") ||
    source.includes("eval_bench_benchmark_sidebar_width")
  );
}

function mainEntryHasRunsImplementation(source) {
  return (
    /function\s+RunsPage\s*\(/.test(source) ||
    /function\s+RunDetailPage\s*\(/.test(source) ||
    source.includes("run-config-panel") ||
    source.includes("import-form") ||
    source.includes("eval_bench_run_sidebar_width")
  );
}

function mainEntryHasComparisonSampleImplementation(source) {
  return (
    /function\s+ComparisonSamplePage\s*\(/.test(source) ||
    /function\s+ComparisonSampleViewer\s*\(/.test(source) ||
    source.includes("comparison-sample-detail") ||
    source.includes("eval_bench_comparison_sample_candidate_width")
  );
}
