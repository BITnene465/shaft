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
const runArtifactSignals = await readSource("src/runArtifactSignals.ts");
const uiSource = await readSource("src/ui.tsx");
const apiSource = await readSource("src/api.ts");
const filterControls = await readSource("src/filterControls.tsx");
const labelSubtaskControls = await readSource("src/labelSubtaskControls.tsx");
const samplePagerSource = await readSource("src/samplePager.tsx");
assert(
  apiSource.includes("export class ApiError extends Error") &&
    apiSource.includes("export function isApiError(") &&
    apiSource.includes("this.status = status") &&
    apiSource.includes("throw error;"),
  "frontend API failures must expose typed ApiError status for structured recovery",
);
assert(
  samplePagerSource.includes("export function PagerControl(") &&
    samplePagerSource.includes("export function clampListPageOffset(") &&
    samplePagerSource.includes("export function SamplePager("),
  "paged list controls must share PagerControl and clampListPageOffset",
);
assert(
  filterControls.includes('import { ActionButton, PanelToggleButton } from "./ui";') &&
    filterControls.includes("function resetAdvancedFilters()") &&
    filterControls.includes("function resetAdvancedFilter(") &&
    filterControls.includes("function defaultFilterValue(") &&
    filterControls.includes("function groupAdvancedControls(") &&
    filterControls.includes('aria-haspopup="dialog"') &&
    filterControls.includes('className="advanced-filter-popover"') &&
    filterControls.includes('className="advanced-filter-directory"') &&
    filterControls.includes('className="advanced-filter-token"') &&
    filterControls.includes('className="advanced-filter-clear"') &&
    filterControls.includes("onClick={() => resetAdvancedFilter(filter.control)}") &&
    filterControls.includes("<PanelToggleButton") &&
    !/<button[\s\S]{0,260}advanced-filter-head/.test(filterControls),
  "advanced filter reset, token clear, popup layout, and grouping must be centralized in AdvancedFilterBar",
);
assert(
  uiSource.includes("export function PanelToggleButton("),
  "collapsible panel toggles must share PanelToggleButton",
);
assert(
  uiSource.includes("export function DisclosurePanel(") &&
    uiSource.includes("<details {...props} className={className}>") &&
    uiSource.includes("<summary>{summary}</summary>"),
  "collapsible details shells must share DisclosurePanel",
);
assert(
  uiSource.includes("export function IconNavLink(") &&
    uiSource.includes('className: joinClassNames("icon-button", dense && "dense", className)'),
  "router icon links must share IconNavLink",
);
assert(
  uiSource.includes("export function InlineNavLink(") &&
    uiSource.includes('className: joinClassNames("mini-link", className)'),
  "router inline links must share InlineNavLink",
);
assert(
  uiSource.includes("export function InlineAnchor(") &&
    uiSource.includes('className={joinClassNames("mini-link", className)}'),
  "href inline links must share InlineAnchor",
);
assert(
  uiSource.includes("export function SelectableRowButton("),
  "sample row selection must be centralized in SelectableRowButton",
);
assert(
  uiSource.includes("export function SelectableTableRow("),
  "table row selection must be centralized in SelectableTableRow",
);
assert(
  uiSource.includes("export function OptionChipButton("),
  "query chip selection must be centralized in OptionChipButton",
);
assert(
  uiSource.includes("const DIALOG_FOCUSABLE_SELECTOR =") &&
    uiSource.includes("document.body.style.overflow = \"hidden\"") &&
    uiSource.includes("previouslyFocused?.focus()") &&
    uiSource.includes("tabIndex={-1}") &&
    uiSource.includes("aria-describedby={meta ? metaId : undefined}"),
  "WorkspaceDialog must own focus trapping, body scroll lock, and accessibility wiring",
);
assert(
  labelSubtaskControls.includes('<ActionButton variant="mini" onClick={() => onChange(labelOptions)}>'),
  "label subtask select-all action must use ActionButton",
);
assert(
  labelSubtaskControls.includes('<ActionButton variant="mini" onClick={() => onChange([])}>'),
  "label subtask default-policy action must use ActionButton",
);
assert(
  !labelSubtaskControls.includes('type="submit"') &&
    !labelSubtaskControls.includes('<button type="submit">添加</button>'),
  "label subtask panel must not expose a custom-label submit path",
);
assert(
  jobsPage.includes("import { CompactSelectControl } from \"./controlPrimitives\";"),
  "manifest toolbar selects must use CompactSelectControl",
);
assert(
  jobsPage.includes("const JOB_PAGE_SIZE = 80;") &&
    jobsPage.includes('import { PagerControl, clampListPageOffset } from "./samplePager";') &&
    jobsPage.includes('<PagerControl\n          className="rank-board-pager job-list-pager"') &&
    jobsPage.includes("offset: compact ? 0 : pageOffset") &&
    jobsPage.includes("limit: compact ? 12 : JOB_PAGE_SIZE") &&
    !jobsPage.includes("function JobListPager(") &&
    !jobsPage.includes("limit: compact ? 12 : 200") &&
    !jobsPage.includes("limit: 200"),
  "jobs queue page must use paged API requests instead of a fixed 200-job slice",
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
  labelSubtaskControls.includes("OptionChipButton") &&
    labelSubtaskControls.includes("DetectionLabelSubtaskPanel") &&
    labelSubtaskControls.includes('if (task !== "detection")') &&
    labelSubtaskControls.includes("return null;") &&
    !labelSubtaskControls.includes("label-subtask-add") &&
    !labelSubtaskControls.includes("自定义 label"),
  "label subtask chips must use controlled candidates and must not expose free-text label entry",
);
assert(
  labelSubtaskControls.includes("<OptionChipButton") &&
    !labelSubtaskControls.includes('className={selectedSet.has(label) ? "query-chip active" : "query-chip"}'),
  "label subtask chips must use OptionChipButton instead of raw query-chip buttons",
);
assert(
  jobsPage.includes("DetectionLabelSubtaskPanel") &&
    jobsPage.includes("<DetectionLabelSubtaskPanel"),
  "label subtask panel must stay detection-only; keypoint jobs must not expose label subset UI",
);
assert(
  jobsPage.includes("DisclosurePanel") &&
    jobsPage.includes('className="prompt-template-panel"') &&
    !/<details\b/.test(jobsPage) &&
    !/<summary\b/.test(jobsPage),
  "jobs prompt template panel must use DisclosurePanel instead of a local details shell",
);
assert(
  jobsPage.includes('import { recentRunsByCreatedAt, runArtifactReadiness } from "./runArtifactSignals";') &&
    jobsPage.includes("recent-run-artifacts") &&
    !jobsPage.includes("recent-run-metrics") &&
    !jobsPage.includes("formatMetric") &&
    !jobsPage.includes("precision_iou50") &&
    !jobsPage.includes("recall_iou50") &&
    !jobsPage.includes("mean_iou"),
  "jobs recent results must stay a compact artifact stream, not a fine metric panel",
);

const settingsControls = await readSource("src/settingsControls.tsx");
assert(
  settingsControls.includes('<ActionButton variant="mini" onClick={() => onReset(action.id)}>'),
  "shortcut reset action must use ActionButton",
);
assert(
  settingsControls.includes('<ActionButton') &&
    settingsControls.includes('className="shortcut-capture"') &&
    !/<button\b/.test(settingsControls),
  "settings controls must not use raw buttons for shortcut capture or reset actions",
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
const styleSource = await readSource("src/styles.css");
const designSource = await readSource("src/design.css");
const formattersSource = await readSource("src/formatters.ts");
assert(
  overviewPage.includes("export function OverviewPage()"),
  "overview page module must export OverviewPage",
);
assert(
  overviewPage.includes("overview-home-v15") &&
    overviewPage.includes("overview-command-deck") &&
    overviewPage.includes("overview-decision-panel") &&
    overviewPage.includes("overview-pulse-panel") &&
    overviewPage.includes("overview-loop-panel") &&
    overviewPage.includes("OverviewDecisionMetrics") &&
    overviewPage.includes("overview-decision-metrics") &&
    overviewPage.includes("overview-decision-metric") &&
    overviewPage.includes("OverviewScoreDial") &&
    overviewPage.includes("overview-score-dial") &&
    overviewPage.includes("OverviewRunFocus") &&
    overviewPage.includes("overview-run-focus") &&
    overviewPage.includes("bestF1Run") &&
    overviewPage.includes('import { formatMetric, runF1Score } from "./formatters";') &&
    overviewPage.includes("OverviewNextAction") &&
    overviewPage.includes("OverviewFlowSpine") &&
    overviewPage.includes("overview-flow-spine") &&
    overviewPage.includes("overview-flow-node") &&
    overviewPage.includes("overviewPostureLine") &&
    overviewPage.includes("recentRunsByCreatedAt(data.runs") &&
    overviewPage.includes("OverviewSignalStack") &&
    overviewPage.includes("overview-signal-stack") &&
    overviewPage.includes("OverviewRecentRunsPanel") &&
    overviewPage.includes("overview-run-counts") &&
    overviewPage.includes("overview-run-artifacts") &&
    overviewPage.includes("overview-run-state") &&
    overviewPage.includes('import { recentRunsByCreatedAt, runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";') &&
    overviewPage.includes("updateOverviewPointer") &&
    !overviewPage.includes("overview-home-v13") &&
    !overviewPage.includes("overview-command-shell") &&
    !overviewPage.includes("overview-now-panel") &&
    !overviewPage.includes("overview-live-panel") &&
    !overviewPage.includes("OverviewProofStrip") &&
    !overviewPage.includes("overview-proof-strip") &&
    !overviewPage.includes("overview-proof-card") &&
    !overviewPage.includes("OverviewTriageRail") &&
    !overviewPage.includes("overviewTriageActions") &&
    !overviewPage.includes("overview-triage-rail") &&
    !overviewPage.includes("overview-triage-link") &&
    !overviewPage.includes("overview-home-v6") &&
    !overviewPage.includes("overview-home-v7") &&
    !overviewPage.includes("overview-home-v8") &&
    !overviewPage.includes("overview-home-v9") &&
    !overviewPage.includes("overview-home-v10") &&
    !overviewPage.includes("overview-home-v11") &&
    !overviewPage.includes("overview-home-v12") &&
    !overviewPage.includes("overview-home-v14") &&
    !overviewPage.includes("OverviewRouteList") &&
    !overviewPage.includes("overviewRouteActions") &&
    !overviewPage.includes("overview-route-panel") &&
    !overviewPage.includes("overview-command-center-redesign") &&
    !overviewPage.includes("OverviewHeroMap") &&
    !overviewPage.includes("overview-orbit-map") &&
    !overviewPage.includes("OverviewReadinessPanel") &&
    !overviewPage.includes("overview-right-rail") &&
    !overviewPage.includes("overview-action-panel") &&
    !overviewPage.includes("overview-hero-route") &&
    !overviewPage.includes("OverviewSignalStrip") &&
    !overviewPage.includes("overview-signal-strip") &&
    !overviewPage.includes("OverviewHealthStrip") &&
    !overviewPage.includes("overview-health-strip") &&
    !overviewPage.includes("OverviewReadinessList") &&
    !overviewPage.includes("overviewReadinessItems") &&
    !overviewPage.includes("OverviewBottleneckPanel") &&
    !overviewPage.includes("overview-bottleneck-panel") &&
    !overviewPage.includes("overview-flow-and-bottleneck") &&
    !overviewPage.includes("overview-focus-panel") &&
    !overviewPage.includes("overview-side-stack") &&
    !overviewPage.includes("OverviewActivityMatrix") &&
    !overviewPage.includes("overview-activity-matrix") &&
    !overviewPage.includes("OverviewTrackGroup") &&
    !overviewPage.includes("OverviewMiniChartPanel") &&
    !overviewPage.includes("overviewCharts") &&
    !overviewPage.includes("overview-chart-matrix") &&
    !/Notes|Tasks|Label footprint|样本\/label|模型分布|Job 日历|Scheduler 资源|Benchmark 任务|Run 日历/.test(
      overviewPage,
    ),
  "overview must stay a curated high-value command deck instead of a low-value panel wall",
);
assert(
  runArtifactSignals.includes("export function recentRunsByCreatedAt") &&
    runArtifactSignals.includes("export function runArtifactReadiness") &&
    runArtifactSignals.includes("export function runAgeLabel") &&
    !overviewPage.includes("function overviewRecentRuns") &&
    !overviewPage.includes("function overviewRunReadiness") &&
    !overviewPage.includes("function overviewRunAge") &&
    !jobsPage.includes("function recentJobRuns") &&
    !jobsPage.includes("function recentRunReadiness"),
  "recent run artifact sorting/readiness/age logic must have one shared source",
);
assert(
  styleSource.includes("Overview v15: four-module operations cockpit") &&
    styleSource.includes(".overview-home-v15 .overview-decision-metric:hover") &&
    styleSource.includes("@keyframes overview-v15-sweep") &&
    styleSource.includes("@keyframes overview-v15-live") &&
    styleSource.includes("@keyframes overview-v15-float") &&
    styleSource.includes("@keyframes overview-v15-radar") &&
    designSource.includes("@keyframes eval-bench-surface-in") &&
    designSource.includes("@keyframes eval-bench-live-pulse") &&
    designSource.includes(".workspace-card:not(.fill):hover") &&
    designSource.includes(".nav-item:hover .app-icon") &&
    designSource.includes(".nav-item:hover::after") &&
    designSource.includes(".user-profile-chip:hover") &&
    designSource.includes(".status-pill:hover"),
  "overview and shared controls must keep tactile hover and motion feedback",
);
assert(
  ![
    "overview-home-v6",
    "overview-home-v7",
    "overview-home-v8",
    "overview-home-v9",
    "overview-home-v10",
    "overview-home-v11",
    "overview-home-v12",
    "overview-home-v13",
    "overview-home-v14",
    "overview-command-shell",
    "overview-now-panel",
    "overview-live-panel",
    "overview-proof-strip",
    "overview-proof-card",
    "overview-triage-rail",
    "overview-triage-link",
    "overview-command-center-redesign",
    "overview-focus-panel",
    "overview-side-stack",
    "overview-right-rail",
    "overview-workband",
    "overview-hero-board",
    "overview-signal-board",
    "overview-route-panel",
    "overview-activity-matrix",
    "overview-chart-matrix",
    "overview-mini-chart"
  ].some((token) => styleSource.includes(token)),
  "overview stylesheet must expose the active v15 surface and block deprecated design tracks",
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
  benchmarksPage.includes("const BENCHMARK_PAGE_SIZE = 80;") &&
    benchmarksPage.includes("PagerControl, SamplePager, clampListPageOffset") &&
    benchmarksPage.includes('className="rank-board-pager benchmark-list-pager"') &&
    benchmarksPage.includes("offset: pageOffset") &&
    benchmarksPage.includes("limit: BENCHMARK_PAGE_SIZE") &&
    !benchmarksPage.includes("function BenchmarkListPager(") &&
    !benchmarksPage.includes("limit: 200"),
  "benchmarks page must use paged API requests instead of a fixed 200-row slice",
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
  runsPage.includes("const RUN_PAGE_SIZE = 80;") &&
    runsPage.includes("PagerControl, SamplePager, clampListPageOffset") &&
    runsPage.includes('className="rank-board-pager run-list-pager"') &&
    runsPage.includes("offset: pageOffset") &&
    runsPage.includes("limit: RUN_PAGE_SIZE") &&
    !runsPage.includes("function RunListPager(") &&
    !runsPage.includes("limit: 200"),
  "runs page must use paged API requests instead of a fixed 200-row slice",
);
assert(
  runsPage.includes("SelectableRowButton") &&
    !runsPage.includes('className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}'),
  "run sample list rows must use SelectableRowButton",
);
assert(
  runsPage.includes("const RUN_NOTE_TEMPLATES = [") &&
    runsPage.includes("const RUN_NOTE_APPEND_HEADINGS = [") &&
    runsPage.includes("function insertNoteTemplate(") &&
    runsPage.includes("appendRunNote(run.run_id, note, heading)") &&
    runsPage.includes("const appendMutation = useMutation(") &&
    runsPage.includes('className="run-note-append-panel"') &&
    runsPage.includes('aria-label="追加 run note"') &&
    runsPage.includes("isApiError(error) && error.status === 409") &&
    (runsPage.match(/invalidateQueries\(\{ queryKey: \["dashboard-state"\] \}\)/g) ?? []).length >= 2 &&
    runsPage.includes('className="run-note-template-bar"') &&
    runsPage.includes("<ActionButton") &&
    !runsPage.includes('error.message.includes("409")') &&
    !runsPage.includes("setNoteDraft(noteDraft +"),
  "run note editor must expose structured templates and refresh dashboard state after 409 conflicts",
);
assert(
  runsPage.includes("DisclosurePanel") &&
    runsPage.includes('className="run-config-panel"') &&
    runsPage.includes('className="prompt-details"') &&
    !/<details\b/.test(runsPage) &&
    !/<summary\b/.test(runsPage),
  "run config and prompt snapshot panels must use DisclosurePanel instead of local details shells",
);
const runTables = await readSource("src/runTables.tsx");
assert(
  runTables.includes('hash="run-note"') &&
    runTables.includes('className={row.original.note ? "run-note-preview" : "run-note-preview empty"}') &&
    runsPage.includes("function shouldOpenRunNotePanel(") &&
    runsPage.includes('id="run-note"') &&
    runsPage.includes("open={configOpen}") &&
    runsPage.includes("setConfigOpen(event.currentTarget.open)") &&
    runsPage.includes("defaultOpen={shouldOpenRunNotePanel()}"),
  "run note previews must deep-link to the editable run note panel",
);
assert(
  formattersSource.includes("export function f1Score(") &&
    formattersSource.includes("export function runF1Score(") &&
    formattersSource.includes("/ F1 ${formatMetric(runF1Score(run))}") &&
    runTables.includes('import { formatDate, formatMetric, runF1Score, unique } from "./formatters";') &&
    runTables.includes('header: "F1@.50"') &&
    runTables.includes("formatMetric(runF1Score(row.original))"),
  "run option labels and run tables must foreground F1 as the default direct metric",
);
assert(
  runTables.includes("footer?: ReactNode") &&
    runTables.includes("{footer}") &&
    runTables.includes("import type { ReactNode }"),
  "run table must expose a footer slot for paged result controls",
);
assert(
  runTables.includes("IconNavLink") &&
    !runTables.includes('className="icon-button dense"'),
  "run table row icon links must use IconNavLink instead of ad hoc icon-button links",
);
const rankBoardMiniLinkSource = await readSource("src/rankBoardPage.tsx");
const jobsMiniLinkSource = await readSource("src/jobsPage.tsx");
const compareMiniLinkSource = await readSource("src/comparePage.tsx");
const comparisonSampleMiniLinkSource = await readSource("src/comparisonSamplePage.tsx");
assert(
  runTables.includes("InlineNavLink") &&
    rankBoardMiniLinkSource.includes("InlineNavLink") &&
    jobsMiniLinkSource.includes("InlineNavLink") &&
    compareMiniLinkSource.includes("InlineNavLink") &&
    !/<Link[^>]+className="mini-link/.test(runTables) &&
    !/<Link[^>]+className="mini-link/.test(rankBoardMiniLinkSource) &&
    !/<Link[^>]+className="mini-link/.test(jobsMiniLinkSource) &&
    !/<Link[^>]+className="mini-link/.test(compareMiniLinkSource),
  "router mini links must use InlineNavLink instead of ad hoc mini-link classes",
);
assert(
  jobsMiniLinkSource.includes("SelectableTableRow") &&
    jobsMiniLinkSource.includes("selected={job.job_id === selectedJob?.job_id}") &&
    !jobsMiniLinkSource.includes('className={job.job_id === selectedJob?.job_id ? "selectable-row selected" : "selectable-row"}'),
  "jobs queue selectable rows must use SelectableTableRow instead of ad hoc selectable-row class composition",
);
assert(
  compareMiniLinkSource.includes("InlineAnchor") &&
    comparisonSampleMiniLinkSource.includes("InlineAnchor") &&
    !/<a[^>]+className="mini-link/.test(compareMiniLinkSource) &&
    !/<a[^>]+className="mini-link/.test(comparisonSampleMiniLinkSource),
  "href mini links must use InlineAnchor instead of ad hoc mini-link anchors",
);
assertNoLegacyFormSubmitClass(runsPage, "runsPage.tsx");
assertNoRawSelectElement(runsPage, "runsPage.tsx");
assert(
  runsPage.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    (runsPage.match(/<FormSelectControl/g) ?? []).length >= 2,
  "runs import dialog selects must use FormSelectControl",
);
assert(
  runsPage.includes("DetectionLabelSubtaskPanel") &&
    runsPage.includes("const [targetLabels, setTargetLabels] = useState<string[]>([])") &&
    runsPage.includes("target_labels: targetLabels") &&
    !runsPage.includes("function parseTargetLabels("),
  "runs import dialog must use the shared detection label subtask panel instead of a free-text target label field",
);
const servicesPage = await readSource("src/servicesPage.tsx");
assert(
  servicesPage.includes("const SERVICE_PAGE_SIZE = 80;") &&
    servicesPage.includes('import { PagerControl, clampListPageOffset } from "./samplePager";') &&
    servicesPage.includes('className="rank-board-pager service-list-pager"') &&
    servicesPage.includes("offset: pageOffset") &&
    servicesPage.includes("limit: SERVICE_PAGE_SIZE") &&
    !servicesPage.includes("function ServiceListPager(") &&
    !servicesPage.includes("limit: 200"),
  "services page must use paged API requests instead of a fixed 200-service slice",
);
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
  comparePage.includes("const COMPARE_RUN_PAGE_SIZE = 80;") &&
    comparePage.includes('import { PagerControl, clampListPageOffset } from "./samplePager";') &&
    comparePage.includes('className="rank-board-pager compare-run-pager"') &&
    comparePage.includes("offset: pageOffset") &&
    comparePage.includes("limit: COMPARE_RUN_PAGE_SIZE") &&
    comparePage.includes("已选择；当前页未加载该 run") &&
    !comparePage.includes("function CompareRunPager(") &&
    !comparePage.includes("limit: 200"),
  "compare run rail must use paged API requests while preserving selected run ids",
);
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
    rankBoardPage.includes('import { PagerControl, clampListPageOffset } from "./samplePager";') &&
    rankBoardPage.includes('className="rank-board-pager"') &&
    rankBoardPage.includes("offset: pageOffset") &&
    rankBoardPage.includes("limit: RANK_PAGE_SIZE") &&
    !rankBoardPage.includes("function RankBoardPager(") &&
    !rankBoardPage.includes("limit: 200"),
  "rank board page must use paged API requests instead of a fixed 200-row slice",
);
assert(
  rankBoardPage.includes("OptionChipButton") &&
    rankBoardPage.includes('className="rank-facet-button"') &&
    rankBoardPage.includes('className="rank-facet-toggle"') &&
    rankBoardPage.includes("const visibleItems = expanded ? items : items.slice(0, 5)") &&
    rankBoardPage.includes("items.length > 5") &&
    rankBoardPage.includes('onClick={() => onSelect(active ? "all" : item.value)}') &&
    rankBoardPage.includes("onFilterChange.task") &&
    rankBoardPage.includes("onFilterChange.benchmark") &&
    rankBoardPage.includes("onFilterChange.status") &&
    rankBoardPage.includes("onFilterChange.label") &&
    rankBoardPage.includes("onFilterChange.metricProfile") &&
    rankBoardPage.includes("board.facets.tasks") &&
    rankBoardPage.includes("board.facets.benchmarks") &&
    rankBoardPage.includes("board.facets.statuses"),
  "rank board facet rail must expose all backend facets as clickable filter chips",
);
assert(
  rankBoardPage.includes("primaryMetricLabel") &&
    rankBoardPage.includes('primaryMetric !== "f1_iou50"') &&
    rankBoardPage.includes('className="rank-primary-score"') &&
    rankBoardPage.includes("formatScoreDelta(row.original.score_delta)") &&
    rankBoardPage.includes("function rankDeltaClassName") &&
    !rankBoardPage.includes('header: "Weighted"'),
  "rank board table must render the active primary metric and leader-relative delta columns",
);
assert(
  rankBoardPage.includes("function RankDecisionPanel(") &&
    rankBoardPage.includes('className="rank-decision-panel"') &&
    rankBoardPage.includes("const RANK_PRIMARY_METRICS = [") &&
    rankBoardPage.includes("const RANK_AUXILIARY_SORTS = [") &&
    rankBoardPage.includes("const RANK_DIRECT_METRICS = [...RANK_PRIMARY_METRICS, ...RANK_AUXILIARY_SORTS];") &&
    rankBoardPage.includes('className="rank-sort-section"') &&
    rankBoardPage.includes('className="rank-sort-section auxiliary"') &&
    rankBoardPage.includes('aria-label="排行榜主指标"') &&
    rankBoardPage.includes('aria-label="排行榜辅助排序字段"') &&
    rankBoardPage.includes('className="rank-sort-chip primary"') &&
    rankBoardPage.includes('className="rank-sort-chip auxiliary"') &&
    rankBoardPage.includes('className="rank-top-panel"') &&
    rankBoardPage.includes('className="rank-spread-panel"') &&
    !rankBoardPage.includes('id: "rank-sort-by"') &&
    !rankBoardPage.includes('id: "rank-sort-order"'),
  "rank board primary metric controls must live in the visible rank decision panel, not inside advanced filters",
);
assert(
  rankBoardPage.includes("DisclosurePanel") &&
    rankBoardPage.includes('className="rank-scheme-panel"') &&
    !/<details\b/.test(rankBoardPage) &&
    !/<summary\b/.test(rankBoardPage),
  "rank weighted scheme panel must use DisclosurePanel instead of a local details shell",
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
const viewerMetrics = await readSource("src/viewerMetrics.ts");
assert(
  viewerPanels.includes('import { CompactSelectControl, ToggleButton } from "./controlPrimitives";'),
  "viewer layer preset select must use CompactSelectControl",
);
assert(
  viewerPanels.includes("OptionChipButton"),
  "viewer label chips must import OptionChipButton",
);
assert(
  viewerPanels.includes("DisclosurePanel") &&
    viewerPanels.includes('className="control-popover"') &&
    !/<details\b/.test(viewerPanels) &&
    !/<summary\b/.test(viewerPanels),
  "viewer control popovers must use DisclosurePanel instead of local details shells",
);
assert(
  !sampleViewer.includes("visibleLabelMetrics") &&
    !sampleViewer.includes("LabelMetricTable") &&
    !viewerPanels.includes("LabelMetricTable") &&
    !viewerPanels.includes('className="label-metric-card"') &&
    !viewerPanels.includes("P@.50") &&
    !viewerPanels.includes("R@.50"),
  "viewer must not expose a resident per-label metric table in the sample inspector",
);
assert(
  !viewerMetrics.includes("matchedCount") &&
    !viewerMetrics.includes("falsePositiveCount") &&
    !viewerMetrics.includes("falseNegativeCount") &&
    !viewerMetrics.includes("meanIou"),
  "viewer visible metrics must stay count-only so fine metrics cannot leak back into the side strip",
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
