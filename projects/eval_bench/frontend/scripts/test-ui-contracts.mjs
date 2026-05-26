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
  assertNoRawButtonElement(source, relativePath);
  assertNoRawInputOutsidePrimitives(source, relativePath);
  assertNoRawSelectOutsidePrimitives(source, relativePath);
  assertNoRawTextareaOutsidePrimitives(source, relativePath);
  assertNoRawDisclosureElement(source, relativePath);
}

const jobsPage = await readSource("src/jobsPage.tsx");
const runArtifactSignals = await readSource("src/runArtifactSignals.ts");
const uiSource = await readSource("src/ui.tsx");
const apiSource = await readSource("src/api.ts");
const filterControls = await readSource("src/filterControls.tsx");
const controlPrimitives = await readSource("src/controlPrimitives.tsx");
const labelSubtaskControls = await readSource("src/labelSubtaskControls.tsx");
const samplePagerSource = await readSource("src/samplePager.tsx");
const styleSource = await readSource("src/styles.css");
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
  filterControls.includes("FilterSelectControl") &&
    filterControls.includes("SearchInputControl") &&
    filterControls.includes("TextInputControl") &&
    filterControls.includes('import { ActionButton, DIALOG_FOCUSABLE_SELECTOR, PanelToggleButton } from "./ui";') &&
    filterControls.includes("<FilterSelectControl") &&
    filterControls.includes("<SearchInputControl") &&
    filterControls.includes("<TextInputControl") &&
    !/<input\b/.test(filterControls) &&
    !/<select\b/.test(filterControls) &&
    controlPrimitives.includes("export function FilterSelectControl(") &&
    controlPrimitives.includes("export function SearchInputControl(") &&
    filterControls.includes("function resetAdvancedFilters()") &&
    filterControls.includes("function resetAdvancedFilter(") &&
    filterControls.includes("function openAdvancedFilter()") &&
    filterControls.includes("function closeAdvancedFilter(") &&
    filterControls.includes("function toggleAdvancedFilter()") &&
    filterControls.includes("function defaultFilterValue(") &&
    filterControls.includes("function groupAdvancedControls(") &&
    filterControls.includes("const ADVANCED_FILTER_CONTROL_FOCUS_SELECTOR = [") &&
    filterControls.includes(".advanced-filter-controls input:not([disabled])") &&
    filterControls.includes("const popoverRef = useRef<HTMLDivElement | null>(null);") &&
    filterControls.includes("const previouslyFocusedRef = useRef<HTMLElement | null>(null);") &&
    filterControls.includes("event.key !== \"Tab\"") &&
    filterControls.includes("previouslyFocusedRef.current?.focus()") &&
    filterControls.includes('aria-haspopup="dialog"') &&
    filterControls.includes('className="advanced-filter-popover"') &&
    filterControls.includes("tabIndex={-1}") &&
    filterControls.includes('className="advanced-filter-directory"') &&
    filterControls.includes('className="advanced-filter-token"') &&
    filterControls.includes('className="advanced-filter-clear"') &&
    filterControls.includes("onClick={() => resetAdvancedFilter(filter.control)}") &&
    filterControls.includes("<PanelToggleButton") &&
    !/<button[\s\S]{0,260}advanced-filter-head/.test(filterControls),
  "advanced filter reset, token clear, popup layout, and grouping must be centralized in AdvancedFilterBar",
);
assert(
  !styleSource.includes(".filter-bar"),
  "legacy filter-bar CSS must not return; page filters must use AdvancedFilterBar",
);
assert(
  controlPrimitives.includes("export function TextInputControl(") &&
    controlPrimitives.includes("export function SearchInputControl(") &&
    controlPrimitives.includes("export function NumberInputControl(") &&
    controlPrimitives.includes("export function TextareaControl(") &&
    controlPrimitives.includes("export function StandaloneTextareaControl(") &&
    controlPrimitives.includes("export function StandaloneTextInputControl(") &&
    controlPrimitives.includes("export function CheckboxFieldControl("),
  "dialog and form fields must share text, number, textarea, and checkbox primitives",
);
assert(
  controlPrimitives.includes("export function StandaloneCheckboxControl(") &&
    controlPrimitives.includes("export function StandaloneColorControl(") &&
    controlPrimitives.includes("export function InlineColorControl("),
  "table selection and color controls must share standalone primitives",
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
  uiSource.includes("export function NavigationCardAnchor(") &&
    uiSource.includes("export function NavigationCardFrame("),
  "card-style navigation rows must share NavigationCardAnchor/NavigationCardFrame",
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
  jobsPage.includes("CompactSelectControl"),
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
    jobsPage.includes("TextareaControl") &&
    jobsPage.includes('className="manifest-editor-field"') &&
    !/<details\b/.test(jobsPage) &&
    !/<summary\b/.test(jobsPage) &&
    !/<textarea\b/.test(jobsPage),
  "jobs prompt template panel and manifest editor must use shared disclosure/textarea controls",
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
  settingsControls.includes("StandaloneTextInputControl") &&
    settingsControls.includes("StandaloneColorControl") &&
    !/<input\b/.test(settingsControls),
  "settings label quick-add must use standalone text/color primitives",
);
assert(
  settingsControls.includes(
    '<ActionButton variant="secondary" className="settings-inline-action" onClick={onResetAll}>',
  ),
  "shortcut reset-all action must use ActionButton",
);
assertNoRawSelectElement(settingsControls, "settingsControls.tsx");
assert(
  settingsControls.includes("FormSelectControl") &&
    settingsControls.includes('className="inline-select-control"') &&
    settingsControls.includes("hideLabel"),
  "settings inline label color role select must use FormSelectControl",
);

const settingsPage = await readSource("src/settingsPage.tsx");
assert(
  settingsPage.includes("CompactSelectControl") &&
    settingsPage.includes("NumberSettingControl") &&
    settingsPage.includes("SearchInputControl"),
  "settings page selects must use CompactSelectControl",
);
assert(
  /<CompactSelectControl\s+dense\s+label="预测线型"/.test(settingsPage),
  "settings prediction line style select must use CompactSelectControl",
);
assert(
  settingsPage.includes("<SearchInputControl") &&
    settingsPage.includes('className="settings-search-box"') &&
    settingsPage.includes('className="settings-search-clear"') &&
    !/<input[\s\S]{0,160}settingsQuery/.test(settingsPage),
  "settings search must use SearchInputControl and IconActionButton",
);
assert(
  settingsPage.includes("InlineColorControl") &&
    !/<input\b/.test(settingsPage),
  "settings label color grid must use InlineColorControl instead of raw color inputs",
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
const designSource = await readSource("src/design.css");
const formattersSource = await readSource("src/formatters.ts");
assert(
  apiSource.includes("export type TargetLabelResolution =") &&
    apiSource.includes("export type TargetLabelResolutionParams =") &&
    apiSource.includes("export function fetchTargetLabelResolution(") &&
    apiSource.includes('params.append("target_label", value)') &&
    apiSource.includes('fetchJson<TargetLabelResolution>(`/api/target-labels'),
  "api client must expose agent-safe target label resolution endpoint",
);
assert(
  overviewPage.includes("export function OverviewPage()"),
  "overview page module must export OverviewPage",
);
assert(
  overviewPage.includes("overview-home-v17") &&
    overviewPage.includes("overview-ops-board") &&
    overviewPage.includes("overview-rank-console") &&
    overviewPage.includes("overview-evidence-row") &&
    overviewPage.includes("overview-loop-panel") &&
    overviewPage.includes("OverviewDecisionMetrics") &&
    overviewPage.includes("overview-decision-metrics") &&
    overviewPage.includes("overview-decision-metric") &&
    overviewPage.includes("overview-decision-icon") &&
    overviewPage.includes("OverviewTelemetryTrace") &&
    overviewPage.includes("overview-telemetry-trace") &&
    overviewPage.includes("overview-telemetry-bar") &&
    overviewPage.includes("overview-resource-chips") &&
    overviewPage.includes("OverviewStateStrip") &&
    overviewPage.includes("overview-state-strip") &&
    !overviewPage.includes("overviewHeroTitle") &&
    !overviewPage.includes("可以看排行") &&
    !overviewPage.includes("可以进入排行") &&
    !overviewPage.includes("查看排行榜") &&
    !overviewPage.includes("等待报告进入排行") &&
    !overviewPage.includes("主指标 F1 可排行") &&
    !overviewPage.includes("从样本到排行") &&
    !overviewPage.includes("rankable") &&
    !overviewPage.includes("F1 ready") &&
    overviewPage.includes("OverviewScoreDial") &&
    overviewPage.includes("overview-score-dial") &&
    overviewPage.includes("OverviewRunFocus") &&
    overviewPage.includes("overview-run-focus") &&
    overviewPage.includes("bestF1Run") &&
    overviewPage.includes('import { formatMetric, runF1Score } from "./formatters";') &&
    overviewPage.includes("OverviewOpsSignal") &&
    overviewPage.includes("overview-ops-signal") &&
    overviewPage.includes("OverviewFlowSpine") &&
    overviewPage.includes("overview-flow-spine") &&
    overviewPage.includes("overview-flow-node") &&
    overviewPage.includes("overviewPostureLine") &&
    overviewPage.includes("recentRunsByCreatedAt(data.runs") &&
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
    !overviewPage.includes("overview-home-v15") &&
    !overviewPage.includes("overview-home-v16") &&
    !overviewPage.includes("overview-pulse-panel") &&
    !overviewPage.includes("overview-operating-row") &&
    !overviewPage.includes("OverviewSignalStack") &&
    !overviewPage.includes("overview-signal-stack") &&
    !overviewPage.includes("overview-signal-card") &&
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
  styleSource.includes("Overview v17: decision-first command desk") &&
    styleSource.includes(".overview-home-v17 .overview-ops-board") &&
    styleSource.includes(".overview-home-v17 .overview-decision-metric:hover") &&
    styleSource.includes(".overview-home-v17 .overview-decision-icon") &&
    styleSource.includes(".overview-home-v17 .overview-telemetry-trace") &&
    styleSource.includes(".overview-home-v17 .overview-telemetry-bar:hover") &&
    styleSource.includes(".overview-home-v17 .overview-resource-chips") &&
    styleSource.includes("@keyframes overview-v17-scan") &&
    styleSource.includes("@keyframes overview-v17-live") &&
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
    "overview-home-v15",
    "overview-home-v16",
    "overview-command-shell",
    "overview-command-deck",
    "overview-pulse-panel",
    "overview-operating-row",
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
    "overview-signal-stack",
    "overview-signal-card",
    "overview-route-panel",
    "overview-activity-matrix",
    "overview-chart-matrix",
    "overview-mini-chart"
  ].some((token) => styleSource.includes(token)),
  "overview stylesheet must expose the active v16 surface and block deprecated design tracks",
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
  benchmarksPage.includes('import { CheckboxFieldControl, TextInputControl } from "./controlPrimitives";') &&
    (benchmarksPage.match(/<TextInputControl/g) ?? []).length >= 5 &&
    (benchmarksPage.match(/<CheckboxFieldControl/g) ?? []).length >= 3,
  "benchmark creation dialog must use shared text and checkbox form controls",
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
  runsPage.includes("CheckboxFieldControl") &&
    runsPage.includes("TextInputControl") &&
    (runsPage.match(/<TextInputControl/g) ?? []).length >= 6 &&
    (runsPage.match(/<CheckboxFieldControl/g) ?? []).length >= 3,
  "run import dialog must use shared text and checkbox form controls",
);
assert(
  runsPage.includes("StandaloneTextareaControl") &&
    (runsPage.match(/<StandaloneTextareaControl/g) ?? []).length >= 2 &&
    !/<textarea\b/.test(runsPage),
  "run note editor must use shared standalone textarea controls",
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
    runsPage.includes("appendRunNote(run.run_id, note, heading, noteVersion)") &&
    runsPage.includes("const appendMutation = useMutation(") &&
    runsPage.includes('className="run-note-append-panel"') &&
    runsPage.includes('label="追加 run note"') &&
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
const comparePage = await readSource("src/comparePage.tsx");
assert(
  runTables.includes("StandaloneCheckboxControl") &&
    runTables.includes('className="row-select-checkbox"') &&
    !/<input\b/.test(runTables),
  "run table row selection must use StandaloneCheckboxControl",
);
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
    runTables.includes("formatMetric(runF1Score(row.original))") &&
    comparePage.includes("runF1Score") &&
    comparePage.includes('className="compare-run-primary-metric"') &&
    comparePage.includes("F1 {formatMetric(runF1Score(selected))}"),
  "run option labels, run tables, and compare run cards must foreground F1 as the default direct metric",
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
    runTables.includes("InlineAnchor") &&
    !/<a[^>]+className="mini-link/.test(compareMiniLinkSource) &&
    !/<a[^>]+className="mini-link/.test(comparisonSampleMiniLinkSource) &&
    !/<a[^>]+className=\{[^}]*mini-link/.test(runTables) &&
    !runTables.includes('"mini-link compare-ready"'),
  "href mini links must use InlineAnchor instead of ad hoc mini-link anchors",
);
assertNoLegacyFormSubmitClass(runsPage, "runsPage.tsx");
assertNoRawSelectElement(runsPage, "runsPage.tsx");
assert(
  runsPage.includes("FormSelectControl") &&
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
assert(
  servicesPage.includes("TextInputControl") &&
    servicesPage.includes("NumberInputControl") &&
    (servicesPage.match(/<TextInputControl/g) ?? []).length >= 5 &&
    (servicesPage.match(/<NumberInputControl/g) ?? []).length >= 5,
  "service registration dialog must use shared text and number form controls",
);
assertNoLegacyFormSubmitClass(servicesPage, "servicesPage.tsx");
assertNoRawSelectElement(servicesPage, "servicesPage.tsx");
assert(
  servicesPage.includes("FormSelectControl") &&
    (servicesPage.match(/<FormSelectControl/g) ?? []).length >= 1,
  "service registration dialog selects must use FormSelectControl",
);
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
assert(
  comparePage.includes("NavigationCardAnchor") &&
    comparePage.includes("NavigationCardFrame") &&
    comparePage.includes('<NavigationCardAnchor\n                className="comparison-sample-row"') &&
    comparePage.includes('<NavigationCardFrame className="comparison-sample-row disabled"') &&
    !/<a[\s\S]{0,160}className="comparison-sample-row"/.test(comparePage) &&
    !/<div[\s\S]{0,120}className="comparison-sample-row disabled"/.test(comparePage),
  "compare sample navigation rows must use shared navigation card primitives",
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
  styleSource.includes(".rank-facet-group.expanded > div") &&
    styleSource.includes("max-height: 126px") &&
    styleSource.includes("flex-wrap: wrap") &&
    styleSource.includes("overflow: auto"),
  "expanded rank board facets must wrap inside a bounded scroll pane instead of stretching the page",
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
    rankBoardPage.includes("StandaloneTextareaControl") &&
    rankBoardPage.includes("ToggleButton") &&
    rankBoardPage.includes('className="rank-scheme-panel"') &&
    !/<details\b/.test(rankBoardPage) &&
    !/<summary\b/.test(rankBoardPage) &&
    !/<textarea\b/.test(rankBoardPage) &&
    !/<input\b/.test(rankBoardPage),
  "rank weighted scheme panel must use shared disclosure, toggle, and textarea controls",
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

function assertNoRawButtonElement(source, relativePath) {
  if (relativePath === "src/ui.tsx") {
    return;
  }
  assert(!/<button\b/.test(source), `${relativePath}: buttons must use shared UI primitives`);
}

function assertNoRawInputOutsidePrimitives(source, relativePath) {
  if (relativePath === "src/controlPrimitives.tsx") {
    return;
  }
  assert(!/<input\b/.test(source), `${relativePath}: inputs must use controlPrimitives`);
}

function assertNoRawSelectOutsidePrimitives(source, relativePath) {
  if (relativePath === "src/controlPrimitives.tsx") {
    return;
  }
  assert(!/<select\b/.test(source), `${relativePath}: selects must use controlPrimitives`);
}

function assertNoRawTextareaOutsidePrimitives(source, relativePath) {
  if (relativePath === "src/controlPrimitives.tsx") {
    return;
  }
  assert(!/<textarea\b/.test(source), `${relativePath}: textareas must use controlPrimitives`);
}

function assertNoRawDisclosureElement(source, relativePath) {
  if (relativePath === "src/ui.tsx") {
    return;
  }
  assert(!/<(?:details|summary)\b/.test(source), `${relativePath}: disclosures must use DisclosurePanel`);
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
