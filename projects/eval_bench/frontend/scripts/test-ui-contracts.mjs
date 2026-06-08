import { strict as assert } from "node:assert";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const srcRoot = path.join(root, "src");
const sourceFiles = await collectSourceFiles(srcRoot);
const styleFiles = await collectStyleFiles(srcRoot);
const maxSourceLines = 900;
const maxStyleLines = 480;
const forbiddenUiCopy = [
  "只保留系统运行态、数据规模和近期写入节奏",
  "精细指标进入排行榜与对比页",
  "首页只保留",
  "可以看排行",
  "可以进入排行",
  "查看排行榜",
  "等待报告进入排行",
  "主指标 F1 可排行",
  "从样本到排行",
  "Run 写入节奏",
  "写入节奏",
  "rankable",
  "F1 ready"
];

for (const filePath of sourceFiles) {
  const source = await readFile(filePath, "utf8");
  const relativePath = path.relative(root, filePath);
  assertMaxLines(source, relativePath, maxSourceLines);
  assertNoForbiddenUiCopy(source, relativePath, forbiddenUiCopy);
  assertNoBlockingBrowserDialogs(source, relativePath);
  assertNoBusinessDialogShell(source, relativePath);
  assertNoLegacySampleFilters(source, relativePath);
  assertNoRawButtonElement(source, relativePath);
  assertNoRawInputOutsidePrimitives(source, relativePath);
  assertNoRawSelectOutsidePrimitives(source, relativePath);
  assertNoRawTextareaOutsidePrimitives(source, relativePath);
  assertNoRawDisclosureElement(source, relativePath);
  assertQueryFnsUseAbortSignal(source, relativePath);
}

for (const filePath of styleFiles) {
  const source = await readFile(filePath, "utf8");
  const relativePath = path.relative(root, filePath);
  assertMaxLines(source, relativePath, maxStyleLines);
  assert.notEqual(relativePath, "src/styles.css", "legacy monolithic styles.css must stay removed");
}

const jobsPage = await readSource("src/jobsPage.tsx");
const jobsCreatePanelSource = await readSource("src/jobsCreatePanel.tsx");
const jobsQueuePanelSource = await readSource("src/jobsQueuePanel.tsx");
const jobsQueueTableSource = await readSource("src/jobsQueueTable.tsx");
const runsPage = await readSource("src/runsPage.tsx");
const runDetailPageSource = await readSource("src/runDetailPage.tsx");
const benchmarksPage = await readSource("src/benchmarksPage.tsx");
const benchmarkCreatePanelSource = await readSource("src/benchmarkCreatePanel.tsx");
const benchmarkModelSource = await readSource("src/benchmarkModel.ts");
const benchmarkSampleInspectorSource = await readSource("src/benchmarkSampleInspector.tsx");
const comparePage = await readSource("src/comparePage.tsx");
const compareControllerSource = await readSource("src/compareController.ts");
const compareFiltersSource = await readSource("src/compareFilters.tsx");
const compareReportComponentsSource = await readSource("src/compareReportComponents.tsx");
const compareReportMetricsSource = await readSource("src/compareReportMetrics.tsx");
const compareReportSamplesSource = await readSource("src/compareReportSamples.tsx");
const compareRunRailComponentsSource = await readSource("src/compareRunRailComponents.tsx");
const suiteReportPage = await readSource("src/suiteReportPage.tsx");
const overviewPage = await readSource("src/overviewPage.tsx");
const overviewModelSource = await readSource("src/overviewModel.ts");
const dashboardStateSource = await readSource("src/dashboardState.ts");
const rankBoardPage = await readSource("src/rankBoardPage.tsx");
const rankThemeStyleSource = await readSource("src/rankTheme.css");
const rankBoardPageStyleSource = await readSource("src/rankBoardPage.css");
const rankBoardSummaryStyleSource = await readSource("src/rankBoardSummary.css");
const rankBoardFacetsStyleSource = await readSource("src/rankBoardFacets.css");
const rankBoardTablesStyleSource = await readSource("src/rankBoardTables.css");
const rankBoardControllerSource = await readSource("src/rankBoardController.ts");
const rankBoardFiltersSource = await readSource("src/rankBoardFilters.tsx");
const rankBoardFacetsSource = await readSource("src/rankBoardFacets.tsx");
const servicesPage = await readSource("src/servicesPage.tsx");
const servicesCreatePanelSource = await readSource("src/servicesCreatePanel.tsx");
const servicesGridSource = await readSource("src/servicesGrid.tsx");
const mainEntry = await readSource("src/main.tsx");
const appShellSource = await readSource("src/appShell.tsx");
const routePrefetchSource = await readSource("src/routePrefetch.ts");
const routeWarmupSource = await readSource("src/routeWarmup.ts");
const runTables = await readSource("src/runTables.tsx");
const runsImportPanelSource = await readSource("src/runsImportPanel.tsx");
const runConfigPanelSource = await readSource("src/runConfigPanel.tsx");
const runSampleSidebarSource = await readSource("src/runSampleSidebar.tsx");
const runArtifactSignals = await readSource("src/runArtifactSignals.ts");
const uiSource = await readSource("src/ui.tsx");
const uiActionsSource = await readSource("src/uiActions.tsx");
const uiDataTableSource = await readSource("src/uiDataTable.tsx");
const uiDialogSource = await readSource("src/uiDialog.tsx");
const manifestToolsSource = await readSource("src/manifestTools.ts");
const apiSource = await readSource("src/api.ts");
const apiTypesSource = await readSource("src/apiTypes.ts");
const formattersSource = await readSource("src/formatters.ts");
const filterControls = await readSource("src/filterControls.tsx");
const advancedFilterTypesSource = await readSource("src/advancedFilterTypes.ts");
const advancedFilterFieldsSource = await readSource("src/advancedFilterFields.tsx");
const advancedFilterModelSource = await readSource("src/advancedFilterModel.ts");
const advancedFilterStorageSource = await readSource("src/advancedFilterStorage.ts");
const workspaceSettingsSource = await readSource("src/workspaceSettings.ts");
const workspaceSettingsSchemaSource = await readSource("src/workspaceSettingsSchema.ts");
const workspaceSettingsStorageSource = await readSource("src/workspaceSettingsStorage.ts");
const typographySettingsSource = await readSource("src/typographySettings.ts");
const controlPrimitives = await readSource("src/controlPrimitives.tsx");
const selectPopoverControl = await readSource("src/selectPopoverControl.tsx");
const selectPopoverModelSource = await readSource("src/selectPopoverModel.ts");
const selectPopoverModelCheckSource = await readSource("scripts/test-select-popover-model.mjs");
const selectPopoverSmokeSource = await readSource("scripts/select-popover-smoke-check.mjs");
const useDebouncedValueSource = await readSource("src/useDebouncedValue.ts");
const themeToggleCheckSource = await readSource("scripts/theme-toggle-check.mjs");
const rankBoardViewStateSource = await readSource("src/rankBoardViewState.ts");
const rankBoardModelSource = await readSource("src/rankBoardModel.ts");
const rankBoardTablesSource = await readSource("src/rankBoardTables.tsx");
const runsViewStateSource = await readSource("src/runsViewState.ts");
const jobsViewStateSource = await readSource("src/jobsViewState.ts");
const compareViewStateSource = await readSource("src/compareViewState.ts");
const labelSubtaskControls = await readSource("src/labelSubtaskControls.tsx");
const labelSubtaskControlsStyleSource = await readSource("src/labelSubtaskControls.css");
const samplePagerSource = await readSource("src/samplePager.tsx");
const styleSource = await readCssSource();
const appBaseStyleSource = await readSource("src/appBase.css");
const appChromeStyleSource = await readSource("src/appChrome.css");
const appChromeVisualStyleSource = await readSource("src/appChromeVisual.css");
const appChromeCollapsedStyleSource = await readSource("src/appChromeCollapsed.css");
const appTypographyStyleSource = await readSource("src/appTypography.css");
const interactionFeedbackStyleSource = await readSource("src/interactionFeedback.css");
const sharedControlsThemeStyleSource = await readSource("src/sharedControlsTheme.css");
const sharedControlsStyleSource = await readSource("src/sharedControls.css");
const sharedButtonsStyleSource = await readSource("src/sharedButtons.css");
const sharedIndicatorsStyleSource = await readSource("src/sharedIndicators.css");
const sharedMetricsStyleSource = await readSource("src/sharedMetrics.css");
const sharedPagerStyleSource = await readSource("src/sharedPager.css");
const sharedSplitStyleSource = await readSource("src/sharedSplit.css");
const controlPrimitiveStyleSource = await readSource("src/controlPrimitiveStyles.css");
const selectPopoverStyleSource = await readSource("src/selectPopover.css");
const labelColorControlsStyleSource = await readSource("src/labelColorControls.css");
const appThemeStyleSource = await readSource("src/appTheme.css");
const themeSurfaceOverridesStyleSource = await readSource("src/themeSurfaceOverrides.css");
const adaptiveContentStyleSource = await readSource("src/adaptiveContent.css");
const dataTableStyleSource = await readSource("src/dataTable.css");
const runTablesStyleSource = await readSource("src/runTables.css");
const runsStyleSource = await readSource("src/runsPage.css");
const servicesPageStyleSource = await readSource("src/servicesPage.css");
const jobsStyleSource = await readSource("src/jobsPage.css");
const jobsQueueStyleSource = await readSource("src/jobsQueue.css");
const jobsRecentRunsStyleSource = await readSource("src/jobsRecentRuns.css");
const jobsDetailStyleSource = await readSource("src/jobsDetail.css");
const jobsManifestStyleSource = await readSource("src/jobsManifest.css");
const formControlsStyleSource = await readSource("src/formControls.css");
const filterThemeStyleSource = await readSource("src/filterTheme.css");
const filterControlsStyleSource = await readSource("src/filterControls.css");
const workspaceThemeStyleSource = await readSource("src/workspaceTheme.css");
const workspaceShellStyleSource = await readSource("src/workspaceShell.css");
const workspaceDialogStyleSource = await readSource("src/workspaceDialog.css");
const pageCommandStyleSource = await readSource("src/pageCommand.css");
const settingsThemeStyleSource = await readSource("src/settingsTheme.css");
const settingsWorkbenchStyleSource = await readSource("src/settingsWorkbench.css");
const settingsPreviewStyleSource = await readSource("src/settingsPreview.css");
const settingsDrawerStyleSource = await readSource("src/settingsDrawer.css");
const settingsEditorStyleSource = await readSource("src/settingsEditor.css");
const settingsTypographyStyleSource = await readSource("src/settingsTypography.css");
const settingsLabelsStyleSource = await readSource("src/settingsLabels.css");
const settingsShortcutsStyleSource = await readSource("src/settingsShortcuts.css");
const overviewStyleSource = await readSource("src/overviewPage.css");
const overviewShellStyleSource = await readSource("src/overviewShell.css");
const overviewPrimaryStyleSource = await readSource("src/overviewPrimary.css");
const overviewConsoleStyleSource = await readSource("src/overviewConsole.css");
const overviewOperationsStyleSource = await readSource("src/overviewOperations.css");
const overviewResponsiveStyleSource = await readSource("src/overviewResponsive.css");
const compareStyleSource = await readSource("src/comparePage.css");
const compareThemeStyleSource = await readSource("src/compareTheme.css");
const compareRunRailStyleSource = await readSource("src/compareRunRail.css");
const compareReportPanelStyleSource = await readSource("src/compareReportPanel.css");
const comparisonSampleStyleSource = await readSource("src/comparisonSampleStyles.css");
const inspectorPageStyleSource = await readSource("src/inspectorPage.css");
const compositeThemeStyleSource = await readSource("src/compositeTheme.css");
const compositeMicroMeterStyleSource = await readSource("src/compositeMicroMeter.css");
const compositePanelPrimitivesStyleSource = await readSource("src/compositePanelPrimitives.css");
const compositeReportStyleSource = await readSource("src/compositeReport.css");
const compositeComposerDockStyleSource = await readSource("src/compositeComposerDock.css");
const compositeComposerDockPreviewStyleSource = await readSource("src/compositeComposerDockPreview.css");
const compositeComposerDrawerStyleSource = await readSource("src/compositeComposerDrawer.css");
const compositeReportPanelStyleSource = await readSource("src/compositeReportPanel.css");
const compositeReportRunPoolStyleSource = await readSource("src/compositeReportRunPool.css");
const compositeReportLayerPlanStyleSource = await readSource("src/compositeReportLayerPlan.css");
const compositeImageNavigatorStyleSource = await readSource("src/compositeImageNavigator.css");
const compositeImageJumpControlStyleSource = await readSource("src/compositeImageJumpControl.css");
const compositeImageSearchBarStyleSource = await readSource("src/compositeImageSearchBar.css");
const compositeImagePanelStyleSource = await readSource("src/compositeImagePanel.css");
const compositeImageJumpItemStyleSource = await readSource("src/compositeImageJumpItem.css");
const compositeImageSearchResultItemStyleSource = await readSource("src/compositeImageSearchResultItem.css");
const compositeInteractionPaletteStyleSource = await readSource("src/compositeInteractionPalette.css");
const compositeImageAtlasStyleSource = await readSource("src/compositeImageAtlas.css");
const compositeImageTimelineStyleSource = await readSource("src/compositeImageTimeline.css");
const compositeImageIndexMeterStyleSource = await readSource("src/compositeImageIndexMeter.css");
const compositeImageScrubTrackStyleSource = await readSource("src/compositeImageScrubTrack.css");
const compositeImageNearbyRailStyleSource = "";
const compositeImageSearchPopoverStyleSource = await readSource("src/compositeImageSearchPopover.css");
const compositeImageSearchResultsStyleSource = await readSource("src/compositeImageSearchResults.css");
const compositeImageSearchPreviewStyleSource = await readSource("src/compositeImageSearchPreview.css");
const compositeImageSearchStatusStyleSource = await readSource("src/compositeImageSearchStatus.css");
const compositeReportStageStyleSource = await readSource("src/compositeReportStage.css");
const compositeStageWorkbenchStyleSource = await readSource("src/compositeStageWorkbench.css");
const compositeLayerFocusToolbarStyleSource = "";
const compositeObjectHudStyleSource = "";
const compositeObjectContextMenuStyleSource = await readSource("src/compositeObjectContextMenu.css");
const compositeOverlayStageStyleSource = await readSource("src/compositeOverlayStage.css");
const compositeLayerCanvasStyleSource = await readSource("src/compositeLayerCanvas.css");
const compositeCanvasOverlayStyleSource = await readSource("src/compositeCanvasOverlay.css");
const compositeCanvasGestureHudStyleSource = await readSource("src/compositeCanvasGestureHud.css");
const compositeCanvasPointerReticleStyleSource = await readSource("src/compositeCanvasPointerReticle.css");
const compositeLayerInspectorStyleSource = await readSource("src/compositeLayerInspector.css");
const compositeLayerObjectStripStyleSource = await readSource("src/compositeLayerObjectStrip.css");
const compositeSplitStageStyleSource = "";
const compositeSplitPaneStyleSource = "";
const compositeReportModelSource = await readSource("src/compositeReportModel.ts");
const compositeLayerPaletteSource = await readSource("src/compositeLayerPalette.ts");
const compositeImageNavigationModelSource = await readSource("src/compositeImageNavigationModel.ts");
const compositeImageNavigationControllerSource = await readSource(
  "src/compositeImageNavigationController.ts",
);
const compositeImageSearchControllerSource = await readSource("src/compositeImageSearchController.ts");
const compositeImageScrubTrack = await readSource("src/compositeImageScrubTrack.tsx");
const compositeImageNearbyRailControllerSource = "";
const compositeImageTimelineControllerSource = await readSource(
  "src/compositeImageTimelineController.ts",
);
const compositeObjectInteractionSource = await readSource("src/compositeObjectInteraction.ts");
const compositeObjectModelSource = await readSource("src/compositeObjectModel.ts");
const compositeObjectInteractionControllerSource = await readSource(
  "src/compositeObjectInteractionController.ts",
);
const compositeObjectKeyboardNavigationSource = await readSource("src/compositeObjectKeyboardNavigation.ts");
const compositeObjectContextMenuLifecycleSource = await readSource(
  "src/compositeObjectContextMenuLifecycle.ts",
);
const compositeCanvasObjectMappingSource = await readSource("src/compositeCanvasObjectMapping.ts");
const keyboardTargetsSource = await readSource("src/keyboardTargets.ts");
const designSource = await readSource("src/design.css");
const removedCompositeApiToken = ["rank", "scheme"].join("_");
const removedCompositeCamelToken = ["rank", "Scheme"].join("");
const removedCompositeClassToken = ["rank", "scheme"].join("-");
const removedCompositePanelToken = ["Rank", "Scheme", "Panel"].join("");
const removedCompositeComponentsToken = ["score", "components"].join("_");
const readProjectFile = (relativePath) => readFile(path.join(root, relativePath), "utf8");
const readRepoFile = (relativePath) => readFile(path.join(root, "..", "..", "..", relativePath), "utf8");
const packageJsonSource = await readProjectFile("package.json");
const shortcutCoverageSource = await readProjectFile("scripts/shortcut-coverage-check.mjs");
const viewerPerformanceSource = await readProjectFile("scripts/viewer-performance-check.mjs");
const dialogSmokeSource = await readProjectFile("scripts/dialog-smoke-check.mjs");
const toastSmokeSource = await readProjectFile("scripts/toast-smoke-check.mjs");
const routeWarmupSmokeSource = await readProjectFile("scripts/route-warmup-check.mjs");
const navPrefetchSmokeSource = await readProjectFile("scripts/nav-prefetch-check.mjs");
const loadingStateSmokeSource = await readProjectFile("scripts/loading-state-check.mjs");
const settingsPreviewSmokeSource = await readProjectFile("scripts/settings-preview-check.mjs");
const layoutSmokeSource = await readProjectFile("scripts/layout-smoke-check.mjs");
const compositeReportSmokeSource = await readProjectFile("scripts/composite-report-smoke-check.mjs");
const readmeSource = await readProjectFile("../README.md");
const scriptsDocSource = await readRepoFile("docs/scripts.md");
const evalBenchArchitectureSource = await readRepoFile("docs/eval_bench_architecture.md");
const rawVisualPageControlGeometryPattern =
  /(?:\bfont-size:\s*(?:\d|0\.|[0-9.]+rem)|\b(?:gap|padding):\s*(?:1|4|6|8|9|10|12|14)px\b|\bmin-height:\s*(?:28|34|38|52)px\b|\bborder-radius:\s*(?:2|3)px\b)/;
const rawAdvancedFilterGeometryPattern =
  /(?:\bfont-size:\s*(?:10|11|12|13)px\b|\bgap:\s*(?:2|3|5|6|8|10)px\b|\bpadding:\s*(?:0 7px|0 9px 0 7px|1px 1px 7px|4px|7px|8px)\b|\bmin-height:\s*(?:26|30|32)px\b|\bborder-radius:\s*2px\b)/;
const rawCompareReportGeometryPattern =
  /(?:\bfont-size:\s*(?:11|12|13|15|19|20)px\b|\bgap:\s*(?:3|4|6|8|10|12|16|18)px\b|\bpadding:\s*(?:0 8px|3px 0|4px 0|4px 7px|7px 0|7px 10px 8px 0|12px)\b|\b(?:height|line-height|min-height):\s*(?:26|30|34|44|56)px\b|\bborder-radius:\s*2px\b)/;
const rawRankBoardGeometryPattern =
  /(?:\bfont-size:\s*(?:10|11|15)px\b|\bgap:\s*(?:5|6|8|10)px\b|\bpadding:\s*(?:2px 8px|3px 6px|4px 7px|4px 8px|7px 10px|7px|8px|10px)\b|\bmin-height:\s*(?:24|28|30|32|38|46)px\b|\bborder-radius:\s*(?:2|3|4|999)px\b)/;
const rawWorkspaceShellGeometryPattern =
  /(?:\bfont-size:\s*(?:10|11|12|13|14|16)px\b|\bgap:\s*(?:2|8|10|12|14|16)px\b|\bpadding:\s*(?:10px|12px|14px|18px|24px|32px|0 10px|0 12px|10px 12px 10px 14px)\b|\bmin-height:\s*(?:32|44|46|54)px\b|\bborder-radius:\s*(?:2|3|8)px\b)/;
const rawSharedControlsGeometryPattern =
  /(?:\bfont-size:\s*(?:11|12|13|14|28)px\b|\bgap:\s*(?:2|4|6|8|10|12|16)px\b|\bpadding:\s*(?:8px|10px|14px|0 8px|0 10px|0 11px|0 12px|0 14px|2px 8px|7px 10px)\b|\bmin-height:\s*(?:24|26|28|30|32|34|36|42|76)px\b|\bborder-radius:\s*(?:2|3|4|999)px\b|\b(?:width|height):\s*(?:28|36|38)px\b)/;
assert(
  apiSource.includes("export class ApiError extends Error") &&
    apiSource.includes("export function isApiError(") &&
    apiSource.includes("export function apiErrorDetailText(") &&
    apiSource.includes("this.status = status") &&
    apiSource.includes("Array.isArray(value)") &&
    apiSource.includes("function errorDetailPayload(") &&
    apiSource.includes('\"detail\" in payload ? payload.detail : payload') &&
    apiSource.includes("function errorItemText(") &&
    apiSource.includes("Array.isArray(item.loc)") &&
    apiSource.includes("`${location}: ${message}`") &&
    apiSource.includes("detailText ? `: ${detailText}` : \"\"") &&
    apiSource.includes("JSON.stringify(detail)") &&
    apiSource.includes("throw error;"),
  "frontend API failures must expose typed ApiError status and structured details for recovery",
);
assert(
    mainEntry.includes('import "./appBase.css";') &&
    mainEntry.includes('import "./appChrome.css";') &&
    mainEntry.includes('import "./sharedControls.css";') &&
    mainEntry.includes('import "./labelColorControls.css";') &&
    mainEntry.includes('import "./appTheme.css";') &&
    mainEntry.includes('import "./appChromeVisual.css";') &&
    mainEntry.includes('import "./appChromeCollapsed.css";') &&
    mainEntry.includes('import "./design.css";') &&
    mainEntry.includes('import "./interactionFeedback.css";') &&
    mainEntry.includes('import "./appTypography.css";') &&
    mainEntry.includes('import "./viewerTheme.css";') &&
    mainEntry.includes('import "./compositeTheme.css";') &&
    mainEntry.includes('import { AppErrorBoundary, AppShell } from "./appShell";') &&
    appShellSource.includes("bootstrapTypographySettings") &&
    appShellSource.includes("useTypographySettings();") &&
    appShellSource.includes("bootstrapTypographySettings();") &&
    mainEntry.includes('import "./adaptiveContent.css";') &&
    mainEntry.indexOf('import "./appBase.css";') < mainEntry.indexOf('import "./appTheme.css";') &&
    mainEntry.indexOf('import "./appChrome.css";') < mainEntry.indexOf('import "./appTheme.css";') &&
    mainEntry.indexOf('import "./sharedControls.css";') < mainEntry.indexOf('import "./appTheme.css";') &&
    mainEntry.indexOf('import "./labelColorControls.css";') <
      mainEntry.indexOf('import "./appTheme.css";') &&
    mainEntry.indexOf('import "./appTheme.css";') <
      mainEntry.indexOf('import "./appChromeVisual.css";') &&
    mainEntry.indexOf('import "./appChromeVisual.css";') <
      mainEntry.indexOf('import "./appChromeCollapsed.css";') &&
    mainEntry.indexOf('import "./appChromeCollapsed.css";') <
      mainEntry.indexOf('import "./design.css";') &&
    mainEntry.indexOf('import "./design.css";') <
      mainEntry.indexOf('import "./interactionFeedback.css";') &&
    mainEntry.indexOf('import "./interactionFeedback.css";') <
      mainEntry.indexOf('import "./appTypography.css";') &&
    mainEntry.indexOf('import "./appTypography.css";') <
      mainEntry.indexOf('import "./viewerTheme.css";') &&
    mainEntry.indexOf('import "./viewerTheme.css";') <
      mainEntry.indexOf('import "./compositeTheme.css";') &&
    mainEntry.indexOf('import "./compositeTheme.css";') <
      mainEntry.indexOf('import "./adaptiveContent.css";') &&
    appChromeCollapsedStyleSource.includes(".app-shell.sidebar-collapsed") &&
    appChromeCollapsedStyleSource.includes(".sidebar.collapsed .brand-logo") &&
    appChromeCollapsedStyleSource.includes(".sidebar-toggle:hover") &&
    appBaseStyleSource.includes("--app-font-family") &&
    appBaseStyleSource.includes("--app-base-font-size: 12px") &&
    appBaseStyleSource.includes("font-family: var(--app-font-family)") &&
    appBaseStyleSource.includes("font-size: var(--app-base-font-size)") &&
    appThemeStyleSource.includes("font-family: var(--app-font-family)") &&
    appThemeStyleSource.includes("font-size: var(--app-base-font-size)") &&
    designSource.includes("--bench-bg") &&
    designSource.includes("--bench-action") &&
    designSource.includes("--bench-shadow-tight") &&
    interactionFeedbackStyleSource.includes("outline: 2px solid var(--control-focus-outline)") &&
    interactionFeedbackStyleSource.includes("border-color: var(--control-focus-line)") &&
    !interactionFeedbackStyleSource.includes(".primary-button {\n  background:") &&
    !interactionFeedbackStyleSource.includes(".primary-button:hover {\n  background:") &&
    !interactionFeedbackStyleSource.includes("transform 160ms ease") &&
    !interactionFeedbackStyleSource.includes("transform: none") &&
    interactionFeedbackStyleSource.includes("@media (prefers-reduced-motion: reduce)") &&
    interactionFeedbackStyleSource.includes(".search-box:focus-within") &&
    appTypographyStyleSource.includes("--text-3xs") &&
    appTypographyStyleSource.includes("--text-xs") &&
    appTypographyStyleSource.includes("--metric-display-size") &&
    appTypographyStyleSource.includes("font-family: var(--app-font-family)") &&
    appTypographyStyleSource.includes("h1 {\n  font-size: var(--text-xl);") &&
    appTypographyStyleSource.includes(".topbar h1 {\n  font-size: var(--text-lg);") &&
    !appTypographyStyleSource.includes("h1,\n.topbar h1") &&
    appTypographyStyleSource.includes(".table-shell th") &&
    appTypographyStyleSource.includes(".table-shell td") &&
    appTypographyStyleSource.includes(".job-form input") &&
    !appTypographyStyleSource.includes(".report-layer") &&
    !appTypographyStyleSource.includes(".image-jump") &&
    !appTypographyStyleSource.includes(".composite-object") &&
    !appTypographyStyleSource.includes(".object-context") &&
    appBaseStyleSource.includes(".sr-only") &&
    appBaseStyleSource.includes("button,\ninput,\ntextarea") &&
    appChromeStyleSource.includes(".app-shell") &&
    appChromeStyleSource.includes(".toast-stack") &&
    appChromeStyleSource.includes(".sidebar") &&
    appChromeStyleSource.includes(".topbar") &&
    !appShellSource.includes("user-profile-chip") &&
    !appShellSource.includes("Profile</span>") &&
    !appChromeStyleSource.includes(".user-profile-chip") &&
    appChromeStyleSource.includes("min-height: 40px") &&
    appChromeStyleSource.includes("padding-bottom: 8px") &&
    appChromeStyleSource.includes("grid-template-columns: 232px minmax(0, 1fr)") &&
    appChromeStyleSource.includes("gap: 14px") &&
    appChromeStyleSource.includes("padding: 16px 12px") &&
    appChromeStyleSource.includes("min-height: 54px") &&
    appChromeStyleSource.includes("font-weight: 760") &&
    appChromeStyleSource.includes("text-transform: uppercase") &&
    appChromeVisualStyleSource.includes(".sidebar::after") &&
    appChromeVisualStyleSource.includes(".content::before") &&
    appChromeVisualStyleSource.includes("min-height: 50px") &&
    appChromeVisualStyleSource.includes("padding: 8px 16px") &&
    !appChromeVisualStyleSource.includes("min-height: 58px") &&
    !appChromeVisualStyleSource.includes("padding: 10px 18px") &&
    appChromeVisualStyleSource.includes(".nav-item:hover .app-icon") &&
    !appChromeVisualStyleSource.includes(".user-profile-chip") &&
    appChromeVisualStyleSource.includes(".topbar .status-pill {\n  position: relative;\n  overflow: hidden;") &&
    appChromeVisualStyleSource.includes(".topbar .status-pill::before") &&
    !appChromeVisualStyleSource.includes(".topbar .status-pill:hover") &&
    !appChromeVisualStyleSource.includes(".status-pill.online") &&
    !appChromeVisualStyleSource.includes(".status-pill.loading") &&
    !appChromeVisualStyleSource.includes("var(--bench-status-success-soft)") &&
    !appChromeVisualStyleSource.includes("var(--bench-status-warning-soft)") &&
    !appChromeStyleSource.includes("translateX(") &&
    !appChromeVisualStyleSource.includes("translateX(") &&
    !appChromeStyleSource.includes("transform 150ms ease") &&
    !appChromeVisualStyleSource.includes("transform 150ms ease") &&
    !appChromeVisualStyleSource.includes("transform 140ms ease") &&
    !appChromeVisualStyleSource.includes(".sidebar.collapsed") &&
    !designSource.includes(".primary-button") &&
    !designSource.includes(".secondary-button") &&
    !designSource.includes(".mini-button") &&
    !designSource.includes(".mini-link") &&
    !designSource.includes(".icon-button") &&
    !designSource.includes(".query-chip") &&
    !designSource.includes(".badge") &&
    !designSource.includes(".metric-card") &&
    !designSource.includes(".metric-icon") &&
    !designSource.includes(".metric-value") &&
    !designSource.includes(".rank-primary-score") &&
    !designSource.includes(".rank-sort-header") &&
    !designSource.includes(".search-box") &&
    !designSource.includes(".select-popover-trigger") &&
    !designSource.includes("@keyframes") &&
    !appThemeStyleSource.includes(".app-shell") &&
    !appThemeStyleSource.includes(".sidebar") &&
    !appThemeStyleSource.includes(".brand") &&
    !appThemeStyleSource.includes(".nav-list") &&
    !appThemeStyleSource.includes(".nav-item") &&
    !appThemeStyleSource.includes(".store-chip") &&
    !appThemeStyleSource.includes(".content") &&
    !appThemeStyleSource.includes(".topbar") &&
    !appThemeStyleSource.includes(".eyebrow") &&
    !appThemeStyleSource.includes(".status-pill") &&
    !appThemeStyleSource.includes("h1") &&
    !appThemeStyleSource.includes("h2") &&
    !appThemeStyleSource.includes(".app-shell.sidebar-collapsed") &&
    !appThemeStyleSource.includes(".sidebar.collapsed") &&
    !appThemeStyleSource.includes(".sidebar-toggle") &&
    !appThemeStyleSource.includes(".dashboard-home") &&
    !appThemeStyleSource.includes(".home-grid") &&
    !appThemeStyleSource.includes(".summary-grid") &&
    !appThemeStyleSource.includes(".metric-card") &&
    !appThemeStyleSource.includes(".metric-icon") &&
    !appThemeStyleSource.includes(".primary-button") &&
    !appThemeStyleSource.includes(".secondary-button") &&
    !appThemeStyleSource.includes(".mini-button") &&
    !appThemeStyleSource.includes(".mini-link") &&
    !appThemeStyleSource.includes(".icon-button") &&
    !appThemeStyleSource.includes(".query-chip") &&
    !appThemeStyleSource.includes(".search-box") &&
    !appThemeStyleSource.includes(".filter-select") &&
    !appThemeStyleSource.includes(".compact-select") &&
    !designSource.includes(".sidebar::after") &&
    !designSource.includes(".content::before") &&
    !designSource.includes(".user-profile-chip") &&
    !designSource.includes("@keyframes status-breathe") &&
    !appThemeStyleSource.includes(".toast-stack") &&
    !appThemeStyleSource.includes(".toast-message") &&
    !appThemeStyleSource.includes(".user-profile-chip") &&
    !appThemeStyleSource.includes(".sr-only"),
  "base reset, app chrome, and app theme styles must live in explicit CSS modules",
);
assert(
  !mainEntry.includes("refetchInterval: 10_000") &&
    mainEntry.includes("staleTime: 15_000") &&
    mainEntry.includes("refetchOnWindowFocus: false") &&
    dashboardStateSource.includes('queryKey: ["dashboard-state"]') &&
    dashboardStateSource.includes("refetchInterval = false") &&
    !dashboardStateSource.includes("refetchInterval: 10_000") &&
    appShellSource.includes("const STATUS_SYNC_DELAY_MS = 450;") &&
    appShellSource.includes("const TOAST_AUTO_DISMISS_MS = 8_000;") &&
    appShellSource.includes("const TOAST_MAX_ITEMS = 3;") &&
    appShellSource.includes("__EVAL_BENCH_TOAST_AUTO_DISMISS_MS__?: number;") &&
    appShellSource.includes("function toastAutoDismissMs()") &&
    appShellSource.includes("? override") &&
    appShellSource.includes("dismissTimersRef") &&
    appShellSource.includes("current.findIndex((item) => item.message === message)") &&
    appShellSource.includes("{ ...item, count: item.count + 1 }") &&
    appShellSource.includes("window.clearTimeout(dismissTimersRef.current[message])") &&
    appShellSource.includes("current.slice(-(TOAST_MAX_ITEMS - 1))") &&
    appShellSource.includes("`操作失败 x${item.count}`") &&
    !appShellSource.includes("current.slice(-3), { id, message, tone: \"danger\" }") &&
    appShellSource.includes("function useDelayedTruthy(value: boolean, delayMs: number)") &&
    appShellSource.includes("const delayedLoading = useDelayedTruthy(loading, STATUS_SYNC_DELAY_MS);") &&
    appShellSource.includes("window.setTimeout(() => setDelayedValue(true), delayMs)") &&
    appShellSource.includes("window.clearTimeout(timeout)") &&
    !appShellSource.includes('className={loading ? "status-pill loading" : "status-pill online"}') &&
    appShellSource.includes('import { prefetchEvalBenchRouteData } from "./routePrefetch";') &&
    appShellSource.includes("const queryClient = useQueryClient();") &&
    appShellSource.includes("onIntent={(pathname) => prefetchEvalBenchRouteData(queryClient, pathname)}") &&
    appShellSource.includes("onMouseEnter={handleIntent}") &&
    appShellSource.includes("onFocus={handleIntent}") &&
    appShellSource.includes("onTouchStart={handleIntent}") &&
    routePrefetchSource.includes("const ROUTE_PREFETCH_STALE_MS = 15_000;") &&
    routePrefetchSource.includes("type SaveDataNavigator = Navigator &") &&
    routePrefetchSource.includes("const prefetchInFlightPathnames = new Set<string>();") &&
    routePrefetchSource.includes("export function prefetchEvalBenchRouteData") &&
    routePrefetchSource.includes("if (shouldSkipRoutePrefetch())") &&
    routePrefetchSource.includes("connection?.saveData") &&
    routePrefetchSource.includes("prefetchInFlightPathnames.has(normalizedPathname)") &&
    routePrefetchSource.includes("const prefetchQueries = prefetchQueriesForPathname(queryClient, normalizedPathname)") &&
    routePrefetchSource.includes("void Promise.allSettled(prefetchQueries).finally") &&
    routePrefetchSource.includes("prefetchInFlightPathnames.delete(normalizedPathname)") &&
    routePrefetchSource.includes("silent: true") &&
    apiSource.includes("silent?: boolean;") &&
    apiSource.includes("if (!silent)") &&
    apiSource.includes("notifyApiError(error.message)") &&
    routePrefetchSource.includes('"rank-board",') &&
    routePrefetchSource.includes('queryKey: ["runs", filters]') &&
    routePrefetchSource.includes('queryKey: ["jobs", filters]') &&
    routePrefetchSource.includes('queryKey: ["scheduler-status"]') &&
    routePrefetchSource.includes('queryKey: ["services", filters]') &&
    routePrefetchSource.includes('queryKey: ["benchmarks", filters]') &&
    routePrefetchSource.includes('queryKey: ["runs", "compare", runFilters]') &&
    routePrefetchSource.includes('queryKey: ["comparisons", comparisonFilters]') &&
    routePrefetchSource.includes('queryKey: ["settings-preview-sample"]') &&
    appShellSource.includes('import { warmupEvalBenchRoutes } from "./routeWarmup";') &&
    appShellSource.includes("useEffect(() => warmupEvalBenchRoutes(), []);") &&
    routeWarmupSource.includes("const ROUTE_WARMUP_TIMEOUT_MS = 2_500;") &&
    routeWarmupSource.includes("const ROUTE_WARMUP_FALLBACK_DELAY_MS = 1_200;") &&
    routeWarmupSource.includes("requestIdleCallback") &&
    routeWarmupSource.includes("cancelIdleCallback") &&
    routeWarmupSource.includes("connection?.saveData") &&
    routeWarmupSource.includes("Promise.allSettled") &&
    routeWarmupSource.includes("WARMUP_CORE_ROUTE_MODULES.map((loadRouteModule) => loadRouteModule())") &&
    !routeWarmupSource.includes("for (const loadRouteModule of WARMUP_CORE_ROUTE_MODULES)") &&
    routeWarmupSource.includes('() => import("./benchmarksPage")') &&
    routeWarmupSource.includes('() => import("./rankBoardPage")') &&
    routeWarmupSource.includes('() => import("./runsPage")') &&
    routeWarmupSource.includes('() => import("./jobsPage")') &&
    routeWarmupSource.includes('() => import("./suiteReportPage")') &&
    routeWarmupSource.includes('() => import("./comparePage")') &&
    routeWarmupSource.includes('() => import("./comparisonSamplePage")') &&
    routeWarmupSource.includes('() => import("./servicesPage")') &&
    routeWarmupSource.includes('() => import("./settingsPage")') &&
    appShellSource.includes('preload="intent"') &&
    appShellSource.includes("preloadDelay={80}") &&
    mainEntry.includes('lazyRouteComponent(() => import("./jobsPage"), "JobsPage")') &&
    mainEntry.includes(
      'lazyRouteComponent(() => import("./comparisonSamplePage"), "ComparisonSamplePage")',
    ) &&
    mainEntry.includes('lazyRouteComponent(() => import("./servicesPage"), "ServicesPage")') &&
    mainEntry.includes('lazyRouteComponent(() => import("./settingsPage"), "SettingsPage")') &&
    !mainEntry.includes('import { JobsPage } from "./jobsPage";') &&
    !mainEntry.includes('import { ServicesPage } from "./servicesPage";') &&
    !mainEntry.includes('import { SettingsPage } from "./settingsPage";') &&
    overviewModelSource.includes("const OVERVIEW_QUEUE_REFRESH_MS = 5_000;") &&
    overviewModelSource.includes("const OVERVIEW_SERVICE_REFRESH_MS = 10_000;") &&
    overviewModelSource.includes("useDashboardState({") &&
    overviewModelSource.includes("refetchInterval: OVERVIEW_SERVICE_REFRESH_MS") &&
    apiSource.includes("type FetchRequestOptions = {") &&
    apiSource.includes("signal?: AbortSignal") &&
    apiSource.includes("fetchTargetLabelResolution(\n  options: TargetLabelResolutionParams = {},\n  request: FetchRequestOptions = {}") &&
    apiSource.includes("export function fetchSuites(options: FetchRequestOptions = {})") &&
    apiSource.includes("export function fetchSuite(\n  suiteId: string,\n  options: FetchRequestOptions = {}") &&
    apiSource.includes("export function fetchCampaigns(\n  options: FetchRequestOptions = {}") &&
    apiSource.includes("export function fetchCampaign(\n  campaignId: string,\n  options: FetchRequestOptions = {}") &&
    apiSource.includes("export function fetchRunNote(\n  runId: string,\n  options: FetchRequestOptions = {}") &&
    dashboardStateSource.includes("queryFn: ({ signal }) => fetchState({ signal })") &&
    overviewModelSource.includes("queryFn: ({ signal }) => fetchJobs({ limit: 1 }, { signal })") &&
    overviewModelSource.includes("queryFn: ({ signal }) => fetchServices({ limit: 1 }, { signal })") &&
    overviewModelSource.includes("queryFn: ({ signal }) => fetchSchedulerStatus({ signal })") &&
    jobsCreatePanelSource.includes("queryFn: ({ signal }) => fetchJobTemplates({ signal })") &&
    jobsCreatePanelSource.includes("queryFn: ({ signal }) => fetchPromptTemplates({ signal })") &&
    jobsQueuePanelSource.includes("queryFn: ({ signal }) => fetchJobs(jobFilters, { signal })") &&
    jobsQueuePanelSource.includes("queryFn: ({ signal }) => fetchJobLogs(selectedJob?.job_id ?? \"\", 0, { signal })") &&
    servicesGridSource.includes("queryFn: ({ signal }) => fetchServiceLogs(service.service_id, { signal })") &&
    rankBoardControllerSource.includes("queryFn: ({ signal }) =>") &&
    rankBoardControllerSource.includes("{ signal }") &&
    useDebouncedValueSource.includes("export function useDebouncedValue<T>") &&
    useDebouncedValueSource.includes("export function useDebouncedValueState<T>") &&
    useDebouncedValueSource.includes("pending: debouncedValue !== value") &&
    useDebouncedValueSource.includes("window.setTimeout") &&
    [
      runsPage,
      benchmarksPage,
      servicesPage,
      jobsQueuePanelSource,
      rankBoardControllerSource,
      compareControllerSource
    ].every((source) => source.includes("useDebouncedValueState(searchText)")) &&
    [
      runsPage,
      benchmarksPage,
      servicesPage,
      jobsQueuePanelSource,
      rankBoardControllerSource,
      compareControllerSource
    ].every((source) => source.includes("debouncedSearch.pending")) &&
    runsPage.includes("refreshing={runsQuery.isPlaceholderData || debouncedSearch.pending}") &&
    benchmarksPage.includes("refreshing={benchmarksQuery.isPlaceholderData || debouncedSearch.pending}") &&
    servicesPage.includes("refreshing={servicesQuery.isPlaceholderData || debouncedSearch.pending}") &&
    jobsQueuePanelSource.includes("const queueRefreshing = Boolean((isPlaceholderData && data) || debouncedSearch.pending)") &&
    jobsQueuePanelSource.includes("refreshing={queueRefreshing}") &&
    rankBoardControllerSource.includes("tableRefreshing: (boardQuery.isPlaceholderData && Boolean(board)) || debouncedSearch.pending") &&
    comparePage.includes("runsRefreshing ?") &&
    comparePage.includes("refreshing={comparisonHistoryRefreshing}") &&
    compareRunRailComponentsSource.includes("refreshing?: boolean;") &&
    compareRunRailComponentsSource.includes("refreshing={refreshing}") &&
    rankBoardControllerSource.includes("query: debouncedSearch.value") &&
    rankBoardControllerSource.includes("writeRankBoardViewState({\n      boardMode,\n      searchText,") &&
    rankBoardControllerSource.includes("const filterValues = useMemo<RankBoardFilterValues>(\n    () => ({\n      searchText,") &&
    compareControllerSource.includes("query: debouncedSearch.value.trim() || undefined") &&
    compareControllerSource.includes("writeCompareViewState({\n      searchText,") &&
    compareControllerSource.includes("const filterValues = useMemo<CompareFilterValues>(\n    () => ({\n      searchText,") &&
    ![
      runsPage,
      benchmarksPage,
      servicesPage,
      jobsQueuePanelSource,
      rankBoardControllerSource,
      compareControllerSource
    ].some((source) => source.includes("query: searchText")),
    overviewModelSource.includes("function facetCount(") &&
    overviewModelSource.includes("jobTotalQuery.data?.facets?.statuses") &&
    !overviewModelSource.includes("overview-jobs-queued") &&
    !overviewModelSource.includes("overview-jobs-running") &&
    !overviewModelSource.includes("overview-jobs-failed") &&
    !overviewModelSource.includes("overview-services-running") &&
    jobsQueuePanelSource.includes("const JOB_QUEUE_REFRESH_MS = 4_000;") &&
    jobsQueuePanelSource.includes("refetchInterval: JOB_QUEUE_REFRESH_MS") &&
    !overviewModelSource.includes("refetchInterval: 2_000") &&
    !jobsQueuePanelSource.includes("refetchInterval: 2_000"),
  "query refresh policy must avoid global polling while preserving explicit live dashboard refresh and nav preloading",
);
assert(
  adaptiveContentStyleSource.includes("td,\ntbody td") &&
    adaptiveContentStyleSource.includes("max-width: clamp(96px, 18vw, 320px)") &&
    adaptiveContentStyleSource.includes("overflow-wrap: anywhere") &&
    adaptiveContentStyleSource.includes(".config-item strong") &&
    adaptiveContentStyleSource.includes(".select-popover-value") &&
    !designSource.includes("td,\ntbody td") &&
    !designSource.includes("max-width: clamp(96px, 18vw, 320px)") &&
    !designSource.includes(".select-popover-value"),
  "adaptive text and table wrapping rules must live in adaptiveContent.css instead of design.css",
);
assert(
  mainEntry.includes('import "./sharedControlsTheme.css";') &&
    mainEntry.includes('import "./sharedControls.css";') &&
    mainEntry.includes('import "./sharedButtons.css";') &&
    mainEntry.includes('import "./sharedIndicators.css";') &&
    mainEntry.includes('import "./sharedMetrics.css";') &&
    mainEntry.includes('import "./sharedPager.css";') &&
    mainEntry.includes('import "./sharedSplit.css";') &&
    mainEntry.indexOf('import "./sharedControlsTheme.css";') <
      mainEntry.indexOf('import "./sharedControls.css";') &&
    mainEntry.indexOf('import "./sharedControls.css";') <
      mainEntry.indexOf('import "./sharedButtons.css";') &&
    mainEntry.indexOf('import "./sharedButtons.css";') <
      mainEntry.indexOf('import "./sharedIndicators.css";') &&
    mainEntry.indexOf('import "./sharedIndicators.css";') <
      mainEntry.indexOf('import "./sharedMetrics.css";') &&
    mainEntry.indexOf('import "./sharedMetrics.css";') <
      mainEntry.indexOf('import "./sharedPager.css";') &&
    mainEntry.indexOf('import "./sharedPager.css";') <
      mainEntry.indexOf('import "./sharedSplit.css";') &&
    sharedControlsThemeStyleSource.includes("--control-primary-height") &&
    sharedControlsThemeStyleSource.includes("--control-icon-dense-size") &&
    sharedControlsThemeStyleSource.includes("--control-chip-height") &&
    sharedControlsThemeStyleSource.includes("--control-metric-value-size") &&
    sharedControlsThemeStyleSource.includes("--control-button-primary-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-button-secondary-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-button-danger-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-button-shadow-hover: 0 4px 10px") &&
    sharedControlsThemeStyleSource.includes("--control-status-success-ink:") &&
    sharedControlsThemeStyleSource.includes("--control-badge-warning-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-query-chip-active-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-pager-bg:") &&
    sharedControlsThemeStyleSource.includes("--control-pager-line:") &&
    sharedControlsThemeStyleSource.includes("--control-focus-outline:") &&
    sharedControlsThemeStyleSource.includes(':root[data-theme="dark"]') &&
    sharedControlsThemeStyleSource.includes("--control-button-primary-bg: color-mix") &&
    sharedControlsThemeStyleSource.includes("--control-button-secondary-ink: var(--bench-cyan-strong)") &&
    sharedControlsThemeStyleSource.includes("--control-status-success-ink: var(--bench-status-success)") &&
    sharedControlsThemeStyleSource.includes("--control-query-chip-bg: var(--bench-surface-raised)") &&
    sharedControlsThemeStyleSource.includes("--control-pager-bg: var(--bench-surface-raised)") &&
    !rawSharedControlsGeometryPattern.test(sharedControlsStyleSource) &&
    sharedControlsStyleSource.includes(".section-header") &&
    sharedControlsStyleSource.includes(".config-item") &&
    sharedControlsStyleSource.includes(".row-actions") &&
    sharedButtonsStyleSource.includes("var(--control-primary-height)") &&
    sharedButtonsStyleSource.includes("var(--control-icon-dense-size)") &&
    sharedButtonsStyleSource.includes(".primary-button") &&
    sharedButtonsStyleSource.includes(".secondary-button") &&
    sharedButtonsStyleSource.includes(".mini-button") &&
    sharedButtonsStyleSource.includes(".icon-button") &&
    sharedButtonsStyleSource.includes(".mini-link.compare-ready") &&
    sharedButtonsStyleSource.includes("var(--control-button-primary-bg)") &&
    sharedButtonsStyleSource.includes("var(--control-button-secondary-bg)") &&
    sharedButtonsStyleSource.includes("var(--control-button-icon-bg)") &&
    sharedButtonsStyleSource.includes("var(--control-button-danger-bg)") &&
    sharedButtonsStyleSource.includes("var(--control-button-disabled-bg)") &&
    sharedButtonsStyleSource.includes("var(--control-button-shadow-hover)") &&
    sharedButtonsStyleSource.includes("var(--control-button-shadow-active)") &&
    !sharedButtonsStyleSource.includes("box-shadow: 0 10px 22px") &&
    !sharedButtonsStyleSource.includes("box-shadow: 0 8px 18px") &&
    !sharedButtonsStyleSource.includes("background: #0b5f82") &&
    !sharedButtonsStyleSource.includes("background: #155f84") &&
    !sharedButtonsStyleSource.includes("background: #9f2532") &&
    !/translateY\(/.test(sharedButtonsStyleSource) &&
    !sharedButtonsStyleSource.includes("transform 140ms ease") &&
    sharedIndicatorsStyleSource.includes("var(--control-chip-height)") &&
    sharedIndicatorsStyleSource.includes(".status-pill") &&
    sharedIndicatorsStyleSource.includes(".query-chip") &&
    sharedIndicatorsStyleSource.includes(".badge.live::before") &&
    sharedIndicatorsStyleSource.includes("@keyframes badge-live-pulse") &&
    !/translateY\(/.test(sharedIndicatorsStyleSource) &&
    !sharedIndicatorsStyleSource.includes("transform 140ms ease") &&
    !sharedIndicatorsStyleSource.includes("transform 150ms ease") &&
    sharedIndicatorsStyleSource.includes("var(--control-status-success-ink)") &&
    sharedIndicatorsStyleSource.includes("var(--control-status-warning-bg)") &&
    sharedIndicatorsStyleSource.includes(".status-pill.loading") &&
    sharedIndicatorsStyleSource.includes("var(--control-status-danger-bg)") &&
    sharedIndicatorsStyleSource.includes("var(--control-badge-warning-bg)") &&
    sharedIndicatorsStyleSource.includes("var(--control-query-chip-bg)") &&
    sharedIndicatorsStyleSource.includes("var(--control-query-chip-active-bg)") &&
    sharedIndicatorsStyleSource.includes("var(--control-indicator-shadow-hover)") &&
    !sharedIndicatorsStyleSource.includes("box-shadow: 0 12px 24px") &&
    !sharedIndicatorsStyleSource.includes("background: #f7f9fb") &&
    !sharedIndicatorsStyleSource.includes("background: #235a78") &&
    sharedIndicatorsStyleSource.includes("border-color: color-mix(in srgb, currentColor 20%, transparent);") &&
    sharedMetricsStyleSource.includes("var(--control-metric-value-size)") &&
    sharedMetricsStyleSource.includes(".summary-grid") &&
    sharedMetricsStyleSource.includes(".metric-card") &&
    sharedMetricsStyleSource.includes(".metric-icon") &&
    sharedPagerStyleSource.includes(".rank-board-pager") &&
    sharedPagerStyleSource.includes("var(--control-pager-bg)") &&
    sharedPagerStyleSource.includes("var(--control-pager-line)") &&
    sharedPagerStyleSource.includes("var(--control-pager-ink)") &&
    !sharedPagerStyleSource.includes("background: #ffffff") &&
    !sharedPagerStyleSource.includes("border-bottom: 1px solid #e5ebf1") &&
    sharedSplitStyleSource.includes(".resizable-split") &&
    sharedSplitStyleSource.includes(".split-resizer") &&
    sharedSplitStyleSource.includes("--split-resizer-bg") &&
    sharedSplitStyleSource.includes("--split-resizer-active-bg") &&
    sharedSplitStyleSource.includes(':root[data-theme="dark"] .resizable-split') &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(sharedSplitStyleSource) &&
    controlPrimitives.includes('import "./controlPrimitiveStyles.css";') &&
    controlPrimitiveStyleSource.includes(".number-setting-control") &&
    controlPrimitiveStyleSource.includes(".compact-select") &&
    controlPrimitiveStyleSource.includes(".compact-select select") &&
    controlPrimitiveStyleSource.includes(".control-popover") &&
    controlPrimitiveStyleSource.includes(".control-check") &&
    controlPrimitiveStyleSource.includes(".color-control") &&
    !appThemeStyleSource.includes(".select-popover-menu") &&
    !appThemeStyleSource.includes(".select-popover-list") &&
    !appThemeStyleSource.includes(".badge.live::before") &&
    !appThemeStyleSource.includes("@keyframes badge-live-pulse") &&
    !appThemeStyleSource.includes(".resizable-split") &&
    !appThemeStyleSource.includes(".icon-button.dense") &&
    !sharedControlsStyleSource.includes(".primary-button") &&
    !sharedControlsStyleSource.includes(".secondary-button") &&
    !sharedControlsStyleSource.includes(".mini-button") &&
    !sharedControlsStyleSource.includes(".icon-button") &&
    !sharedControlsStyleSource.includes(".mini-link") &&
    !sharedControlsStyleSource.includes(".status-pill") &&
    !sharedControlsStyleSource.includes(".query-chip") &&
    !sharedControlsStyleSource.includes(".badge") &&
    !sharedControlsStyleSource.includes(".metric-card") &&
    !sharedControlsStyleSource.includes(".rank-board-pager") &&
    !sharedControlsStyleSource.includes(".resizable-split") &&
    !sharedControlsStyleSource.includes(".number-setting-control") &&
    !sharedControlsStyleSource.includes(".compact-select") &&
    !sharedControlsStyleSource.includes(".control-popover") &&
    !sharedControlsStyleSource.includes(".control-check") &&
    !sharedControlsStyleSource.includes(".color-control") &&
    !sharedControlsStyleSource.includes(".select-popover-menu") &&
    !sharedControlsStyleSource.includes(".select-popover-list"),
  "shared UI primitives and control primitive styles must live in focused CSS modules while appTheme.css remains a theme override layer",
);
assert(
  filterControlsStyleSource.includes(".search-box,\n.filter-select") &&
    filterControlsStyleSource.includes(".filter-select span") &&
    filterControlsStyleSource.includes(".filter-select select,\n.search-box input"),
  "filter/search visual overrides must live in filterControls.css instead of appTheme.css",
);
assert(
    selectPopoverControl.includes('import "./selectPopover.css";') &&
    selectPopoverControl.includes('import { createPortal } from "react-dom";') &&
    selectPopoverControl.includes('data-select-popover-menu="true"') &&
    selectPopoverControl.includes("SELECT_MENU_MAX_HEIGHT") &&
    selectPopoverControl.includes('closest(\'[role="dialog"], .settings-drawer-scroll\')') &&
    selectPopoverControl.includes("safeTop") &&
    selectPopoverControl.includes("safeBottom") &&
    selectPopoverControl.includes("availableBelow") &&
    selectPopoverControl.includes("availableAbove") &&
    selectPopoverControl.includes("data-placement={menuPlacement}") &&
    selectPopoverControl.includes("useDeferredValue") &&
    selectPopoverControl.includes("indexedOptions") &&
    selectPopoverControl.includes('} from "./selectPopoverModel";') &&
    selectPopoverControl.includes('export type { SelectOption } from "./selectPopoverModel";') &&
    selectPopoverModelSource.includes("export type VisibleSelectWindow =") &&
    selectPopoverModelSource.includes("export const SELECT_VISIBLE_LIMIT = 80") &&
    selectPopoverModelSource.includes("export function selectVisibleWindow(") &&
    selectPopoverModelSource.includes("export function clampSelectWindowStart(") &&
    selectPopoverModelSource.includes("export function enabledIndexNear(") &&
    selectPopoverModelSource.includes("export function pagedEnabledIndex(") &&
    selectPopoverModelSource.includes("export function centeredSelectWindowStart(") &&
    selectPopoverModelSource.includes("export function selectWindowStartForActiveIndex(") &&
    selectPopoverModelSource.includes("activeIndex < currentStart") &&
    selectPopoverModelSource.includes("activeIndex >= currentStart + SELECT_VISIBLE_LIMIT") &&
    selectPopoverModelSource.includes("activeIndex - SELECT_VISIBLE_LIMIT + 1") &&
    !selectPopoverControl.includes("function selectVisibleWindow(") &&
    !selectPopoverControl.includes("function enabledIndexNear(") &&
    !selectPopoverControl.includes("function selectWindowStartForActiveIndex(") &&
    selectPopoverControl.includes("const [windowStart, setWindowStart] = useState(0);") &&
    selectPopoverControl.includes("selectVisibleWindow(filteredOptions, windowStart)") &&
    selectPopoverControl.includes("centeredSelectWindowStart(Math.max(nextActiveIndex, 0), filteredOptions.length)") &&
    selectPopoverControl.includes("function focusTrigger()") &&
    selectPopoverControl.includes('querySelector<HTMLButtonElement>(".select-popover-trigger")') &&
    selectPopoverControl.includes("focus({ preventScroll: true })") &&
    selectPopoverControl.includes("function focusSearchInput()") &&
    selectPopoverControl.includes("searchRef.current?.focus({ preventScroll: true })") &&
    selectPopoverControl.includes("function clearSearch()") &&
    selectPopoverControl.includes("onClick={clearSearch}") &&
    selectPopoverControl.includes("function closePopover({ restoreFocus = false }") &&
    selectPopoverControl.includes("closePopover({ restoreFocus: true })") &&
    selectPopoverControl.includes("function togglePopover()") &&
    selectPopoverControl.includes("onClick={togglePopover}") &&
    selectPopoverControl.includes("closePopover();") &&
    selectPopoverControl.includes("const activeOptionInVisibleWindow =") &&
    selectPopoverControl.includes("activeOptionIndex >= visibleWindow.start") &&
    selectPopoverControl.includes("activeOptionIndex < visibleWindow.start + visibleOptions.length") &&
    selectPopoverControl.includes("&& activeOptionInVisibleWindow") &&
    selectPopoverControl.includes("const activeOptionId =") &&
    selectPopoverControl.includes("`${listboxId}-option-${activeOptionIndex + 1}`") &&
    selectPopoverModelSource.includes("hiddenBefore: start") &&
    selectPopoverModelSource.includes("hiddenAfter: Math.max(options.length - start - visibleOptions.length, 0)") &&
    selectPopoverControl.includes("data-select-window-start={visibleWindow.start}") &&
    selectPopoverControl.includes("aria-activedescendant={activeOptionId}") &&
    selectPopoverControl.includes("id={`${listboxId}-option-${absoluteIndex + 1}`}") &&
    selectPopoverControl.includes('aria-autocomplete="list"') &&
    selectPopoverControl.includes("aria-controls={listboxId}") &&
    selectPopoverControl.includes("aria-posinset={absoluteIndex + 1}") &&
    selectPopoverControl.includes("aria-setsize={filteredOptions.length}") &&
    selectPopoverControl.includes("data-select-window-index={absoluteIndex + 1}") &&
    selectPopoverControl.includes("activeOptionIndex") &&
    selectPopoverControl.includes("nextEnabledIndex(filteredOptions, activeOptionIndex, step)") &&
    selectPopoverControl.includes("function pageActive(direction: 1 | -1)") &&
    selectPopoverControl.includes("pagedEnabledIndex(filteredOptions, activeOptionIndex, direction)") &&
    selectPopoverControl.includes("function isSearchInputEvent(event: ReactKeyboardEvent<HTMLDivElement>)") &&
    selectPopoverControl.includes("return event.target === searchRef.current;") &&
    selectPopoverControl.includes('isSearchInputEvent(event) && (event.key === "Home" || event.key === "End")') &&
    selectPopoverControl.includes("event.stopPropagation();") &&
    selectPopoverControl.includes("event.nativeEvent.stopImmediatePropagation?.();") &&
    !selectPopoverControl.includes("currentIndex + direction * (SELECT_VISIBLE_LIMIT - 1)") &&
    selectPopoverModelSource.includes("activeIndex + direction * (SELECT_VISIBLE_LIMIT - 1)") &&
    selectPopoverModelCheckSource.includes("SELECT_VISIBLE_LIMIT") &&
    selectPopoverModelCheckSource.includes("selectVisibleWindow(options, centeredStart)") &&
    selectPopoverModelCheckSource.includes("selectVisibleWindow(shortOptions, 99)") &&
    selectPopoverModelCheckSource.includes("selectVisibleWindow(options, -40)") &&
    selectPopoverModelCheckSource.includes("selectVisibleWindow(options, 999)") &&
    selectPopoverModelCheckSource.includes("selectWindowStartForActiveIndex(80, 120, options.length)") &&
    selectPopoverModelCheckSource.includes("selectWindowStartForActiveIndex(80, -1, options.length)") &&
    selectPopoverModelCheckSource.includes("nextEnabledIndex(options, 78, 1)") &&
    selectPopoverModelCheckSource.includes("enabledIndexNear(options, 79, 1)") &&
    selectPopoverModelCheckSource.includes("pagedEnabledIndex(options, 0, 1)") &&
    selectPopoverModelCheckSource.includes("pagedEnabledIndex(options, 120, -1)") &&
    selectPopoverModelCheckSource.includes("pagedEnabledIndex(allDisabled, -1, 1)") &&
    selectPopoverSmokeSource.includes('const candidateRoutes = ["/rank-board", "/runs", "/jobs", "/benchmarks", "/services", "/compare"]') &&
    selectPopoverSmokeSource.includes("async function openFirstAdvancedFilterWithSelect(page)") &&
    selectPopoverSmokeSource.includes('".advanced-filter-controls .select-popover-trigger"') &&
    selectPopoverSmokeSource.includes('data-select-popover-menu="true"') &&
    selectPopoverSmokeSource.includes("async function assertSelectStructure(page, scope)") &&
    selectPopoverSmokeSource.includes('trigger?.getAttribute("aria-controls")') &&
    selectPopoverSmokeSource.includes('listbox?.getAttribute("role")') &&
    selectPopoverSmokeSource.includes('searchInput?.getAttribute("aria-controls")') &&
    selectPopoverSmokeSource.includes('searchInput?.getAttribute("aria-autocomplete")') &&
    selectPopoverSmokeSource.includes("async function assertSearchClearKeepsInputFocus(page)") &&
    selectPopoverSmokeSource.includes('page.getByRole("button", { name: "清空搜索" })') &&
    selectPopoverSmokeSource.includes("async function assertKeyboardActiveDescendant(page)") &&
    selectPopoverSmokeSource.includes('page.keyboard.press("PageDown")') &&
    selectPopoverSmokeSource.includes('document.getElementById(activeId)') &&
    selectPopoverSmokeSource.includes('activeNode?.getAttribute("data-select-window-index")') &&
    selectPopoverSmokeSource.includes("async function assertEscapeRestoresTriggerFocus(page)") &&
    selectPopoverSmokeSource.includes('page.keyboard.press("Escape")') &&
    selectPopoverSmokeSource.includes('classList.contains("select-popover-trigger")') &&
    selectPopoverControl.includes('event.key === "PageDown"') &&
    selectPopoverControl.includes("pageActive(1)") &&
    selectPopoverControl.includes('event.key === "PageUp"') &&
    selectPopoverControl.includes("pageActive(-1)") &&
    selectPopoverControl.includes("selectWindowStartForActiveIndex(currentStart, nextIndex, filteredOptions.length)") &&
    selectPopoverControl.includes("selectWindowStartForActiveIndex(0, nextIndex, filteredOptions.length)") &&
    selectPopoverControl.includes("selectWindowStartForActiveIndex(windowStart, index, filteredOptions.length)") &&
    selectPopoverControl.includes("for (let index = filteredOptions.length - 1; index >= 0; index -= 1)") &&
    selectPopoverControl.includes("const activeOption = filteredOptions[activeOptionIndex]") &&
    selectPopoverControl.includes("const active = absoluteIndex === activeOptionIndex") &&
    selectPopoverControl.includes("onMouseEnter={() => {") &&
    selectPopoverStyleSource.includes(".select-popover-control") &&
    selectPopoverStyleSource.includes("--select-popover-menu-bg: #ffffff") &&
    selectPopoverStyleSource.includes("--select-popover-trigger-bg: #ffffff") &&
    selectPopoverStyleSource.includes("--select-popover-menu-shadow: 0 8px 18px") &&
    selectPopoverStyleSource.includes("--select-popover-menu-shadow-top: 0 -8px 18px") &&
    selectPopoverStyleSource.includes(':root[data-theme="dark"]') &&
    selectPopoverStyleSource.includes("--select-popover-menu-bg: var(--bench-surface-raised)") &&
    selectPopoverStyleSource.includes("--select-popover-menu-shadow: 0 10px 22px") &&
    selectPopoverStyleSource.includes("--select-popover-trigger-hover-ink: var(--bench-cyan-strong)") &&
    selectPopoverStyleSource.includes(".select-popover-filter.compact {\n  width: 100%;") &&
    selectPopoverStyleSource.includes("max-width: 100%;") &&
    !selectPopoverStyleSource.includes("width: min(150px, 18vw)") &&
    selectPopoverStyleSource.includes(".select-popover-menu") &&
    selectPopoverStyleSource.includes("position: fixed;") &&
    selectPopoverStyleSource.includes("z-index: 260;") &&
    selectPopoverStyleSource.includes(".select-popover-list") &&
    selectPopoverStyleSource.includes("contain: layout paint;") &&
    selectPopoverStyleSource.includes("overscroll-behavior: contain;") &&
    selectPopoverStyleSource.includes("scrollbar-gutter: stable;") &&
    selectPopoverStyleSource.includes(".select-popover-window-note") &&
    selectPopoverStyleSource.includes('.select-popover-menu[data-placement="top"]') &&
    selectPopoverStyleSource.includes(".select-popover-filter .select-popover-trigger") &&
    selectPopoverStyleSource.includes("max-height: var(--select-menu-max-height") &&
    selectPopoverStyleSource.includes("background: var(--select-popover-menu-bg);") &&
    selectPopoverStyleSource.includes("box-shadow: var(--select-popover-menu-shadow);") &&
    selectPopoverStyleSource.includes("color: var(--select-popover-option-selected-ink);") &&
    !selectPopoverStyleSource.includes(':root[data-theme="dark"] .select-popover-menu') &&
    !selectPopoverStyleSource.includes(':root[data-theme="dark"] .select-popover-option.selected') &&
    !selectPopoverStyleSource.includes("0 14px 30px") &&
    !selectPopoverStyleSource.includes("background: #ffffff;") &&
    !selectPopoverStyleSource.includes("color: #1d2a38;") &&
    !selectPopoverStyleSource.includes("rgba(99, 127, 149") &&
    !appThemeStyleSource.includes(".select-popover-menu") &&
    !designSource.includes(".select-popover-menu"),
  "select popover styles must stay colocated with selectPopoverControl instead of shared theme files",
);
assert(
  labelColorControlsStyleSource.includes(".label-color-add-row") &&
    labelColorControlsStyleSource.includes(".inline-select-control") &&
    labelColorControlsStyleSource.includes(".select-control-label-hidden > span") &&
    labelColorControlsStyleSource.includes(".label-color-grid") &&
    labelColorControlsStyleSource.includes(".label-color-row") &&
    labelColorControlsStyleSource.includes(".label-color-role-grid") &&
    labelColorControlsStyleSource.includes(':root[data-theme="dark"] .label-color-add-row') &&
    labelColorControlsStyleSource.includes("--label-color-field-bg") &&
    labelColorControlsStyleSource.includes("--label-color-name-ink") &&
    !appThemeStyleSource.includes(".label-color-add-row") &&
    !appThemeStyleSource.includes(".inline-select-control") &&
    !appThemeStyleSource.includes(".select-control-label-hidden > span") &&
    !appThemeStyleSource.includes(".label-color-row") &&
    !appThemeStyleSource.includes(".label-color-role-grid"),
  "label color controls must live in labelColorControls.css instead of the legacy theme layer",
);
assert(
  apiSource.includes("`/api/runs/${encodeURIComponent(runId)}/evaluate`") &&
    apiSource.includes("`/api/runs/${encodeURIComponent(runId)}/samples?${params.toString()}`") &&
    apiSource.includes("`/api/runs/${encodeURIComponent(runId)}/samples/${index}`") &&
    apiSource.includes(
      "`/api/benchmarks/${encodeURIComponent(benchmarkId)}/samples?${params.toString()}`",
    ) &&
    apiSource.includes(
      "`/api/benchmarks/${encodeURIComponent(benchmarkId)}/samples/${index}${query ? `?${query}` : \"\"}`",
    ),
  "frontend API must encode run and benchmark ids in sample/evaluate paths",
);
assert(
  samplePagerSource.includes("export function PagerControl(") &&
    samplePagerSource.includes("export function clampListPageOffset(") &&
    samplePagerSource.includes("export function updatePagedFilterValue<T>(") &&
    samplePagerSource.includes("...resetOffsets: Array<(offset: number) => void>") &&
    samplePagerSource.includes("export function SamplePager("),
  "paged list controls must share PagerControl, clampListPageOffset, and filter offset reset semantics",
);
assert(
    !uiSource.includes('export * from "./uiDataTable";') &&
    uiDataTableSource.includes("export type TableColumnWidth =") &&
    uiDataTableSource.includes('import "./dataTable.css";') &&
    uiDataTableSource.includes("export function TableEmptyState(") &&
    uiDataTableSource.includes("export function TableLoadingState(") &&
    uiDataTableSource.includes("table-skeleton-line") &&
    uiDataTableSource.includes("declare module \"@tanstack/react-table\"") &&
    uiDataTableSource.includes("export function tableColumnClassName(") &&
    uiDataTableSource.includes("data-table-cell") &&
    uiDataTableSource.includes('"table-shell",\n        "empty"') &&
    uiDataTableSource.includes("refreshing && \"refreshing\"") &&
    uiDataTableSource.includes("<div className=\"empty-panel\">{emptyText}</div>") &&
    !mainEntry.includes('import "./dataTable.css";') &&
    runTables.includes('import { DataTable } from "./uiDataTable";') &&
    rankBoardTablesSource.includes('import { DataTable } from "./uiDataTable";') &&
    compareRunRailComponentsSource.includes('import { DataTable } from "./uiDataTable";') &&
    jobsQueuePanelSource.includes("TableEmptyState") &&
    servicesGridSource.includes('import { TableEmptyState } from "./uiDataTable";') &&
    jobsQueueTableSource.includes('import { tableColumnClassName } from "./uiDataTable";') &&
    dataTableStyleSource.includes(".table-shell .table-col-id") &&
    dataTableStyleSource.includes(".table-shell .table-col-metric") &&
    dataTableStyleSource.includes(".table-shell .table-wrap-wrap") &&
    dataTableStyleSource.includes(".table-shell.empty .empty-panel") &&
    dataTableStyleSource.includes("background: transparent;") &&
    dataTableStyleSource.includes(".table-shell.refreshing::after") &&
    dataTableStyleSource.includes(".table-shell.table-loading") &&
    dataTableStyleSource.includes("--table-loading-glint") &&
    dataTableStyleSource.includes("--table-hover-border") &&
    dataTableStyleSource.includes("--table-hover-shadow: var(--bench-shadow-tight)") &&
    dataTableStyleSource.includes("--table-row-hover-rail") &&
    dataTableStyleSource.includes("--table-row-selected-rail") &&
    dataTableStyleSource.includes("--table-checkbox-bg") &&
    dataTableStyleSource.includes("--table-checkbox-check-ring") &&
    dataTableStyleSource.includes("contain: layout;") &&
    dataTableStyleSource.includes("outline: 2px solid var(--control-focus-outline)") &&
    !dataTableStyleSource.includes("transform 150ms ease") &&
    !dataTableStyleSource.includes("animation: table-region-refresh") &&
    !dataTableStyleSource.includes("@keyframes table-region-refresh") &&
    !dataTableStyleSource.includes("translateX(-70%)") &&
    !dataTableStyleSource.includes("0 16px 34px") &&
    dataTableStyleSource.includes(':root[data-theme="dark"] .table-shell.table-loading') &&
    dataTableStyleSource.includes(':root[data-theme="dark"] .table-shell') &&
    dataTableStyleSource.includes("@keyframes table-skeleton-sheen") &&
    dataTableStyleSource.includes("@media (prefers-reduced-motion: reduce)") &&
    !dataTableStyleSource.includes(".table-shell.refreshing::after,\n  .table-skeleton-line") &&
    dataTableStyleSource.includes(".table-refresh-indicator") &&
    dataTableStyleSource.includes(".row-select-checkbox") &&
    dataTableStyleSource.includes(".selectable-row") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(dataTableStyleSource) &&
    runTables.includes('import { useCallback, useMemo, useState } from "react";') &&
    runTables.includes("const BENCHMARK_TABLE_COLUMNS: ColumnDef<BenchmarkSummary>[] = [") &&
    runTables.includes("columns={BENCHMARK_TABLE_COLUMNS}") &&
    runTables.includes("const toggleRunSelection = useCallback((runId: string) => {") &&
    runTables.includes("const columns = useMemo<ColumnDef<RunSummary>[]>(\n    () => [") &&
    runTables.includes("const { mutate: evaluateRunMutate, isPending: evaluatePending } = evaluateMutation;") &&
    runTables.includes("const { mutate: archiveRunMutate, isPending: archivePending } = archiveMutation;") &&
    runTables.includes("const { isPending: deletePending } = deleteMutation;") &&
    compareRunRailComponentsSource.includes("const COMPARISON_HISTORY_COLUMNS: ColumnDef<ComparisonSummary>[] = [") &&
    compareRunRailComponentsSource.includes("columns={COMPARISON_HISTORY_COLUMNS}") &&
    !compareRunRailComponentsSource.includes("const columns: ColumnDef<ComparisonSummary>[] = [") &&
    servicesGridSource.includes("TableEmptyState") &&
    servicesGridSource.includes('emptyText="没有符合高级检索条件的模型服务。"') &&
    servicesGridSource.includes('refreshLabel="服务列表更新中"') &&
    !styleSource.includes("animation: table-region-refresh") &&
    !styleSource.includes("@keyframes table-region-refresh") &&
    !servicesGridSource.includes("return <EmptyState title=\"没有符合高级检索条件的模型服务。\" />") &&
    jobsQueuePanelSource.includes("TableEmptyState") &&
    jobsQueuePanelSource.includes('emptyText="没有符合高级检索条件的任务。"') &&
    jobsQueuePanelSource.includes('refreshLabel="队列更新中"') &&
    !jobsQueuePanelSource.includes('<div className="empty-panel">没有符合高级检索条件的任务。</div>') &&
    benchmarksPage.includes("<TableLoadingState label=\"正在加载基准集\"") &&
    runsPage.includes("<TableLoadingState label=\"正在加载评测记录\"") &&
    rankBoardPage.includes("<TableLoadingState label=\"正在加载排行榜\"") &&
    servicesPage.includes("<TableLoadingState label=\"正在加载服务\"") &&
    jobsQueuePanelSource.includes("<TableLoadingState") &&
    !benchmarksPage.includes('<EmptyState title="正在加载基准集"') &&
    !runsPage.includes('<EmptyState title="正在加载评测记录"') &&
    !rankBoardPage.includes('<EmptyState title="正在加载排行榜"') &&
    !servicesPage.includes('<EmptyState title="正在加载服务"') &&
    !jobsQueuePanelSource.includes('className="empty-panel">正在加载队列状态') &&
    runTables.includes('meta: { width: "id", wrap: "wrap" }') &&
    rankBoardTablesSource.includes('meta: { width: "metric", align: "end" }') &&
    compareRunRailComponentsSource.includes('meta: { width: "date" }') &&
    jobsQueueTableSource.includes("JOB_QUEUE_COLUMN_CLASS_NAMES") &&
    jobsQueueTableSource.includes('tableColumnClassName({ width: "id", wrap: "wrap" })') &&
    jobsQueueTableSource.includes("className={JOB_QUEUE_COLUMN_CLASS_NAMES.identity}"),
  "shared data tables must use column metadata for adaptive width, wrapping, and alignment",
);
assert(
  apiTypesSource.includes("export type FacetBuckets = Record<string, FacetBucket>;") &&
    formattersSource.includes("export function facetValues(") &&
    [runsPage, benchmarksPage, compareControllerSource, rankBoardControllerSource, servicesPage, jobsQueuePanelSource].every((source) =>
      source.includes("facetValues(")
    ) &&
    rankBoardControllerSource.includes('facetValues(board?.facets, "tasks"') &&
    !rankBoardPage.includes("const tasks = unique(runs") &&
    !rankBoardControllerSource.includes("const tasks = unique(runs") &&
    !runTables.includes("const tasks = unique(runs") &&
    !runTables.includes("filterControls ?? [") &&
    !runsPage.includes('fetchRuns({ limit: 500 })') &&
    !comparePage.includes('fetchRuns({ limit: 500 })') &&
    !compareControllerSource.includes('fetchRuns({ limit: 500 })') &&
    !benchmarksPage.includes('fetchBenchmarks({ limit: 500 })') &&
    !servicesPage.includes('fetchServices({ limit: 500 })') &&
    !jobsQueuePanelSource.includes('fetchJobs({ limit: 500 })'),
  "advanced filter option directories must use backend facets instead of truncated 500-item list requests",
);
assert(
  advancedFilterTypesSource.includes("export type AdvancedFilterControl") &&
    filterControls.includes('import type { AdvancedFilterControl } from "./advancedFilterTypes";') &&
    filterControls.includes('import { renderAdvancedControl } from "./advancedFilterFields";') &&
    filterControls.includes('export type { AdvancedFilterControl } from "./advancedFilterTypes";') &&
    filterControls.includes('export { FilterSelect } from "./advancedFilterFields";') &&
    advancedFilterFieldsSource.includes("FilterSelectControl") &&
    advancedFilterFieldsSource.includes("SearchInputControl") &&
    advancedFilterFieldsSource.includes("TextInputControl") &&
    filterControls.includes('import { ActionButton, PanelToggleButton } from "./ui";') &&
    filterControls.includes('import { DIALOG_FOCUSABLE_SELECTOR } from "./uiDialog";') &&
    advancedFilterFieldsSource.includes("<FilterSelectControl") &&
    advancedFilterFieldsSource.includes("<SearchInputControl") &&
    advancedFilterFieldsSource.includes("<TextInputControl") &&
    !/<input\b/.test(filterControls) &&
    !/<select\b/.test(filterControls) &&
    !/<input\b/.test(advancedFilterFieldsSource) &&
    !/<select\b/.test(advancedFilterFieldsSource) &&
    controlPrimitives.includes("FilterSelectControl") &&
    selectPopoverControl.includes("export function FilterSelectControl(") &&
    controlPrimitives.includes("export function SearchInputControl(") &&
    filterControls.includes("function resetAdvancedFilters()") &&
    advancedFilterModelSource.includes("export function resetAdvancedFilter(") &&
    filterControls.includes("function resetSingleAdvancedFilter(") &&
    filterControls.includes("function updateDraftValue(") &&
    filterControls.includes("function applyDraftFilters(") &&
    filterControls.includes("function applyDraftFiltersFromKeyboard(") &&
    advancedFilterModelSource.includes("export function applyAdvancedFilterValues(") &&
    filterControls.includes('import "./filterTheme.css";') &&
    filterControls.includes('import "./filterControls.css";') &&
    advancedFilterStorageSource.includes("export function readAdvancedFilterDraftValues(") &&
    advancedFilterStorageSource.includes("export function writeAdvancedFilterDraftValues(") &&
    advancedFilterModelSource.includes("export function advancedFilterDirtyControlIds(") &&
    advancedFilterStorageSource.includes("window.sessionStorage") &&
    filterControls.includes("event.composedPath().includes(rootRef.current)") &&
    filterControls.includes("const latestDraftValuesRef = useRef<Record<string, string>>(draftValues);") &&
    filterControls.includes("const dirtyControlIdsRef = useRef<Set<string>>(") &&
    advancedFilterModelSource.includes("export function syncDraftValuesWithApplied(") &&
    filterControls.includes("function openAdvancedFilter()") &&
    filterControls.includes("function closeAdvancedFilter(") &&
    filterControls.includes("function toggleAdvancedFilter()") &&
    advancedFilterModelSource.includes("export function defaultFilterValue(") &&
    advancedFilterModelSource.includes("export function displayFilterValue(") &&
    advancedFilterModelSource.includes("export function groupAdvancedControls(") &&
    advancedFilterStorageSource.includes("export function advancedFilterOpenStateKey(") &&
    advancedFilterStorageSource.includes("export function readAdvancedFilterOpenState(") &&
    advancedFilterStorageSource.includes("export function writeAdvancedFilterOpenState(") &&
    advancedFilterModelSource.includes("export function advancedFilterValues(") &&
    advancedFilterModelSource.includes("export function advancedFilterValuesKey(") &&
    advancedFilterStorageSource.includes("eval_bench_advanced_filter_open") &&
    !filterControls.includes("window.sessionStorage") &&
    !filterControls.includes("const ADVANCED_FILTER_OPEN_STORAGE_PREFIX") &&
    !filterControls.includes("function groupAdvancedControls(") &&
    !filterControls.includes("function advancedFilterValues(") &&
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
    advancedFilterFieldsSource.includes('className="advanced-filter-search-control"') &&
    advancedFilterFieldsSource.includes('className="advanced-filter-number-control"') &&
    advancedFilterFieldsSource.includes('className="advanced-filter-text-control"') &&
    filterControls.includes('className="advanced-filter-token"') &&
    filterControls.includes('className="advanced-filter-clear"') &&
    filterControls.includes('className="advanced-filter-apply"') &&
    filterControls.includes("onKeyDown={applyDraftFiltersFromKeyboard}") &&
    filterControls.includes('hasDraftChanges ? "dirty" : ""') &&
    filterControls.includes('aria-keyshortcuts="Enter"') &&
    filterControls.includes("onClick={() => resetSingleAdvancedFilter(filter.control)}") &&
    filterControls.includes("onClick={applyDraftFilters}") &&
    filterControls.includes("<PanelToggleButton") &&
    !filterControls.includes('className="search-box advanced-search-box"') &&
    !filterControls.includes('className="filter-select compact advanced-number-box"') &&
    !/<button[\s\S]{0,260}advanced-filter-head/.test(filterControls),
  "advanced filter chrome must stay in AdvancedFilterBar while field rendering, model rules, and storage stay in dedicated modules",
);
assert(
  !styleSource.includes(".filter-bar"),
  "legacy filter-bar CSS must not return; page filters must use AdvancedFilterBar",
);
assert(
  filterControlsStyleSource.includes(".advanced-filter-search-control") &&
    filterControlsStyleSource.includes(".advanced-filter-number-control") &&
    filterControlsStyleSource.includes(".advanced-filter-text-control") &&
    filterThemeStyleSource.includes("--filter-control-min: 26px") &&
    filterThemeStyleSource.includes("--filter-head-min: 30px") &&
    filterThemeStyleSource.includes("--filter-text-input: var(--text-sm)") &&
    filterThemeStyleSource.includes("--filter-radius-control: 2px") &&
    filterThemeStyleSource.includes("--filter-surface: #ffffff") &&
    filterThemeStyleSource.includes("--filter-field-bg: #ffffff") &&
    filterThemeStyleSource.includes("--filter-popover-shadow: 0 8px 18px") &&
    filterThemeStyleSource.includes(':root[data-theme="dark"]') &&
    filterThemeStyleSource.includes("--filter-surface: var(--bench-surface)") &&
    filterThemeStyleSource.includes("--filter-focus: var(--bench-cyan)") &&
    filterThemeStyleSource.includes("--filter-shadow: 0 6px 14px rgb(0 0 0 / 18%)") &&
    filterThemeStyleSource.includes("--filter-popover-shadow: 0 10px 22px rgb(0 0 0 / 24%)") &&
    filterControlsStyleSource.includes("var(--filter-control-min)") &&
    filterControlsStyleSource.includes("var(--filter-text-label)") &&
    filterControlsStyleSource.includes("var(--filter-radius-control)") &&
    filterControlsStyleSource.includes("background: var(--filter-surface);") &&
    filterControlsStyleSource.includes("background: var(--filter-field-bg);") &&
    filterControlsStyleSource.includes("border-color: var(--filter-field-line-strong);") &&
    filterControlsStyleSource.includes("box-shadow: var(--filter-popover-shadow);") &&
    !filterControlsStyleSource.includes("background: #ffffff;") &&
    !filterControlsStyleSource.includes("color: #101828;") &&
    !filterControlsStyleSource.includes("rgba(99, 127, 149") &&
    !filterThemeStyleSource.includes("0 14px 30px") &&
    !rawAdvancedFilterGeometryPattern.test(filterControlsStyleSource) &&
    !styleSource.includes(".advanced-search-box") &&
    !styleSource.includes(".advanced-number-box") &&
    !appThemeStyleSource.includes(".advanced-filter-search-control") &&
    !appThemeStyleSource.includes(".advanced-filter-number-control") &&
    !appThemeStyleSource.includes(".advanced-filter-text-control") &&
    !designSource.includes(".advanced-filter-search-control") &&
    !designSource.includes(".advanced-filter-number-control") &&
    !designSource.includes(".advanced-filter-text-control"),
  "advanced filter controls must use dedicated semantic classes instead of legacy search/filter shells",
);
assert(
    styleSource.includes("scrollbar-gutter: stable") &&
    styleSource.includes("table-layout: auto") &&
    styleSource.includes("max-width: clamp(96px, 18vw, 320px)") &&
    styleSource.includes("overflow-wrap: anywhere") &&
    dataTableStyleSource.includes(".table-shell td:has(.row-actions)") &&
    dataTableStyleSource.includes("max-width: clamp(96px, 18vw, 320px)") &&
    styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr))") &&
    styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr))") &&
    filterControls.includes('data-filter-group={group.id}') &&
    styleSource.includes("grid-template-columns: minmax(74px, max-content) minmax(0, 1fr)") &&
    advancedFilterModelSource.includes('{ id: "search", title: "搜索"') &&
    advancedFilterModelSource.includes('{ id: "tune", title: "阈值排序"') &&
    styleSource.includes('.advanced-filter-group[data-filter-group="scope"] .advanced-filter-controls') &&
    styleSource.includes('.advanced-filter-group[data-filter-group="search"] {') &&
    styleSource.includes('.advanced-filter-group[data-filter-group="search"] .advanced-filter-controls') &&
    styleSource.includes('.advanced-filter-group[data-filter-group="search"] .advanced-filter-group-title') &&
    styleSource.includes('.advanced-filter-group[data-filter-group="tune"] .advanced-filter-controls') &&
    styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 118px), 1fr))") &&
    styleSource.includes("--service-card-column-min: 360px") &&
    styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, var(--service-card-column-min)), 1fr))") &&
    !styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr))") &&
    styleSource.includes(".service-form") &&
    filterControlsStyleSource.includes(".advanced-filter-bar.dirty .advanced-filter-head") &&
    jobsPage.includes('import "./jobsPage.css";') &&
    !mainEntry.includes('import "./formControls.css";') &&
    benchmarkCreatePanelSource.includes('import "./formControls.css";') &&
    servicesCreatePanelSource.includes('import "./formControls.css";') &&
    servicesGridSource.includes('import "./formControls.css";') &&
    runsImportPanelSource.includes('import "./formControls.css";') &&
    jobsCreatePanelSource.includes('import "./formControls.css";') &&
    mainEntry.includes('import "./themeSurfaceOverrides.css";') &&
    mainEntry.includes('import "./workspaceTheme.css";') &&
    mainEntry.includes('import "./workspaceShell.css";') &&
    !mainEntry.includes('import "./workspaceDialog.css";') &&
    uiDialogSource.includes('import "./workspaceDialog.css";') &&
    mainEntry.includes('import "./pageCommand.css";') &&
    mainEntry.indexOf('import "./workspaceTheme.css";') <
      mainEntry.indexOf('import "./workspaceShell.css";') &&
    mainEntry.indexOf('import "./workspaceShell.css";') <
      mainEntry.indexOf('import "./pageCommand.css";') &&
    themeSurfaceOverridesStyleSource.includes(':root[data-theme="dark"]') &&
    themeSurfaceOverridesStyleSource.includes("scrollbar-color:") &&
    !themeSurfaceOverridesStyleSource.includes(':root[data-theme="dark"] *') &&
    themeSurfaceOverridesStyleSource.includes(".dashboard-home.overview-home-v18") &&
    !themeSurfaceOverridesStyleSource.includes(".advanced-filter-bar") &&
    !themeSurfaceOverridesStyleSource.includes(".advanced-filter-directory") &&
    !themeSurfaceOverridesStyleSource.includes(".advanced-filter-search-control") &&
    !themeSurfaceOverridesStyleSource.includes(".advanced-filter-popover") &&
    !themeSurfaceOverridesStyleSource.includes(".advanced-filter-token") &&
    themeSurfaceOverridesStyleSource.includes(".rank-board-table-card") &&
    themeSurfaceOverridesStyleSource.includes(".scheduler-strip") &&
    themeSurfaceOverridesStyleSource.includes(".compare-run-rail") &&
    themeSurfaceOverridesStyleSource.includes(".compare-report-pane") &&
    themeSurfaceOverridesStyleSource.includes(".compare-context-pane") &&
    themeSurfaceOverridesStyleSource.includes(".composite-report-shell") &&
    themeSurfaceOverridesStyleSource.includes(".composite-composer-dock") &&
    themeSurfaceOverridesStyleSource.includes(".composite-stage-region") &&
    themeSurfaceOverridesStyleSource.includes(".settings-workbench-shell") &&
    themeSurfaceOverridesStyleSource.includes(".settings-preference-drawer") &&
    themeSurfaceOverridesStyleSource.includes(".settings-drawer-head") &&
    themeSurfaceOverridesStyleSource.includes(".icon-button:focus-visible") &&
    themeSurfaceOverridesStyleSource.includes(".run-config-panel") &&
    !themeSurfaceOverridesStyleSource.includes(".badge.success") &&
    !themeSurfaceOverridesStyleSource.includes(".badge.warning") &&
    !themeSurfaceOverridesStyleSource.includes(".badge.danger") &&
    !themeSurfaceOverridesStyleSource.includes(".badge.info") &&
    !themeSurfaceOverridesStyleSource.includes(".status-pill.online") &&
    !themeSurfaceOverridesStyleSource.includes(".status-pill.loading") &&
    !themeSurfaceOverridesStyleSource.includes(".status-pill.danger") &&
    !themeSurfaceOverridesStyleSource.includes(".rank-mode-switch .query-chip") &&
    !themeSurfaceOverridesStyleSource.includes(".query-chip.active") &&
    !themeSurfaceOverridesStyleSource.includes(".app-shell") &&
    !themeSurfaceOverridesStyleSource.includes("@keyframes") &&
    workspaceThemeStyleSource.includes("--workspace-tab-height") &&
    workspaceThemeStyleSource.includes("--workspace-dialog-head-height") &&
    workspaceThemeStyleSource.includes("--workspace-radius-panel") &&
    workspaceThemeStyleSource.includes("--workspace-command-button-height") &&
    workspaceThemeStyleSource.includes("--workspace-text-caption") &&
    workspaceShellStyleSource.includes("var(--workspace-tab-height)") &&
    workspaceShellStyleSource.includes("var(--workspace-radius-panel)") &&
    workspaceShellStyleSource.includes("var(--workspace-text-caption)") &&
    !rawWorkspaceShellGeometryPattern.test(workspaceShellStyleSource) &&
    !rawWorkspaceShellGeometryPattern.test(workspaceDialogStyleSource) &&
    workspaceShellStyleSource.includes(".workspace-card") &&
    workspaceShellStyleSource.includes(".workspace-tabs") &&
    workspaceShellStyleSource.includes(".action-panel") &&
    workspaceShellStyleSource.includes(".fatal-panel") &&
    !workspaceShellStyleSource.includes(".workspace-dialog-backdrop") &&
    !workspaceShellStyleSource.includes(".danger-confirm-panel") &&
    workspaceDialogStyleSource.includes(".workspace-dialog-backdrop") &&
    workspaceDialogStyleSource.includes(".workspace-dialog-head") &&
    workspaceDialogStyleSource.includes("var(--workspace-dialog-head-height)") &&
    workspaceDialogStyleSource.includes("var(--workspace-text-caption)") &&
    workspaceDialogStyleSource.includes(".danger-confirm-panel") &&
    workspaceDialogStyleSource.includes("box-shadow: 0 16px 36px") &&
    !workspaceDialogStyleSource.includes("backdrop-filter") &&
    !workspaceDialogStyleSource.includes("0 26px 70px") &&
    pageCommandStyleSource.includes(".density-page") &&
    pageCommandStyleSource.includes(".page-command-row") &&
    pageCommandStyleSource.includes(".command-button") &&
    pageCommandStyleSource.includes(".compact-form-card") &&
    pageCommandStyleSource.includes("var(--workspace-command-button-height)") &&
    !pageCommandStyleSource.includes("transform: none") &&
    !appThemeStyleSource.includes(".workspace-card") &&
    !appThemeStyleSource.includes(".workspace-tabs") &&
    !appThemeStyleSource.includes(".action-panel") &&
    !appThemeStyleSource.includes(".workspace-dialog-backdrop") &&
    !appThemeStyleSource.includes(".danger-confirm-panel") &&
    !designSource.includes(".workspace-card") &&
    !designSource.includes(".workspace-tabs") &&
    !designSource.includes(".action-panel") &&
    !designSource.includes(".workspace-dialog-backdrop") &&
    !designSource.includes(".danger-confirm-panel") &&
    !designSource.includes(".page-command-row") &&
    !designSource.includes(".command-button") &&
    !designSource.includes(".compact-form-card") &&
    !designSource.includes(".number-setting-control") &&
    formControlsStyleSource.includes(".job-form") &&
    formControlsStyleSource.includes(".form-result") &&
    formControlsStyleSource.includes(".form-error") &&
    formControlsStyleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr))") &&
    formControlsStyleSource.includes("border-color: var(--control-focus-line)") &&
    formControlsStyleSource.includes("outline: 2px solid var(--control-focus-outline)") &&
    formControlsStyleSource.includes("color: var(--control-status-danger-ink)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(formControlsStyleSource) &&
    jobsStyleSource.includes('@import "./jobsQueue.css";') &&
    jobsStyleSource.includes('@import "./jobsRecentRuns.css";') &&
    jobsStyleSource.includes('@import "./jobsDetail.css";') &&
    jobsStyleSource.includes('@import "./jobsManifest.css";') &&
    jobsManifestStyleSource.includes(".manifest-job-form label") &&
    jobsManifestStyleSource.includes(".manifest-toolbar") &&
    jobsQueueStyleSource.includes(".queue-stack") &&
    jobsRecentRunsStyleSource.includes(".recent-run-card") &&
    !jobsRecentRunsStyleSource.includes("transform 160ms ease") &&
    !jobsRecentRunsStyleSource.includes("translateY(") &&
    !jobsRecentRunsStyleSource.includes("0 14px 28px") &&
    jobsRecentRunsStyleSource.includes(':root[data-theme="dark"] .recent-run-list') &&
    jobsRecentRunsStyleSource.includes("--recent-run-card-accent") &&
    jobsRecentRunsStyleSource.includes("--recent-run-artifact-gradient") &&
    jobsRecentRunsStyleSource.includes("content-visibility: auto;") &&
    jobsRecentRunsStyleSource.includes("contain-intrinsic-size: auto 58px;") &&
    jobsRecentRunsStyleSource.includes("box-shadow: inset 3px 0 0 var(--recent-run-card-accent)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(jobsRecentRunsStyleSource) &&
    jobsQueueStyleSource.includes(':root[data-theme="dark"] .job-activity-grid') &&
    jobsQueueStyleSource.includes("--job-strip-bg") &&
    jobsQueueStyleSource.includes("--job-eval-strong-ink") &&
    jobsQueueStyleSource.includes("box-shadow: 0 0 0 3px var(--job-status-live-ring)") &&
    jobsQueueStyleSource.includes("grid-template-columns: minmax(0, 1.45fr) minmax(min(100%, var(--job-activity-side-min)), 0.85fr)") &&
    jobsQueueStyleSource.includes("flex-wrap: wrap;") &&
    jobsQueueStyleSource.includes("content-visibility: auto;") &&
    jobsQueueStyleSource.includes("contain-intrinsic-size: auto 360px;") &&
    jobsQueueStyleSource.includes(".job-eval-cell .run-id-text") &&
    jobsDetailStyleSource.includes(".job-detail-panel") &&
    jobsDetailStyleSource.includes(':root[data-theme="dark"] .job-detail-panel') &&
    jobsDetailStyleSource.includes("--job-progress-start") &&
    jobsDetailStyleSource.includes("content-visibility: auto;") &&
    jobsDetailStyleSource.includes("contain-intrinsic-size: auto 420px;") &&
    jobsDetailStyleSource.includes("font-size: var(--text-xs);") &&
    !jobsDetailStyleSource.includes("font-size: 13px") &&
    jobsManifestStyleSource.includes(".prompt-template-panel") &&
    jobsManifestStyleSource.includes(':root[data-theme="dark"] .manifest-card') &&
    jobsManifestStyleSource.includes("--manifest-surface") &&
    jobsManifestStyleSource.includes("--manifest-code-bg") &&
    jobsManifestStyleSource.includes("background: var(--manifest-surface)") &&
    jobsManifestStyleSource.includes("content-visibility: auto;") &&
    jobsManifestStyleSource.includes("contain-intrinsic-size: auto 180px;") &&
    !jobsManifestStyleSource.includes("background: #ffffff") &&
    !jobsManifestStyleSource.includes("color: #607080") &&
    !appThemeStyleSource.includes(".manifest-toolbar") &&
    !appThemeStyleSource.includes(".recent-run-card") &&
    !appThemeStyleSource.includes(".job-eval-cell") &&
    !appThemeStyleSource.includes(".job-form") &&
    !appThemeStyleSource.includes(".form-result") &&
    !appThemeStyleSource.includes(".form-error") &&
    !appThemeStyleSource.includes(".data-table-cell") &&
    !appThemeStyleSource.includes(".table-col-id") &&
    !appThemeStyleSource.includes(".row-select-checkbox") &&
    !appThemeStyleSource.includes(".selectable-row") &&
    !appThemeStyleSource.includes(".table-shell.refreshing") &&
    !appThemeStyleSource.includes("@keyframes table-region-refresh") &&
    !designSource.includes(".job-form") &&
    !designSource.includes(".form-result") &&
    !designSource.includes(".form-error") &&
    !designSource.includes(".table-shell") &&
    !designSource.includes(".data-table-cell") &&
    !designSource.includes(".table-col-id") &&
    !designSource.includes(".row-select-checkbox") &&
    !designSource.includes(".selectable-row") &&
    !designSource.includes("@keyframes table-region-refresh") &&
    styleSource.includes(".row-actions {\n  display: flex;\n  flex-wrap: wrap;") &&
    !styleSource.includes("grid-template-columns: repeat(4, minmax(160px, 1fr)) auto") &&
    !styleSource.includes("minmax(430px, 1fr)") &&
    !styleSource.includes("minmax(360px, 1fr)") &&
    styleSource.includes(".service-card") &&
    styleSource.includes(':root[data-theme="dark"] .service-grid') &&
    styleSource.includes("--service-card-bg") &&
    styleSource.includes("--service-card-column-min: 360px") &&
    styleSource.includes("--service-code-bg") &&
    styleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, var(--service-card-column-min)), 1fr))") &&
    styleSource.includes("max-height: var(--service-command-max)") &&
    styleSource.includes("max-height: var(--service-log-max)") &&
    styleSource.includes("scrollbar-gutter: stable both-edges;") &&
    styleSource.includes("content-visibility: auto;") &&
    styleSource.includes("contain-intrinsic-size: auto 360px;") &&
    !styleSource.includes("box-shadow: 0 12px 30px rgb(15 23 42 / 5%)") &&
    !servicesPageStyleSource.includes("box-shadow 150ms ease") &&
    !servicesPageStyleSource.includes("minmax(min(100%, 430px), 1fr)") &&
    !designSource.includes(".service-card") &&
    jobsManifestStyleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr))") &&
    adaptiveContentStyleSource.includes("max-width: clamp(96px, 18vw, 320px)") &&
    filterControlsStyleSource.includes(".advanced-filter-bar.dirty .advanced-filter-head") &&
    !designSource.includes(".advanced-filter-bar.dirty") &&
    !designSource.includes(".manifest-toolbar") &&
    !designSource.includes(".recent-run-card") &&
    !designSource.includes(".manifest-job-form") &&
    styleSource.includes("overflow-wrap: normal") &&
    !designSource.includes("grid-template-columns: repeat(12, minmax(0, 1fr))"),
  "tables and form grids must auto-fit in both base and design styles without late overrides",
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
  jobsPage.includes('import { JobQueuePanel } from "./jobsQueuePanel";') &&
    jobsPage.includes('import { JobCreatePanel } from "./jobsCreatePanel";') &&
    jobsPage.includes("<JobQueuePanel />") &&
    jobsPage.includes("<JobCreatePanel") &&
    !jobsPage.includes("export function JobQueuePanel(") &&
    !jobsPage.includes("export function JobCreatePanel(") &&
    !jobsPage.includes("function SchedulerStrip(") &&
    !jobsPage.includes("function JobDetailPanel(") &&
    !jobsPage.includes("manifestBenchmarkSplit") &&
    !jobsPage.includes("createJob") &&
    !jobsPage.includes("fetchJobTemplates") &&
    !jobsPage.includes("CompactSelectControl") &&
    !jobsPage.includes("TextareaControl") &&
    jobsQueuePanelSource.includes("export function JobQueuePanel(") &&
    jobsQueuePanelSource.includes("function SchedulerStrip(") &&
    jobsQueuePanelSource.includes("function JobDetailPanel(") &&
    !jobsQueuePanelSource.includes("function JobProgressInline(") &&
    jobsQueuePanelSource.includes('import { JobQueueTable, jobRunId } from "./jobsQueueTable";') &&
    jobsQueuePanelSource.includes("<JobQueueTable") &&
    jobsQueueTableSource.includes("export function JobQueueTable(") &&
    jobsQueueTableSource.includes("function JobQueueRow(") &&
    jobsQueueTableSource.includes("function JobProgressInline(") &&
    jobsCreatePanelSource.includes("export function JobCreatePanel(") &&
    jobsCreatePanelSource.includes("function PromptTemplatePanel(") &&
    jobsCreatePanelSource.includes("function PreflightPanel("),
  "jobs queue and manifest create workflows must live in focused submodules instead of jobsPage",
);
assert(
  controlPrimitives.includes('} from "./selectPopoverControl";') &&
    !controlPrimitives.includes("function SelectPopoverControl(") &&
    selectPopoverControl.includes("function SelectPopoverControl(") &&
    selectPopoverModelSource.includes("export const SELECT_VISIBLE_LIMIT = 80") &&
    selectPopoverControl.includes("selectVisibleWindow") &&
    selectPopoverControl.includes("pagedEnabledIndex") &&
    !selectPopoverControl.includes("const SELECT_VISIBLE_LIMIT") &&
    selectPopoverControl.includes("deferredQuery") &&
    selectPopoverControl.includes('role="listbox"') &&
    selectPopoverControl.includes('role="option"') &&
    selectPopoverControl.includes('placeholder="搜索选项"') &&
    selectPopoverControl.includes("visibleWindow = useMemo(") &&
    selectPopoverControl.includes("const visibleOptions = visibleWindow.options") &&
    selectPopoverControl.includes("const options = useMemo(\n    () =>\n      values.map((item) => ({") &&
    selectPopoverControl.includes("[labels, values]") &&
    !selectPopoverControl.includes("nextEnabledIndex(visibleOptions") &&
    !selectPopoverControl.includes("const activeOption = visibleOptions[activeIndex]") &&
    !/<select\b/.test(controlPrimitives) &&
    !/<select\b/.test(selectPopoverControl),
  "select primitives must render a searchable bounded custom popover instead of native browser selects",
);
assert(
  styleSource.includes(".select-popover-filter .select-popover-trigger {\n  min-height: 26px;") &&
    styleSource.includes("max-height: var(--select-menu-max-height") &&
    filterControlsStyleSource.includes(".advanced-filter-bar {\n  position: relative;") &&
    filterControlsStyleSource.includes("transition: none;") &&
    filterControlsStyleSource.includes(".advanced-filter-popover {\n  position: absolute;") &&
    filterControlsStyleSource.includes("grid-template-rows: auto minmax(0, 1fr);") &&
    filterControlsStyleSource.includes("overflow: visible;") &&
    !filterControlsStyleSource.includes("grid-template-areas:") &&
    !filterControlsStyleSource.includes("2.35fr") &&
    filterControlsStyleSource.includes("grid-template-columns: minmax(74px, max-content) minmax(0, 1fr);") &&
    filterControlsStyleSource.includes('.advanced-filter-group[data-filter-group="scope"] .advanced-filter-controls') &&
    filterControlsStyleSource.includes('.advanced-filter-group[data-filter-group="search"] {') &&
    filterControlsStyleSource.includes('.advanced-filter-group[data-filter-group="search"] .advanced-filter-controls') &&
    filterControlsStyleSource.includes(".advanced-filter-group[data-filter-group=\"search\"] .advanced-filter-controls {\n  grid-template-columns: minmax(0, 1fr);") &&
    filterControlsStyleSource.includes(".advanced-filter-group[data-filter-group=\"search\"] .advanced-filter-group-title {\n  display: none;") &&
    filterControlsStyleSource.includes('.advanced-filter-group[data-filter-group="tune"] .advanced-filter-controls') &&
    !filterControlsStyleSource.includes("minmax(min(100%, 280px), 1fr) repeat(auto-fit") &&
    filterControlsStyleSource.includes("grid-template-columns: repeat(auto-fit, minmax(min(100%, 118px), 1fr));") &&
    filterControls.includes('closest(\'[data-select-popover-menu="true"]\')') &&
    !styleSource.includes("animation: advanced-filter-enter") &&
    !styleSource.includes("@keyframes advanced-filter-enter") &&
    !appThemeStyleSource.includes(".advanced-filter-bar {\n  position: relative;") &&
    !designSource.includes(".advanced-filter-bar {\n  position: relative;") &&
    interactionFeedbackStyleSource.includes("outline: 2px solid var(--control-focus-outline)"),
  "filter and select controls must stay compact, neutral, and free of decorative open animations",
);
assert(
  controlPrimitives.includes("export function StandaloneCheckboxControl(") &&
    controlPrimitives.includes("export function StandaloneColorControl(") &&
    controlPrimitives.includes("export function InlineColorControl("),
  "table selection and color controls must share standalone primitives",
);
assert(
  uiSource.includes('export * from "./uiActions";') &&
    uiActionsSource.includes("export function PanelToggleButton("),
  "collapsible panel toggles must share PanelToggleButton",
);
assert(
  uiSource.includes("export function DisclosurePanel(") &&
    uiSource.includes("<details {...props} className={className}>") &&
    uiSource.includes("<summary>{summary}</summary>"),
  "collapsible details shells must share DisclosurePanel",
);
assert(
  uiActionsSource.includes("export function IconNavLink(") &&
    uiActionsSource.includes('className: joinClassNames("icon-button", dense && "dense", className)'),
  "router icon links must share IconNavLink",
);
assert(
  uiActionsSource.includes("export function InlineNavLink(") &&
    uiActionsSource.includes('className: joinClassNames("mini-link", className)'),
  "router inline links must share InlineNavLink",
);
assert(
  uiActionsSource.includes("export function InlineAnchor(") &&
    uiActionsSource.includes('className={joinClassNames("mini-link", className)}'),
  "href inline links must share InlineAnchor",
);
assert(
  uiActionsSource.includes("export function NavigationCardAnchor(") &&
    uiActionsSource.includes("export function NavigationCardFrame("),
  "card-style navigation rows must share NavigationCardAnchor/NavigationCardFrame",
);
assert(
  uiActionsSource.includes("export function SelectableRowButton("),
  "sample row selection must be centralized in SelectableRowButton",
);
assert(
  uiActionsSource.includes("export function SelectableTableRow("),
  "table row selection must be centralized in SelectableTableRow",
);
assert(
  uiActionsSource.includes("export function OptionChipButton("),
  "query chip selection must be centralized in OptionChipButton",
);
assert(
  !uiSource.includes('export * from "./uiDialog";') &&
    uiDialogSource.includes("export const DIALOG_FOCUSABLE_SELECTOR =") &&
    uiDialogSource.includes('import "./workspaceDialog.css";') &&
    filterControls.includes('import { DIALOG_FOCUSABLE_SELECTOR } from "./uiDialog";') &&
    runsPage.includes('import { WorkspaceDialog } from "./uiDialog";') &&
    benchmarksPage.includes('import { WorkspaceDialog } from "./uiDialog";') &&
    jobsPage.includes('import { WorkspaceDialog } from "./uiDialog";') &&
    servicesPage.includes('import { WorkspaceDialog } from "./uiDialog";') &&
    runTables.includes('import { DangerConfirmDialog } from "./uiDialog";') &&
    jobsQueuePanelSource.includes('import { DangerConfirmDialog } from "./uiDialog";') &&
    servicesGridSource.includes('import { DangerConfirmDialog } from "./uiDialog";') &&
    uiDialogSource.includes("document.body.style.overflow = \"hidden\"") &&
    uiDialogSource.includes("previouslyFocused?.focus()") &&
    uiDialogSource.includes("tabIndex={-1}") &&
    uiDialogSource.includes("aria-describedby={meta ? metaId : undefined}"),
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
  jobsCreatePanelSource.includes("CompactSelectControl"),
  "manifest toolbar selects must use CompactSelectControl",
);
assert(
  jobsQueuePanelSource.includes("const JOB_PAGE_SIZE = 80;") &&
    jobsQueuePanelSource.includes(
      'import { PagerControl, clampListPageOffset, updatePagedFilterValue } from "./samplePager";',
    ) &&
    jobsQueuePanelSource.includes('<PagerControl\n          className="rank-board-pager job-list-pager"') &&
    jobsQueuePanelSource.includes("offset: compact ? 0 : pageOffset") &&
    jobsQueuePanelSource.includes("limit: compact ? 12 : JOB_PAGE_SIZE") &&
    !jobsQueuePanelSource.includes("function JobListPager(") &&
    !jobsQueuePanelSource.includes("limit: compact ? 12 : 200") &&
    !jobsQueuePanelSource.includes("limit: 200"),
  "jobs queue page must use paged API requests instead of a fixed 200-job slice",
);
assert(
  jobsQueuePanelSource.includes("readJobsViewState") &&
    jobsQueuePanelSource.includes("writeJobsViewState") &&
    jobsQueuePanelSource.includes("JOBS_VIEW_STATE_RESET_EVENT") &&
    jobsQueuePanelSource.includes("updatePagedFilterValue(searchText, value, setSearchText, setPageOffset)") &&
    jobsQueuePanelSource.includes("updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)") &&
    jobsQueuePanelSource.includes("updatePagedFilterValue(kindFilter, value, setKindFilter, setPageOffset)") &&
    jobsViewStateSource.includes("export const JOBS_VIEW_STATE_KEY") &&
    jobsViewStateSource.includes("export const JOBS_VIEW_STATE_RESET_EVENT") &&
    jobsViewStateSource.includes("selectedJobId") &&
    jobsViewStateSource.includes("window.sessionStorage") &&
    appShellSource.includes('import { resetJobsViewState } from "./jobsViewState";') &&
    appShellSource.includes("onNavigate={resetJobsViewState}") &&
    !jobsQueuePanelSource.includes("useEffect(() => {\n    if (!compact) {\n      setPageOffset(0);"),
  "jobs queue must preserve filters, paging, and selected job across back-navigation and reset from main nav",
);
assert(
  jobsQueuePanelSource.includes("errorMessage(error)") &&
    !jobsQueuePanelSource.includes("<div className=\"empty-panel danger-text\">队列状态加载失败</div>"),
  "jobs queue page must show concrete API errors for queue loading failures",
);
assert(
  apiTypesSource.includes("run_id: string | null;") &&
    jobsQueueTableSource.includes("<th className={JOB_QUEUE_COLUMN_CLASS_NAMES.identity}>评测</th>") &&
    jobsQueueTableSource.includes("export function jobRunId(job: JobSummary)") &&
    jobsQueueTableSource.includes("job.run_id || stringValue(job.metadata.run_id) || stringValue(job.payload.run_id)") &&
    jobsQueueTableSource.includes('className="job-eval-cell"') &&
    jobsQueueTableSource.includes("title={runId || job.job_id}") &&
    jobsQueuePanelSource.includes("const linkedRunId = stringValue(job.metadata.run_manifest_path) ? jobRunId(job) : \"\";") &&
    runTables.includes('header: "评测"') &&
    !runTables.includes('header: "记录"'),
  "eval identity must be run_id-first in both jobs queue and result library tables",
);
assert(
    runTables.includes('meta: { width: "id", wrap: "wrap" }') &&
    runTables.includes('className="run-id-link"') &&
    rankBoardTablesSource.includes('width: "id"') &&
    rankBoardTablesSource.includes('wrap: "wrap"') &&
    rankBoardTablesSource.includes('className="run-id-link"') &&
    jobsQueueTableSource.includes('identity: tableColumnClassName({ width: "id", wrap: "wrap" })') &&
    jobsQueueTableSource.includes('className="run-id-text"') &&
    mainEntry.includes('import "./runTables.css";') &&
    runTablesStyleSource.includes(".run-id-link,") &&
    runTablesStyleSource.includes("text-overflow: clip;") &&
    runTablesStyleSource.includes(".table-shell .run-id-link") &&
    !appThemeStyleSource.includes(".run-id-link,") &&
    !dataTableStyleSource.includes(".table-shell .run-id-link") &&
    styleSource.includes(".job-eval-cell .run-id-text") &&
    styleSource.includes(".overview-v18-run-id .run-id-text"),
  "run names must wrap across table and card surfaces instead of being truncated with ellipsis",
);
assert(
  (jobsCreatePanelSource.match(/<CompactSelectControl/g) ?? []).length >= 2,
  "manifest toolbar must render template and prompt through CompactSelectControl",
);
assert(
  !jobsCreatePanelSource.includes('className="filter-select compact"'),
  "jobs create panel must not create ad hoc compact filter selects outside shared controls",
);
assertNoLegacyFormSubmitClass(jobsCreatePanelSource, "jobsCreatePanel.tsx");
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
    labelSubtaskControls.includes('import "./labelSubtaskControls.css";') &&
    labelSubtaskControlsStyleSource.includes(".label-subtask-panel") &&
    labelSubtaskControlsStyleSource.includes(".label-subtask-chips") &&
    !designSource.includes(".label-subtask-panel") &&
    !designSource.includes(".label-subtask-chips") &&
    !labelSubtaskControls.includes('className={selectedSet.has(label) ? "query-chip active" : "query-chip"}'),
  "label subtask controls must own their component styles and use OptionChipButton instead of raw query-chip buttons",
);
assert(
  jobsCreatePanelSource.includes("DetectionLabelSubtaskPanel") &&
    jobsCreatePanelSource.includes("<DetectionLabelSubtaskPanel"),
  "label subtask panel must stay detection-only; keypoint jobs must not expose label subset UI",
);
assert(
  jobsCreatePanelSource.includes("manifestBenchmarkSplit") &&
    jobsCreatePanelSource.includes("updateManifestBenchmarkSplit") &&
    jobsCreatePanelSource.includes("jobBenchmarkSplitOptions(selectedBenchmark, manifestBenchmarkSplitValue)") &&
    jobsCreatePanelSource.includes('label="Benchmark split"') &&
    jobsCreatePanelSource.includes("onChange={updateBenchmarkSplit}") &&
    manifestToolsSource.includes("export function manifestBenchmarkSplit(") &&
    manifestToolsSource.includes("export function updateManifestBenchmarkSplit(") &&
    manifestToolsSource.includes("section.benchmark_split = normalized") &&
    manifestToolsSource.includes("delete section.split") &&
    manifestToolsSource.includes("function normalizeManifestBenchmarkSplit(") &&
    manifestToolsSource.includes("clearManifestBenchmarkSplit(section)") &&
    manifestToolsSource.includes("function clearManifestBenchmarkSplit("),
  "jobs manifest editor must expose explicit benchmark_split selection for suite benchmarks",
);
assert(
  jobsCreatePanelSource.includes("function resetPreflightResult()") &&
    jobsCreatePanelSource.includes("resetPreflightResult();") &&
    jobsCreatePanelSource.includes("}, [manifestDraft, manifestTaskValue]);") &&
    !jobsCreatePanelSource.includes("}, [manifestText, manifestDraft, manifestTaskValue, preflightMutation]);"),
  "manifest preflight result must reset only on semantic draft changes, not on mutation object rerenders",
);
assert(
  formattersSource.includes("export function errorMessage(value: unknown)") &&
    jobsCreatePanelSource.includes("errorMessage(preflightMutation.error)") &&
    jobsCreatePanelSource.includes("errorMessage(mutation.error)") &&
    jobsCreatePanelSource.includes("setParseError(errorMessage(error))") &&
    jobsCreatePanelSource.includes("saveErrorMessage={errorMessage(promptMutation.error)}") &&
    appShellSource.includes("import { errorMessage } from \"./formatters\";") &&
    appShellSource.includes("return { error: errorMessage(error) };") &&
    !jobsCreatePanelSource.includes("<div className=\"form-error\">预检查请求失败。</div>") &&
    !jobsCreatePanelSource.includes("<div className=\"form-error\">任务入队失败。</div>") &&
    !jobsCreatePanelSource.includes("<div className=\"form-error\">Prompt 模板保存失败。</div>") &&
    !appShellSource.includes("return { error: error instanceof Error ? error.message : String(error) };"),
  "manifest job form and render boundary errors must use shared concrete error text",
);
assert(
  jobsCreatePanelSource.includes("DisclosurePanel") &&
    jobsCreatePanelSource.includes('className="prompt-template-panel"') &&
    jobsCreatePanelSource.includes("TextareaControl") &&
    jobsCreatePanelSource.includes('className="manifest-editor-field"') &&
    !/<details\b/.test(jobsCreatePanelSource) &&
    !/<summary\b/.test(jobsCreatePanelSource) &&
    !/<textarea\b/.test(jobsCreatePanelSource),
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
assert(
  shortcutCoverageSource.includes("async function discoverShortcutTargets(") &&
    shortcutCoverageSource.includes('fetch(`${rootUrl}/api/state`)') &&
    shortcutCoverageSource.includes('path.join(root, "src", "workspaceSettingsSchema.ts")') &&
    shortcutCoverageSource.includes("process.env.EVAL_BENCH_BENCHMARK_ID ?? shortcutTargets.benchmarkId") &&
    shortcutCoverageSource.includes("process.env.EVAL_BENCH_RUN_ID ?? shortcutTargets.runId") &&
    shortcutCoverageSource.includes("shortcut dynamic benchmark checks skipped") &&
    shortcutCoverageSource.includes("shortcut dynamic run checks skipped") &&
    shortcutCoverageSource.includes("benchmarkSampleInspector.tsx") &&
    shortcutCoverageSource.includes("runDetailPage.tsx") &&
    shortcutCoverageSource.includes("viewerViewportController.ts") &&
    !shortcutCoverageSource.includes('throw new Error("shortcut coverage requires') &&
    !shortcutCoverageSource.includes('?? "multitask_val_v1"') &&
    !shortcutCoverageSource.includes('?? "config_smoke_prompt_params"'),
  "shortcut coverage must discover current store targets and skip data-dependent checks cleanly when fixtures are absent",
);
assert(
  shortcutCoverageSource.includes("appUrl(`/benchmarks/${encodeURIComponent(benchmarkId)}?sample=0`)") &&
    shortcutCoverageSource.includes("appUrl(`/runs/${encodeURIComponent(runId)}?sample=0`)") &&
    shortcutCoverageSource.includes(
      "appUrl(`/compare/${encodeURIComponent(runId)}/${encodeURIComponent(runId)}/0`)",
    ),
  "shortcut coverage must encode dynamic benchmark/run route ids",
);
assert(
    viewerPerformanceSource.includes("async function resolveViewerPerformanceUrl(") &&
    viewerPerformanceSource.includes('fetch(new URL("/api/state", parsed.origin))') &&
    viewerPerformanceSource.includes('new URL(`/runs/${encodeURIComponent(run.run_id)}`, parsed.origin)') &&
    viewerPerformanceSource.includes("viewer performance check skipped: no evaluated run with sample detail") &&
    viewerPerformanceSource.includes("process.exit(0)") &&
    viewerPerformanceSource.includes('.viewer-pointer-surface[data-pointer-reticle="active"]') &&
    viewerPerformanceSource.includes(".viewer-pointer-surface .composite-canvas-pointer-reticle") &&
    viewerPerformanceSource.includes(".viewer-pointer-surface .composite-canvas-coordinate-tag") &&
    !viewerPerformanceSource.includes("config_smoke_prompt_params"),
  "viewer performance smoke must discover current runs and protect ordinary viewer pointer feedback",
);
assert(
  packageJsonSource.includes('"test:composite-report": "node scripts/composite-report-smoke-check.mjs"') &&
    packageJsonSource.includes('"test:dialogs": "node scripts/dialog-smoke-check.mjs"') &&
    packageJsonSource.includes('"test:toast": "node scripts/toast-smoke-check.mjs"') &&
    packageJsonSource.includes('"test:route-warmup": "node scripts/route-warmup-check.mjs"') &&
    packageJsonSource.includes('"test:nav-prefetch": "node scripts/nav-prefetch-check.mjs"') &&
    packageJsonSource.includes('"test:loading-state": "node scripts/loading-state-check.mjs"') &&
    packageJsonSource.includes('"test:settings-preview": "node scripts/settings-preview-check.mjs"') &&
    packageJsonSource.includes(
      '"test:smoke": "npm run test:theme && npm run test:route-warmup && npm run test:nav-prefetch && npm run test:loading-state && npm run test:toast && npm run test:dialogs && npm run test:settings-preview && npm run test:select-popover-ui"',
    ) &&
    dialogSmokeSource.includes('process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/"') &&
    dialogSmokeSource.includes('!text.includes("/api/")') &&
    dialogSmokeSource.includes('!text.includes("Failed to load resource")') &&
    toastSmokeSource.includes('new CustomEvent("eval-bench-api-error"') &&
    toastSmokeSource.includes("window.__EVAL_BENCH_TOAST_AUTO_DISMISS_MS__ = 600;") &&
    toastSmokeSource.includes("duplicate API errors should coalesce into one toast") &&
    toastSmokeSource.includes("操作失败 x3") &&
    toastSmokeSource.includes("duplicate API error should refresh the auto-dismiss timer") &&
    toastSmokeSource.includes("操作失败 x4") &&
    toastSmokeSource.includes("different API errors should remain separately visible") &&
    toastSmokeSource.includes("manual close should remove only the selected toast") &&
    toastSmokeSource.includes("toast stack should keep only the latest three distinct API errors") &&
    toastSmokeSource.includes('"422 Unprocessable Entity: manifest invalid"') &&
    toastSmokeSource.includes('getByRole("button", { name: "关闭提醒" })') &&
    routeWarmupSmokeSource.includes('"benchmarksPage"') &&
    routeWarmupSmokeSource.includes('"rankBoardPage"') &&
    routeWarmupSmokeSource.includes('"runsPage"') &&
    routeWarmupSmokeSource.includes('"jobsPage"') &&
    routeWarmupSmokeSource.includes('"suiteReportPage"') &&
    routeWarmupSmokeSource.includes('"comparePage"') &&
    routeWarmupSmokeSource.includes('"comparisonSamplePage"') &&
    routeWarmupSmokeSource.includes('"servicesPage"') &&
    routeWarmupSmokeSource.includes('"settingsPage"') &&
    routeWarmupSmokeSource.includes('performance.getEntriesByType("resource")') &&
    routeWarmupSmokeSource.includes("core route chunks must warm up after idle") &&
    routeWarmupSmokeSource.includes('Object.defineProperty(navigator, "connection"') &&
    routeWarmupSmokeSource.includes("saveData: true") &&
    routeWarmupSmokeSource.includes("route warmup must not fetch core chunks when navigator.connection.saveData is true") &&
    routeWarmupSmokeSource.includes('!text.includes("/api/")') &&
    navPrefetchSmokeSource.includes('const expectedPrefetches = [') &&
    navPrefetchSmokeSource.includes('{ label: "评测中心", pathPrefix: "/api/jobs", query: "limit=80" }') &&
    navPrefetchSmokeSource.includes('{ label: "排行榜", pathPrefix: "/api/rank-board", query: "sort_by=f1_iou50" }') &&
    navPrefetchSmokeSource.includes("async function waitForApiRequest(prefetch)") &&
    navPrefetchSmokeSource.includes("async function assertPrefetchFailureStaysSilent()") &&
    navPrefetchSmokeSource.includes("prefetch failure must not show toast") &&
    navPrefetchSmokeSource.includes("failed route prefetch should retry on the next intent") &&
    navPrefetchSmokeSource.includes("async function waitForPrefetchCount") &&
    navPrefetchSmokeSource.includes("async function assertSaveDataSkipsPrefetch()") &&
    navPrefetchSmokeSource.includes("nav prefetch must not fetch API data when navigator.connection.saveData is true") &&
    navPrefetchSmokeSource.includes("runs endpoint should be prefetched once for results and once for compare, not once per hover") &&
    loadingStateSmokeSource.includes('delayedApi: "/api/benchmarks", label: "正在加载基准集"') &&
    loadingStateSmokeSource.includes('delayedApi: "/api/runs", label: "正在加载评测记录"') &&
    loadingStateSmokeSource.includes('delayedApi: "/api/rank-board", label: "正在加载排行榜"') &&
    loadingStateSmokeSource.includes('delayedApi: "/api/jobs", label: "正在加载队列状态"') &&
    loadingStateSmokeSource.includes('delayedApi: "/api/services", label: "正在加载服务"') &&
    loadingStateSmokeSource.includes(".table-loading .table-refresh-indicator") &&
    loadingStateSmokeSource.includes("await page.emulateMedia({ reducedMotion: \"reduce\" })") &&
    loadingStateSmokeSource.includes("reduced-motion loading skeleton must not animate") &&
    loadingStateSmokeSource.includes("reduced-motion table refresh line must not animate") &&
    loadingStateSmokeSource.includes("async function assertDarkThemeLoadingState()") &&
    loadingStateSmokeSource.includes("dark loading table shell must not fall back to a bright surface") &&
    loadingStateSmokeSource.includes("dark loading skeleton glint must not use white shimmer") &&
    loadingStateSmokeSource.includes("dark loading indicator must not use a bright pill surface") &&
    loadingStateSmokeSource.includes("dark run note preview must not use a bright filled surface") &&
    loadingStateSmokeSource.includes("dark empty run note preview must not use a bright empty surface") &&
    loadingStateSmokeSource.includes("dark row selection checkbox must not use a bright native surface") &&
    loadingStateSmokeSource.includes("async function assertDarkRankBoardTableControls()") &&
    loadingStateSmokeSource.includes("dark rank primary score must not use a bright score pill") &&
    loadingStateSmokeSource.includes("dark rank delta must not use a bright delta pill") &&
    loadingStateSmokeSource.includes("dark rank active metric cell must not use a bright active background") &&
    loadingStateSmokeSource.includes("dark rank mode switch must not use a bright switch surface") &&
    loadingStateSmokeSource.includes("dark rank pager must not use a bright pagination surface") &&
    loadingStateSmokeSource.includes("dark rank pager text must stay readable") &&
    loadingStateSmokeSource.includes("dark rank summary title must not use light-theme dark ink") &&
    loadingStateSmokeSource.includes("dark rank facet group must not use a bright group surface") &&
    loadingStateSmokeSource.includes("dark rank facet chip must not use a bright chip surface") &&
    loadingStateSmokeSource.includes("dark rank facet toggle must not use a bright toggle surface") &&
    loadingStateSmokeSource.includes("function sampleRankEntries()") &&
    loadingStateSmokeSource.includes("function sampleRankFacets()") &&
    loadingStateSmokeSource.includes("function sampleRuns()") &&
    loadingStateSmokeSource.includes("must not use a full-page empty panel for first load") &&
    layoutSmokeSource.includes('{ name: "wide", width: 1920, height: 1080 }') &&
    layoutSmokeSource.includes('{ name: "compact", width: 1024, height: 760 }') &&
    layoutSmokeSource.includes('{ name: "landscape", width: 900, height: 430 }') &&
    layoutSmokeSource.includes("await assertPagersFit(page") &&
    layoutSmokeSource.includes("async function assertPagersFit(page, scope)") &&
    layoutSmokeSource.includes('page.locator(".rank-board-pager")') &&
    layoutSmokeSource.includes("pager ${pager.index} clips horizontally") &&
    layoutSmokeSource.includes("${name} leaks outside pager") &&
    layoutSmokeSource.includes("if (state.sortHeaderCount === 0)") &&
    layoutSmokeSource.includes("rank table header sorting contract failed") &&
    layoutSmokeSource.includes("if (!collapsedState.hasIndexMeter)") &&
    layoutSmokeSource.includes("composite report visual stage did not render") &&
    layoutSmokeSource.includes("const recentMinimumHeight = overviewState.runRows.length > 0 ? 120 : 80") &&
    layoutSmokeSource.includes("const minimumStageHeight = state.viewportHeight <= 460 ? 140 : 180") &&
    packageJsonSource.includes('"test:select-popover": "node scripts/test-select-popover-model.mjs"') &&
    packageJsonSource.includes('"test:select-popover-ui": "node scripts/select-popover-smoke-check.mjs"') &&
    readmeSource.includes("npm run test:select-popover") &&
    readmeSource.includes("npm run test:select-popover-ui") &&
    readmeSource.includes("npm run test:route-warmup") &&
    readmeSource.includes("npm run test:nav-prefetch") &&
    readmeSource.includes("npm run test:loading-state") &&
    readmeSource.includes("npm run test:toast") &&
    readmeSource.includes("npm run test:dialogs") &&
    readmeSource.includes("npm run test:settings-preview") &&
    readmeSource.includes("npm run test:smoke") &&
    readmeSource.includes("`test:select-popover` 会检查共享下拉控件的窗口化渲染和键盘导航 model（`src/selectPopoverModel.ts`）") &&
    readmeSource.includes("`test:select-popover-ui` 是共享下拉控件的浏览器 smoke") &&
    readmeSource.includes("`test:route-warmup` 是首屏后路由预热 smoke") &&
    readmeSource.includes("`test:nav-prefetch` 是主导航意图预取 smoke") &&
    readmeSource.includes("`test:loading-state` 是列表首屏加载态 smoke") &&
    readmeSource.includes("`test:toast` 是 API 错误提示 smoke") &&
    readmeSource.includes("settings preview 和 select popover UI smoke") &&
    compositeReportSmokeSource.includes('const url = new URL("/suite-report", baseUrl).toString();') &&
    compositeReportSmokeSource.includes('{ name: "wide", width: 1440, height: 900 }') &&
    compositeReportSmokeSource.includes('{ name: "desktop-narrow", width: 1180, height: 760 }') &&
    compositeReportSmokeSource.includes('{ name: "short-console", width: 980, height: 720 }') &&
    compositeReportSmokeSource.includes("composite-report-shell.sidebar-collapsed") &&
    compositeReportSmokeSource.includes("composite-composer-dock.collapsed") &&
    compositeReportSmokeSource.includes("composite-sidebar-drawer") &&
    compositeReportSmokeSource.includes("composite-sidebar-backdrop") &&
    compositeReportSmokeSource.includes("const hasImageNavigator =") &&
    compositeReportSmokeSource.includes("if (!hasImageNavigator)") &&
    compositeReportSmokeSource.includes("image-navigator-search input") &&
    compositeReportSmokeSource.includes("image-jump-popover") &&
    compositeReportSmokeSource.includes("image-jump-result") &&
    compositeReportSmokeSource.includes('composite-workbench-canvas[data-pointer-reticle="active"]') &&
    compositeReportSmokeSource.includes("composite-canvas-pointer-reticle") &&
    compositeReportSmokeSource.includes("composite-canvas-gesture-hud") &&
    readmeSource.includes("npm run test:composite-report") &&
    readmeSource.includes("组合报告专项浏览器 smoke") &&
    scriptsDocSource.includes("npm run test:composite-report") &&
    scriptsDocSource.includes("`test:composite-report` 专门覆盖组合报告页 `/suite-report`") &&
    evalBenchArchitectureSource.includes("`test:composite-report`"),
  "composite report smoke must protect collapsed composer, image jumping, and pointer feedback across core desktop viewports",
);
assertNoRawSelectElement(settingsControls, "settingsControls.tsx");
assert(
  settingsControls.includes("FormSelectControl") &&
    settingsControls.includes('className="inline-select-control"') &&
    settingsControls.includes("hideLabel"),
  "settings inline label color role select must use FormSelectControl",
);

const settingsPage = await readSource("src/settingsPage.tsx");
const settingsPreferenceDrawer = await readSource("src/settingsPreferenceDrawer.tsx");
const rawSettingsEditorGeometryPattern =
  /(?:\bfont-size:\s*(?:10|11|12|13|15|20)px\b|\bgap:\s*(?:2|3|4|6|8|10|18|24)px\b|\bpadding:\s*(?:14px 22px 12px|12px 16px|10px 14px|8px 10px|0 7px|0 9px|9px 10px|0 10px|2px|6px 8px|1px|0 22px 20px|3px 7px|7px 8px)\b|\bmin-height:\s*(?:24|28|30|36|38|46|58|68)px\b|\bborder-radius:\s*(?:2|4)px\b)/;
assert(
  settingsPage.includes("SearchInputControl") &&
    settingsPreferenceDrawer.includes("CompactSelectControl") &&
    settingsPreferenceDrawer.includes("NumberSettingControl") &&
    settingsPreferenceDrawer.includes("TextInputControl"),
  "settings page must keep search in the page shell and delegate setting controls to SettingsPreferenceDrawer",
);
assert(
  settingsPage.includes('import { SettingsPreferenceDrawer } from "./settingsPreferenceDrawer";') &&
    settingsPage.includes("<SettingsPreferenceDrawer") &&
    settingsPage.includes("visiblePanels={visiblePanels}") &&
    settingsPreferenceDrawer.includes("export function SettingsPreferenceDrawer") &&
    settingsPreferenceDrawer.includes("export type SettingsPanelId") &&
    settingsPreferenceDrawer.includes("export type SettingsSectionSummary") &&
    settingsPreferenceDrawer.includes("SettingsEditorSection") &&
    settingsPreferenceDrawer.includes("SettingsPreferenceRow") &&
    settingsPreferenceDrawer.includes("ShortcutSettingsPanel") &&
    !settingsPage.includes("SettingsEditorSection") &&
    !settingsPage.includes("ShortcutSettingsPanel") &&
    !settingsPage.includes("LabelColorQuickAdd") &&
    !settingsPage.includes("function isTypographyPresetActive"),
  "settings page must delegate preference drawer groups to a focused settingsPreferenceDrawer module",
);
assert(
  settingsPreferenceDrawer.includes("CompactSelectControl") &&
    settingsPreferenceDrawer.includes("NumberSettingControl") &&
    settingsPage.includes("SearchInputControl") &&
    settingsPreferenceDrawer.includes("TextInputControl"),
  "settings drawer selects must use CompactSelectControl",
);
assert(
  controlPrimitives.includes("export function NumberSettingControl") &&
    controlPrimitives.includes("formatNumberInputValue") &&
    !controlPrimitives.includes("<strong>{Number.isInteger(value)") &&
    !controlPrimitiveStyleSource.includes(".number-setting-control strong"),
  "settings number controls must not duplicate the current value beside the numeric input",
);
assert(
  typographySettingsSource.includes("export type TypographySettings") &&
    typographySettingsSource.includes("export const DEFAULT_TYPOGRAPHY_SETTINGS") &&
    typographySettingsSource.includes("export const TYPOGRAPHY_PRESETS") &&
    typographySettingsSource.includes("export function bootstrapTypographySettings") &&
    typographySettingsSource.includes("TYPOGRAPHY_STORAGE_VERSION_KEY") &&
    typographySettingsSource.includes("CURRENT_TYPOGRAPHY_STORAGE_VERSION") &&
    typographySettingsSource.includes("12px-default") &&
    typographySettingsSource.includes("localStorage.setItem(TYPOGRAPHY_STORAGE_VERSION_KEY") &&
    typographySettingsSource.includes("fontCssUrl") &&
    typographySettingsSource.includes("customFontName") &&
    typographySettingsSource.includes("customFontFileUrl") &&
    typographySettingsSource.includes("const LEGACY_DEFAULT_BASE_FONT_SIZES = [") &&
    typographySettingsSource.includes(
      "20.5",
    ) &&
    typographySettingsSource.includes("baseFontSize: 12") &&
    typographySettingsSource.includes("baseFontSize: 11") &&
    typographySettingsSource.includes("baseFontSize: 14") &&
    typographySettingsSource.includes("Math.max(10, Math.min(20, numeric))") &&
    typographySettingsSource.includes("isStoredOldDefaultTypography(parsed, normalized)") &&
    typographySettingsSource.includes("CUSTOM_FONT_LINK_ID") &&
    typographySettingsSource.includes("CUSTOM_FONT_FACE_STYLE_ID") &&
    typographySettingsSource.includes("applyTypographySettings") &&
    typographySettingsSource.includes("updateCustomFontLink") &&
    typographySettingsSource.includes("updateCustomFontFaceStyle") &&
    typographySettingsSource.includes("effectiveFontFamily") &&
    typographySettingsSource.includes("TYPOGRAPHY_CHANGED_EVENT") &&
    typographySettingsSource.includes("sameTypographySettings") &&
    settingsPage.includes('id: "typography"') &&
    settingsPage.includes("typographySettings.baseFontSize") &&
    settingsPreferenceDrawer.includes("TYPOGRAPHY_PRESETS.map") &&
    settingsPreferenceDrawer.includes("isTypographyPresetActive") &&
    settingsPage.includes("useTypographySettings") &&
    settingsPreferenceDrawer.includes('settingKey="evalBench.typography"') &&
    settingsPreferenceDrawer.includes('label="界面字体族"') &&
    settingsPreferenceDrawer.includes('label="等宽字体族"') &&
    settingsPreferenceDrawer.includes('label="字体 CSS URL"') &&
    settingsPreferenceDrawer.includes('label="自定义字体名称"') &&
    settingsPreferenceDrawer.includes('label="字体文件 URL"') &&
    settingsPreferenceDrawer.includes('label="基础字号"') &&
    settingsPreferenceDrawer.includes("min={10}") &&
    settingsPreferenceDrawer.includes("max={20}") &&
    settingsPage.includes("resetTypographySettings") &&
    settingsPage.includes('import "./settingsTypography.css";') &&
    settingsTypographyStyleSource.includes(".typography-preset-grid") &&
    settingsTypographyStyleSource.includes(".typography-preset-card") &&
    settingsTypographyStyleSource.includes(".settings-typography-grid") &&
    settingsTypographyStyleSource.includes(".typography-preview-strip") &&
    settingsTypographyStyleSource.includes(':root[data-theme="dark"] .typography-preset-grid') &&
    settingsTypographyStyleSource.includes("--settings-typography-card-bg") &&
    settingsTypographyStyleSource.includes("--settings-typography-preview-code") &&
    settingsTypographyStyleSource.includes("box-shadow: var(--settings-typography-card-active-shadow)") &&
    !settingsTypographyStyleSource.includes("linear-gradient(180deg, #ffffff 0%, #f5f9fc 100%)") &&
    settingsTypographyStyleSource.includes("font-family: var(--app-font-family)") &&
    settingsTypographyStyleSource.includes("font-family: var(--mono-font)") &&
    !settingsEditorStyleSource.includes(".typography-preset-grid") &&
    !settingsEditorStyleSource.includes(".typography-preset-card") &&
    !settingsEditorStyleSource.includes(".settings-typography-grid") &&
    !settingsEditorStyleSource.includes(".typography-preview-strip"),
  "settings page must expose typography density controls plus loadable font CSS and font files through typographySettings",
);
assert(
  settingsPage.includes('import "./settingsTheme.css";') &&
    settingsPage.indexOf('import "./settingsTheme.css";') <
      settingsPage.indexOf('import "./settingsWorkbench.css";') &&
    settingsThemeStyleSource.includes("--settings-gap-24") &&
    settingsThemeStyleSource.includes("--settings-pad-22") &&
    settingsThemeStyleSource.includes("--settings-text-caption") &&
    settingsThemeStyleSource.includes("--settings-editor-head-min") &&
    settingsThemeStyleSource.includes("--settings-shortcut-row-min") &&
    settingsThemeStyleSource.includes("--settings-preview-foot-min") &&
    settingsEditorStyleSource.includes("var(--settings-text-caption)") &&
    settingsEditorStyleSource.includes("var(--settings-editor-head-min)") &&
    settingsEditorStyleSource.includes(':root[data-theme="dark"] .settings-editor-pane') &&
    settingsEditorStyleSource.includes("--settings-editor-bg") &&
    settingsEditorStyleSource.includes("--settings-inline-action-bg") &&
    !settingsEditorStyleSource.includes("linear-gradient(180deg, #ffffff 0%, #ecf6fa 100%)") &&
    settingsShortcutsStyleSource.includes("var(--settings-shortcut-row-min)") &&
    settingsShortcutsStyleSource.includes(':root[data-theme="dark"] .shortcut-map-table') &&
    settingsShortcutsStyleSource.includes("--shortcut-map-button-bg") &&
    settingsShortcutsStyleSource.includes("content-visibility: auto;") &&
    settingsShortcutsStyleSource.includes("contain-intrinsic-size: auto 48px;") &&
    settingsWorkbenchStyleSource.includes("var(--settings-preset-min)") &&
    settingsWorkbenchStyleSource.includes(':root[data-theme="dark"] .settings-workbench-shell') &&
    settingsWorkbenchStyleSource.includes("--settings-command-bg") &&
    settingsWorkbenchStyleSource.includes("--settings-section-active-shadow") &&
    settingsWorkbenchStyleSource.includes("box-shadow: var(--settings-shell-shadow)") &&
    !settingsWorkbenchStyleSource.includes("0 22px 54px") &&
    !settingsWorkbenchStyleSource.includes("linear-gradient(180deg, #ffffff 0%, #f5f8fb 100%)") &&
    settingsPreviewStyleSource.includes("var(--settings-preview-foot-min)") &&
    settingsDrawerStyleSource.includes("var(--settings-drawer-head-min)") &&
    settingsDrawerStyleSource.includes(':root[data-theme="dark"] .settings-preference-drawer') &&
    settingsDrawerStyleSource.includes("--settings-drawer-bg") &&
    settingsDrawerStyleSource.includes("scrollbar-gutter: stable") &&
    !settingsDrawerStyleSource.includes("background: #ffffff") &&
    !settingsDrawerStyleSource.includes("color: #162536") &&
    !rawSettingsEditorGeometryPattern.test(settingsEditorStyleSource),
  "settings workbench styles must use the shared settings theme tokens instead of repeated editor geometry literals",
);
assert(
  settingsPage.includes('import "./settingsWorkbench.css";') &&
    settingsPage.includes('import "./settingsPreview.css";') &&
    settingsPage.includes('import "./settingsDrawer.css";') &&
    settingsPage.includes('import "./settingsEditor.css";') &&
    settingsPage.includes('import "./settingsTypography.css";') &&
    settingsPage.includes('import "./settingsLabels.css";') &&
    settingsPage.includes('import "./settingsShortcuts.css";') &&
    settingsPage.indexOf('import "./settingsEditor.css";') <
      settingsPage.indexOf('import "./settingsTypography.css";') &&
    settingsPage.indexOf('import "./settingsTypography.css";') <
      settingsPage.indexOf('import "./settingsLabels.css";') &&
    settingsPage.indexOf('import "./settingsLabels.css";') <
      settingsPage.indexOf('import "./settingsShortcuts.css";') &&
    !settingsPage.includes('import "./settingsPage.css";') &&
    settingsWorkbenchStyleSource.includes(".settings-workbench-page") &&
    settingsWorkbenchStyleSource.includes(".settings-command-bar") &&
    settingsWorkbenchStyleSource.includes(".settings-search-box") &&
    settingsPreviewStyleSource.includes(".settings-preview-pane") &&
    settingsPreviewStyleSource.includes(".settings-preview-stage") &&
    !settingsPreviewStyleSource.includes("backdrop-filter") &&
    !settingsPreviewStyleSource.includes("rgb(7 13 20 / 76%)") &&
    settingsPreviewSmokeSource.includes('process.env.EVAL_BENCH_URL ?? "http://127.0.0.1:4173/settings"') &&
    settingsPreviewSmokeSource.includes('!text.includes("/api/")') &&
    settingsPreviewSmokeSource.includes('!text.includes("Failed to load resource")') &&
    settingsDrawerStyleSource.includes(".settings-preference-drawer") &&
    settingsDrawerStyleSource.includes(".settings-drawer-head") &&
    settingsLabelsStyleSource.includes(".settings-label-row") &&
    settingsLabelsStyleSource.includes(".settings-label-role-grid") &&
    settingsLabelsStyleSource.includes(':root[data-theme="dark"] .settings-label-table') &&
    settingsLabelsStyleSource.includes("--settings-label-ink") &&
    settingsLabelsStyleSource.includes("content-visibility: auto;") &&
    settingsLabelsStyleSource.includes("contain-intrinsic-size: auto 42px;") &&
    settingsShortcutsStyleSource.includes(".shortcut-map-row") &&
    settingsShortcutsStyleSource.includes(".shortcut-capture") &&
    settingsShortcutsStyleSource.includes("--shortcut-map-conflict-bg") &&
    settingsEditorStyleSource.includes(".settings-inline-action .app-icon") &&
    settingsEditorStyleSource.includes(".settings-preference-row") &&
    !settingsWorkbenchStyleSource.includes(".settings-preview-pane") &&
    !settingsPreviewStyleSource.includes(".settings-preference-row") &&
    !settingsDrawerStyleSource.includes(".shortcut-map-row") &&
    !settingsEditorStyleSource.includes(".settings-command-bar") &&
    !settingsEditorStyleSource.includes(".settings-label-row") &&
    !settingsEditorStyleSource.includes(".settings-label-role-grid") &&
    !settingsEditorStyleSource.includes(".shortcut-map-row") &&
    !settingsEditorStyleSource.includes(".shortcut-capture") &&
    !settingsEditorStyleSource.includes(".settings-command-list") &&
    !settingsEditorStyleSource.includes(".settings-color-strip") &&
    !settingsWorkbenchStyleSource.includes(".settings-grid") &&
    !settingsPreviewStyleSource.includes(".settings-preview-card") &&
    !settingsEditorStyleSource.includes(".settings-control-card") &&
    !settingsEditorStyleSource.includes(".settings-workflow-card") &&
    !settingsEditorStyleSource.includes(".settings-note-grid") &&
    !settingsPreviewStyleSource.includes(".settings-preview-svg") &&
    !appThemeStyleSource.includes(".settings-workbench-page") &&
    !appThemeStyleSource.includes(".settings-command-bar") &&
    !appThemeStyleSource.includes(".shortcut-map-row") &&
    !appThemeStyleSource.includes(".settings-preview-pane") &&
    !appThemeStyleSource.includes(".settings-grid") &&
    !appThemeStyleSource.includes(".settings-preview-card") &&
    !appThemeStyleSource.includes(".settings-control-card") &&
    !appThemeStyleSource.includes(".settings-workflow-card") &&
    !appThemeStyleSource.includes(".settings-note-grid") &&
    !appThemeStyleSource.includes(".settings-preview-svg") &&
    !designSource.includes(".settings-workbench-page") &&
    !designSource.includes(".settings-command-bar") &&
    !designSource.includes(".shortcut-map-row") &&
    !designSource.includes(".settings-preview-pane") &&
    !designSource.includes(".settings-grid") &&
    !designSource.includes(".settings-preview-card") &&
    !designSource.includes(".settings-control-card") &&
    !designSource.includes(".settings-workflow-card") &&
    !designSource.includes(".settings-note-grid") &&
    !designSource.includes(".settings-preview-svg") &&
    !designSource.includes(".settings-inline-action"),
  "settings workbench styles must live in focused settings CSS modules instead of one page blob or global base/design CSS",
);
assert(
  /<CompactSelectControl\s+dense\s+label="预测线型"/.test(settingsPreferenceDrawer),
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
  settingsPreferenceDrawer.includes("InlineColorControl") &&
    !/<input\b/.test(settingsPreferenceDrawer),
  "settings label color grid must use InlineColorControl instead of raw color inputs",
);
assert(
  settingsPage.includes("SelectableCardButton") &&
    !/<button[\s\S]{0,220}settings-section-button/.test(settingsPage),
  "settings section navigation must use SelectableCardButton instead of raw section buttons",
);
assert(
  !settingsPreferenceDrawer.includes('className="compact-select dense"'),
  "settings page must not create ad hoc compact select shells",
);
assert(
  !/<button[^>]+className="settings-inline-action"/.test(settingsPreferenceDrawer),
  "settings inline standard actions must use ActionButton",
);
assert(
  !/<button[^>]+onRemoveLabelColor/.test(settingsPreferenceDrawer),
  "settings label clear action must use ActionButton",
);
assert(
  apiSource.includes('import type {') &&
    apiSource.includes('} from "./apiTypes";') &&
    apiSource.includes('export type * from "./apiTypes";') &&
    apiTypesSource.includes("export type TargetLabelResolution =") &&
    apiTypesSource.includes("export type TargetLabelResolutionParams =") &&
    apiTypesSource.includes("export type CompositeSampleView =") &&
    apiTypesSource.includes("export type RunSampleDetail =") &&
    apiTypesSource.includes("export type JobPreflightResult =") &&
    !apiSource.includes("export type CompositeSampleView =") &&
    !apiSource.includes("export type RunSampleDetail =") &&
    apiSource.includes("export function fetchTargetLabelResolution(") &&
    apiSource.includes('params.append("target_label", value)') &&
    apiSource.includes('fetchJson<TargetLabelResolution>(`/api/target-labels'),
  "api client must keep schema types in apiTypes while exposing agent-safe target label resolution endpoint",
);
assert(
  overviewPage.includes("export function OverviewPage()"),
  "overview page module must export OverviewPage",
);
assert(
    overviewModelSource.includes("export function useOverviewModel()") &&
    overviewPage.includes("const overview = useOverviewModel();") &&
    overviewModelSource.includes('queryKey: ["overview-jobs-total"]') &&
    overviewModelSource.includes("queryFn: ({ signal }) => fetchJobs({ limit: 1 }, { signal })") &&
    overviewModelSource.includes('queryKey: ["overview-services-total"]') &&
    overviewModelSource.includes("queryFn: ({ signal }) => fetchServices({ limit: 1 }, { signal })") &&
    overviewModelSource.includes("const jobStatusFacets = jobTotalQuery.data?.facets?.statuses") &&
    overviewModelSource.includes('facetCount(jobStatusFacets, "queued")') &&
    overviewModelSource.includes('facetCount(jobStatusFacets, "running")') &&
    overviewModelSource.includes('facetCount(jobStatusFacets, "failed")') &&
    overviewModelSource.includes('facetCount(serviceTotalQuery.data?.facets?.statuses, "running")') &&
    overviewModelSource.includes("function jobPageTotal(") &&
    overviewModelSource.includes("function servicePageTotal(") &&
    overviewModelSource.includes("function facetCount(") &&
    overviewModelSource.includes("const totalJobs = Math.max(") &&
    overviewModelSource.includes("const serviceCount = Math.max(servicePageTotal(serviceTotalQuery.data), liveServices)") &&
    !overviewPage.includes("useQuery(") &&
    !overviewPage.includes("fetchSchedulerStatus") &&
    !overviewPage.includes("function jobPageTotal(") &&
    !overviewPage.includes("function servicePageTotal(") &&
    !overviewPage.includes("function overviewNextAction(") &&
    !overviewPage.includes("function bestF1Run(") &&
    !overviewModelSource.includes('fetchJobs({ limit: 500 })') &&
    !overviewPage.includes('fetchServices({ limit: 500 })') &&
    !overviewModelSource.includes('fetchJobs({ status: "queued", limit: 1 })') &&
    !overviewModelSource.includes('fetchJobs({ status: "running", limit: 1 })') &&
    !overviewModelSource.includes('fetchJobs({ status: "failed", limit: 1 })') &&
    !overviewModelSource.includes('fetchServices({ status: "running", limit: 1 })') &&
    !overviewPage.includes('jobs.filter((job) => job.status === "queued").length') &&
    !overviewPage.includes('services.filter((service) => service.status === "running").length') &&
    !overviewPage.includes("serviceCount: services.length") &&
    !overviewPage.includes("Math.max(jobs.length, 1)") &&
    !overviewPage.includes("Math.max(services.length, 1)"),
  "overview job/service runtime counts must use backend facet totals instead of repeated filtered requests or first-page list estimates",
);
assert(
  overviewPage.includes("overview-home-v18") &&
    overviewPage.includes("overview-v18-grid") &&
    overviewPage.includes("OverviewPrimaryCard") &&
    overviewPage.includes("overview-v18-primary") &&
    overviewPage.includes("OverviewQueueCard") &&
    overviewPage.includes("overview-v18-queue") &&
    overviewPage.includes("OverviewResourceCard") &&
    overviewPage.includes("overview-v18-resources") &&
    overviewPage.includes("OverviewRecentRunsPanel") &&
    overviewPage.includes("overview-v18-recent") &&
    overviewPage.includes("OverviewFlowItem") &&
    overviewPage.includes("overview-v18-flow-item") &&
    overviewPage.includes("OverviewLens") &&
    overviewPage.includes("overview-v18-console") &&
    overviewPage.includes("overview-v18-surface-tab") &&
    overviewPage.includes("overview-v18-surface-body") &&
    overviewPage.includes("overview-v18-signal-map") &&
    overviewPage.includes("overview-v18-signal-node") &&
    overviewPage.includes("overview-v18-signal-inspector") &&
    overviewPage.includes("overviewSignalNodes") &&
    overviewPage.includes("function overviewActionIcon(") &&
    overviewPage.includes("OptionChipButton") &&
    !overviewPage.includes("overviewHeroTitle") &&
    overviewPage.includes("overview-v18-score") &&
    overviewModelSource.includes("function bestF1Run(") &&
    overviewModelSource.includes("recentRunsByCreatedAt(data.runs") &&
    overviewModelSource.includes("export type OverviewActionIcon") &&
    overviewPage.includes('import { errorMessage, formatMetric, runF1Score } from "./formatters";') &&
    overviewPage.includes("errorMessage(overview.error)") &&
    overviewPage.includes("overview-v18-run-artifacts") &&
    overviewPage.includes("overview-v18-run-score") &&
    overviewPage.includes('import { runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";') &&
    !overviewPage.includes("updateOverviewPointer") &&
    !overviewPage.includes("overview-home-v17") &&
    !overviewPage.includes("overview-ops-board") &&
    !overviewPage.includes("overview-rank-console") &&
    !overviewPage.includes("overview-decision-metric") &&
    !overviewPage.includes("overview-telemetry-trace") &&
    !overviewPage.includes("overview-resource-chips") &&
    !overviewPage.includes("overview-state-strip") &&
    !overviewPage.includes("overview-score-dial") &&
    !overviewPage.includes("overview-run-focus") &&
    !overviewPage.includes("overview-ops-signal") &&
    !overviewPage.includes("overview-flow-spine") &&
    !overviewPage.includes("overview-flow-node") &&
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
  overviewStyleSource.includes('@import "./overviewShell.css";') &&
    overviewStyleSource.includes('@import "./overviewPrimary.css";') &&
    overviewStyleSource.includes('@import "./overviewConsole.css";') &&
    overviewStyleSource.includes('@import "./overviewOperations.css";') &&
    overviewStyleSource.includes('@import "./overviewResponsive.css";') &&
    overviewShellStyleSource.includes("Overview v18: compact operator workspace") &&
    overviewShellStyleSource.includes(".dashboard-home.overview-home-v18") &&
    overviewShellStyleSource.includes(".overview-home-v18") &&
    overviewShellStyleSource.includes(".overview-v18-grid") &&
    overviewShellStyleSource.includes(".overview-v18-primary") &&
    overviewShellStyleSource.includes(".overview-v18-recent") &&
    overviewShellStyleSource.includes(".overview-v18-card::before") &&
    overviewShellStyleSource.includes(".overview-v18-icon-link:hover") &&
    overviewShellStyleSource.includes("--overview-home-bg: #f4f7f9") &&
    overviewShellStyleSource.includes(':root[data-theme="dark"] .dashboard-home.overview-home-v18') &&
    overviewShellStyleSource.includes("background: var(--overview-home-bg)") &&
    overviewPrimaryStyleSource.includes(".overview-v18-flow-item:hover") &&
    overviewPrimaryStyleSource.includes(".overview-v18-score:hover") &&
    overviewPrimaryStyleSource.includes("color: var(--overview-tile-strong)") &&
    overviewPrimaryStyleSource.includes("background: var(--overview-tile-bg)") &&
    overviewPrimaryStyleSource.includes("box-shadow: none") &&
    overviewConsoleStyleSource.includes(".overview-v18-console") &&
    overviewConsoleStyleSource.includes("--overview-console-bg") &&
    overviewConsoleStyleSource.includes(':root[data-theme="dark"] .overview-v18-console') &&
    overviewConsoleStyleSource.includes("background: var(--overview-console-bg)") &&
    !overviewConsoleStyleSource.includes("background: #f8fafb") &&
    !overviewConsoleStyleSource.includes("background: #ffffff") &&
    overviewConsoleStyleSource.includes(".overview-v18-signal-map") &&
    overviewConsoleStyleSource.includes(".overview-v18-signal-node.active") &&
    overviewConsoleStyleSource.includes(".overview-v18-signal-inspector") &&
    overviewOperationsStyleSource.includes(".overview-v18-run-list") &&
    overviewOperationsStyleSource.includes(".overview-v18-service-line") &&
    overviewOperationsStyleSource.includes("background: var(--overview-tile-bg)") &&
    overviewOperationsStyleSource.includes("background: var(--overview-meter-fill)") &&
    overviewOperationsStyleSource.includes("content-visibility: auto;") &&
    overviewOperationsStyleSource.includes("contain-intrinsic-size: auto 58px;") &&
    !overviewPrimaryStyleSource.includes("color: #111827") &&
    !overviewPrimaryStyleSource.includes("background: #f8fafb") &&
    !overviewOperationsStyleSource.includes("background: #f8fafb") &&
    !overviewOperationsStyleSource.includes("color: #111827") &&
    overviewResponsiveStyleSource.includes("@media (min-width: 1320px)") &&
    appChromeVisualStyleSource.includes(".content::before") &&
    appChromeVisualStyleSource.includes("display: none") &&
    workspaceShellStyleSource.includes(".workspace-card:not(.fill):hover") &&
    appChromeVisualStyleSource.includes(".nav-item:hover .app-icon") &&
    appChromeVisualStyleSource.includes(".nav-item::after") &&
    appChromeVisualStyleSource.includes("content: none") &&
    !appChromeVisualStyleSource.includes("translateX(") &&
    !appChromeVisualStyleSource.includes(".user-profile-chip") &&
    appChromeVisualStyleSource.includes(".topbar .status-pill {\n  position: relative;\n  overflow: hidden;") &&
    !appChromeVisualStyleSource.includes(".topbar .status-pill:hover") &&
    !appChromeVisualStyleSource.includes("@keyframes status-breathe") &&
    !appChromeVisualStyleSource.includes("@keyframes status-online-breathe") &&
    !appChromeVisualStyleSource.includes(".status-pill.online") &&
    !appChromeVisualStyleSource.includes(".status-pill.loading"),
  "overview and shared controls must keep focused layout while suppressing decorative motion",
);
assert(
  ![
    "overview-home-v17",
    "overview-ops-board",
    "overview-rank-console",
    "overview-decision-metric",
    "overview-telemetry-trace",
    "overview-resource-chips",
    "overview-state-strip",
    "overview-score-dial",
    "overview-run-focus",
    "overview-ops-signal",
    "overview-flow-spine",
    "overview-flow-node",
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
    "overview-evidence-row",
    "overview-loop-panel",
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
  "overview stylesheet must expose only the active v18 surface and block deprecated design tracks",
);
assert(
  !readmeSource.includes("overview-evidence-row`、\n`overview-ops-signal") &&
    !readmeSource.includes("overview-telemetry-trace`、`overview-loop-panel") &&
    !scriptsDocSource.includes("overview-evidence-row`、`overview-ops-signal") &&
    !scriptsDocSource.includes("overview-telemetry-trace`、`overview-loop-panel") &&
    !scriptsDocSource.includes("rank console、loop 和最近 run 面板") &&
    readmeSource.includes("overview-command-shell`、`overview-evidence-row`、`overview-loop-panel`") &&
    scriptsDocSource.includes("`overview-evidence-row`、`overview-loop-panel`、`overview-signal-stack`"),
  "README and scripts docs must document overview evidence/loop panels as deprecated, not active v17 contract",
);
assert(
  !/recall|precision|mIoU|R@\.50|P@\.50/i.test(overviewPage) &&
    !/overview-home-v18[\s\S]*grid-template-columns:\s*repeat\((?:[5-9]|\d{2,})/.test(overviewStyleSource),
  "overview command desk must avoid fine metric copy and five-plus column grids",
);
assert(
  mainEntry.includes('import { OverviewPage } from "./overviewPage";'),
  "main.tsx must route to the extracted OverviewPage module",
);
assert(
  appShellSource.includes('className="sidebar-toggle"') &&
    appShellSource.includes("<IconActionButton") &&
    !/<button[\s\S]{0,180}className="sidebar-toggle"/.test(appShellSource),
  "sidebar collapse control must use IconActionButton instead of a raw button",
);
assert(
  appShellSource.includes("this.setState({ error: null })") &&
    appShellSource.includes("重试渲染") &&
    !appShellSource.includes("window.location.reload()"),
  "fatal render boundary must retry in place instead of forcing a full page reload",
);
assert(
  benchmarksPage.includes("export function BenchmarksPage()") &&
    benchmarksPage.includes('export { BenchmarkDetailPage } from "./benchmarkSampleInspector";') &&
    benchmarksPage.includes('import { BenchmarkCreatePanel } from "./benchmarkCreatePanel";') &&
    benchmarksPage.includes('import { benchmarkSplitValues } from "./benchmarkModel";') &&
    !benchmarksPage.includes("function BenchmarkCreatePanel(") &&
    !benchmarksPage.includes("function BenchmarkSampleViewer(") &&
    !benchmarksPage.includes("function parseBenchmarkSlices("),
  "benchmarks page module must stay as the list-page shell and delegate create/detail responsibilities",
);
assert(
  benchmarkCreatePanelSource.includes("CheckboxFieldControl, TextareaControl, TextInputControl") &&
    (benchmarkCreatePanelSource.match(/<TextInputControl/g) ?? []).length >= 5 &&
    (benchmarkCreatePanelSource.match(/<CheckboxFieldControl/g) ?? []).length >= 3 &&
    benchmarkCreatePanelSource.includes("<TextareaControl"),
  "benchmark creation dialog must use shared text, textarea, and checkbox form controls",
);
assert(
  benchmarkCreatePanelSource.includes('placeholder="grounding_layout_main"') &&
    !benchmarkCreatePanelSource.includes("multitask_val_v1"),
  "benchmark creation dialog must avoid legacy fixture benchmark placeholders",
);
assert(
  benchmarkCreatePanelSource.includes("parseBenchmarkSlices(") &&
    benchmarkCreatePanelSource.includes("slices: suiteMode ? slices : undefined") &&
    benchmarkCreatePanelSource.includes("default_slice: slices[0]?.split") &&
    benchmarkCreatePanelSource.includes('label="Suite slices"') &&
    benchmarkCreatePanelSource.includes("suiteSliceParse.error") &&
    benchmarkCreatePanelSource.includes('normalizedSplit === "val"') &&
    benchmarkCreatePanelSource.includes('? "suite"') &&
    benchmarkCreatePanelSource.includes("suiteSliceParse.slices.length === 0") &&
    benchmarkCreatePanelSource.includes("&& tasks.length === 0") &&
    benchmarkCreatePanelSource.includes("Boolean(suiteSliceParse.error)") &&
    benchmarkModelSource.includes("export function parseBenchmarkSlices(") &&
    benchmarkModelSource.includes("Suite slices 至少需要一行 split=manifest") &&
    benchmarkModelSource.includes("Suite slices 第") &&
    benchmarkModelSource.includes("Suite slices split 重复") &&
    benchmarkModelSource.includes("Suite slices 不支持的任务"),
  "benchmark creation dialog must support suite slice payloads",
);
assert(
  benchmarksPage.includes("const BENCHMARK_PAGE_SIZE = 80;") &&
    benchmarksPage.includes("PagerControl") &&
    benchmarkSampleInspectorSource.includes("SamplePager") &&
    benchmarksPage.includes("clampListPageOffset") &&
    benchmarksPage.includes("updatePagedFilterValue") &&
    benchmarksPage.includes('className="rank-board-pager benchmark-list-pager"') &&
    benchmarksPage.includes("offset: pageOffset") &&
    benchmarksPage.includes("limit: BENCHMARK_PAGE_SIZE") &&
    !benchmarksPage.includes("function BenchmarkListPager(") &&
    !benchmarksPage.includes("limit: 200"),
  "benchmarks page must use paged API requests instead of a fixed 200-row slice",
);
assert(
  benchmarksPage.includes("updatePagedFilterValue(searchText, value, setSearchText, setPageOffset)") &&
    benchmarksPage.includes("updatePagedFilterValue(taskFilter, value, setTaskFilter, setPageOffset)") &&
    benchmarkSampleInspectorSource.includes("updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset)") &&
    benchmarkSampleInspectorSource.includes("updatePagedFilterValue(splitFilter, value, setSplitFilter, setPageOffset)") &&
    !benchmarksPage.includes("function updateBenchmarkFilter(") &&
    !benchmarkSampleInspectorSource.includes("setLabelFilter(value);\n    setPageOffset(0);") &&
    !benchmarkSampleInspectorSource.includes("setSplitFilter(value);\n    setPageOffset(0);") &&
    !benchmarkSampleInspectorSource.includes("useEffect(() => {\n    setPageOffset(0);"),
  "benchmarks page list and sample filters must use shared same-batch paging reset instead of issuing a stale-offset refresh",
);
assert(
  benchmarkSampleInspectorSource.includes("SelectableRowButton") &&
    !benchmarkSampleInspectorSource.includes('className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}'),
  "benchmark sample list rows must use SelectableRowButton",
);
assert(
  benchmarksPage.includes("errorMessage(benchmarksQuery.error)") &&
    benchmarkSampleInspectorSource.includes("errorMessage(samplesQuery.error)") &&
    benchmarkSampleInspectorSource.includes("errorMessage(detailQuery.error)") &&
    benchmarkCreatePanelSource.includes("errorMessage(mutation.error)") &&
    !benchmarkCreatePanelSource.includes("mutation.error.message") &&
    !benchmarksPage.includes('return <EmptyState title="基准集加载失败" tone="danger" />;') &&
    !benchmarkSampleInspectorSource.includes("<div className=\"empty-panel\">样本详情加载失败</div>"),
  "benchmark pages must show concrete API errors for list, detail, and mutation failures",
);
assertNoLegacyFormSubmitClass(benchmarksPage, "benchmarksPage.tsx");
assertNoLegacyFormSubmitClass(benchmarkCreatePanelSource, "benchmarkCreatePanel.tsx");
assert(
  mainEntry.includes('lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarksPage")') &&
    mainEntry.includes('lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarkDetailPage")'),
  "main.tsx must lazy-route to the extracted benchmarks page module",
);
assert(
  runsPage.includes("export function RunsPage()") &&
    runsPage.includes('export { RunDetailPage } from "./runDetailPage";') &&
    runDetailPageSource.includes("export function RunDetailPage()") &&
    runsPage.includes('import { ImportPredictionsPanel } from "./runsImportPanel";') &&
    runDetailPageSource.includes('import { RunConfigPanel, shouldOpenRunNotePanel } from "./runConfigPanel";') &&
    runDetailPageSource.includes('import { SampleFilters, SampleList } from "./runSampleSidebar";') &&
    !runsPage.includes("function ImportPredictionsPanel(") &&
    !runsPage.includes("function RunDetailPage(") &&
    !runsPage.includes('from "./runConfigPanel"') &&
    !runsPage.includes('from "./runSampleSidebar"') &&
    !runsPage.includes("function RunConfigPanel(") &&
    !runsPage.includes("function SampleFilters(") &&
    !runsPage.includes("function SampleList("),
  "runs page module must keep list concerns separate and re-export the extracted run detail module",
);
assert(
  runsImportPanelSource.includes("CheckboxFieldControl") &&
    runsImportPanelSource.includes("TextInputControl") &&
    (runsImportPanelSource.match(/<TextInputControl/g) ?? []).length >= 6 &&
    (runsImportPanelSource.match(/<CheckboxFieldControl/g) ?? []).length >= 3,
  "run import dialog must use shared text and checkbox form controls",
);
assert(
  runConfigPanelSource.includes("StandaloneTextareaControl") &&
    (runConfigPanelSource.match(/<StandaloneTextareaControl/g) ?? []).length >= 2 &&
    !/<textarea\b/.test(runConfigPanelSource),
  "run note editor must use shared standalone textarea controls",
);
assert(
  runsPage.includes("const RUN_PAGE_SIZE = 80;") &&
    runsPage.includes("PagerControl") &&
    runDetailPageSource.includes("SamplePager") &&
    runsPage.includes("clampListPageOffset") &&
    runsPage.includes("updatePagedFilterValue") &&
    runsPage.includes('className="rank-board-pager run-list-pager"') &&
    runsPage.includes("offset: pageOffset") &&
    runsPage.includes("limit: RUN_PAGE_SIZE") &&
    runsPage.includes("benchmarkSplitFilter") &&
    runsPage.includes("benchmarkSplit: benchmarkSplitFilter") &&
    runsPage.includes('id: "run-benchmark-split"') &&
    !runsPage.includes("function RunListPager(") &&
    !runsPage.includes("limit: 200"),
  "runs page must use paged API requests instead of a fixed 200-row slice",
);
assert(
  runsPage.includes("updatePagedFilterValue(searchText, value, setSearchText, setPageOffset)") &&
    runsPage.includes("updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)") &&
    runDetailPageSource.includes("updatePagedFilterValue(errorFilter, value, setErrorFilter, setPageOffset)") &&
    runDetailPageSource.includes("updatePagedFilterValue(labelFilter, value, setLabelFilter, setPageOffset)") &&
    !runsPage.includes("function updateRunFilter(") &&
    !runsPage.includes("setErrorFilter(value);\n    setPageOffset(0);") &&
    !runsPage.includes("setLabelFilter(value);\n    setPageOffset(0);") &&
    !runsPage.includes("useEffect(() => {\n    setPageOffset(0);"),
  "runs page list and sample filters must use shared same-batch paging reset instead of issuing a stale-offset refresh",
);
assert(
  runsPage.includes("readRunsViewState") &&
    runsPage.includes("writeRunsViewState") &&
    runsPage.includes("RUNS_VIEW_STATE_RESET_EVENT") &&
    runsViewStateSource.includes("export const RUNS_VIEW_STATE_KEY") &&
    runsViewStateSource.includes("export const RUNS_VIEW_STATE_RESET_EVENT") &&
    runsViewStateSource.includes("window.sessionStorage") &&
    appShellSource.includes('import { resetRunsViewState } from "./runsViewState";') &&
    appShellSource.includes("onNavigate={resetRunsViewState}"),
  "runs page must preserve filters and paging across run-detail back-navigation and reset from main nav",
);
assert(
  runSampleSidebarSource.includes("SelectableRowButton") &&
    !runSampleSidebarSource.includes('className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}'),
  "run sample list rows must use SelectableRowButton",
);
assert(
  runsPage.includes("errorMessage(runsQuery.error)") &&
    runDetailPageSource.includes("errorMessage(samplesQuery.error)") &&
    runDetailPageSource.includes("errorMessage(detailQuery.error)") &&
    runsImportPanelSource.includes("errorMessage(mutation.error)") &&
    runConfigPanelSource.includes("errorMessage(noteMutation.error)") &&
    runConfigPanelSource.includes("errorMessage(appendMutation.error)") &&
    !runsImportPanelSource.includes("mutation.error.message") &&
    !runConfigPanelSource.includes("noteMutation.error.message") &&
    !runConfigPanelSource.includes("appendMutation.error.message") &&
    !runsPage.includes('return <EmptyState title="评测记录加载失败" tone="danger" />;') &&
    !runDetailPageSource.includes("<div className=\"empty-panel\">样本详情加载失败</div>"),
  "run pages must show concrete API errors for list, detail, and mutation failures",
);
assert(
    runConfigPanelSource.includes("const RUN_NOTE_TEMPLATES = [") &&
    runConfigPanelSource.includes("const RUN_NOTE_APPEND_HEADINGS = [") &&
    runConfigPanelSource.includes("function insertNoteTemplate(") &&
    runConfigPanelSource.includes("appendRunNote(run.run_id, note, heading, noteVersion)") &&
    runConfigPanelSource.includes("const appendMutation = useMutation(") &&
    runConfigPanelSource.includes('className="run-note-append-panel"') &&
    runConfigPanelSource.includes('label="追加 run note"') &&
    runConfigPanelSource.includes("isApiError(error) && error.status === 409") &&
    (runConfigPanelSource.match(/invalidateQueries\(\{ queryKey: \["dashboard-state"\] \}\)/g) ?? []).length >= 2 &&
    runConfigPanelSource.includes('className="run-note-template-bar"') &&
    runConfigPanelSource.includes("<ActionButton") &&
    !runConfigPanelSource.includes('error.message.includes("409")') &&
    !runConfigPanelSource.includes("setNoteDraft(noteDraft +"),
  "run note editor must expose structured templates and refresh dashboard state after 409 conflicts",
);
assert(
  runConfigPanelSource.includes("DisclosurePanel") &&
    runDetailPageSource.includes('import "./runsPage.css";') &&
    runConfigPanelSource.includes('className="run-config-panel"') &&
    runConfigPanelSource.includes('className="prompt-details"') &&
    runsStyleSource.includes(".run-config-panel") &&
    runsStyleSource.includes(':root[data-theme="dark"] .run-config-panel') &&
    runsStyleSource.includes("--run-config-bg") &&
    runsStyleSource.includes("--run-note-editor-bg") &&
    runsStyleSource.includes("--run-prompt-code-bg") &&
    runsStyleSource.includes("box-shadow: 0 0 0 3px var(--run-note-field-focus-ring)") &&
    !runsStyleSource.includes("rgba(27, 94, 122, 0.06)") &&
    runsStyleSource.includes(".run-note-editor") &&
    runsStyleSource.includes(".run-note-template-bar") &&
    runsStyleSource.includes(".run-note-append-panel") &&
    runsStyleSource.includes(".run-config-grid") &&
    runsStyleSource.includes(".prompt-details") &&
    !appThemeStyleSource.includes(".run-config-panel") &&
    !appThemeStyleSource.includes(".run-note-editor") &&
    !appThemeStyleSource.includes(".run-note-template-bar") &&
    !appThemeStyleSource.includes(".run-config-grid") &&
    !appThemeStyleSource.includes(".prompt-details") &&
    !appThemeStyleSource.includes(".run-query-bar") &&
    !designSource.includes(".run-config-panel") &&
    !designSource.includes(".run-query-bar") &&
    !/<details\b/.test(runConfigPanelSource) &&
    !/<summary\b/.test(runConfigPanelSource),
  "run config, note, and prompt snapshot panels must use DisclosurePanel and keep page styles in runsPage.css",
);
assert(
  runTables.includes("StandaloneCheckboxControl") &&
    runTables.includes('className="row-select-checkbox"') &&
    !/<input\b/.test(runTables),
  "run table row selection must use StandaloneCheckboxControl",
);
assert(
  runTables.includes('hash="run-note"') &&
    runTables.includes('className={hasNote ? "run-note-preview" : "run-note-preview empty"}') &&
    runTables.includes('title={hasNote ? "有备注" : "无备注"}') &&
    runTables.includes("<FileText size={14} />") &&
    runTables.includes("<FileX size={14} />") &&
    runTablesStyleSource.includes(".run-note-preview") &&
    runTablesStyleSource.includes("--run-note-preview-bg") &&
    runTablesStyleSource.includes("--run-note-preview-empty-bg") &&
    runTablesStyleSource.includes(':root[data-theme="dark"] .run-note-preview') &&
    runTablesStyleSource.includes(".run-table-stack") &&
    runTablesStyleSource.includes(".workspace-card.fill .run-table-stack") &&
    !appThemeStyleSource.includes(".run-note-preview") &&
    !appThemeStyleSource.includes(".run-table-stack") &&
    !dataTableStyleSource.includes("run-note-preview") &&
    !designSource.includes(".run-note-preview") &&
    runConfigPanelSource.includes("export function shouldOpenRunNotePanel(") &&
    runConfigPanelSource.includes('id="run-note"') &&
    runConfigPanelSource.includes("open={configOpen}") &&
    runConfigPanelSource.includes("setConfigOpen(event.currentTarget.open)") &&
    runDetailPageSource.includes("defaultOpen={shouldOpenRunNotePanel()}"),
  "run note previews must deep-link to the editable run note panel",
);
assert(
  formattersSource.includes("export function f1Score(") &&
    formattersSource.includes("export function runF1Score(") &&
    formattersSource.includes("parts.push(run.model_id, `F1 ${formatMetric(runF1Score(run))}`)") &&
    formattersSource.includes("if (run.benchmark_id)") &&
    runTables.includes('import { formatDate, formatMetric, runF1Score } from "./formatters";') &&
    !runTables.includes('import { formatDate, formatMetric, runF1Score, unique } from "./formatters";') &&
    runTables.includes('header: "F1@.50"') &&
    runTables.includes("formatMetric(runF1Score(row.original))") &&
    compareRunRailComponentsSource.includes("runF1Score") &&
    compareRunRailComponentsSource.includes('className="compare-run-primary-metric"') &&
    compareRunRailComponentsSource.includes("F1 {formatMetric(runF1Score(selected))}"),
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
const jobsMiniLinkSource = jobsQueuePanelSource;
const compareMiniLinkSource = await readSource("src/comparePage.tsx");
const comparisonSampleMiniLinkSource = await readSource("src/comparisonSamplePage.tsx");
assert(
  runTables.includes("InlineNavLink") &&
    jobsMiniLinkSource.includes("InlineNavLink") &&
    compareMiniLinkSource.includes("InlineNavLink") &&
    !/<Link[^>]+className="mini-link/.test(runTables) &&
    !/<Link[^>]+className="mini-link/.test(rankBoardMiniLinkSource) &&
    !/<Link[^>]+className="mini-link/.test(jobsMiniLinkSource) &&
    !/<Link[^>]+className="mini-link/.test(compareMiniLinkSource),
  "router mini links must use InlineNavLink instead of ad hoc mini-link classes",
);
assert(
  jobsQueueTableSource.includes("SelectableTableRow") &&
    jobsQueueTableSource.includes("selected={selected}") &&
    !jobsQueueTableSource.includes('className={job.job_id === selectedJob?.job_id ? "selectable-row selected" : "selectable-row"}'),
  "jobs queue selectable rows must use SelectableTableRow instead of ad hoc selectable-row class composition",
);
assert(
  compareReportSamplesSource.includes("InlineAnchor") &&
    comparisonSampleMiniLinkSource.includes("InlineAnchor") &&
    runTables.includes("InlineAnchor") &&
    !/<a[^>]+className="mini-link/.test(compareReportSamplesSource) &&
    !/<a[^>]+className="mini-link/.test(comparisonSampleMiniLinkSource) &&
    !/<a[^>]+className=\{[^}]*mini-link/.test(runTables) &&
    !runTables.includes('"mini-link compare-ready"'),
  "href mini links must use InlineAnchor instead of ad hoc mini-link anchors",
);
assertNoLegacyFormSubmitClass(runsPage, "runsPage.tsx");
assertNoRawSelectElement(runsPage, "runsPage.tsx");
assert(
  runsImportPanelSource.includes("FormSelectControl") &&
    (runsImportPanelSource.match(/<FormSelectControl/g) ?? []).length >= 2,
  "runs import dialog selects must use FormSelectControl",
);
assert(
  runsImportPanelSource.includes("DetectionLabelSubtaskPanel") &&
    runsImportPanelSource.includes("const [targetLabels, setTargetLabels] = useState<string[]>([])") &&
    runsImportPanelSource.includes("target_labels: targetLabels") &&
    !runsImportPanelSource.includes("function parseTargetLabels("),
  "runs import dialog must use the shared detection label subtask panel instead of a free-text target label field",
);
assert(
  runsImportPanelSource.includes('const [benchmarkSplit, setBenchmarkSplit] = useState("auto")') &&
    runsImportPanelSource.includes("benchmarkImportSplitOptions(selectedBenchmark)") &&
    runsImportPanelSource.includes('split: benchmarkSplit === "auto" ? undefined : benchmarkSplit') &&
    runsImportPanelSource.includes('label="Benchmark split"') &&
    runsImportPanelSource.includes('{ value: "auto", label: "自动推断" }') &&
    runsImportPanelSource.includes("benchmark?.split_manifests"),
  "runs import dialog must allow explicit benchmark split selection for suite benchmarks",
);
assert(
  servicesPage.includes("const SERVICE_PAGE_SIZE = 80;") &&
    servicesPage.includes("PagerControl") &&
    servicesPage.includes("clampListPageOffset") &&
    servicesPage.includes("updatePagedFilterValue") &&
    servicesPage.includes('className="rank-board-pager service-list-pager"') &&
    servicesPage.includes("offset: pageOffset") &&
    servicesPage.includes("limit: SERVICE_PAGE_SIZE") &&
    !servicesPage.includes("function ServiceListPager(") &&
    !servicesPage.includes("limit: 200"),
  "services page must use paged API requests instead of a fixed 200-service slice",
);
assert(
  servicesPage.includes("updatePagedFilterValue(searchText, value, setSearchText, setPageOffset)") &&
    servicesPage.includes("updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)") &&
    !servicesPage.includes("function updateServiceFilter(") &&
    !servicesPage.includes("useEffect(() => {\n    setPageOffset(0);"),
  "services page filter changes must use shared same-batch paging reset instead of issuing a stale-offset refresh",
);
assert(
  servicesPage.includes('import { ServiceCreatePanel } from "./servicesCreatePanel";') &&
    servicesCreatePanelSource.includes("TextInputControl") &&
    servicesCreatePanelSource.includes("NumberInputControl") &&
    (servicesCreatePanelSource.match(/<TextInputControl/g) ?? []).length >= 5 &&
    (servicesCreatePanelSource.match(/<NumberInputControl/g) ?? []).length >= 5 &&
    !servicesPage.includes("function ServiceCreatePanel("),
  "service registration dialog must use shared text and number form controls",
);
assertNoLegacyFormSubmitClass(servicesPage, "servicesPage.tsx");
assertNoRawSelectElement(servicesPage, "servicesPage.tsx");
assertNoLegacyFormSubmitClass(servicesCreatePanelSource, "servicesCreatePanel.tsx");
assertNoRawSelectElement(servicesCreatePanelSource, "servicesCreatePanel.tsx");
assert(
  servicesCreatePanelSource.includes("FormSelectControl") &&
    (servicesCreatePanelSource.match(/<FormSelectControl/g) ?? []).length >= 1,
  "service registration dialog selects must use FormSelectControl",
);
assert(
  servicesCreatePanelSource.includes("errorMessage(mutation.error)") &&
    servicesGridSource.includes("errorMessage(query.error)") &&
    servicesPage.includes("errorMessage(servicesQuery.error)") &&
    !servicesPage.includes('<EmptyState title="服务加载失败" tone="danger" />') &&
    !servicesCreatePanelSource.includes("<div className=\"form-error full-field\">服务保存失败。</div>") &&
    !servicesGridSource.includes("<div className=\"service-log-panel form-error\">日志加载失败。</div>"),
  "services page must show concrete API errors for list, service save, and log loading failures",
);
assert(
  servicesPage.includes('import { ServiceGrid } from "./servicesGrid";') &&
    servicesPage.includes("<ServiceGrid") &&
    servicesGridSource.includes("export function ServiceGrid(") &&
    servicesGridSource.includes("function ServiceCard(") &&
    servicesGridSource.includes("function ServiceLogPanel(") &&
    servicesGridSource.includes("startService(service.service_id)") &&
    servicesGridSource.includes("checkServiceHealth(service.service_id)") &&
    servicesGridSource.includes("stopService(service.service_id)") &&
    servicesGridSource.includes("deleteService(service.service_id)") &&
    servicesGridSource.includes("fetchServiceLogs(service.service_id, { signal })") &&
    !servicesPage.includes("function ServiceCard(") &&
    !servicesPage.includes("function ServiceLogPanel("),
  "services page must delegate service cards, runtime actions, and logs to the service grid module",
);
assertNoRawSelectElement(comparePage, "comparePage.tsx");
assert(
  compareControllerSource.includes("export const COMPARE_RUN_PAGE_SIZE = 80;") &&
    comparePage.includes("PagerControl") &&
    compareControllerSource.includes("clampListPageOffset") &&
    compareFiltersSource.includes("updatePagedFilterValue") &&
    comparePage.includes('className="rank-board-pager compare-run-pager"') &&
    compareControllerSource.includes("offset: pageOffset") &&
    compareControllerSource.includes("limit: COMPARE_RUN_PAGE_SIZE") &&
    compareRunRailComponentsSource.includes("已选择；当前页未加载该 run") &&
    comparePage.includes('import { ComparisonHistoryPanel, RunSelectRail } from "./compareRunRailComponents";') &&
    !comparePage.includes("function RunSelectRail(") &&
    !comparePage.includes("function ComparisonHistoryPanel(") &&
    !comparePage.includes("function CompareRunPager(") &&
    !comparePage.includes("limit: 200"),
  "compare run rail must use paged API requests while preserving selected run ids",
);
assert(
  compareFiltersSource.includes("setters.setSearchText") &&
    compareFiltersSource.includes("setters.setPageOffset") &&
    compareFiltersSource.includes("setters.setHistoryOffset") &&
    compareFiltersSource.includes("historyBaselineFilter") &&
    compareFiltersSource.includes("setters.setHistoryBaselineFilter") &&
    !comparePage.includes("function updateCompareRunFilter(") &&
    !comparePage.includes("function updateCompareSharedFilter(") &&
    !comparePage.includes("function updateComparisonHistoryFilter(") &&
    !comparePage.includes("useEffect(() => {\n    setPageOffset(0);") &&
    !comparePage.includes("useEffect(() => {\n    setHistoryOffset(0);"),
  "compare page filter changes must use shared same-batch paging reset for run/history offsets",
);
assert(
  compareViewStateSource.includes("export const COMPARE_VIEW_STATE_KEY") &&
    compareViewStateSource.includes("export const COMPARE_VIEW_STATE_RESET_EVENT") &&
    compareViewStateSource.includes("export function readCompareViewState(") &&
    compareViewStateSource.includes("export function writeCompareViewState(") &&
    compareViewStateSource.includes("export function resetCompareViewState(") &&
    compareViewStateSource.includes("activeLabel") &&
    compareViewStateSource.includes("window.sessionStorage") &&
    compareControllerSource.includes("readCompareViewState") &&
    compareControllerSource.includes("writeCompareViewState") &&
    compareControllerSource.includes("COMPARE_VIEW_STATE_RESET_EVENT") &&
    compareControllerSource.includes("window.addEventListener(COMPARE_VIEW_STATE_RESET_EVENT, resetViewState)") &&
    !comparePage.includes("readCompareViewState") &&
    comparePage.includes("activeLabel={activeLabel}") &&
    comparePage.includes("onActiveLabelChange={setActiveLabel}") &&
    appShellSource.includes('import { resetCompareViewState } from "./compareViewState";') &&
    appShellSource.includes("onNavigate={resetCompareViewState}"),
  "compare page must preserve filters, selected runs, offsets, and active label across back-navigation and reset from main nav",
);
assert(
  apiTypesSource.includes("baselineRunId?: string;") &&
    apiTypesSource.includes("candidateRunId?: string;") &&
    apiSource.includes('params.set("list", "1");') &&
    apiSource.includes('params.set("baseline_run_id", filters.baselineRunId.trim());') &&
    apiSource.includes('params.set("candidate_run_id", filters.candidateRunId.trim());') &&
    compareFiltersSource.includes("historyBaselineFilter") &&
    compareFiltersSource.includes("historyCandidateFilter") &&
    compareControllerSource.includes("export const COMPARISON_HISTORY_PAGE_SIZE = 50;") &&
    compareControllerSource.includes("const [historyOffset, setHistoryOffset] = useState(initialViewState.historyOffset);") &&
    compareControllerSource.includes("offset: historyOffset") &&
    compareControllerSource.includes("limit: COMPARISON_HISTORY_PAGE_SIZE") &&
    compareRunRailComponentsSource.includes('className="rank-board-pager compare-history-pager"') &&
    comparePage.includes("onPageChange={setHistoryOffset}") &&
    compareFiltersSource.includes('id: "compare-history-baseline"') &&
    compareFiltersSource.includes('id: "compare-history-candidate"') &&
    comparePage.includes("active={hasComparisonHistoryFilters}"),
  "compare history advanced search must expose baseline/candidate filters and pagination through the list API",
);
assert(
  compareRunRailComponentsSource.includes('import { FormSelectControl } from "./controlPrimitives";') &&
    (compareRunRailComponentsSource.match(/<FormSelectControl/g) ?? []).length >= 1,
  "compare run rail selects must use FormSelectControl",
);
assert(
  compareControllerSource.includes("placeholderData: (previousData) => previousData") &&
    compareControllerSource.includes("const comparisonReport = comparisonQuery.data;") &&
    compareControllerSource.includes("comparisonQuery.isPlaceholderData && Boolean(comparisonReport)") &&
    comparePage.includes("正在切换对比报告") &&
    comparePage.includes("<ComparisonPanel") &&
    comparePage.includes("report={comparisonReport}") &&
    styleSource.includes(".compare-report-pane {\n  position: relative;"),
  "compare report pane must keep the previous report visible while loading a new pair",
);
assert(
  comparePage.includes('import { ComparisonPanel } from "./compareReportComponents";') &&
    !comparePage.includes("function ComparisonPanel(") &&
    compareReportComponentsSource.includes("<ComparisonReportTabs />") &&
    compareReportComponentsSource.includes('from "./compareReportMetrics"') &&
    compareReportComponentsSource.includes('from "./compareReportSamples"') &&
    compareReportComponentsSource.includes(
      "<ComparisonMetricTable report={report} showsEndpointMetric={showsEndpointMetric} />",
    ) &&
    compareReportComponentsSource.includes("<ComparisonOutcomeBand summary={report.summary} />") &&
    !compareReportComponentsSource.includes("function ComparisonMetricTable(") &&
    !compareReportComponentsSource.includes("function ComparisonSampleTable(") &&
    compareReportMetricsSource.includes("export function ComparisonReportTabs(") &&
    compareReportMetricsSource.includes("export function ComparisonMetricTable(") &&
    compareReportMetricsSource.includes("export function ComparisonOutcomeBand(") &&
    compareReportSamplesSource.includes("export function ComparisonQuickActions(") &&
    compareReportSamplesSource.includes("export function ComparisonLabelDeltaStrip(") &&
    compareReportSamplesSource.includes("export function ComparisonSampleTable(") &&
    comparePage.includes('import "./compareTheme.css";') &&
    comparePage.includes('import "./compareRunRail.css";') &&
    comparePage.includes('import "./compareReportPanel.css";') &&
    comparePage.indexOf('import "./comparePage.css";') <
      comparePage.indexOf('import "./compareTheme.css";') &&
    comparePage.indexOf('import "./compareTheme.css";') <
      comparePage.indexOf('import "./compareRunRail.css";') &&
    compareThemeStyleSource.includes("--compare-gap-18: 18px") &&
    compareThemeStyleSource.includes("--compare-tab-min: 34px") &&
    compareThemeStyleSource.includes("--compare-delta-min: 56px") &&
    compareThemeStyleSource.includes("--compare-text-value: var(--text-xl)") &&
    compareReportPanelStyleSource.includes(".comparison-report-tabs") &&
    compareReportPanelStyleSource.includes(".comparison-metric-table table") &&
    compareReportPanelStyleSource.includes(".comparison-outcome-band") &&
    compareReportPanelStyleSource.includes(':root[data-theme="dark"] .comparison-panel') &&
    compareReportPanelStyleSource.includes("--compare-report-active") &&
    compareReportPanelStyleSource.includes("--compare-report-warning-bg") &&
    compareReportPanelStyleSource.includes("color: var(--compare-report-ink)") &&
    compareReportPanelStyleSource.includes("border-bottom: 1px solid var(--compare-report-label-line)") &&
    compareReportPanelStyleSource.includes("var(--compare-tab-min)") &&
    compareReportPanelStyleSource.includes("var(--compare-text-small)") &&
    compareReportPanelStyleSource.includes("var(--compare-radius-control)") &&
    !rawCompareReportGeometryPattern.test(compareReportPanelStyleSource) &&
    compareReportPanelStyleSource.includes(".compare-page .comparison-metric-table,") &&
    compareReportPanelStyleSource.includes(".compare-page .comparison-outcome-band,") &&
    compareReportPanelStyleSource.includes(".compare-page .sample-count-chip,") &&
    compareReportPanelStyleSource.includes(".compare-page .label-delta-card") &&
    !compareReportPanelStyleSource.includes("transform 130ms ease") &&
    !compareReportPanelStyleSource.includes("transform: none") &&
    styleSource.includes("--compare-shell-surface: #ffffff") &&
    styleSource.includes(':root[data-theme="dark"] .compare-page') &&
    styleSource.includes("background: var(--compare-shell-surface)") &&
    styleSource.includes(".compare-run-rail,\n.compare-report-pane,\n.compare-context-pane {\n  min-height: 0;\n  overflow: auto;\n  background: var(--compare-shell-surface);\n  border: 0;") &&
    styleSource.includes(".compare-context-card {\n  display: grid;") &&
    styleSource.includes("background: transparent;\n  border: 0;\n  border-bottom: 1px solid var(--compare-shell-line-soft);") &&
    compareReportPanelStyleSource.includes(".label-delta-card {\n  display: grid;") &&
    !compareReportPanelStyleSource.includes("border-bottom: 1px solid #d8e2eb;") &&
    !compareReportPanelStyleSource.includes("color: #607080") &&
    !compareReportPanelStyleSource.includes("background: #fff7e6") &&
    compareRunRailStyleSource.includes(".compare-run-select") &&
    compareRunRailStyleSource.includes(".compare-run-card") &&
    compareRunRailStyleSource.includes(".history-block") &&
    compareRunRailStyleSource.includes(':root[data-theme="dark"] .compare-run-rail') &&
    compareRunRailStyleSource.includes("--compare-rail-card-bg") &&
    compareRunRailStyleSource.includes("scrollbar-gutter: stable") &&
    compareRunRailStyleSource.includes("content-visibility: auto;") &&
    compareRunRailStyleSource.includes("contain-intrinsic-size: auto 56px;") &&
    !compareRunRailStyleSource.includes("background: #ffffff") &&
    !compareRunRailStyleSource.includes("color: #607080") &&
    compareStyleSource.includes(".compare-page .compare-run-rail,") &&
    compareStyleSource.includes("border: 0;\n  border-top: 1px solid var(--bench-line);") &&
    compareStyleSource.includes("animation: none;\n  transition: none;") &&
    compareStyleSource.includes("--compare-shell-topbar-bg") &&
    !compareStyleSource.includes("background: #ffffff") &&
    !compareStyleSource.includes("background: rgb(255 255 255 / 92%)") &&
    !compareStyleSource.includes("color: #637486") &&
    !compareStyleSource.includes(".comparison-metric-table table") &&
    !compareStyleSource.includes(".label-delta-card {\n  display: grid;") &&
    !compareStyleSource.includes(".compare-run-card {\n  display: grid;") &&
    !compareStyleSource.includes(".history-block") &&
    comparePage.includes('import "./comparisonSampleStyles.css";') &&
    comparisonSampleStyleSource.includes(".comparison-sample-row") &&
    comparisonSampleStyleSource.includes(".comparison-run-panel") &&
    comparisonSampleStyleSource.includes(".metric-delta") &&
    comparisonSampleStyleSource.includes("var(--viewer-gap-8)") &&
    comparisonSampleStyleSource.includes("var(--viewer-text-small)") &&
    comparisonSampleStyleSource.includes("var(--viewer-radius-control)") &&
    !comparisonSampleStyleSource.includes("transform 140ms ease") &&
    !comparisonSampleStyleSource.includes("translateX(") &&
    comparisonSampleStyleSource.includes(':root[data-theme="dark"] .comparison-sample-page') &&
    comparisonSampleStyleSource.includes("--comparison-sample-row-accent") &&
    comparisonSampleStyleSource.includes("contain-intrinsic-size: auto 92px;") &&
    comparisonSampleStyleSource.includes("box-shadow: inset 3px 0 0 var(--comparison-sample-row-accent)") &&
    !rawVisualPageControlGeometryPattern.test(comparisonSampleStyleSource) &&
    !compareStyleSource.includes(".comparison-sample-row") &&
    !compareStyleSource.includes(".comparison-run-panel") &&
    !compareStyleSource.includes(".metric-delta") &&
    !designSource.includes(".compare-page .compare-run-rail,") &&
    !designSource.includes(".compare-workspace") &&
    !designSource.includes(".label-delta-card"),
  "compare page must use restrained section dividers instead of stacking boxed cards",
);
assert(
  compareControllerSource.includes("errorMessage(runsQuery.error)") &&
    comparePage.includes("errorMessage(comparisonError)") &&
    compareControllerSource.includes("comparisonError: comparisonQuery.error") &&
    comparisonSampleMiniLinkSource.includes("errorMessage(query.error)") &&
    !comparePage.includes("<div className=\"empty-panel danger-text\">对比报告加载失败。</div>") &&
    !comparisonSampleMiniLinkSource.includes('return <EmptyState title="对比样本加载失败" tone="danger" />;'),
  "compare report and comparison sample pages must show concrete API errors",
);
assert(
  rankBoardControllerSource.includes("errorMessage(dashboardQuery.error || boardQuery.error)") &&
    rankBoardPage.includes("errorTitle") &&
    !rankBoardPage.includes('return <EmptyState title="排行榜加载失败" tone="danger" />;'),
  "rank board page must show concrete API errors for leaderboard failures",
);
assert(
  compareReportSamplesSource.includes("SelectableCardButton") &&
    (compareReportSamplesSource.match(/<SelectableCardButton/g) ?? []).length >= 2 &&
    !compareReportComponentsSource.includes("SelectableCardButton") &&
    !comparePage.includes("SelectableCardButton") &&
    !/<button[\s\S]{0,240}label-delta-card/.test(compareReportSamplesSource),
  "compare label delta cards must use SelectableCardButton instead of raw buttons",
);
assert(
  compareReportSamplesSource.includes("NavigationCardAnchor") &&
    compareReportSamplesSource.includes("NavigationCardFrame") &&
    compareReportSamplesSource.includes('className="comparison-sample-row"') &&
    compareReportSamplesSource.includes('className="comparison-sample-row disabled"') &&
    !compareReportComponentsSource.includes("NavigationCardAnchor") &&
    !compareReportComponentsSource.includes("NavigationCardFrame") &&
    !comparePage.includes("NavigationCardAnchor") &&
    !comparePage.includes("NavigationCardFrame") &&
    !/<a[\s\S]{0,160}className="comparison-sample-row"/.test(compareReportSamplesSource) &&
    !/<div[\s\S]{0,120}className="comparison-sample-row disabled"/.test(
      compareReportSamplesSource,
    ),
  "compare sample navigation rows must use shared navigation card primitives",
);
assert(
  rankBoardModelSource.includes("export const RANK_PAGE_SIZE = 80;") &&
    rankBoardPage.includes('import "./rankTheme.css";') &&
    rankBoardPage.includes('import "./rankBoardPage.css";') &&
    rankBoardPage.indexOf('import "./rankTheme.css";') <
      rankBoardPage.indexOf('import "./rankBoardPage.css";') &&
    rankThemeStyleSource.includes("--rank-gap-10: 10px") &&
    rankThemeStyleSource.includes("--rank-radius-pill: 999px") &&
    rankThemeStyleSource.includes("--rank-toolbar-min: 46px") &&
    rankThemeStyleSource.includes("--rank-facet-board-column-min: 118px") &&
    rankThemeStyleSource.includes("--rank-facet-expanded-max: 112px") &&
    rankThemeStyleSource.includes("--rank-text-title: var(--text-md)") &&
    [
      rankBoardPageStyleSource,
      rankBoardSummaryStyleSource,
      rankBoardFacetsStyleSource,
      rankBoardTablesStyleSource
    ].every((source) => !rawRankBoardGeometryPattern.test(source)) &&
    rankBoardPageStyleSource.includes("var(--rank-gap-8)") &&
    rankBoardPageStyleSource.includes("--rank-mode-switch-bg") &&
    rankBoardPageStyleSource.includes("--rank-toolbar-line") &&
    rankBoardPageStyleSource.includes(':root[data-theme="dark"] .rank-board-page') &&
    rankBoardPageStyleSource.includes("contain: layout paint;") &&
    rankBoardPageStyleSource.includes("scrollbar-gutter: stable both-edges;") &&
    rankBoardFacetsStyleSource.includes("var(--rank-radius-pill)") &&
    rankBoardFacetsStyleSource.includes("repeat(auto-fit, minmax(min(100%, var(--rank-facet-board-column-min)), 1fr))") &&
    rankBoardFacetsStyleSource.includes("grid-auto-flow: column;") &&
    rankBoardFacetsStyleSource.includes("content-visibility: auto;") &&
    rankBoardFacetsStyleSource.includes("contain-intrinsic-size: auto 76px;") &&
    !rankBoardFacetsStyleSource.includes("repeat(7, minmax(0, 1fr))") &&
    !rankBoardFacetsStyleSource.includes("transform 150ms ease") &&
    !rankBoardFacetsStyleSource.includes("transform: none") &&
    !rankBoardFacetsStyleSource.includes("translateY(") &&
    rankBoardSummaryStyleSource.includes("var(--rank-text-caption)") &&
    rankBoardSummaryStyleSource.includes("--rank-summary-title-ink") &&
    rankBoardSummaryStyleSource.includes(':root[data-theme="dark"] .rank-board-summary') &&
    rankBoardTablesStyleSource.includes("var(--rank-radius-pill)") &&
    rankBoardControllerSource.includes("RANK_PAGE_SIZE") &&
    rankBoardControllerSource.includes("RANK_SORTABLE_FIELDS") &&
    rankBoardPage.includes("PagerControl") &&
    rankBoardControllerSource.includes("clampListPageOffset") &&
    rankBoardControllerSource.includes("updatePagedFilterValue") &&
    rankBoardPage.includes('className="rank-board-pager"') &&
    rankBoardControllerSource.includes("offset: pageOffset") &&
    rankBoardControllerSource.includes("limit: RANK_PAGE_SIZE") &&
    rankBoardPage.includes("useRankBoardController") &&
    !rankBoardPage.includes("function RankBoardPager(") &&
    !rankBoardPage.includes("limit: 200"),
  "rank board page must use paged API requests instead of a fixed 200-row slice",
);
assert(
    rankBoardControllerSource.includes("const handleSortChange = useCallback((value: string) => {") &&
    rankBoardControllerSource.includes("}, [sortBy, sortOrder]);") &&
    rankBoardControllerSource.includes("const handleSuiteSortChange = useCallback((value: string) => {") &&
    rankBoardControllerSource.includes("}, [suiteSortBy, suiteSortOrder]);") &&
    rankBoardControllerSource.includes("const setRunMode = useCallback(() => {") &&
    rankBoardControllerSource.includes("const setSuiteMode = useCallback(() => {") &&
    rankBoardControllerSource.includes("setRunMode,\n    setSuiteMode,") &&
    rankBoardControllerSource.includes("setPageOffset(0);") &&
    rankBoardModelSource.includes("export function defaultRankSortOrder(") &&
    rankBoardModelSource.includes("export function toggleSortOrder(") &&
    rankBoardControllerSource.includes("defaultRankSortOrder(value)") &&
    rankBoardControllerSource.includes("toggleSortOrder(sortOrder)") &&
    rankBoardFiltersSource.includes("updatePagedFilterValue(") &&
    rankBoardFiltersSource.includes("values.searchText") &&
    rankBoardFiltersSource.includes("setters.setSearchText") &&
    rankBoardPage.includes("<RankBoardFilterBar") &&
    !rankBoardPage.includes("function updateRankFilter(") &&
    !rankBoardPage.includes('id: "rank-query"') &&
    !rankBoardPage.includes("useEffect(() => {\n    setPageOffset(0);"),
  "rank board filter changes must use shared same-batch paging reset instead of issuing a stale-offset refresh",
);
assert(
  rankBoardViewStateSource.includes("export const RANK_BOARD_VIEW_STATE_KEY") &&
    rankBoardViewStateSource.includes("export const RANK_BOARD_VIEW_STATE_RESET_EVENT") &&
    rankBoardViewStateSource.includes("export function readRankBoardViewState(") &&
    rankBoardViewStateSource.includes("export function writeRankBoardViewState(") &&
    rankBoardViewStateSource.includes("export function resetRankBoardViewState(") &&
    rankBoardViewStateSource.includes("window.sessionStorage") &&
    rankBoardControllerSource.includes("readRankBoardViewState") &&
    rankBoardControllerSource.includes("writeRankBoardViewState") &&
    rankBoardControllerSource.includes("RANK_BOARD_VIEW_STATE_RESET_EVENT") &&
    rankBoardControllerSource.includes("window.addEventListener(RANK_BOARD_VIEW_STATE_RESET_EVENT, resetViewState)") &&
    !rankBoardPage.includes("readRankBoardViewState") &&
    !rankBoardPage.includes("writeRankBoardViewState") &&
    appShellSource.includes('import { resetRankBoardViewState } from "./rankBoardViewState";') &&
    appShellSource.includes("onNavigate={resetRankBoardViewState}"),
  "rank board must preserve table/filter state across detail back-navigation and reset only from the main nav item",
);
assert(
  rankBoardControllerSource.includes("tableRefreshing: (boardQuery.isPlaceholderData && Boolean(board)) || debouncedSearch.pending") &&
    rankBoardPage.includes('className="rank-board-table-toolbar"') &&
    rankBoardPage.includes("refreshing={tableRefreshing}") &&
    !rankBoardPage.includes("rank-board-table-card refreshing") &&
    !rankBoardPage.includes('className="rank-board-table-refresh"') &&
    !styleSource.includes(".rank-board-table-card.refreshing::after") &&
    !styleSource.includes("@keyframes rank-board-table-refresh"),
  "rank board refresh feedback must use the shared table-region indicator only",
);
assert(
  [
    runsPage,
    benchmarksPage,
    compareControllerSource,
    servicesPage,
    jobsQueuePanelSource,
  ].every((source) => source.includes("placeholderData: (previousData) => previousData")),
  "list pages must keep previous data while filters refetch instead of replacing the whole workspace",
);
assert(
  rankBoardFacetsSource.includes("OptionChipButton") &&
    rankBoardFacetsSource.includes('import "./rankBoardSummary.css";') &&
    rankBoardFacetsSource.includes('import "./rankBoardFacets.css";') &&
    rankBoardFacetsSource.includes('className="rank-facet-button"') &&
    rankBoardFacetsSource.includes('className="rank-facet-toggle"') &&
    rankBoardFacetsSource.includes("const visibleItems = expanded ? items : items.slice(0, 5)") &&
    rankBoardFacetsSource.includes("items.length > 5") &&
    rankBoardFacetsSource.includes('onClick={() => onSelect(active ? "all" : item.value)}') &&
    rankBoardFacetsSource.includes("onFilterChange.task") &&
    rankBoardFacetsSource.includes("onFilterChange.benchmark") &&
    rankBoardFacetsSource.includes("onFilterChange.split") &&
    rankBoardFacetsSource.includes("onFilterChange.status") &&
    rankBoardFacetsSource.includes("onFilterChange.label") &&
    rankBoardFacetsSource.includes("onFilterChange.metricProfile") &&
    rankBoardFacetsSource.includes("board.facets.tasks") &&
    rankBoardFacetsSource.includes("board.facets.benchmarks") &&
    rankBoardFacetsSource.includes("board.facets.splits") &&
    rankBoardFacetsSource.includes("board.facets.statuses") &&
    !rankBoardPage.includes("function RankFacetGroup(") &&
    !rankBoardPage.includes('className="rank-facet-button"'),
  "rank board facet rail must expose all backend facets as clickable filter chips",
);
assert(
  styleSource.includes(".rank-facet-group.expanded > div") &&
    styleSource.includes("max-height: var(--rank-facet-expanded-max)") &&
    styleSource.includes("grid-auto-flow: row") &&
    styleSource.includes("overflow: auto"),
  "expanded rank board facets must wrap inside a bounded scroll pane instead of stretching the page",
);
assert(
  rankBoardTablesSource.includes("function SortableHeader(") &&
    rankBoardTablesSource.includes('import { useMemo } from "react";') &&
    rankBoardTablesSource.includes('import "./rankBoardTables.css";') &&
    rankBoardTablesSource.includes("const RANK_METRIC_COLUMNS") &&
    rankBoardTablesSource.includes("const columns = useMemo<ColumnDef<SuiteRankEntry>[]>(() => {") &&
    rankBoardTablesSource.includes("const columns = useMemo<ColumnDef<RankBoardEntry>[]>(() => {") &&
    rankBoardTablesSource.includes("}, [onSortChange, sortBy, sortOrder]);") &&
    rankBoardTablesSource.includes("}, [onSortChange, primaryMetric, sortBy, sortOrder]);") &&
    rankBoardTablesSource.includes('id: `metric_${metric.id}`') &&
    rankBoardTablesSource.includes('id: "created_at"') &&
    rankBoardTablesSource.includes('header: () => auxiliaryHeader("Run", "run_id")') &&
    rankBoardTablesSource.includes('header: () => auxiliaryHeader("创建时间", "created_at")') &&
    rankBoardTablesSource.includes("rank-sort-active-cell") &&
    rankBoardTablesSource.includes('id: "leader_delta"') &&
    rankBoardTablesSource.includes('"rank-primary-score"') &&
    rankBoardTablesSource.includes("formatScoreDelta(row.original.score_delta)") &&
    rankBoardTablesSource.includes("function rankDeltaClassName") &&
    rankBoardControllerSource.includes("defaultRankSortOrder(value)") &&
    rankBoardControllerSource.includes("toggleSortOrder(sortOrder)") &&
    !rankBoardPage.includes("function SortableHeader(") &&
    !rankBoardPage.includes("const RANK_METRIC_COLUMNS") &&
    !rankBoardPage.includes('header: "Weighted"') &&
    !rankBoardTablesSource.includes('header: "Weighted"') &&
    !rankBoardTablesSource.includes(removedCompositeComponentsToken) &&
    !rankBoardTablesSource.includes(removedCompositeApiToken),
  "rank board table components must expose metric and auxiliary sorting through stable sortable headers",
);
assert(
  rankBoardFacetsSource.includes("export function RankBoardSummary(") &&
    rankBoardFacetsSource.includes('className="rank-board-summary"') &&
    rankBoardPage.includes("workspace-card fill rank-board-table-card") &&
    rankBoardModelSource.includes("export const RANK_PRIMARY_METRICS = [") &&
    rankBoardModelSource.includes("export const RANK_AUXILIARY_SORTS = [") &&
    rankBoardPage.includes('import { RankBoardTable, SuiteRankBoardTable } from "./rankBoardTables";') &&
    !rankBoardPage.includes("CompactSelectControl") &&
    !rankBoardPage.includes("function RankBoardSummary(") &&
    !rankBoardPage.includes("function SuiteRankSummary(") &&
    !rankBoardPage.includes("function SuiteRankFacetRail(") &&
    !rankBoardPage.includes('className="rank-sort-section primary"') &&
    !rankBoardPage.includes('className="rank-sort-section auxiliary"') &&
    !rankBoardPage.includes('className="rank-sort-chip primary"') &&
    !rankBoardPage.includes('className="rank-sort-chip auxiliary"') &&
    !rankBoardPage.includes('className="rank-order-chip"') &&
    !rankBoardPage.includes("rankBoardOrderLabel") &&
    !rankBoardPage.includes("best={") &&
    !rankBoardPage.includes('className="rank-top-panel"') &&
    !rankBoardPage.includes('className="rank-spread-panel"') &&
    !rankBoardPage.includes("rank-metric-strip") &&
    !rankBoardPage.includes('id: "rank-sort-by"') &&
    !rankBoardPage.includes('id: "rank-sort-order"'),
  "rank board sorting must live in table headers without separate dropdown or button controls",
);
assert(
  rankBoardPageStyleSource.includes(".rank-mode-switch") &&
    rankBoardPageStyleSource.includes(".rank-board-table-card") &&
    !rankBoardPageStyleSource.includes(".rank-facet-group") &&
    !rankBoardPageStyleSource.includes(".rank-primary-score") &&
    !rankBoardPageStyleSource.includes(".rank-board-summary") &&
    rankBoardFacetsStyleSource.includes(".rank-facet-group.expanded > div") &&
    rankBoardFacetsStyleSource.includes("--rank-facet-group-bg") &&
    rankBoardFacetsStyleSource.includes("--rank-facet-chip-bg") &&
    rankBoardFacetsStyleSource.includes("--rank-facet-toggle-bg") &&
    rankBoardFacetsStyleSource.includes(':root[data-theme="dark"] .rank-board-page .rank-facet-group') &&
    rankBoardFacetsStyleSource.includes("content-visibility: auto;") &&
    !rankBoardFacetsStyleSource.includes("transform 150ms ease") &&
    !rankBoardFacetsStyleSource.includes("transform: none") &&
    rankBoardSummaryStyleSource.includes(".rank-board-summary") &&
    rankBoardTablesStyleSource.includes(".rank-primary-score") &&
    rankBoardTablesStyleSource.includes("--rank-score-bg") &&
    rankBoardTablesStyleSource.includes("--rank-sort-active-cell-bg") &&
    rankBoardTablesStyleSource.includes("--rank-delta-bg") &&
    rankBoardTablesStyleSource.includes(':root[data-theme="dark"] .rank-board-page') &&
    rankBoardTablesStyleSource.includes(':root[data-theme="dark"] .rank-score-delta.neutral') &&
    rankBoardTablesStyleSource.includes(".rank-board-table-card tbody tr") &&
    rankBoardTablesStyleSource.includes("content-visibility: auto;") &&
    rankBoardTablesStyleSource.includes("contain-intrinsic-size: auto 46px;") &&
    !rankBoardTablesStyleSource.includes("var(--rank-table-index-ink, #174d65)") &&
    !rankBoardTablesStyleSource.includes("var(--rank-sort-active-head-bg, #ecf7fb)") &&
    rankBoardTablesStyleSource.includes(".rank-sort-header"),
  "rank board styles must keep page frame, summary, facets, and table concerns in separate CSS modules",
);
assert(
  !rankBoardPage.includes(removedCompositePanelToken) &&
    !rankBoardPage.includes(removedCompositeCamelToken) &&
    !rankBoardPage.includes(removedCompositeClassToken) &&
    !apiSource.includes(removedCompositeCamelToken) &&
    !apiSource.includes(removedCompositeApiToken) &&
    !apiSource.includes(removedCompositeComponentsToken),
  "rank composite metric controls and API parameters must be removed",
);
const sampleViewer = await readSource("src/sampleViewer.tsx");
const viewerPointerSurfaceSource = await readSource("src/viewerPointerSurface.tsx");
const visualStatusBar = await readSource("src/visualStatusBar.tsx");
const visualStatusBarStyleSource = await readSource("src/visualStatusBar.css");
assert(
  sampleViewer.includes("export function SampleViewer("),
  "sample viewer module must export the shared SampleViewer",
);
assert(
  sampleViewer.includes("OptionChipButton") && !sampleViewer.includes('className="query-chip"'),
  "sample viewer utility chips must use OptionChipButton",
);
assert(
    sampleViewer.includes('import { VisualStatusBar } from "./visualStatusBar";') &&
    benchmarkSampleInspectorSource.includes('import { VisualStatusBar } from "./visualStatusBar";') &&
    sampleViewer.includes('className="viewer-visual-status"') &&
    benchmarkSampleInspectorSource.includes('className="viewer-visual-status"') &&
    sampleViewer.includes('className="viewer-stage-shell"') &&
    benchmarkSampleInspectorSource.includes('className="viewer-stage-shell benchmark-stage"') &&
    !sampleViewer.includes("viewer-toolbar") &&
    visualStatusBar.includes("export function VisualStatusBar") &&
    visualStatusBarStyleSource.includes(".visual-status-bar") &&
    visualStatusBarStyleSource.includes(".visual-status-bar.composite-visual-status") &&
    visualStatusBarStyleSource.includes(".visual-status-bar.viewer-visual-status") &&
    visualStatusBarStyleSource.includes('data-refreshing="true"') &&
    visualStatusBarStyleSource.includes("position: absolute") &&
    visualStatusBarStyleSource.includes("--visual-status-bg") &&
    visualStatusBarStyleSource.includes("--visual-status-good-bg") &&
    visualStatusBarStyleSource.includes(':root[data-theme="dark"]') &&
    visualStatusBarStyleSource.includes("box-shadow: var(--visual-status-shadow)") &&
    visualStatusBarStyleSource.includes("contain: layout paint style") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(visualStatusBarStyleSource) &&
    !visualStatusBarStyleSource.includes("0 10px 24px rgb(0 0 0 / 24%)"),
  "sample viewers must use the shared in-canvas visual status bar instead of a layout-consuming toolbar",
);
assert(
  sampleViewer.includes('import { ViewerPointerSurface } from "./viewerPointerSurface";') &&
    benchmarkSampleInspectorSource.includes('import { ViewerPointerSurface } from "./viewerPointerSurface";') &&
    sampleViewer.includes("<ViewerPointerSurface>") &&
    benchmarkSampleInspectorSource.includes("<ViewerPointerSurface>") &&
    viewerPointerSurfaceSource.includes("export function ViewerPointerSurface") &&
    viewerPointerSurfaceSource.includes('import { CompositeCanvasPointerReticle } from "./compositeCanvasPointerReticle";') &&
    viewerPointerSurfaceSource.includes('import { useCompositeCanvasPointerTracker } from "./compositeCanvasPointerTracker";') &&
    viewerPointerSurfaceSource.includes("const pointer = useCompositeCanvasPointerTracker();") &&
    viewerPointerSurfaceSource.includes("{...pointer.pointerHandlers}") &&
    viewerPointerSurfaceSource.includes("<CompositeCanvasPointerReticle coordinateRef={pointer.coordinateRef} />"),
  "ordinary visual workbenches must share the composite pointer reticle and coordinate tracker",
);
const viewerPanels = await readSource("src/viewerPanels.tsx");
const viewerMetrics = await readSource("src/viewerMetrics.ts");
const viewerThemeStyleSource = await readSource("src/viewerTheme.css");
const viewerCanvasStyleSource = await readSource("src/viewerCanvas.css");
const viewerOverlayCanvasStyleSource = await readSource("src/viewerOverlayCanvas.css");
const viewerInspectorStyleSource = await readSource("src/viewerInspector.css");
const viewerComponentStyleSources = [
  viewerCanvasStyleSource,
  viewerOverlayCanvasStyleSource,
  viewerInspectorStyleSource
];
const rawViewerControlGeometryPattern =
  /(?:\bfont-size:\s*\d|\b(?:gap|padding):\s*(?:2|3|4|5|6|7|8|9|10|11)px\b|\bmin-height:\s*(?:19|20|22|24|28|42|46)px\b|\bborder-radius:\s*(?:2|3)px\b)/;
assert(
  viewerCanvasStyleSource.includes(".viewer-stage-shell") &&
    viewerCanvasStyleSource.includes(".viewer-stage-shell > .image-stage") &&
    viewerCanvasStyleSource.includes(".viewer-stage-shell > .viewer-pointer-surface") &&
    viewerCanvasStyleSource.includes(".viewer-pointer-surface > .image-stage"),
  "viewer canvas CSS must provide a fixed in-canvas shell for status overlays",
);
assert(
  viewerPanels.includes('import { CompactSelectControl, ToggleButton } from "./controlPrimitives";'),
  "viewer layer preset select must use CompactSelectControl",
);
assert(
  benchmarkSampleInspectorSource.includes('import "./inspectorPage.css";') &&
    runDetailPageSource.includes('import "./inspectorPage.css";') &&
    inspectorPageStyleSource.includes(".visual-inspector-page") &&
    inspectorPageStyleSource.includes(".sample-row") &&
    inspectorPageStyleSource.includes("var(--viewer-sample-row-min)") &&
    inspectorPageStyleSource.includes("var(--viewer-filter-min)") &&
    inspectorPageStyleSource.includes("var(--viewer-radius-control)") &&
    !rawVisualPageControlGeometryPattern.test(inspectorPageStyleSource),
  "visual inspector page styles must use the shared viewer token layer instead of page-local control geometry",
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
const viewerInstanceLayer = await readSource("src/viewerInstanceLayer.tsx");
const viewerRenderMetricsSource = await readSource("src/viewerRenderMetrics.ts");
const viewerTileLayer = await readSource("src/viewerTileLayer.tsx");
const viewerViewportController = await readSource("src/viewerViewportController.ts");
const viewerViewportPointerInteractionSource = await readSource(
  "src/viewerViewportPointerInteraction.ts",
);
const viewerViewportSyncSource = await readSource("src/viewerViewportSync.ts");
const viewerViewportTileLevelSource = await readSource("src/viewerViewportTileLevel.ts");
const viewerViewportWheelZoomSource = await readSource("src/viewerViewportWheelZoom.ts");
const viewerViewportCommandsSource = await readSource("src/viewerViewportCommands.ts");
assert(
  viewerCanvas.includes('import { ActionButton } from "./ui";') &&
    viewerCanvas.includes('className="canvas-reset-button"') &&
    viewerCanvas.includes("onInspect?: (objectId: string | null) => void") &&
    viewerCanvas.includes('import { MemoizedInstanceLayer } from "./viewerInstanceLayer";') &&
    viewerCanvas.includes("onObjectContextMenu?: (request: CanvasObjectContextMenuRequest) => void") &&
    viewerCanvas.includes("const overlayInteractive = Boolean(onHover || onLock || onInspect || onObjectContextMenu);") &&
    viewerCanvas.includes("event.target !== event.currentTarget || !activeObjectId") &&
    viewerCanvas.includes("onLock?.(activeObjectId)") &&
    viewerCanvas.includes("onDoubleClick={(event) =>") &&
    viewerCanvas.includes("resetViewport();") &&
    !viewerCanvas.includes("onInspect?.(activeObjectId)") &&
    viewerCanvas.includes("objectId: activeObjectId") &&
    viewerViewportPointerInteractionSource.includes("function isOverlayInteractionTarget") &&
    viewerViewportPointerInteractionSource.includes(
      'target.closest(".overlay-instance, .canvas-hud")',
    ) &&
    !viewerViewportPointerInteractionSource.includes(".overlay-svg.has-active, .canvas-hud") &&
    viewerCanvas.includes("allowOverlaySurfacePan = false") &&
    viewerCanvas.includes("allowOverlaySurfacePan?: boolean") &&
    viewerCanvas.includes("allowOverlaySurfacePan,") &&
    viewerViewportPointerInteractionSource.includes("allowOverlaySurfacePan = false") &&
    viewerViewportPointerInteractionSource.includes("function canPanOverlaySurface") &&
    viewerViewportPointerInteractionSource.includes("event.button === 1 || event.altKey || event.shiftKey") &&
    viewerViewportPointerInteractionSource.includes("!canPanOverlaySurface(event, allowOverlaySurfacePan)") &&
    viewerViewportController.includes('from "./viewerViewportPointerInteraction"') &&
    viewerViewportController.includes('from "./viewerViewportWheelZoom"') &&
    viewerViewportController.includes("useViewerViewportPointerInteraction({") &&
    viewerViewportController.includes("useViewerViewportWheelZoom({") &&
    !viewerViewportController.includes("function isOverlayInteractionTarget") &&
    viewerInstanceLayer.includes("export type CanvasObjectContextMenuRequest") &&
    viewerInstanceLayer.includes("export const MemoizedInstanceLayer = React.memo(InstanceLayer);") &&
    viewerInstanceLayer.includes("function InstanceLayer(") &&
    viewerInstanceLayer.includes('className="overlay-hitbox"') &&
    viewerInstanceLayer.includes('className="overlay-hitline"') &&
    viewerInstanceLayer.includes("onDoubleClick={(event) =>") &&
    viewerInstanceLayer.includes("onInspect?.(objectId)") &&
    viewerInstanceLayer.includes("onContextMenu={(event) =>") &&
    viewerInstanceLayer.includes("event.preventDefault();") &&
    viewerInstanceLayer.includes("onObjectContextMenu?.({") &&
    viewerInstanceLayer.includes("recordViewerRenderMetric(`instanceLayer:${kind}`)") &&
    viewerRenderMetricsSource.includes("export function recordViewerRenderMetric") &&
    !viewerCanvas.includes("function InstanceLayer(") &&
    !viewerCanvas.includes("arrowHeadPoints") &&
    !viewerCanvas.includes("normalizeBbox") &&
    !viewerCanvas.includes("resolveInstanceColor") &&
    !/<button[\s\S]{0,120}resetViewport/.test(viewerCanvas),
  "viewer canvas reset control must use ActionButton while instance overlay rendering lives in viewerInstanceLayer",
);
assert(
  viewerCanvas.includes('import { PyramidTileLayer } from "./viewerTileLayer";') &&
    viewerViewportTileLevelSource.includes('import { tileLevelForZoom } from "./viewerTileLayer";') &&
    viewerViewportTileLevelSource.includes("export function useViewerViewportTileLevel") &&
    viewerViewportTileLevelSource.includes("const TILE_LOAD_IDLE_DELAY_MS = 700") &&
    viewerViewportTileLevelSource.includes("const clearScheduledTileLevel = useCallback") &&
    viewerViewportTileLevelSource.includes("const resetTileLevel = useCallback") &&
    viewerViewportTileLevelSource.includes("const scheduleTileLevelUpdate = useCallback") &&
    viewerViewportTileLevelSource.includes("tileLevelForZoom({") &&
    viewerViewportTileLevelSource.includes("tileLoadTimeoutRef") &&
    viewerViewportController.includes('from "./viewerViewportTileLevel"') &&
    viewerViewportController.includes("useViewerViewportTileLevel({") &&
    viewerViewportController.includes("resetTileLevel") &&
    viewerViewportController.includes("scheduleTileLevelUpdate") &&
    !viewerViewportController.includes('import { tileLevelForZoom } from "./viewerTileLayer";') &&
    !viewerViewportController.includes("TILE_LOAD_IDLE_DELAY_MS") &&
    !viewerViewportController.includes("tileLoadTimeoutRef") &&
    !viewerCanvas.includes("function PyramidTileLayer(") &&
    !viewerCanvas.includes("function tileLevelForZoom(") &&
    viewerTileLayer.includes("export function PyramidTileLayer(") &&
    viewerTileLayer.includes("export function tileLevelForZoom(") &&
    viewerTileLayer.includes("MAX_RENDERED_TILES = 24") &&
    viewerTileLayer.includes("TILE_ZOOM_THRESHOLD = 2.1"),
  "viewer tile pyramid rendering and zoom-level selection must live in viewerTileLayer",
);
assert(
  viewerCanvas.includes('import { useViewerViewportController } from "./viewerViewportController";') &&
    viewerCanvas.includes("useViewerViewportController({") &&
    viewerViewportController.includes("export function useViewerViewportController") &&
    viewerViewportController.includes("ResizeObserver") &&
    viewerViewportController.includes("clampPan") &&
    viewerViewportController.includes("computeFitSize") &&
    viewerViewportController.includes("useWorkspaceShortcuts") &&
    viewerViewportController.includes("viewportSyncKey?: string | null") &&
    viewerViewportController.includes("allowOverlaySurfacePan?: boolean") &&
    viewerViewportController.includes("allowOverlaySurfacePan = false") &&
    viewerViewportController.includes('from "./viewerViewportSync"') &&
    viewerViewportController.includes("currentSyncedViewport(viewportSyncKey)") &&
    viewerViewportController.includes("subscribeSyncedViewport") &&
    viewerViewportController.includes("publishSyncedViewport") &&
    viewerViewportController.includes('from "./viewerViewportCommands"') &&
    viewerViewportController.includes("VIEWPORT_RESET_COMMAND") &&
    viewerViewportController.includes("viewportResetCommandDetail(event)") &&
    viewerViewportController.includes("detail.viewportSyncKey !== viewportSyncKey") &&
    viewerViewportController.includes("currentSyncedSnapshot") &&
    viewerViewportController.includes("applySyncedViewport") &&
    viewerViewportPointerInteractionSource.includes("export function useViewerViewportPointerInteraction") &&
    viewerViewportPointerInteractionSource.includes("dragRef") &&
    viewerViewportPointerInteractionSource.includes("setPointerCapture") &&
    viewerViewportPointerInteractionSource.includes("releasePointerCapture") &&
    viewerViewportPointerInteractionSource.includes("pendingPanRef.current") &&
    viewerViewportPointerInteractionSource.includes("interactionSettings.panSensitivity") &&
    viewerViewportPointerInteractionSource.includes("event.preventDefault();") &&
    viewerViewportPointerInteractionSource.includes("event.stopPropagation();") &&
    viewerViewportWheelZoomSource.includes("export function useViewerViewportWheelZoom") &&
    viewerViewportWheelZoomSource.includes("normalizedWheelDelta") &&
    viewerViewportWheelZoomSource.includes("addEventListener(\"wheel\", handleWheel, { passive: false })") &&
    viewerViewportWheelZoomSource.includes("onWheelZoomRef") &&
    viewerViewportWheelZoomSource.includes("interactionSettings.wheelZoomSensitivity") &&
    !viewerViewportController.includes("setPointerCapture") &&
    !viewerViewportController.includes("releasePointerCapture") &&
    !viewerViewportController.includes("function handleWheel(") &&
    !viewerViewportController.includes("normalizedWheelDelta") &&
    viewerViewportSyncSource.includes("export type SyncedViewportSnapshot") &&
    viewerViewportSyncSource.includes("const syncedViewports = new Map") &&
    viewerViewportSyncSource.includes("const syncedViewportSubscribers = new Map") &&
    viewerViewportSyncSource.includes("export function currentSyncedViewport") &&
    viewerViewportSyncSource.includes("export function subscribeSyncedViewport") &&
    viewerViewportSyncSource.includes("export function publishSyncedViewport") &&
    viewerViewportCommandsSource.includes("export const VIEWPORT_RESET_COMMAND") &&
    viewerViewportCommandsSource.includes("export function requestViewportReset") &&
    viewerViewportCommandsSource.includes("export function viewportResetCommandDetail") &&
    viewerViewportCommandsSource.includes("new CustomEvent<ViewportResetCommandDetail>") &&
    !viewerViewportController.includes("const syncedViewports = new Map") &&
    !viewerViewportController.includes("const syncedViewportSubscribers = new Map") &&
    viewerViewportController.includes("function resetViewport()") &&
    !viewerCanvas.includes("function handleWheel(") &&
    !viewerCanvas.includes("function applyZoom(") &&
    !viewerCanvas.includes("new ResizeObserver") &&
    !viewerCanvas.includes("normalizedWheelDelta") &&
    !viewerCanvas.includes("clampPan"),
  "viewer viewport zoom/pan/fit/keyboard state must live in viewerViewportController",
);
assert(
  viewerCanvas.includes('import "./viewerOverlayCanvas.css";') &&
    viewerCanvas.includes('import "./viewerCanvas.css";') &&
    viewerThemeStyleSource.includes("--viewer-gap-8: 8px") &&
    viewerThemeStyleSource.includes("--viewer-gap-12: 12px") &&
    viewerThemeStyleSource.includes("--viewer-pad-1: 1px") &&
    viewerThemeStyleSource.includes("--viewer-pad-7: 7px") &&
    viewerThemeStyleSource.includes("--viewer-radius-control: 2px") &&
    viewerThemeStyleSource.includes("--viewer-filter-min: 34px") &&
    viewerThemeStyleSource.includes("--viewer-object-row-min: 42px") &&
    viewerThemeStyleSource.includes("--viewer-sample-row-min: 52px") &&
    viewerThemeStyleSource.includes("--viewer-text-caption: var(--text-2xs)") &&
    viewerCanvasStyleSource.includes("var(--viewer-gap-8)") &&
    viewerCanvasStyleSource.includes("var(--viewer-text-small)") &&
    viewerCanvasStyleSource.includes(':root[data-theme="dark"] .viewer-panel') &&
    viewerCanvasStyleSource.includes("--viewer-fetch-bg") &&
    viewerCanvasStyleSource.includes("--viewer-control-bg") &&
    viewerCanvasStyleSource.includes("--viewer-label-chip-active-bg") &&
    viewerCanvasStyleSource.includes("background: var(--viewer-control-bg)") &&
    !viewerCanvasStyleSource.includes("background: rgba(255, 255, 255, 0.86)") &&
    !viewerCanvasStyleSource.includes("color: #607080") &&
    !viewerCanvasStyleSource.includes("border: 1px solid #dbe3ec") &&
    !viewerCanvasStyleSource.includes("background: #ffffff;") &&
    viewerInspectorStyleSource.includes("var(--viewer-object-row-min)") &&
    viewerOverlayCanvasStyleSource.includes("var(--viewer-radius-control)") &&
    viewerComponentStyleSources.every(
      (source) => !rawViewerControlGeometryPattern.test(source),
    ) &&
    viewerOverlayCanvasStyleSource.includes(".image-stage") &&
    viewerOverlayCanvasStyleSource.includes("--image-stage-checker-a") &&
    viewerOverlayCanvasStyleSource.includes("border-color: var(--image-stage-checker-line)") &&
    viewerOverlayCanvasStyleSource.includes(".image-zoom-layer") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-svg.interactive") &&
    viewerOverlayCanvasStyleSource.includes("pointer-events: visiblePainted") &&
    !viewerOverlayCanvasStyleSource.includes("pointer-events: visibleStroke") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance .overlay-hitbox") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance .overlay-hitline") &&
    viewerOverlayCanvasStyleSource.includes("pointer-events: all") &&
    viewerOverlayCanvasStyleSource.includes("pointer-events: stroke") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance.active .overlay-box") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-svg.has-active .overlay-instance:not(.active)") &&
    !viewerInstanceLayer.includes('related ? "related"') &&
    !viewerOverlayCanvasStyleSource.includes(".overlay-instance.related") &&
    viewerOverlayCanvasStyleSource.includes(".canvas-hud") &&
    viewerOverlayCanvasStyleSource.includes("--canvas-hud-bg") &&
    viewerOverlayCanvasStyleSource.includes("--canvas-hud-button-hover-bg") &&
    viewerOverlayCanvasStyleSource.includes("background: var(--canvas-hud-bg)") &&
    !viewerCanvasStyleSource.includes(".image-zoom-layer") &&
    !viewerCanvasStyleSource.includes(".overlay-instance") &&
    !viewerCanvasStyleSource.includes(".canvas-hud"),
  "viewer overlay canvas styles must live in viewerOverlayCanvas.css instead of the shell/panel CSS",
);
assert(
  workspaceSettingsSource.includes('export * from "./workspaceSettingsSchema";') &&
    workspaceSettingsSource.includes('export * from "./workspaceSettingsStorage";') &&
    workspaceSettingsSource.includes('from "./workspaceSettingsSchema"') &&
    workspaceSettingsSource.includes('from "./workspaceSettingsStorage"') &&
    workspaceSettingsSchemaSource.includes("export const DEFAULT_OVERLAY_STYLE") &&
    workspaceSettingsSchemaSource.includes("labelFontSize: 8") &&
    workspaceSettingsSchemaSource.includes("labelStrokeWidth: 0.45") &&
    workspaceSettingsSchemaSource.includes("labelBackgroundOpacity: 0.82") &&
    workspaceSettingsSchemaSource.includes("precision?: number") &&
    workspaceSettingsSchemaSource.includes('{ key: "boxStrokeWidth", label: "框线宽", min: 1, max: 10, step: 1, precision: 0 }') &&
    workspaceSettingsSchemaSource.includes('{ key: "activeStrokeWidth", label: "高亮线宽", min: 2, max: 16, step: 1, precision: 0 }') &&
    workspaceSettingsSchemaSource.includes('{ key: "labelFontSize", label: "标签字号", min: 6, max: 14') &&
    workspaceSettingsSchemaSource.includes('themeMode: "eval_bench_theme_mode"') &&
    workspaceSettingsSchemaSource.includes("export const THEME_MODES") &&
    workspaceSettingsSchemaSource.includes("export const SHORTCUT_ACTIONS") &&
    workspaceSettingsSource.includes("export function bootstrapThemePreference") &&
    workspaceSettingsSource.includes("export function useThemePreference") &&
    workspaceSettingsStorageSource.includes("export function loadThemeMode") &&
    workspaceSettingsStorageSource.includes("export function applyThemeMode") &&
    workspaceSettingsStorageSource.includes("document.documentElement.dataset.theme") &&
    workspaceSettingsStorageSource.includes("document.documentElement.style.colorScheme") &&
    appShellSource.includes("bootstrapThemePreference();") &&
    appShellSource.includes("useThemePreference()") &&
    appShellSource.includes('className="theme-toggle"') &&
    appShellSource.includes('aria-pressed={themeMode === "dark"}') &&
    themeToggleCheckSource.includes('themeToggle?.getAttribute("aria-pressed")') &&
    themeToggleCheckSource.includes('light theme toggle must expose aria-pressed=false') &&
    themeToggleCheckSource.includes('dark theme toggle must expose aria-pressed=true') &&
    themeToggleCheckSource.includes("const darkSurfaceRoutes = [") &&
    themeToggleCheckSource.includes('"/benchmarks"') &&
    themeToggleCheckSource.includes('"/suite-report"') &&
    themeToggleCheckSource.includes("function assertNoBrightDarkSurfaces") &&
    themeToggleCheckSource.includes("darkSurfaceCandidateSelector") &&
    themeToggleCheckSource.includes('[class*="pager"]') &&
    themeToggleCheckSource.includes("function isBrightSurfaceFinding(finding)") &&
    themeToggleCheckSource.includes("routeFindings.filter(isBrightSurfaceFinding)") &&
    themeToggleCheckSource.includes("color\\(srgb") &&
    themeToggleCheckSource.includes("function parseCssColor(value)") &&
    themeToggleCheckSource.includes("function isNearWhite(value)") &&
    themeToggleCheckSource.includes("function isLightThemeInk(value)") &&
    themeToggleCheckSource.includes("function isLightNeutralFocus(value)") &&
    !themeToggleCheckSource.includes('assert.notEqual(snapshot.background, "rgb(255, 255, 255)"') &&
    themeToggleCheckSource.includes("parsed.alpha <= 0.2") &&
    themeToggleCheckSource.includes("const TOPBAR_MAX_HEIGHT = 56;") &&
    themeToggleCheckSource.includes("stored dark theme must survive reload") &&
    themeToggleCheckSource.includes("stored light theme must survive reload") &&
    themeToggleCheckSource.includes("function assertTopbarDensity") &&
    themeToggleCheckSource.includes("topbar density regressed") &&
    themeToggleCheckSource.includes('document.querySelector(".user-profile-chip")') &&
    themeToggleCheckSource.includes("snapshot.actionCount !== 2") &&
    themeToggleCheckSource.includes("function assertDarkMicroInteractions") &&
    themeToggleCheckSource.includes("scrollbarColor") &&
    themeToggleCheckSource.includes("focusBoxShadow") &&
    workspaceSettingsStorageSource.includes("export function loadOverlayStyle") &&
    workspaceSettingsStorageSource.includes("export function normalizeShortcutBinding") &&
    workspaceSettingsStorageSource.includes("export function visibleViewerLabels") &&
    workspaceSettingsStorageSource.includes("export function applyViewerVisibleLabelSelection") &&
    workspaceSettingsStorageSource.includes("roundToStep") &&
    workspaceSettingsStorageSource.includes("precisionFromStep") &&
    !workspaceSettingsSource.includes("export const DEFAULT_OVERLAY_STYLE") &&
    !workspaceSettingsSource.includes("export const SHORTCUT_ACTIONS") &&
    !workspaceSettingsSource.includes("function loadOverlayStyle") &&
    !workspaceSettingsSource.includes("function normalizeShortcutBinding") &&
    !workspaceSettingsSource.includes("function migrateLegacyOverlayLabelStyle") &&
    workspaceSettingsStorageSource.includes("export function migrateLegacyOverlayLabelStyle(") &&
    workspaceSettingsStorageSource.includes("raw.labelFontSize === 14") &&
    workspaceSettingsStorageSource.includes("raw.labelStrokeWidth === 4") &&
    workspaceSettingsStorageSource.includes("raw.labelBackgroundOpacity === 0.86") &&
    workspaceSettingsStorageSource.includes("raw.labelFontSize === 11") &&
    workspaceSettingsStorageSource.includes("raw.labelStrokeWidth === 0.9") &&
    workspaceSettingsStorageSource.includes("raw.labelBackgroundOpacity === 0.64") &&
    workspaceSettingsStorageSource.includes("raw.labelStrokeWidth === 0.45") &&
    workspaceSettingsStorageSource.includes("raw.labelBackgroundOpacity === 0.82") &&
    viewerInstanceLayer.includes("export function compactOverlayLabel") &&
    viewerInstanceLayer.includes("export function overlayLabelBounds") &&
    viewerInstanceLayer.includes("fitOverlayLabel") &&
    viewerInstanceLayer.includes("LABEL_MAX_BOX_RATIO") &&
    viewerInstanceLayer.includes("fontSize={labelBounds.fontSize}") &&
    viewerInstanceLayer.includes("onPointerEnter={() => onHover?.(objectId)}") &&
    !viewerInstanceLayer.includes("onPointerEnter={() => onHover?.(objectId)}\n            onPointerLeave") &&
    viewerInstanceLayer.includes('className="overlay-box"') &&
    viewerCanvas.includes("adaptiveOverlayStyle") &&
    viewerCanvas.includes("fitSize.width") &&
    viewerInstanceLayer.includes("<title>{instance.label}</title>") &&
    !viewerInstanceLayer.includes("instance.label.length * overlayStyle.labelFontSize * 0.58 + 8") &&
    viewerOverlayCanvasStyleSource.includes("--overlay-label-backplate: #fffffe;") &&
    viewerOverlayCanvasStyleSource.includes("fill: var(--overlay-label-backplate);") &&
    viewerOverlayCanvasStyleSource.includes("font-size: var(--overlay-label-size, 8px);") &&
    viewerOverlayCanvasStyleSource.includes("stroke-width: var(--overlay-label-stroke, 0.45px);") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance .overlay-label") &&
    viewerOverlayCanvasStyleSource.includes("pointer-events: all") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance.active .overlay-label") &&
    viewerOverlayCanvasStyleSource.includes("fill-opacity: 0.96;") &&
    viewerOverlayCanvasStyleSource.includes("stroke-opacity: 0.86;") &&
    viewerOverlayCanvasStyleSource.includes("font-weight: 760;") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance.gt.match .overlay-label text") &&
    viewerOverlayCanvasStyleSource.includes("--overlay-label-text: #172033;") &&
    viewerOverlayCanvasStyleSource.includes("fill: var(--overlay-label-text);") &&
    viewerOverlayCanvasStyleSource.includes(".overlay-instance.active .label-backplate") &&
    !viewerOverlayCanvasStyleSource.includes("font-size: calc(var(--overlay-label-size") &&
    viewerOverlayCanvasStyleSource.includes("--overlay-label-gt-backplate: #f7fffb;") &&
    viewerOverlayCanvasStyleSource.includes("--overlay-label-pred-backplate: #fffaf3;") &&
    viewerOverlayCanvasStyleSource.includes("--overlay-label-fn-backplate: #fff8fa;") &&
    viewerOverlayCanvasStyleSource.includes("fill: var(--overlay-label-gt-backplate);") &&
    !viewerOverlayCanvasStyleSource.includes("fill: #0b1118;\n  fill-opacity: var(--overlay-label-bg-opacity, 0.86)") &&
    !viewerOverlayCanvasStyleSource.includes("fill: #0b1118;\n  fill-opacity: var(--overlay-label-bg-opacity, 0.64)") &&
    !viewerOverlayCanvasStyleSource.includes("font-size: var(--overlay-label-size, 11px);") &&
    !viewerOverlayCanvasStyleSource.includes("stroke-width: var(--overlay-label-stroke, 4px)") &&
    !viewerOverlayCanvasStyleSource.includes("font-size: var(--overlay-label-size, 14px);"),
  "viewer bbox labels must use compact light backplates instead of large black labels",
);
assert(
  viewerPanels.includes('import "./viewerInspector.css";') &&
    viewerInspectorStyleSource.includes(".viewer-side-panel") &&
    viewerInspectorStyleSource.includes(".object-row") &&
    viewerInspectorStyleSource.includes(".instance-card") &&
    viewerInspectorStyleSource.includes(".label-chip") &&
    viewerInspectorStyleSource.includes(':root[data-theme="dark"] .viewer-side-panel') &&
    viewerInspectorStyleSource.includes("--viewer-object-row-bg") &&
    viewerInspectorStyleSource.includes("--viewer-status-good-bg") &&
    viewerInspectorStyleSource.includes("content-visibility: auto;") &&
    viewerInspectorStyleSource.includes("contain-intrinsic-size: auto 44px;") &&
    !viewerInspectorStyleSource.includes("box-shadow: inset 3px 0 0 #32a7d8") &&
    !viewerInspectorStyleSource.includes("transform 130ms ease") &&
    !viewerInspectorStyleSource.includes("transform: none") &&
    !viewerCanvasStyleSource.includes(".viewer-side-panel") &&
    !viewerCanvasStyleSource.includes(".object-row") &&
    !viewerCanvasStyleSource.includes(".instance-card"),
  "viewer inspector/object panel styles must live in viewerInspector.css instead of viewerCanvas.css",
);
const compositeReportStage = await readSource("src/compositeReportStage.tsx");
const compositeReportStageControllerSource = await readSource("src/compositeReportStageController.ts");
const compositeStageWorkbench = await readSource("src/compositeStageWorkbench.tsx");
const compositeMicroMeter = await readSource("src/compositeMicroMeter.tsx");
const compositeObjectHud = "";
const compositeObjectContextMenu = await readSource("src/compositeObjectContextMenu.tsx");
const compositeImageNavigator = await readSource("src/compositeImageNavigator.tsx");
const compositeImageNavigatorKeyboard = await readSource("src/compositeImageNavigatorKeyboard.ts");
const compositeImageNavigatorPrimary = await readSource("src/compositeImageNavigatorPrimary.tsx");
const compositeImageJumpControl = await readSource("src/compositeImageJumpControl.tsx");
const compositeInteractionPalette = await readSource("src/compositeInteractionPalette.tsx");
const compositeImageSearchBar = await readSource("src/compositeImageSearchBar.tsx");
const compositePanelPrimitives = await readSource("src/compositePanelPrimitives.tsx");
const compositeImagePanel = await readSource("src/compositeImagePanel.tsx");
const compositeImageJumpItem = await readSource("src/compositeImageJumpItem.tsx");
const compositeImageSearchActiveScrollSource = await readSource("src/compositeImageSearchActiveScroll.ts");
const compositeImageSearchResultItem = await readSource("src/compositeImageSearchResultItem.tsx");
const compositeImageAtlas = await readSource("src/compositeImageAtlas.tsx");
const compositeImageAtlasPanel = await readSource("src/compositeImageAtlasPanel.tsx");
const compositeImageAtlasControllerSource = await readSource("src/compositeImageAtlasController.ts");
const compositeImageTimeline = await readSource("src/compositeImageTimeline.tsx");
const compositeImageIndexMeter = await readSource("src/compositeImageIndexMeter.tsx");
const compositeImageNearbyRail = "";
const compositeImageSearchPopover = await readSource("src/compositeImageSearchPopover.tsx");
const compositeImageSearchResults = await readSource("src/compositeImageSearchResults.tsx");
const compositeImageSearchResultList = await readSource("src/compositeImageSearchResultList.tsx");
const compositeImageSearchScanRail = await readSource("src/compositeImageSearchScanRail.tsx");
const compositeImageSearchResultDragSource = await readSource("src/compositeImageSearchResultDrag.ts");
const compositeImageSearchWheelSource = await readSource("src/compositeImageSearchWheel.ts");
const compositeImageSearchPreview = await readSource("src/compositeImageSearchPreview.tsx");
const compositeImageSearchStatus = await readSource("src/compositeImageSearchStatus.tsx");
const compositeReportShell = await readSource("src/compositeReportShell.tsx");
const compositeReportComposer = await readSource("src/compositeReportComposer.tsx");
const compositeReportComposerDock = await readSource("src/compositeReportComposerDock.tsx");
const compositeReportComposerDockPreview = await readSource("src/compositeReportComposerDockPreview.tsx");
const compositeReportPanel = await readSource("src/compositeReportPanel.tsx");
const compositeReportRunPool = await readSource("src/compositeReportRunPool.tsx");
const compositeReportLayerPlan = await readSource("src/compositeReportLayerPlan.tsx");
const compositeReportControllerSource = await readSource("src/compositeReportController.ts");
const compositeReportViewStateSource = await readSource("src/compositeReportViewState.ts");
const compositeReportComposerModelSource = await readSource("src/compositeReportComposerModel.ts");
const compositeOverlayStage = await readSource("src/compositeOverlayStage.tsx");
const compositeLayerFocusToolbar = "";
const compositeLayerInspector = await readSource("src/compositeLayerInspector.tsx");
const compositeLayerObjectStrip = await readSource("src/compositeLayerObjectStrip.tsx");
const compositeLayerObjectStripDragSource = await readSource("src/compositeLayerObjectStripDrag.ts");
const compositePointerSweepSource = await readSource("src/compositePointerSweep.ts");
const compositePointerDragSource = await readSource("src/compositePointerDrag.ts");
const compositeSplitStage = "";
const compositeSplitPane = "";
const compositeSplitLayerCanvas = "";
const compositeLayerCanvas = await readSource("src/compositeLayerCanvas.tsx");
const compositeCanvasOverlay = await readSource("src/compositeCanvasOverlay.tsx");
const compositeCanvasGestureHud = await readSource("src/compositeCanvasGestureHud.tsx");
const compositeCanvasPointerReticle = await readSource("src/compositeCanvasPointerReticle.tsx");
const compositeCanvasPointerTrackerSource = await readSource("src/compositeCanvasPointerTracker.ts");
const compositeLayerCanvasControllerSource = await readSource("src/compositeLayerCanvasController.ts");
assert(
  suiteReportPage.includes('from "./compositeReportController"') &&
    suiteReportPage.includes('from "./compositeReportShell"') &&
    suiteReportPage.includes("const report = useCompositeReportController();") &&
    !suiteReportPage.includes('from "./compositeReportCommandBar"') &&
    !suiteReportPage.includes("<CompositeReportCommandBar") &&
    !suiteReportPage.includes("report.stageMode") &&
    suiteReportPage.includes("<CompositeReportShell report={report} />") &&
    !suiteReportPage.includes("<ReportComposerDock") &&
    !suiteReportPage.includes("<ReportComposerDrawer") &&
    !suiteReportPage.includes("composite-stage-region") &&
    !suiteReportPage.includes('data-sidebar={report.sidebarOpen ? "open" : "collapsed"}') &&
    !suiteReportPage.includes("<CompositeStage") &&
    compositeReportShell.includes("export function CompositeReportShell") &&
    compositeReportShell.includes('from "./compositeReportComposer"') &&
    compositeReportShell.includes('from "./compositeReportStage"') &&
    compositeReportShell.includes("CompositeReportController") &&
    compositeReportShell.includes("useCompositeSidebarDismiss({") &&
    compositeReportShell.includes("function useCompositeSidebarDismiss") &&
    compositeReportShell.includes("window.addEventListener(\"keydown\", handleKeyDown)") &&
    compositeReportShell.includes("window.removeEventListener(\"keydown\", handleKeyDown)") &&
    compositeReportShell.includes('event.key === "Escape"') &&
    compositeReportShell.includes("<ReportComposerDock") &&
    compositeReportShell.includes("<ReportComposerDrawer") &&
    compositeReportShell.includes("composite-stage-region") &&
    compositeReportShell.includes("report.sidebarOpen") &&
    compositeReportShell.includes('data-sidebar={sidebarState}') &&
    compositeReportShell.includes("composite-sidebar-backdrop") &&
    compositeReportShell.includes('aria-label="关闭报告编排器"') &&
    compositeReportShell.includes("onClick={() => report.setSidebarOpen(false)}") &&
    compositeReportShell.includes("report.activeLayerConfigs") &&
    compositeReportShell.includes("report.focusedLayerKey") &&
    compositeReportShell.includes("refreshing={report.compositeQuery.isFetching") &&
    compositeReportShell.includes("!report.compositeQuery.isLoading") &&
    compositeReportShell.includes("activeSlotCount={report.activeSlots.length}") &&
    compositeReportShell.includes("readyLayerCount={report.readyLayerCount}") &&
    compositeReportShell.includes("missingLayerCount={report.missingLayerCount}") &&
    compositeReportShell.includes("<CompositeStage") &&
    compositeReportControllerSource.includes("export function useCompositeReportController") &&
    compositeReportControllerSource.includes("export type CompositeReportController = ReturnType<typeof useCompositeReportController>;") &&
    compositeReportControllerSource.includes('queryKey: ["composite-report-sample", layerRuns, sampleIndex]') &&
    compositeReportControllerSource.includes("fetchCompositeSample({ sampleIndex, layerRuns }, { signal })") &&
    compositeReportControllerSource.includes("const initialViewState = useMemo(() => loadCompositeReportViewState(), []);") &&
    compositeReportControllerSource.includes("useState<LayerSlot[]>(initialViewState.slots)") &&
    !compositeReportControllerSource.includes("StageMode") &&
    !compositeReportControllerSource.includes("stageMode") &&
    compositeReportControllerSource.includes("const [sidebarOpen, setSidebarOpen] = useState(false);") &&
    compositeReportControllerSource.includes("sidebarOpen: false") &&
    compositeReportControllerSource.includes("saveCompositeReportViewState({") &&
    compositeReportControllerSource.includes("reconcileCompositeReportSlots(current, reportRuns)") &&
    compositeReportViewStateSource.includes("export function loadCompositeReportViewState") &&
    compositeReportViewStateSource.includes("export function saveCompositeReportViewState") &&
    compositeReportViewStateSource.includes("export function reconcileCompositeReportSlots") &&
    compositeReportViewStateSource.includes('const COMPOSITE_REPORT_VIEW_STATE_KEY = "eval_bench_composite_report_view";') &&
    !compositeReportViewStateSource.includes("stageMode") &&
    compositeReportViewStateSource.includes("sidebarOpen: false") &&
    !compositeReportControllerSource.includes("useState(initialViewState.sidebarOpen)") &&
    !compositeReportViewStateSource.includes("sidebarOpen: value.sidebarOpen === true") &&
    !compositeReportControllerSource.includes('const [stageMode, setStageMode] = useState<StageMode>("both");') &&
    compositeReportControllerSource.includes("filterReportRuns(reportRuns, query, layerFilter)") &&
    compositeReportControllerSource.includes("groupSlots(slots, runById)") &&
    compositeReportControllerSource.includes('pickLayerPreset(reportRuns, ["layout", "arrow"])') &&
    compositeReportComposer.includes('export { ReportComposerDock } from "./compositeReportComposerDock";') &&
    compositeReportComposer.includes("export function ReportComposerDrawer") &&
    compositeReportComposer.includes("<ReportRunPool") &&
    compositeReportComposer.includes("<ReportLayerPlan") &&
    compositeReportComposerDock.includes("export function ReportComposerDock") &&
    compositePanelPrimitives.includes("export function CompositePanelHeader") &&
    compositePanelPrimitives.includes("export function CompositePanelEmptyState") &&
    compositePanelPrimitives.includes('import "./compositePanelPrimitives.css";') &&
    compositeReportPanel.includes("export function CompositeReportPanelHeader") &&
    compositeReportPanel.includes("export function CompositeReportEmptyState") &&
    compositeReportPanel.includes('import { CompositePanelEmptyState, CompositePanelHeader } from "./compositePanelPrimitives";') &&
    compositeReportPanel.includes("<CompositePanelHeader") &&
    compositeReportPanel.includes("<CompositePanelEmptyState") &&
    compositeReportPanel.includes('import "./compositeReportPanel.css";') &&
    compositeReportRunPool.includes("export function ReportRunPool") &&
    compositeReportLayerPlan.includes("export function ReportLayerPlan") &&
    compositeReportComposerDock.includes('import "./compositeComposerDock.css";') &&
    compositeReportComposer.includes('import "./compositeComposerDrawer.css";') &&
    compositeReportRunPool.includes('import { CompositeReportPanelHeader } from "./compositeReportPanel";') &&
    compositeReportRunPool.includes("<CompositeReportPanelHeader") &&
    compositeReportRunPool.includes('eyebrow="Result Pool"') &&
    compositeReportRunPool.includes('title="评测结果池"') &&
    compositeReportRunPool.includes('className="report-run-filter-tabs"') &&
    !compositeReportRunPool.includes("report-layer-tabs") &&
    !compositeReportRunPool.includes('import "./compositeReportComposerPanels.css";') &&
    compositeReportRunPool.includes('import "./compositeReportRunPool.css";') &&
    compositeReportLayerPlan.includes('import { CompositeReportEmptyState, CompositeReportPanelHeader } from "./compositeReportPanel";') &&
    compositeReportLayerPlan.includes("<CompositeReportPanelHeader") &&
    compositeReportLayerPlan.includes("<CompositeReportEmptyState>") &&
    compositeReportLayerPlan.includes('eyebrow="Report Layers"') &&
    compositeReportLayerPlan.includes('title="分层报告结构"') &&
    !compositeReportLayerPlan.includes('import "./compositeReportComposerPanels.css";') &&
    compositeReportLayerPlan.includes('import "./compositeReportLayerPlan.css";') &&
    !compositeReportRunPool.includes('className="report-panel-head"') &&
    !compositeReportLayerPlan.includes('className="report-panel-head"') &&
    !compositeReportLayerPlan.includes('className="report-empty-state"') &&
    !compositeReportComposer.includes('import "./compositeReportComposer.css";') &&
    compositeReportRunPool.includes("<SearchInputControl") &&
    compositeReportLayerPlan.includes("<TextInputControl") &&
    compositeReportComposerDock.includes("composite-composer-dock") &&
    compositeReportComposerDock.includes("ActionButton") &&
    compositeReportComposerDock.includes("composite-composer-dock open") &&
    compositeReportComposerDock.includes("composite-composer-dock collapsed") &&
    compositeReportComposerDock.includes('data-state={open ? "open" : "collapsed"}') &&
    compositeReportComposerDock.includes("aria-expanded={open}") &&
    compositeReportComposerDock.includes("onDoubleClick={() => onOpenChange(!open)}") &&
    compositeReportComposerDock.includes("composer-dock-grip") &&
    compositeReportComposerDock.includes("PLAN") &&
    !compositeReportComposerDock.includes("COMPOSER") &&
    !compositeReportComposerDock.includes('import { CompositeMicroMeter } from "./compositeMicroMeter";') &&
    !compositeReportComposerDock.includes("<CompositeMicroMeter") &&
    !compositeReportComposerDock.includes("function RailStat") &&
    !compositeReportComposerDock.includes("composer-dock-stats") &&
    !compositeReportComposerDock.includes("composer-dock-stat") &&
    !compositeReportComposerDock.includes("const activeSlotProgress =") &&
    !compositeReportComposerDock.includes("const readyProgress =") &&
    !compositeReportComposerDock.includes("<ReportComposerDockPreview") &&
    !compositeReportComposerDock.includes('from "./compositeReportComposerDockPreview"') &&
    compositeComposerDockStyleSource.includes("position: relative") &&
    compositeComposerDockStyleSource.includes(".composite-composer-dock.collapsed") &&
    compositeComposerDockStyleSource.includes(':root[data-theme="dark"] .composite-composer-dock') &&
    compositeComposerDockStyleSource.includes("--composer-dock-grip-bg") &&
    compositeComposerDockStyleSource.includes("scrollbar-gutter: stable") &&
    !compositeComposerDockStyleSource.includes("background: #f4f8ff") &&
    !compositeComposerDockStyleSource.includes("border: 1px solid #cad7e5") &&
    compositeComposerDockStyleSource.includes(".composer-dock-grip") &&
    compositeComposerDockStyleSource.includes("writing-mode: vertical-rl") &&
    !compositeComposerDockStyleSource.includes(".composer-dock-stats") &&
    !compositeComposerDockStyleSource.includes(".composer-dock-stat") &&
    !compositeComposerDockStyleSource.includes("--dock-stat-progress") &&
    !compositeComposerDockStyleSource.includes(".composer-dock-meter") &&
    compositeReportComposer.includes("composite-sidebar-drawer") &&
    compositeReportComposer.includes("composite-sidebar-grid") &&
    compositeReportRunPool.includes("report-run-pool") &&
    compositeReportLayerPlan.includes("report-layer-plan") &&
    !compositeReportComposer.includes("<SearchInputControl") &&
    !compositeReportComposer.includes("<TextInputControl") &&
    !compositeReportComposer.includes("function ReportRunPool(") &&
    !compositeReportComposer.includes("function ReportLayerPlan(") &&
    compositeReportComposerModelSource.includes("export function filterReportRuns") &&
    compositeReportComposerModelSource.includes("export function groupSlots") &&
    compositeReportComposerModelSource.includes("export function pickLayerPreset") &&
    compositeReportComposerModelSource.includes("export function layerIndex") &&
    compositeReportComposerModelSource.includes("export function fallbackRun") &&
    !suiteReportPage.includes("function ReportRunPool(") &&
    !suiteReportPage.includes("function ReportLayerPlan(") &&
    !suiteReportPage.includes("function ReportComposerDock(") &&
    !suiteReportPage.includes("function ReportSignal(") &&
    !suiteReportPage.includes("STAGE_MODES.map") &&
    !suiteReportPage.includes("errorMessage(") &&
    !suiteReportPage.includes("useState") &&
    !suiteReportPage.includes("useMemo") &&
    !suiteReportPage.includes("useEffect") &&
    !suiteReportPage.includes("useQuery") &&
    !suiteReportPage.includes("fetchCompositeSample") &&
    !suiteReportPage.includes('from "./compositeReportComposerModel"') &&
    !suiteReportPage.includes("function filterReportRuns(") &&
    !suiteReportPage.includes("function groupSlots(") &&
    !suiteReportPage.includes("function pickLayerPreset(") &&
    !suiteReportPage.includes("<SearchInputControl") &&
    !suiteReportPage.includes("<TextInputControl") &&
    !compositeReportModelSource.includes("STAGE_MODES") &&
    !compositeReportModelSource.includes('{ value: "both", label: "总览" }') &&
    !compositeReportModelSource.includes('{ value: "split", label: "分屏" }') &&
    !compositeReportModelSource.includes("LAYER_COLORS") &&
    !compositeReportModelSource.includes("#2563eb") &&
    compositeLayerPaletteSource.includes("export const LAYER_COLORS") &&
    compositeLayerPaletteSource.includes("export const LAYER_UNAVAILABLE_COLOR") &&
    compositeLayerPaletteSource.includes("var(--composite-layer-blue)") &&
    compositeLayerPaletteSource.includes("var(--composite-layer-unavailable)") &&
    !/#(?:2563eb|dc2626|059669|b45309|7c3aed|0891b2|a8b2bd)\b/i.test(compositeLayerPaletteSource) &&
    compositeLayerPaletteSource.includes("export function layerColor") &&
    compositeLayerPaletteSource.includes("export function layerAvailabilityColor") &&
    compositeThemeStyleSource.includes("--composite-layer-blue: #2563eb") &&
    compositeThemeStyleSource.includes("--composite-layer-unavailable: #a8b2bd") &&
    !compositeReportComposerDock.includes('from "./compositeLayerPalette"') &&
    compositeReportComposerDockPreview.includes('from "./compositeLayerPalette"') &&
    compositeReportRunPool.includes('from "./compositeLayerPalette"') &&
    compositeReportLayerPlan.includes('from "./compositeLayerPalette"') &&
    compositeOverlayStage.includes('from "./compositeLayerPalette"') &&
    !compositeLayerFocusToolbar.includes('from "./compositeLayerPalette"') &&
    compositeLayerInspector.includes('from "./compositeLayerPalette"') &&
    !suiteReportPage.includes("<NumberSettingControl") &&
    !suiteReportPage.includes("railCollapsed") &&
    !suiteReportPage.includes("image-union-local"),
  "composite report must default to a collapsed composer dock, keep composition in the drawer, and avoid page-level mode/header controls",
);
assert(
  compositeReportStage.includes('import { CompositeImageNavigator } from "./compositeImageNavigator";') &&
    compositeReportStage.includes('from "./compositeStageWorkbench"') &&
    compositeReportStage.includes('from "./compositeReportStageController"') &&
    compositeReportStage.includes("const stage = useCompositeReportStageController({") &&
    compositeReportStage.includes("<CompositeImageNavigator") &&
    compositeReportStage.includes("navigator={<CompositeImageNavigator") &&
    compositeReportStage.includes("<CompositeStageWorkbench") &&
    compositeReportStage.includes("refreshing: boolean") &&
    compositeReportStage.includes("activeSlotCount: number") &&
    !compositeReportStage.includes('from "./compositeOverlayStage"') &&
    !compositeReportStage.includes('from "./compositeLayerInspector"') &&
    !compositeReportStage.includes('from "./compositeSplitStage"') &&
    !compositeReportStage.includes('from "./compositeObjectHud"') &&
    compositeStageWorkbench.includes('import { CompositeInspector } from "./compositeLayerInspector";') &&
    compositeStageWorkbench.includes('from "./compositeOverlayStage"') &&
    !compositeStageWorkbench.includes('from "./compositeSplitStage"') &&
    !compositeStageWorkbench.includes('from "./compositeObjectHud"') &&
    !compositeStageWorkbench.includes('from "./compositeLayerFocusToolbar"') &&
    compositeStageWorkbench.includes('from "./compositeObjectContextMenu"') &&
    compositeStageWorkbench.includes('from "./visualStatusBar"') &&
    compositeStageWorkbench.includes('import { ResizableSplit } from "./workspaceLayout";') &&
    compositeStageWorkbench.includes("export function CompositeStageWorkbench") &&
    compositeStageWorkbench.includes("navigator: ReactNode") &&
    compositeStageWorkbench.includes("composite-report-focus") &&
    compositeStageWorkbench.includes("<OverlayStage") &&
    compositeStageWorkbench.includes("navigator={navigator}") &&
    compositeStageWorkbench.includes("<ResizableSplit") &&
    compositeStageWorkbench.includes('storageKey="eval_bench_composite_inspector_width"') &&
    compositeStageWorkbench.includes('fixedPane="second"') &&
    compositeStageWorkbench.includes("useCompositeWorkbenchCompactMode") &&
    !compositeStageWorkbench.includes("<SplitStage") &&
    compositeSplitStage === "" &&
    compositeSplitPane === "" &&
    compositeSplitLayerCanvas === "" &&
    !compositeSplitStage.includes("<CompositeLayerCanvas") &&
    !compositeSplitStage.includes("objectKeyForLocalObject") &&
    compositeStageWorkbench.includes("viewportSyncKey={stage.viewportSyncKey}") &&
    compositeStageWorkbench.includes("CompositeInspector") &&
    !compositeStageWorkbench.includes("<CompositeLayerFocusToolbar") &&
    compositeStageWorkbench.includes("<VisualStatusBar") &&
    compositeStageWorkbench.includes("aggregateCompositeDiagnostics") &&
    compositeReportStageControllerSource.includes("export function useCompositeReportStageController") &&
    compositeReportStageControllerSource.includes("export type CompositeReportStageState") &&
    compositeReportStageControllerSource.includes("const viewportSyncKey = composite ? `composite:${composite.image_key}` : null;") &&
    compositeReportStageControllerSource.includes("const focusAvailable = Boolean(") &&
    compositeReportStageControllerSource.includes("const activeFocusedLayerKey = focusAvailable ? focusedLayerKey : null;") &&
    compositeReportStageControllerSource.includes("const focusedLayers = activeFocusedLayerKey") &&
    compositeReportStageControllerSource.includes("const focusedStatuses = activeFocusedLayerKey") &&
    compositeReportStageControllerSource.includes("useCompositeObjectInteraction({") &&
    compositeReportStageControllerSource.includes("onFocusedLayerChange(null)") &&
    compositeLayerFocusToolbar === "" &&
    compositeReportStageControllerSource.includes('from "./compositeObjectInteractionController"') &&
    !compositeReportStage.includes('from "./compositeObjectInteractionController"') &&
    !compositeReportStage.includes("useCompositeObjectInteraction({") &&
    compositeStageWorkbench.includes("<CompositeObjectContextMenu") &&
    compositeStageWorkbench.includes("stage.objectInteraction.openObjectContextMenu") &&
    compositeObjectInteractionControllerSource.includes("export function useCompositeObjectInteraction") &&
    compositeObjectInteractionControllerSource.includes("const [contextMenu, setContextMenu]") &&
    compositeObjectInteractionControllerSource.includes("function openObjectContextMenu") &&
    compositeObjectInteractionControllerSource.includes("setLockedObjectKey(request.objectKey)") &&
    compositeObjectInteractionControllerSource.includes("useCompositeObjectContextMenuLifecycle({ contextMenu, closeContextMenu })") &&
    compositeObjectInteractionControllerSource.includes("useCompositeObjectKeyboardNavigation({ navigateObject })") &&
    compositeObjectContextMenuLifecycleSource.includes('event.key === "Escape"') &&
    compositeObjectContextMenuLifecycleSource.includes("function closeMenuFromKey") &&
    compositeObjectKeyboardNavigationSource.includes("function navigateObjectFromKey") &&
    compositeObjectInteractionControllerSource.includes("nextCompositeObjectKey(layers, activeObjectKey, direction)") &&
    compositeObjectInteractionControllerSource.includes("const navigateObject = useCallback") &&
    compositeObjectInteractionControllerSource.includes("function handleObjectWheel") &&
    compositeObjectInteractionControllerSource.includes("if (!event.altKey && !event.shiftKey)") &&
    compositeObjectInteractionControllerSource.includes("onObjectWheel: handleObjectWheel") &&
    compositeObjectInteractionControllerSource.includes("event.deltaY") &&
    compositeObjectInteractionControllerSource.includes("event.preventDefault();") &&
    compositeObjectInteractionControllerSource.includes("event.stopPropagation();") &&
    compositeObjectInteractionControllerSource.includes("allCompositeObjectRefs") &&
    compositeObjectInteractionControllerSource.includes("const objectRefs = useMemo") &&
    compositeObjectInteractionControllerSource.includes("const activeObjectIndex = objectRefs.findIndex") &&
    compositeObjectInteractionControllerSource.includes("objectCount: objectRefs.length") &&
    compositeObjectKeyboardNavigationSource.includes("export function objectNavigationDirection") &&
    compositeObjectKeyboardNavigationSource.includes('from "./keyboardTargets"') &&
    keyboardTargetsSource.includes("export function isEditableTarget") &&
    !compositeObjectInteractionControllerSource.includes("function objectNavigationDirection") &&
    !compositeObjectInteractionControllerSource.includes("function isEditableTarget") &&
    compositeObjectInteractionControllerSource.includes("closeContextMenu,") &&
    !compositeStageWorkbench.includes("CompositeObjectHud") &&
    compositeMicroMeter.includes("export function CompositeMicroMeter") &&
    compositeMicroMeter.includes('import "./compositeMicroMeter.css";') &&
    compositeMicroMeter.includes("--composite-meter-progress") &&
    compositeMicroMeter.includes('value ? "has-value" : ""') &&
    compositeMicroMeter.includes("{value ? <strong>{value}</strong> : null}") &&
    compositeMicroMeter.includes("composite-meter-ring") &&
    compositeMicroMeter.includes("Math.max(0, Math.min(1, progress))") &&
    compositeObjectHud === "" &&
    compositeObjectContextMenu.includes("export type CompositeObjectMenuRequest") &&
    compositeObjectContextMenu.includes("export function CompositeObjectContextMenu") &&
    compositeObjectContextMenu.includes('role="menu"') &&
    compositeObjectContextMenu.includes("onPointerDown={(event) => event.stopPropagation()}") &&
    compositeObjectContextMenu.includes("锁定对象") &&
    compositeObjectContextMenu.includes("解锁对象") &&
    compositeObjectContextMenu.includes("聚焦图层") &&
    compositeObjectContextMenu.includes("查看详情") &&
    compositeObjectContextMenu.includes("清除选择") &&
    compositeObjectContextMenu.includes('import "./compositeObjectContextMenu.css";') &&
    compositeObjectInteractionControllerSource.includes("resolveCompositeObjectRef") &&
    compositeObjectInteractionControllerSource.includes("function inspectObject") &&
    compositeObjectInteractionControllerSource.includes("function clearObjectInteraction") &&
    compositeObjectInteractionControllerSource.includes("function toggleObjectLock") &&
    compositeObjectInteractionControllerSource.includes("activeObjectKey") &&
    !compositeStageWorkbench.includes("activeObjectIndex={stage.objectInteraction.activeObjectIndex}") &&
    !compositeStageWorkbench.includes("relatedObjectCount={stage.objectInteraction.relatedObjectKeys.size}") &&
    compositeObjectInteractionControllerSource.includes("lockedObjectKey") &&
    compositeObjectInteractionControllerSource.includes("relatedCompositeObjectKeys") &&
    compositeObjectInteractionControllerSource.includes("relatedObjectKeys") &&
    compositeStageWorkbench.includes("stage.objectInteraction.activeObjectKey") &&
    compositeStageWorkbench.includes("stage.objectInteraction.relatedObjectKeys") &&
    compositeStageWorkbench.includes("stage.objectInteraction.lockedObjectKey") &&
    compositeStageWorkbench.includes("onObjectHover") &&
    compositeStageWorkbench.includes("onObjectLock") &&
    compositeStageWorkbench.includes("onObjectWheel={stage.objectInteraction.onObjectWheel}") &&
    compositeStageWorkbench.includes("onObjectInspect={stage.objectInteraction.inspectObject}") &&
    compositeObjectModelSource.includes("export type CompositeObjectRef") &&
    compositeObjectModelSource.includes("export type CompositeObjectKind") &&
    compositeObjectModelSource.includes("export type CompositeObjectStatus") &&
    compositeObjectModelSource.includes("export function compositeObjectKey") &&
    compositeObjectModelSource.includes("export function parseCompositeObjectKey") &&
    compositeObjectModelSource.includes("export function localCanvasObjectIdToKey") &&
    compositeObjectModelSource.includes("export function objectDiagnosticStatus") &&
    compositeObjectModelSource.includes("export function objectStatusWeight") &&
    compositeObjectModelSource.includes("export function normalizeObjectLabel") &&
    compositeObjectInteractionSource.includes("export function buildOverlayObjects") &&
    compositeObjectInteractionSource.includes("export function buildLayerObjectRefs") &&
    compositeObjectInteractionSource.includes("export function buildLayerObjectRefsForScope") &&
    compositeObjectInteractionSource.includes("export function relatedCompositeObjectKeys") &&
    compositeObjectInteractionSource.includes("export function allCompositeObjectRefs") &&
    compositeObjectInteractionSource.includes("export function nextCompositeObjectKey") &&
    compositeObjectInteractionSource.includes("export function resolveCompositeObjectRef") &&
    compositeObjectInteractionSource.includes('from "./compositeObjectModel"') &&
    !compositeObjectInteractionSource.includes("export function parseCompositeObjectKey") &&
    !compositeObjectInteractionSource.includes("export function localCanvasObjectIdToKey") &&
    !compositeObjectInteractionSource.includes("function objectDiagnosticStatus") &&
    compositeCanvasObjectMappingSource.includes("export function overlayObjectIdForKey") &&
    compositeCanvasObjectMappingSource.includes('from "./compositeObjectModel"') &&
    compositeCanvasObjectMappingSource.includes("export function objectKeyForOverlayObject") &&
    compositeCanvasObjectMappingSource.includes("export function relatedOverlayObjectIds") &&
    compositeCanvasObjectMappingSource.includes("export function localObjectIdForKey") &&
    compositeCanvasObjectMappingSource.includes("export function relatedLocalObjectIds") &&
    compositeCanvasObjectMappingSource.includes("export function objectKeyForLocalObject") &&
    compositeOverlayStage.includes("export function OverlayStage") &&
    compositeOverlayStage.includes("buildOverlayObjects") &&
    compositeOverlayStage.includes('from "./compositeCanvasObjectMapping"') &&
    compositeOverlayStage.includes("<CompositeLayerCanvas") &&
    compositeOverlayStage.includes("navigator?: ReactNode") &&
    compositeOverlayStage.includes("{navigator}") &&
    compositeOverlayStage.includes("viewportSyncKey?: string | null") &&
    compositeOverlayStage.includes("viewportSyncKey={viewportSyncKey}") &&
    compositeOverlayStage.includes("relatedObjectKeys: Set<string>") &&
    compositeOverlayStage.includes("relatedOverlayIds") &&
    compositeOverlayStage.includes("relatedObjectIds={relatedOverlayIds}") &&
    compositeOverlayStage.includes("onObjectInspect") &&
    compositeOverlayStage.includes("onInspect={(objectId) => onObjectInspect(resolveOverlayObjectKey(objectId))}") &&
    compositeOverlayStage.includes("onObjectWheel") &&
    compositeOverlayStage.includes("onObjectWheel={onObjectWheel}") &&
    compositeOverlayStage.includes("onObjectContextMenu") &&
    compositeOverlayStage.includes("resolveOverlayObjectKey(request.objectId)") &&
    compositeOverlayStage.includes('import "./compositeOverlayStage.css";') &&
    compositeLayerInspector.includes("export function CompositeInspector") &&
    compositeLayerInspector.includes('import { LayerObjectStrip } from "./compositeLayerObjectStrip";') &&
    compositeLayerInspector.includes("buildLayerObjectRefs") &&
    compositeLayerInspector.includes("relatedObjectKeys: Set<string>") &&
    compositeLayerInspector.includes("<LayerObjectStrip") &&
    compositeLayerInspector.includes("onObjectInspect={onObjectInspect}") &&
    compositeStageWorkbench.includes("onObjectInspect={stage.objectInteraction.inspectObject}") &&
    !compositeLayerInspector.includes('relatedObjectKeys.has(object.key) ? "related" : ""') &&
    !compositeLayerInspector.includes("function LayerObjectStrip") &&
    !compositeLayerInspector.includes("onWheelCapture={onObjectWheel}") &&
    !compositeLayerInspector.includes("event.deltaY") &&
    compositeLayerInspector.includes('import "./compositeLayerInspector.css";') &&
    compositeLayerObjectStrip.includes("export function LayerObjectStrip") &&
    compositeLayerObjectStrip.includes('import { useLayerObjectStripDrag } from "./compositeLayerObjectStripDrag";') &&
    compositeLayerObjectStrip.includes("const objectDrag = useLayerObjectStripDrag") &&
    compositeLayerObjectStrip.includes("objectDrag.objectStripDragHandlers") &&
    compositeLayerObjectStrip.includes("objectDrag.shouldSuppressClick()") &&
    compositeLayerObjectStrip.includes("layer-object-drag-hint") &&
    compositeLayerObjectStrip.includes("data-object-key={object.key}") &&
    compositeLayerObjectStrip.includes("objectDrag.dragging") &&
    compositeLayerObjectStrip.includes('"layer-object-strip dragging"') &&
    compositeLayerObjectStrip.includes('"layer-object-strip"') &&
    !compositeLayerObjectStrip.includes("objects.slice(0, 6)") &&
    !compositeLayerObjectStrip.includes("overflowCount") &&
    !compositeLayerObjectStrip.includes("setPointerCapture") &&
    compositeLayerObjectStripDragSource.includes("export function useLayerObjectStripDrag") &&
    compositeLayerObjectStripDragSource.includes('import { usePointerSweepSelection } from "./compositePointerSweep";') &&
    compositeLayerObjectStripDragSource.includes("objectKeyFromPointer") &&
    compositeLayerObjectStripDragSource.includes("usePointerSweepSelection") &&
    compositeLayerObjectStripDragSource.includes("resolveValueFromPointer: objectKeyFromPointer") &&
    compositeLayerObjectStripDragSource.includes("pointerSweepHandlers") &&
    compositeLayerObjectStripDragSource.includes("shouldSuppressClick") &&
    !compositeLayerObjectStripDragSource.includes("useRef") &&
    !compositeLayerObjectStripDragSource.includes("useState") &&
    !compositeLayerObjectStripDragSource.includes("setPointerCapture") &&
    !compositeLayerObjectStripDragSource.includes("suppressClickRef") &&
    compositePointerSweepSource.includes("export function usePointerSweepSelection") &&
    compositePointerSweepSource.includes("useRef") &&
    compositePointerSweepSource.includes("useState") &&
    compositePointerSweepSource.includes("resolveValueFromPointer") &&
    compositePointerSweepSource.includes("previewValueAtPointer") &&
    compositePointerSweepSource.includes("event.button !== 0") &&
    compositePointerSweepSource.includes("setPointerCapture") &&
    compositePointerSweepSource.includes("releasePointerCapture") &&
    compositePointerSweepSource.includes("onPointerDown") &&
    compositePointerSweepSource.includes("onPointerMove") &&
    compositePointerSweepSource.includes("onPointerUp") &&
    compositePointerSweepSource.includes("onPointerCancel") &&
    compositePointerSweepSource.includes("activeValueRef") &&
    compositePointerSweepSource.includes("suppressClickRef") &&
    compositePointerSweepSource.includes("pointerSweepHandlers") &&
    compositePointerDragSource.includes("export function usePointerDrag") &&
    compositePointerDragSource.includes("export type PointerDragState") &&
    compositePointerDragSource.includes("thresholdPx = DEFAULT_DRAG_THRESHOLD_PX") &&
    compositePointerDragSource.includes("Math.hypot(deltaX, deltaY) > thresholdPx") &&
    compositePointerDragSource.includes("element.setPointerCapture(event.pointerId)") &&
    compositePointerDragSource.includes("element.releasePointerCapture(event.pointerId)") &&
    compositePointerDragSource.includes("pointerDragHandlers") &&
    compositePointerDragSource.includes("shouldSuppressClick") &&
    compositeLayerObjectStrip.includes("relatedObjectKeys.has(object.key)") &&
    compositeLayerObjectStrip.includes("onWheelCapture={onObjectWheel}") &&
    compositeLayerObjectStrip.includes("onPointerEnter={() => onObjectHover(object.key)}") &&
    compositeLayerObjectStrip.includes("onObjectLock(object.key)") &&
    compositeLayerObjectStrip.includes("onObjectInspect(object.key)") &&
    compositeLayerObjectStrip.includes("onDoubleClick") &&
    compositeLayerObjectStrip.includes("{objects.length.toLocaleString()} objects") &&
    compositeLayerObjectStrip.includes("<OptionChipButton") &&
    compositeLayerObjectStrip.includes('import "./compositeLayerObjectStrip.css";') &&
    [
      compositeMicroMeterStyleSource,
      compositeObjectHudStyleSource,
      compositeObjectContextMenuStyleSource,
      compositeLayerInspectorStyleSource,
      compositeLayerObjectStripStyleSource
    ].every((source) => !/font-size:\s*\d/.test(source)) &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-chip.related") &&
    compositeLayerObjectStripStyleSource.includes("overflow-x: auto") &&
    compositeLayerObjectStripStyleSource.includes("overscroll-behavior-inline: contain") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-strip.dragging") &&
    compositeLayerObjectStripStyleSource.includes("cursor: grab") &&
    compositeLayerObjectStripStyleSource.includes("cursor: grabbing") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-drag-hint") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-drag-hint.active") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-chip b") &&
    compositeThemeStyleSource.includes("--composite-object-strip-line") &&
    compositeThemeStyleSource.includes("--composite-object-fn-ink") &&
    compositeThemeStyleSource.includes("--composite-object-fp-ink") &&
    compositeThemeStyleSource.includes("--composite-object-related-ring") &&
    compositeLayerObjectStripStyleSource.includes("border-color: var(--composite-object-strip-line)") &&
    compositeLayerObjectStripStyleSource.includes("color: var(--composite-object-fn-ink)") &&
    compositeLayerObjectStripStyleSource.includes("color: var(--composite-object-fp-ink)") &&
    compositeLayerObjectStripStyleSource.includes("box-shadow: var(--composite-object-related-ring)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeLayerObjectStripStyleSource) &&
    compositeThemeStyleSource.includes("--composite-inspector-row-line") &&
    compositeThemeStyleSource.includes("--composite-inspector-row-focus-line") &&
    compositeThemeStyleSource.includes("--composite-inspector-row-rail-width") &&
    compositeLayerInspectorStyleSource.includes(
      "grid-template-columns: var(--composite-inspector-row-rail-width) minmax(0, 1fr)",
    ) &&
    compositeLayerInspectorStyleSource.includes("border-bottom: 1px solid var(--composite-inspector-row-line)") &&
    compositeLayerInspectorStyleSource.includes("background: var(--composite-inspector-row-focus-surface)") &&
    !compositeLayerInspectorStyleSource.includes(".layer-report-flags") &&
    !compositeLayerInspectorStyleSource.includes(".layer-report-row dl") &&
    !compositeLayerInspectorStyleSource.includes("color: var(--composite-inspector-flag-ink)") &&
    !compositeLayerInspectorStyleSource.includes("background: var(--composite-inspector-metric-surface)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeLayerInspectorStyleSource) &&
    !compositeLayerInspectorStyleSource.includes(".layer-object-strip") &&
    !compositeLayerInspectorStyleSource.includes(".layer-object-chip") &&
    compositeSplitStage === "" &&
    compositeSplitPane === "" &&
    compositeSplitLayerCanvas === "" &&
    !compositeSplitStage.includes('from "./compositeCanvasObjectMapping"') &&
    !compositeSplitStage.includes("<CompositeLayerCanvas") &&
    !compositeSplitStage.includes("parseCompositeObjectKey") &&
    !compositeSplitStage.includes("localCanvasObjectIdToKey") &&
    compositeLayerCanvas.includes("export function CompositeLayerCanvas") &&
    compositeLayerCanvas.includes('from "./compositeLayerCanvasController"') &&
    compositeLayerCanvas.includes("useCompositeLayerCanvasController({") &&
    compositeLayerCanvas.includes("<CanvasStage") &&
    compositeLayerCanvas.includes("viewportSyncKey?: string | null") &&
    compositeLayerCanvas.includes("viewportSyncKey={viewportSyncKey}") &&
    compositeLayerCanvas.includes("relatedObjectIds?: Set<string>") &&
    compositeLayerCanvas.includes("relatedObjectIds={relatedObjectIds}") &&
    compositeLayerCanvas.includes("onInspect?: (objectId: string | null) => void") &&
    compositeLayerCanvas.includes("onInspect={onInspect}") &&
    compositeLayerCanvas.includes("onObjectWheel?: (event: WheelEvent<HTMLElement>) => void") &&
    compositeLayerCanvas.includes('data-object-wheel-cruise={onObjectWheel ? "modified" : undefined}') &&
    compositeLayerCanvas.includes('data-overlay-surface-pan="modified"') &&
    compositeLayerCanvas.includes("onWheelCapture={onObjectWheel}") &&
    compositeLayerCanvas.includes("allowOverlaySurfacePan") &&
    compositeLayerCanvas.includes('from "./viewerInstanceLayer"') &&
    compositeLayerCanvas.includes("onObjectContextMenu?: (request: CanvasObjectContextMenuRequest) => void") &&
    compositeLayerCanvas.includes("onObjectContextMenu={onObjectContextMenu}") &&
    compositeLayerCanvas.includes("activeObjectId={canvas.resolvedActiveObjectId}") &&
    compositeLayerCanvas.includes('import { CompositeCanvasGestureHud } from "./compositeCanvasGestureHud";') &&
    compositeLayerCanvas.includes("<CompositeCanvasGestureHud") &&
    compositeLayerCanvas.includes("activeObjectId={canvas.resolvedActiveObjectId}") &&
    compositeLayerCanvas.includes("relatedObjectCount={relatedObjectIds?.size ?? 0}") &&
    compositeLayerCanvas.includes("wheelCruise={Boolean(onObjectWheel)}") &&
    compositeLayerCanvas.includes("contextMenu={Boolean(onObjectContextMenu)}") &&
    !compositeLayerCanvas.includes("composite-canvas-gesture-hud") &&
    compositeLayerCanvas.includes('import { CompositeCanvasPointerReticle } from "./compositeCanvasPointerReticle";') &&
    compositeLayerCanvas.includes('import { useCompositeCanvasPointerTracker } from "./compositeCanvasPointerTracker";') &&
    compositeLayerCanvas.includes("const pointer = useCompositeCanvasPointerTracker();") &&
    !compositeLayerCanvas.includes('data-pointer-reticle={pointer.pointerActive ? "active" : undefined}') &&
    compositeLayerCanvas.includes("style={canvas.overlayVars}") &&
    !compositeLayerCanvas.includes("pointer.pointerVars") &&
    compositeLayerCanvas.includes("{...pointer.pointerHandlers}") &&
    compositeLayerCanvas.includes("<CompositeCanvasPointerReticle coordinateRef={pointer.coordinateRef} />") &&
    !compositeLayerCanvas.includes("clientX") &&
    !compositeLayerCanvas.includes("getBoundingClientRect") &&
    compositeCanvasOverlay.includes("export function CompositeCanvasOverlayPanel") &&
    compositeCanvasOverlay.includes("export function CompositeCanvasOverlayChip") &&
    compositeCanvasOverlay.includes("export function CompositeCanvasCoordinateTag") &&
    compositeCanvasOverlay.includes('import "./compositeCanvasOverlay.css";') &&
    compositeCanvasOverlay.includes("joinClassNames") &&
    compositeCanvasGestureHud.includes("export function CompositeCanvasGestureHud") &&
    compositeCanvasGestureHud.includes('from "./compositeCanvasOverlay"') &&
    compositeCanvasGestureHud.includes("<CompositeCanvasOverlayPanel") &&
    compositeCanvasGestureHud.includes("<CompositeCanvasOverlayChip") &&
    compositeCanvasGestureHud.includes("MousePointer2") &&
    compositeCanvasGestureHud.includes("relatedObjectCount.toLocaleString()") &&
    compositeCanvasGestureHud.includes('aria-label="组合画布鼠标交互状态"') &&
    compositeCanvasGestureHud.includes("wheelCruise: boolean") &&
    compositeCanvasGestureHud.includes("surfacePan: boolean") &&
    compositeCanvasGestureHud.includes("contextMenu: boolean") &&
    !compositeCanvasGestureHud.includes("<span") &&
    compositeCanvasPointerTrackerSource.includes("export function useCompositeCanvasPointerTracker") &&
    !compositeCanvasPointerTrackerSource.includes("useState") &&
    compositeCanvasPointerTrackerSource.includes("coordinateRef") &&
    compositeCanvasPointerTrackerSource.includes("handleCanvasPointerMove") &&
    compositeCanvasPointerTrackerSource.includes("getBoundingClientRect") &&
    compositeCanvasPointerTrackerSource.includes("--composite-pointer-x") &&
    compositeCanvasPointerTrackerSource.includes("--composite-pointer-y") &&
    compositeCanvasPointerTrackerSource.includes("dataset.pointerReticle") &&
    compositeCanvasPointerTrackerSource.includes("style.setProperty") &&
    compositeCanvasPointerTrackerSource.includes("pointerHandlers") &&
    compositeCanvasPointerReticle.includes("export function CompositeCanvasPointerReticle") &&
    compositeCanvasPointerReticle.includes('from "./compositeCanvasOverlay"') &&
    compositeCanvasPointerReticle.includes("<CompositeCanvasOverlayPanel") &&
    compositeCanvasPointerReticle.includes("<CompositeCanvasCoordinateTag") &&
    compositeCanvasPointerReticle.includes("composite-canvas-pointer-reticle") &&
    compositeCanvasPointerReticle.includes("coordinateRef") &&
    compositeCanvasPointerReticle.includes('aria-hidden="true"') &&
    compositeCanvasPointerReticle.includes("<span ref={coordinateRef} />") &&
    compositeLayerCanvas.includes("onHover={canvas.handleHover}") &&
    compositeLayerCanvas.includes("onLock={canvas.handleLock}") &&
    compositeLayerCanvasControllerSource.includes("export function useCompositeLayerCanvasController") &&
    compositeLayerCanvasControllerSource.includes("useWorkspaceSettings(labels)") &&
    compositeLayerCanvasControllerSource.includes("localHoveredObjectId") &&
    compositeLayerCanvasControllerSource.includes("localLockedObjectId") &&
    compositeLayerCanvasControllerSource.includes("resolvedActiveObjectId") &&
    compositeLayerCanvasControllerSource.includes("function handleHover") &&
    compositeLayerCanvasControllerSource.includes("function handleLock") &&
    !compositeLayerCanvas.includes("useState") &&
    !compositeLayerCanvas.includes("useWorkspaceSettings") &&
    !compositeLayerCanvas.includes("localHoveredObjectId") &&
    !compositeLayerCanvas.includes("localLockedObjectId") &&
    compositeLayerCanvas.includes('import "./compositeLayerCanvas.css";') &&
    !compositeReportStage.includes("function buildOverlayObjects") &&
    !compositeReportStage.includes("function buildLayerObjectRefs") &&
    !compositeReportStage.includes("function objectDiagnosticStatus") &&
    !compositeReportStage.includes("function OverlayStage") &&
    !compositeReportStage.includes("function CompositeInspector") &&
    !compositeReportStage.includes("function CompositeWorkbenchToolbar") &&
    !compositeReportStage.includes("layerColor") &&
    !compositeReportStage.includes("function SplitStage") &&
    !compositeReportStage.includes("function CompositeLayerCanvas") &&
    !compositeReportStage.includes("function LayerObjectStrip") &&
    !compositeReportStage.includes("const [contextMenu, setContextMenu]") &&
    !compositeReportStage.includes("function navigateObjectFromKey") &&
    !compositeReportStage.includes("function openObjectContextMenu") &&
    !compositeReportStage.includes("function inspectObject") &&
    !compositeReportStage.includes("function clearObjectInteraction") &&
    !compositeReportStage.includes("function toggleObjectLock") &&
    !compositeReportStage.includes("const viewportSyncKey =") &&
    !compositeReportStage.includes("const focusAvailable =") &&
    !compositeReportStage.includes("const focusedLayers =") &&
    !compositeReportStage.includes("const focusedStatuses =") &&
    !compositeReportStage.includes("useEffect") &&
    compositeReportStage.includes("focusedLayerKey") &&
    !compositeReportStage.includes("mode === \"both\" || mode === \"overlay\"") &&
    !compositeReportStage.includes("mode === \"both\" || mode === \"split\"") &&
    !compositeStageWorkbench.includes("mode === \"both\" || mode === \"overlay\"") &&
    !compositeStageWorkbench.includes("mode === \"both\" || mode === \"split\"") &&
    !compositeReportStage.includes("function ImageNavigator") &&
    !compositeReportStage.includes("image-union-local"),
  "composite report stage must orchestrate image jumping, overlay, inspector, and object interaction through dedicated canvas-first modules",
);
assert(
  compositeImageNavigator.includes("<CompositeImageNavigatorPrimary") &&
    compositeImageNavigator.includes("useCompositeNavigatorDensity") &&
    compositeImageNavigator.includes("ResizeObserver") &&
    compositeImageNavigator.includes("data-density={density}") &&
    compositeImageNavigator.includes("<CompositeImageSearchBar") &&
    compositeImageNavigator.includes("useCompositeImageNavigationController") &&
    compositeImageNavigator.includes("navigation.primaryProps") &&
    compositeImageNavigator.includes("navigation.timelineProps") &&
    compositeImageNavigator.includes("navigation.searchProps") &&
    compositeImageNavigator.includes("<CompositeImageTimeline") &&
    compositeImageNavigator.includes("<CompositeInteractionPalette") &&
    compositeImageNavigator.includes("focusCompositeImageSearchInput(navigation.rootRef.current)") &&
    compositeImageNavigator.includes('requestViewportReset(`composite:${composite.image_key}`)') &&
    compositeImageNavigator.includes("onPrevious={() => navigation.primaryProps.onStep(-1)}") &&
    compositeImageNavigator.includes("onNext={() => navigation.primaryProps.onStep(1)}") &&
    compositeInteractionPalette.includes("export function CompositeInteractionPalette") &&
    compositeInteractionPalette.includes("IconActionButton") &&
    compositeInteractionPalette.includes('role="toolbar"') &&
    compositeInteractionPalette.includes('data-tool="previous"') &&
    compositeInteractionPalette.includes('data-tool="next"') &&
    compositeInteractionPalette.includes('data-tool="search"') &&
    compositeInteractionPalette.includes('data-tool="reset"') &&
    compositeInteractionPalette.includes("disabled={!canPrevious}") &&
    compositeInteractionPalette.includes("disabled={!canNext}") &&
    !compositeInteractionPalette.includes('role="list"') &&
    !compositeInteractionPalette.includes('role="listitem"') &&
    compositeInteractionPalette.includes('import "./compositeInteractionPalette.css";') &&
    compositeImageNavigationControllerSource.includes("export function useCompositeImageNavigationController") &&
    compositeImageNavigationControllerSource.includes("useCompositeImageKeyboard") &&
    compositeImageNavigationControllerSource.includes("useCompositeImageSearchController") &&
    compositeImageNavigationControllerSource.includes("useCompositeImageTimelineController") &&
    compositeImageSearchControllerSource.includes("export function useCompositeImageSearchController") &&
    compositeImageSearchControllerSource.includes("imageResultWindow(filteredImages, activeResultIndex)") &&
    compositeImageSearchControllerSource.includes("activeImageResultIndex(filteredImages, composite.image_index)") &&
    compositeImageSearchControllerSource.includes("resultWindow.offset + index") &&
    !compositeImageSearchControllerSource.includes("filteredImages.slice(0, IMAGE_RESULT_LIMIT)") &&
    compositeImageSearchControllerSource.includes("buildImageMapBins") &&
    compositeImageSearchControllerSource.includes("activeResultIndex") &&
    compositeImageSearchControllerSource.includes("closeFromDocument") &&
    compositeImageSearchControllerSource.includes("handleSearchResultWheel") &&
    compositeImageSearchControllerSource.includes("onSearchResultWheel") &&
    compositeImageSearchControllerSource.includes('import { imageSearchWheelStep } from "./compositeImageSearchWheel";') &&
    compositeImageSearchControllerSource.includes("const step = imageSearchWheelStep(event)") &&
    !compositeImageSearchControllerSource.includes("event.deltaY") &&
    !compositeImageSearchControllerSource.includes("event.deltaX") &&
    !compositeImageSearchControllerSource.includes("event.preventDefault()") &&
    compositeImageSearchWheelSource.includes("export function imageSearchWheelStep(event: WheelEvent)") &&
    compositeImageSearchWheelSource.includes('import { useEffect } from "react";') &&
    !compositeImageSearchWheelSource.includes("WheelEvent as") &&
    !compositeImageSearchWheelSource.includes('import type { WheelEvent } from "react";') &&
    compositeImageSearchWheelSource.includes("event.deltaY") &&
    compositeImageSearchWheelSource.includes("event.deltaX") &&
    compositeImageSearchWheelSource.includes("event.preventDefault();") &&
    compositeImageSearchWheelSource.includes("event.stopPropagation();") &&
    compositeImageSearchWheelSource.includes("export function useImageSearchWheelCruise") &&
    compositeImageSearchWheelSource.includes("useEffect") &&
    compositeImageSearchWheelSource.includes("elementRef: RefObject<T | null>") &&
    compositeImageSearchWheelSource.includes('node.addEventListener("wheel", handleNativeWheel, { passive: false })') &&
    compositeImageSearchWheelSource.includes('node.removeEventListener("wheel", handleNativeWheel)') &&
    compositeImageSearchControllerSource.includes("focusCurrentImageSearch") &&
    !compositeImageSearchControllerSource.includes("locateActiveNearbyItem") &&
    !compositeImageSearchControllerSource.includes('querySelector(".image-nearby-card.active")') &&
    compositeImageTimelineControllerSource.includes("export function useCompositeImageTimelineController") &&
    !compositeImageTimelineControllerSource.includes("nearbyImageKeys") &&
    !compositeImageTimelineControllerSource.includes("nearbyImages") &&
    !compositeImageTimelineControllerSource.includes("scrubPreview") &&
    !compositeImageTimelineControllerSource.includes("previewFromScrubPointer") &&
    compositeImageTimelineControllerSource.includes("imageIndex: composite.image_index") &&
    !compositeImageTimelineControllerSource.includes('import { usePointerDrag } from "./compositePointerDrag";') &&
    !compositeImageTimelineControllerSource.includes("const scrubDrag = usePointerDrag<HTMLDivElement>") &&
    !compositeImageTimelineControllerSource.includes("onStart: (event) => jumpFromScrubPointer(event)") &&
    !compositeImageTimelineControllerSource.includes("onMove: (event) => jumpFromScrubPointer(event)") &&
    !compositeImageTimelineControllerSource.includes("scrubbing: scrubDrag.dragging") &&
    !compositeImageTimelineControllerSource.includes("setPointerCapture") &&
    !compositeImageTimelineControllerSource.includes("releasePointerCapture") &&
    !compositeImageTimelineControllerSource.includes("setScrubbing") &&
    !compositeImageTimelineControllerSource.includes("handleScrubPointerMove") &&
    !compositeImageTimelineControllerSource.includes("handleScrubPointerLeave") &&
    compositeImageNavigationControllerSource.includes("primaryProps") &&
    compositeImageNavigationControllerSource.includes("timelineProps") &&
    compositeImageNavigationControllerSource.includes("searchProps") &&
    compositeImageNavigatorPrimary.includes("export function CompositeImageNavigatorPrimary") &&
    compositeImageNavigatorPrimary.includes('import { CompositeImageJumpControl } from "./compositeImageJumpControl";') &&
    compositeImageNavigatorPrimary.includes("<CompositeImageJumpControl") &&
    !compositeImageNavigatorPrimary.includes("<TextInputControl") &&
    !compositeImageNavigatorPrimary.includes("<IconActionButton") &&
    compositeImageJumpControl.includes("export function CompositeImageJumpControl") &&
    compositeImageJumpControl.includes("<TextInputControl") &&
    compositeImageJumpControl.includes("<IconActionButton") &&
    compositeImageJumpControl.includes("image-jump-control") &&
    compositeImageJumpControl.includes("image-jump-step-group") &&
    compositeImageJumpControl.includes("image-jump-step edge") &&
    compositeImageJumpControl.includes('import "./compositeImageJumpControl.css";') &&
    compositeImageSearchBar.includes("export function CompositeImageSearchBar") &&
    compositeImageSearchBar.includes("<SearchInputControl") &&
    compositeImageSearchBar.includes("<CompositeImageSearchPopover") &&
    compositeImageSearchBar.includes("onSearchResultWheel") &&
    compositeImageSearchBar.includes('event.key === "ArrowDown"') &&
    compositeImageSearchBar.includes('event.key === "ArrowUp"') &&
    compositeImageSearchBar.includes('event.key === "Enter"') &&
    compositeImageSearchBar.includes("useImageSearchPlacement") &&
    compositeImageSearchBar.includes("availableBelow") &&
    compositeImageSearchBar.includes("availableAbove") &&
    compositeImageSearchBar.includes('import "./compositeImageSearchBar.css";') &&
    compositeImageTimeline.includes("export function CompositeImageTimeline") &&
    compositeImageTimeline.includes('import { CompositeImageIndexMeter } from "./compositeImageIndexMeter";') &&
    !compositeImageTimeline.includes("CompositeImageNearbyRail") &&
    !compositeImageTimeline.includes('import { CompositeImageScrubTrack } from "./compositeImageScrubTrack";') &&
    compositeImageTimeline.includes("<CompositeImageIndexMeter") &&
    !compositeImageTimeline.includes("<RangeSettingControl") &&
    !compositeImageTimeline.includes("<CompositeImageScrubTrack") &&
    !compositeImageTimeline.includes("image-scrub-track") &&
    !compositeImageTimeline.includes("image-scrub-preview") &&
    compositeImageTimeline.includes("onWheelCapture={onTimelineWheel}") &&
    compositeImageTimeline.includes('import "./compositeImageTimeline.css";') &&
    compositeImageScrubTrack.includes("export function CompositeImageScrubTrack") &&
    compositeImageScrubTrack.includes('className={scrubbing ? "image-scrub-track scrubbing" : "image-scrub-track"}') &&
    compositeImageScrubTrack.includes("image-scrub-preview") &&
    compositeImageScrubTrack.includes("scrubDeltaLabel(scrubPreview.delta)") &&
    compositeImageScrubTrack.includes("onPointerDown={onScrubPointerDown}") &&
    compositeImageScrubTrack.includes("onMouseMove={onScrubMouseMove}") &&
    compositeImageScrubTrack.includes("basename(scrubPreview.image)") &&
    compositeImageScrubTrack.includes('import "./compositeImageScrubTrack.css";') &&
    compositeImageNearbyRail === "" &&
    compositeImageNearbyRailControllerSource === "" &&
    compositeImageNearbyRailStyleSource === "" &&
    !compositeImageTimeline.includes("onStep={onStep}") &&
    compositeImagePanel.includes("export function CompositeImagePanelHeader") &&
    compositeImagePanel.includes('import { CompositePanelHeader } from "./compositePanelPrimitives";') &&
    compositeImagePanel.includes("<CompositePanelHeader") &&
    compositeImagePanel.includes('density="compact"') &&
    compositeImagePanel.includes("framed") &&
    compositeImagePanel.includes("image-panel-head") &&
    compositeImagePanel.includes("image-panel-head-action") &&
    compositeImagePanel.includes('import "./compositeImagePanel.css";') &&
    compositeImageSearchPopover.includes("export function CompositeImageSearchPopover") &&
    compositeImageSearchPopover.includes("image-jump-popover") &&
    compositeImageSearchPopover.includes('placement: "top" | "bottom"') &&
    compositeImageSearchPopover.includes('data-placement={placement}') &&
    compositeImageSearchPopover.includes('import { CompositeImageAtlasPanel } from "./compositeImageAtlasPanel";') &&
    compositeImageSearchPopover.includes("<CompositeImageAtlasPanel") &&
    !compositeImageSearchPopover.includes('import { CompositeImagePanelHeader } from "./compositeImagePanel";') &&
    !compositeImageSearchPopover.includes("<CompositeImagePanelHeader") &&
    !compositeImageSearchPopover.includes("useCompositeImageAtlasController") &&
    !compositeImageSearchPopover.includes("<CompositeImageAtlas\n") &&
    !compositeImageSearchPopover.includes("<CompositeImageAtlas ") &&
    !compositeImageSearchPopover.includes("image-jump-command-head") &&
    compositeImageSearchPopover.includes("image-jump-popover-body") &&
    compositeImageSearchPopover.includes('import { CompositeImageSearchResults } from "./compositeImageSearchResults";') &&
    compositeImageSearchPopover.includes("<CompositeImageSearchResults") &&
    compositeImageSearchPopover.includes("visibleSearchResults={visibleSearchResults}") &&
    compositeImageSearchPopover.includes("imageCount={imageCount}") &&
    compositeImageSearchPopover.includes("onResultWheel={onResultWheel}") &&
    !compositeImageSearchPopover.includes("const activeResult = visibleSearchResults[activeResultIndex]") &&
    !compositeImageSearchPopover.includes('import { CompositeImageSearchPreview } from "./compositeImageSearchPreview";') &&
    !compositeImageSearchPopover.includes("<CompositeImageSearchPreview") &&
    !compositeImageSearchPopover.includes("event.deltaY") &&
    !compositeImageSearchPopover.includes("function jumpToActiveResult") &&
    !compositeImageSearchPopover.includes("Enter / Click") &&
    compositeImageAtlasPanel.includes("export function CompositeImageAtlasPanel") &&
    compositeImageAtlasPanel.includes('import { CompositeImageAtlas } from "./compositeImageAtlas";') &&
    compositeImageAtlasPanel.includes('import { useCompositeImageAtlasController } from "./compositeImageAtlasController";') &&
    compositeImageAtlasPanel.includes('import { CompositeImagePanelHeader } from "./compositeImagePanel";') &&
    compositeImageAtlasPanel.includes("<CompositeImagePanelHeader") &&
    compositeImageAtlasPanel.includes("<CompositeImageAtlas") &&
    compositeImageAtlasPanel.includes("const atlas = useCompositeImageAtlasController") &&
    compositeImageSearchResults.includes("export function CompositeImageSearchResults") &&
    !compositeImageSearchResults.includes("const resultListRef = useRef<HTMLDivElement | null>(null)") &&
    !compositeImageSearchResults.includes('import { useImageSearchWheelCruise } from "./compositeImageSearchWheel";') &&
    !compositeImageSearchResults.includes('import { useImageSearchActiveScroll } from "./compositeImageSearchActiveScroll";') &&
    !compositeImageSearchResults.includes("useImageSearchWheelCruise({") &&
    !compositeImageSearchResults.includes("useImageSearchActiveScroll({") &&
    !compositeImageSearchResults.includes("addEventListener") &&
    !compositeImageSearchResults.includes("removeEventListener") &&
    !compositeImageSearchResults.includes("handleNativeWheel") &&
    !compositeImageSearchResults.includes('data-wheel-cruise="native"') &&
    compositeImageSearchResults.includes("const activeResult = visibleSearchResults[activeResultIndex]") &&
    compositeImageSearchResults.includes("const dragTargetLabel = resultDragTargetLabel(resultDrag.dragging, activeResult)") &&
    compositeImageSearchResults.includes("function resultDragTargetLabel") &&
    !compositeImageSearchResults.includes("function scrollActiveImageResultIntoView") &&
    !compositeImageSearchResults.includes("scrollIntoView") &&
    compositeImageSearchActiveScrollSource.includes("export function useImageSearchActiveScroll") &&
    compositeImageSearchActiveScrollSource.includes("scrollActiveImageResultIntoView(elementRef.current, activeResultIndex)") &&
    compositeImageSearchActiveScrollSource.includes("function scrollActiveImageResultIntoView") &&
    compositeImageSearchActiveScrollSource.includes('?.scrollIntoView({ block: "nearest" })') &&
    compositeImageSearchActiveScrollSource.includes("resultCount") &&
    compositeImageSearchResults.includes('import { CompositeImageSearchPreview } from "./compositeImageSearchPreview";') &&
    compositeImageSearchResults.includes("<CompositeImageSearchPreview") &&
    compositeImageSearchResults.includes("activeResult={activeResult}") &&
    !compositeImageSearchResults.includes("onWheelCapture={onResultWheel}") &&
    compositeImageSearchResults.includes("image-jump-results-panel") &&
    compositeImageSearchResults.includes('import { CompositeImagePanelHeader } from "./compositeImagePanel";') &&
    compositeImageSearchResults.includes("<CompositeImagePanelHeader") &&
    !compositeImageSearchResults.includes("image-jump-results-head") &&
    compositeImageSearchResults.includes(
      'import { CompositeImageSearchMore, CompositeImageSearchStatus } from "./compositeImageSearchStatus";',
    ) &&
    compositeImageSearchResults.includes("<CompositeImageSearchStatus") &&
    compositeImageSearchResults.includes("hiddenBeforeCount={hiddenBeforeCount}") &&
    compositeImageSearchResults.includes("hiddenAfterCount={hiddenAfterCount}") &&
    compositeImageSearchResults.includes("dragging={resultDrag.dragging}") &&
    compositeImageSearchResults.includes("dragTargetLabel={dragTargetLabel}") &&
    compositeImageSearchResults.includes("<CompositeImageSearchMore hiddenCount={hiddenCount} />") &&
    !compositeImageSearchResults.includes("image-jump-window-meter") &&
    !compositeImageSearchResults.includes("image-jump-empty") &&
    !compositeImageSearchResults.includes("image-jump-more") &&
    compositeImageSearchResults.includes('import { CompositeImageSearchResultList } from "./compositeImageSearchResultList";') &&
    compositeImageSearchResults.includes("<CompositeImageSearchResultList") &&
    compositeImageSearchResults.includes("resultDragHandlers={resultDrag.resultDragHandlers}") &&
    compositeImageSearchResults.includes("shouldSuppressClick={resultDrag.shouldSuppressClick}") &&
    compositeImageSearchResults.includes("onActiveResultIndexChange={onActiveResultIndexChange}") &&
    compositeImageSearchResultList.includes("export function CompositeImageSearchResultList") &&
    compositeImageSearchResultList.includes("const resultListRef = useRef<HTMLDivElement | null>(null)") &&
    compositeImageSearchResultList.includes('import { useImageSearchWheelCruise } from "./compositeImageSearchWheel";') &&
    compositeImageSearchResultList.includes('import { useImageSearchActiveScroll } from "./compositeImageSearchActiveScroll";') &&
    compositeImageSearchResultList.includes("useImageSearchWheelCruise({") &&
    compositeImageSearchResultList.includes("useImageSearchActiveScroll({") &&
    compositeImageSearchResultList.includes('data-wheel-cruise="native"') &&
    compositeImageSearchResultList.includes('import { CompositeImageSearchScanRail } from "./compositeImageSearchScanRail";') &&
    compositeImageSearchResultList.includes("<CompositeImageSearchScanRail") &&
    !compositeImageSearchResultList.includes("function CompositeImageSearchResultScanRail") &&
    !compositeImageSearchResultList.includes("image-jump-scan-rail") &&
    compositeImageSearchScanRail.includes("export function CompositeImageSearchScanRail") &&
    compositeImageSearchScanRail.includes("--image-result-scan-progress") &&
    compositeImageSearchScanRail.includes("--image-result-scan-top") &&
    compositeImageSearchScanRail.includes("data-dragging={dragging ? \"true\" : undefined}") &&
    compositeImageSearchScanRail.includes('delta === 0 ? "current" : delta > 0 ? "forward" : "backward"') &&
    compositeImageSearchScanRail.includes("className={`image-jump-scan-rail direction-${direction}`}") &&
    compositeImageSearchResultList.includes("image-jump-empty") &&
    compositeImageSearchResultList.includes("visibleSearchResults.map") &&
    compositeImageSearchResultList.includes("<CompositeImageSearchResultItem") &&
    compositeImageSearchResultList.includes("windowIndex={index}") &&
    compositeImageSearchResultList.includes('import { CompositeImageSearchResultItem } from "./compositeImageSearchResultItem";') &&
    !compositeImageSearchResults.includes("<CompositeImageJumpIdentity") &&
    !compositeImageSearchResults.includes("<CompositeImageJumpPosition") &&
    !compositeImageSearchResults.includes("<CompositeImageJumpDelta") &&
    !compositeImageSearchResults.includes("imageProgressPercent(item.index, imageCount)") &&
    !compositeImageSearchResults.includes("basename(item.image)") &&
    !compositeImageSearchResults.includes("<ActionButton") &&
    compositeImageSearchResults.includes('import { useCompositeImageSearchResultDrag } from "./compositeImageSearchResultDrag";') &&
    compositeImageSearchResults.includes("const resultDrag = useCompositeImageSearchResultDrag") &&
    compositeImageSearchResults.includes("resultDrag.resultDragHandlers") &&
    compositeImageSearchResults.includes("shouldSuppressClick={resultDrag.shouldSuppressClick}") &&
    !compositeImageSearchResults.includes("image-jump-drag-hint") &&
    !compositeImageSearchResults.includes("function resultIndexFromPointer") &&
    !compositeImageSearchResults.includes("function previewResultAtPointer") &&
    !compositeImageSearchResults.includes("setPointerCapture") &&
    compositeImageSearchStatus.includes("export function CompositeImageSearchStatus") &&
    compositeImageSearchStatus.includes("export function CompositeImageSearchMore") &&
    compositeImageSearchStatus.includes("dragTargetLabel") &&
    compositeImageSearchStatus.includes("{dragTargetLabel ? <strong>{dragTargetLabel}</strong> : null}") &&
    compositeImageSearchStatus.includes('import { CompositeMicroMeter } from "./compositeMicroMeter";') &&
    compositeImageSearchStatus.includes('import "./compositeImageSearchStatus.css";') &&
    compositeImageSearchStatus.includes("image-jump-search-status") &&
    compositeImageSearchStatus.includes("image-jump-search-window") &&
    compositeImageSearchStatus.includes("<CompositeMicroMeter") &&
    compositeImageSearchStatus.includes("beforeProgress") &&
    compositeImageSearchStatus.includes("afterProgress") &&
    compositeImageSearchStatus.includes('className="image-jump-search-window-meter"') &&
    compositeImageSearchStatus.includes('className="image-jump-search-window-meter after"') &&
    compositeImageSearchStatus.includes("image-jump-search-gesture") &&
    compositeImageSearchStatus.includes("image-jump-search-more") &&
    compositeImageSearchStatus.includes('data-dragging={dragging ? "true" : undefined}') &&
    compositeImageSearchResultDragSource.includes("export function useCompositeImageSearchResultDrag") &&
    compositeImageSearchResultDragSource.includes('import { usePointerSweepSelection } from "./compositePointerSweep";') &&
    compositeImageSearchResultDragSource.includes("type SearchResultSweepValue") &&
    compositeImageSearchResultDragSource.includes("resultValueFromPointer") &&
    compositeImageSearchResultDragSource.includes("usePointerSweepSelection") &&
    compositeImageSearchResultDragSource.includes("resolveValueFromPointer: resultValueFromPointer") &&
    compositeImageSearchResultDragSource.includes("pointerSweepHandlers") &&
    compositeImageSearchResultDragSource.includes("shouldSuppressClick") &&
    !compositeImageSearchResultDragSource.includes("useRef") &&
    !compositeImageSearchResultDragSource.includes("useState") &&
    !compositeImageSearchResultDragSource.includes("setPointerCapture") &&
    !compositeImageSearchResultDragSource.includes("suppressClickRef") &&
    !compositeImageSearchResults.includes("data-result-window-index={index}") &&
    !compositeImageSearchResults.includes('role="option"') &&
    !compositeImageSearchResults.includes("aria-selected={active}") &&
    !compositeImageSearchResults.includes("onFocus={() => onActiveResultIndexChange(index)}") &&
    !compositeImageSearchResults.includes("image-jump-result-position") &&
    !compositeImageSearchResults.includes("image-jump-result-delta") &&
    !compositeImageSearchResults.includes("onActiveResultIndexChange(index)") &&
    compositeImageSearchResults.includes('import "./compositeImageSearchResults.css";') &&
    compositeImageJumpItem.includes("export function CompositeImageJumpSummary") &&
    compositeImageJumpItem.includes("export function CompositeImageJumpIdentity") &&
    compositeImageJumpItem.includes("export function CompositeImageJumpPosition") &&
    compositeImageJumpItem.includes("export function CompositeImageJumpDelta") &&
    compositeImageJumpItem.includes("canShowPosition") &&
    compositeImageJumpItem.includes("canShowDelta") &&
    compositeImageJumpItem.includes("export function imageJumpDeltaLabel") &&
    compositeImageJumpItem.includes("basename(item.image)") &&
    compositeImageJumpItem.includes("imageProgressPercent(item.index, imageCount)") &&
    compositeImageJumpItem.includes('import "./compositeImageJumpItem.css";') &&
    compositeImageSearchResultItem.includes("export function CompositeImageSearchResultItem") &&
    compositeImageSearchResultItem.includes('import { CompositeImageJumpSummary } from "./compositeImageJumpItem";') &&
    compositeImageSearchResultItem.includes("<ActionButton") &&
    compositeImageSearchResultItem.includes('role="option"') &&
    compositeImageSearchResultItem.includes("aria-selected={active}") &&
    compositeImageSearchResultItem.includes("data-result-window-index={windowIndex}") &&
    compositeImageSearchResultItem.includes("data-result-direction={direction}") &&
    compositeImageSearchResultItem.includes("onMouseEnter={() => onPreview(windowIndex)}") &&
    compositeImageSearchResultItem.includes("onFocus={() => onPreview(windowIndex)}") &&
    compositeImageSearchResultItem.includes("<CompositeImageJumpSummary") &&
    compositeImageSearchResultItem.includes("currentIndex={imageIndex}") &&
    compositeImageSearchResultItem.includes("showPosition") &&
    compositeImageSearchResultItem.includes("showDelta") &&
    !compositeImageSearchResultItem.includes("<CompositeImageJumpIdentity") &&
    !compositeImageSearchResultItem.includes("<CompositeImageJumpPosition") &&
    !compositeImageSearchResultItem.includes("<CompositeImageJumpDelta") &&
    compositeImageSearchResultItem.includes('import "./compositeImageSearchResultItem.css";') &&
    compositeImageSearchPreview.includes("export function CompositeImageSearchPreview") &&
    compositeImageSearchPreview.includes("<ActionButton") &&
    compositeImageSearchPreview.includes('import { CompositeImageJumpSummary } from "./compositeImageJumpItem";') &&
    compositeImageSearchPreview.includes('<CompositeImageJumpSummary item={activeResult} badge="Active" compact />') &&
    !compositeImageSearchPreview.includes("<CompositeImageJumpIdentity") &&
    !compositeImageSearchPreview.includes("basename(activeResult.image)") &&
    compositeImageSearchPreview.includes("Enter / Click") &&
    compositeImageSearchPreview.includes('import "./compositeImageSearchPreview.css";') &&
    compositeImageSearchPopover.includes("<CompositeImageAtlasPanel") &&
    !compositeImageSearchPopover.includes("useCompositeImageAtlasController") &&
    !compositeImageSearchPopover.includes("{...atlas}") &&
    compositeImageAtlasPanel.includes("useCompositeImageAtlasController") &&
    compositeImageAtlasPanel.includes("<CompositeImageAtlas") &&
    compositeImageAtlasPanel.includes("{...atlas}") &&
    compositeImageSearchPopover.includes('import "./compositeImageSearchPopover.css";') &&
    compositeImageAtlasControllerSource.includes("export function useCompositeImageAtlasController") &&
    compositeImageAtlasControllerSource.includes("atlasDragging") &&
    compositeImageAtlasControllerSource.includes('import { usePointerSweepSelection } from "./compositePointerSweep";') &&
    compositeImageAtlasControllerSource.includes("const sweep = usePointerSweepSelection") &&
    compositeImageAtlasControllerSource.includes("binFromPointer") &&
    compositeImageAtlasControllerSource.includes("[data-image-map-bin-key]") &&
    compositeImageAtlasControllerSource.includes("atlasSweepHandlers: sweep.pointerSweepHandlers") &&
    compositeImageAtlasControllerSource.includes("shouldSuppressAtlasClick: sweep.shouldSuppressClick") &&
    !compositeImageAtlasControllerSource.includes("window.addEventListener") &&
    !compositeImageAtlasControllerSource.includes("function startAtlasDrag") &&
    !compositeImageAtlasControllerSource.includes("function moveAtlasDrag") &&
    compositeImageAtlasControllerSource.includes("function handleAtlasBinKeyDown") &&
    compositeImageAtlasControllerSource.includes('event.key === "ArrowRight"') &&
    compositeImageAtlas.includes("export function CompositeImageAtlas") &&
    compositeImageAtlas.includes("hoveredMapBin") &&
    compositeImageAtlas.includes("atlasDragging") &&
    compositeImageAtlas.includes("image-jump-atlas") &&
    compositeImageAtlas.includes("image-jump-map") &&
    compositeImageAtlas.includes("image-map-bin") &&
    compositeImageAtlas.includes("onPointerDown={atlasSweepHandlers.onPointerDown}") &&
    compositeImageAtlas.includes("onPointerMove={atlasSweepHandlers.onPointerMove}") &&
    compositeImageAtlas.includes("onPointerUp={atlasSweepHandlers.onPointerUp}") &&
    compositeImageAtlas.includes("onPointerCancel={atlasSweepHandlers.onPointerCancel}") &&
    compositeImageAtlas.includes("data-image-map-bin-key={bin.key}") &&
    compositeImageAtlas.includes("shouldSuppressAtlasClick()") &&
    compositeImageAtlas.includes("onPointerEnter={() => onAtlasBinPointerEnter(bin)}") &&
    compositeImageAtlas.includes("onPointerMove={() => onAtlasBinPointerMove(bin)}") &&
    compositeImageAtlas.includes("onAtlasBinClick(bin)") &&
    compositeImageAtlas.includes("onKeyDown={(event) => onAtlasBinKeyDown(bin, event)}") &&
    !compositeImageAtlas.includes("useState") &&
    !compositeImageAtlas.includes("setHoveredMapBin") &&
    !compositeImageAtlas.includes("onJump(bin.midpoint)") &&
    compositeImageAtlas.includes('import "./compositeImageAtlas.css";') &&
    compositeImageNavigatorKeyboard.includes("export function useCompositeImageKeyboard") &&
    compositeImageNavigatorKeyboard.includes('event.key === "/"') &&
    compositeImageNavigatorKeyboard.includes('event.key === "ArrowLeft"') &&
    compositeImageNavigatorKeyboard.includes('event.key === "ArrowRight"') &&
    compositeImageNavigatorKeyboard.includes('event.key === "PageUp"') &&
    compositeImageNavigatorKeyboard.includes('event.key === "PageDown"') &&
    compositeImageNavigatorKeyboard.includes("export function focusCompositeImageSearchInput") &&
    compositeImageNavigatorKeyboard.includes('from "./keyboardTargets"') &&
    !compositeImageNavigatorKeyboard.includes("function isEditableTarget") &&
    !compositeImageNavigator.includes("image-interaction-hints") &&
    !compositeImageNavigator.includes("Right click: 对象菜单") &&
    !compositeImageSearchPopover.includes("Hover 预览") &&
    compositeImageNavigationModelSource.includes("export const IMAGE_RESULT_LIMIT") &&
    compositeImageNavigationModelSource.includes("export const IMAGE_MAP_BIN_COUNT") &&
    !compositeImageNavigationModelSource.includes("NEIGHBOR_RADIUS") &&
    compositeImageNavigationModelSource.includes("export type ImageResultWindow") &&
    compositeImageNavigationModelSource.includes("export type ImageMapBin") &&
    compositeImageNavigationModelSource.includes("export function filterImageKeys") &&
    compositeImageNavigationModelSource.includes("export function imageResultWindow") &&
    compositeImageNavigationModelSource.includes("export function activeImageResultIndex") &&
    !compositeImageNavigationModelSource.includes("export function nearbyImageKeys") &&
    compositeImageNavigationModelSource.includes("export function buildImageMapBins") &&
    compositeImageNavigationModelSource.includes("export function clampImageIndex") &&
    compositeImageNavigationModelSource.includes("export function imageProgressPercent") &&
    compositeImageNavigationModelSource.includes("export function previewFromScrubPointer") &&
    compositeImageNavigationModelSource.includes("delta: index - clampImageIndex(activeIndex, imageKeys.length)") &&
    !compositeImageNavigationControllerSource.includes("IMAGE_RESULT_LIMIT") &&
    !compositeImageNavigationControllerSource.includes("buildImageMapBins") &&
    !compositeImageNavigationControllerSource.includes("activeResultIndex") &&
    !compositeImageNavigationControllerSource.includes("scrubPreview") &&
    !compositeImageNavigationControllerSource.includes("previewFromScrubPointer") &&
    !compositeImageNavigationControllerSource.includes("handleScrubPointerDown") &&
    !compositeImageNavigationControllerSource.includes("handleScrubPointerMove") &&
    !compositeImageNavigationControllerSource.includes("handleScrubPointerLeave") &&
    !compositeImageNavigator.includes("function filterImageKeys") &&
    !compositeImageNavigator.includes("nearbyImageKeys") &&
    !compositeImageNavigator.includes("function clampImageIndex") &&
    !compositeImageNavigator.includes("function indexFromScrubPointer") &&
    !compositeImageNavigator.includes("function handleScrubPointerDown") &&
    !compositeImageNavigator.includes("function handleScrubPointerMove") &&
    !compositeImageNavigator.includes("function moveActiveResult") &&
    !compositeImageNavigator.includes("useCompositeImageKeyboard") &&
    !compositeImageNavigator.includes("IMAGE_RESULT_LIMIT") &&
    !compositeImageNavigator.includes("buildImageMapBins") &&
    !compositeImageNavigator.includes("previewFromScrubPointer") &&
    !compositeImageNavigator.includes("activeResultIndex") &&
    !compositeImageNavigator.includes("scrubPreview") &&
    !compositeImageNavigator.includes("<RangeSettingControl") &&
    !compositeImageNavigator.includes("<CompositeImageAtlas") &&
    !compositeImageNavigator.includes("image-jump-popover") &&
    !compositeImageNavigator.includes("image-scrub-track") &&
    !compositeImageNavigator.includes("<SearchInputControl") &&
    !compositeImageNavigator.includes("<TextInputControl") &&
    !compositeImageNavigator.includes("<IconActionButton") &&
    !compositeImageTimeline.includes("image-filmstrip") &&
    !compositeImageTimeline.includes("image-filmstrip-item") &&
    !compositeImageSearchPopover.includes("image-filmstrip") &&
    !compositeImageTimeline.includes("<ActionButton") &&
    !compositeImageNavigator.includes('event.key === "/"') &&
    !compositeImageNavigator.includes("hoveredMapBin") &&
    !compositeImageNavigator.includes("onPointerEnter={() => setHoveredMapBin(bin)}") &&
    !/<button\b/.test(compositeImageNavigator) &&
    !/<input\b/.test(compositeImageNavigator),
  "composite image navigator must compose command search, atlas navigation, keyboard selection, and wheel stepping without a bottom nearby image list",
);
const compositeComponentStyleSources = [
  compositeReportStyleSource,
  compositeComposerDockStyleSource,
  compositeComposerDrawerStyleSource,
  compositeMicroMeterStyleSource,
  compositePanelPrimitivesStyleSource,
  compositeReportPanelStyleSource,
  compositeReportRunPoolStyleSource,
  compositeReportLayerPlanStyleSource,
  compositeImageNavigatorStyleSource,
  compositeImageJumpControlStyleSource,
  compositeImageSearchBarStyleSource,
  compositeImagePanelStyleSource,
  compositeImageJumpItemStyleSource,
  compositeImageSearchResultItemStyleSource,
  compositeInteractionPaletteStyleSource,
  compositeImageAtlasStyleSource,
  compositeImageTimelineStyleSource,
  compositeImageIndexMeterStyleSource,
  compositeImageSearchPopoverStyleSource,
  compositeImageSearchResultsStyleSource,
  compositeImageSearchPreviewStyleSource,
  compositeImageSearchStatusStyleSource,
  compositeReportStageStyleSource,
  compositeStageWorkbenchStyleSource,
  compositeLayerFocusToolbarStyleSource,
  compositeObjectHudStyleSource,
  compositeObjectContextMenuStyleSource,
  compositeOverlayStageStyleSource,
  compositeLayerCanvasStyleSource,
  compositeCanvasOverlayStyleSource,
  compositeCanvasGestureHudStyleSource,
  compositeCanvasPointerReticleStyleSource,
  compositeLayerInspectorStyleSource,
  compositeLayerObjectStripStyleSource,
  compositeSplitStageStyleSource,
  compositeSplitPaneStyleSource
];
const rawCompositeControlGeometryPattern =
  /(?:\b(?:gap|padding):\s*(?:2|3|4|5|6|7|8|9|10|12)px\b|\bmin-height:\s*(?:22|24|30|40)px\b|\bborder-radius:\s*(?:2|999)px\b)/;
const compositeImageNavigationInteractionStyles = [
  compositeImageScrubTrackStyleSource,
  compositeImageJumpItemStyleSource
];
assert(
  compositeImageNavigator.includes('import "./compositeImageNavigator.css";') &&
    compositeImageJumpControl.includes('import "./compositeImageJumpControl.css";') &&
    compositeImageSearchBar.includes('import "./compositeImageSearchBar.css";') &&
    compositeInteractionPalette.includes('import "./compositeInteractionPalette.css";') &&
    compositeImageTimeline.includes('import "./compositeImageTimeline.css";') &&
    compositeImageNearbyRail === "" &&
    compositeImageSearchPopover.includes('import "./compositeImageSearchPopover.css";') &&
    compositeImageNavigatorStyleSource.includes(".composite-image-navigator") &&
    compositeImageNavigatorStyleSource.includes('.composite-image-navigator[data-density="controls"]') &&
    compositeImageNavigatorStyleSource.includes('.composite-image-navigator[data-density="compact"]') &&
    compositeImageNavigatorStyleSource.includes('.composite-image-navigator[data-density="compact"] .image-navigator-primary') &&
    compositeImageNavigatorStyleSource.includes('grid-template-areas: "timeline search actions"') &&
    compositeImageJumpControlStyleSource.includes(".image-jump-control") &&
    compositeImageJumpControlStyleSource.includes(".image-jump-step-group") &&
    compositeImageJumpControlStyleSource.includes(".image-jump-step.icon-button") &&
    compositeImageJumpControlStyleSource.includes(".image-jump-field") &&
    compositeImageSearchBarStyleSource.includes(".image-navigator-search-row") &&
    compositeImageSearchBarStyleSource.includes("overflow: visible") &&
    compositeImageSearchBarStyleSource.includes(".image-navigator-search") &&
    compositeImageSearchBarStyleSource.includes(".image-navigator-count") &&
    compositeImageSearchBarStyleSource.includes("var(--composite-navigator-search-line)") &&
    compositeImageSearchBarStyleSource.includes("var(--composite-navigator-search-ink)") &&
    compositeImageSearchBarStyleSource.includes("var(--composite-navigator-count-ink)") &&
    compositeImageNavigatorStyleSource.includes("var(--composite-navigator-copy-subtle)") &&
    compositeInteractionPaletteStyleSource.includes(".composite-interaction-palette") &&
    compositeInteractionPaletteStyleSource.includes(".interaction-palette-tool") &&
    compositeInteractionPaletteStyleSource.includes(".interaction-palette-tool.icon-button") &&
    compositeInteractionPaletteStyleSource.includes(".interaction-palette-tool.icon-button:disabled") &&
    compositeInteractionPaletteStyleSource.includes('.interaction-palette-tool[data-tool="reset"]') &&
    compositeInteractionPaletteStyleSource.includes("var(--composite-tool-line)") &&
    compositeInteractionPaletteStyleSource.includes("var(--composite-tool-anchor-surface)") &&
    compositeInteractionPaletteStyleSource.includes("var(--composite-tool-hover-surface)") &&
    compositeInteractionPaletteStyleSource.includes("var(--composite-tool-disabled-ink)") &&
    !compositeInteractionPaletteStyleSource.includes("transform 120ms ease") &&
    !compositeInteractionPaletteStyleSource.includes("translateY(") &&
    compositeComposerDockPreviewStyleSource.includes(".composer-dock-preview") &&
    !compositeComposerDockPreviewStyleSource.includes("transform 120ms ease") &&
    !compositeComposerDockPreviewStyleSource.includes("translateX(") &&
    !compositeInteractionPaletteStyleSource.includes("display: none") &&
    !compositeImageNavigatorStyleSource.includes("image-nearby") &&
    !compositeImageTimelineStyleSource.includes("image-nearby") &&
    !compositeImageSearchBarStyleSource.includes("image-nearby") &&
    compositePanelPrimitivesStyleSource.includes(".composite-panel-head") &&
    compositePanelPrimitivesStyleSource.includes(".composite-panel-head.framed") &&
    compositePanelPrimitivesStyleSource.includes(".composite-panel-action") &&
    compositePanelPrimitivesStyleSource.includes(".composite-panel-empty") &&
    !compositeImagePanelStyleSource.includes(".image-panel-head {") &&
    !compositeImagePanelStyleSource.includes(".image-panel-head-action {") &&
    compositeImagePanelStyleSource.includes(".image-panel-head-action kbd") &&
    compositeImageJumpItemStyleSource.includes(".image-jump-identity") &&
    compositeImageJumpItemStyleSource.includes(".image-jump-identity-main") &&
    compositeImageJumpItemStyleSource.includes(".image-jump-position") &&
    compositeImageJumpItemStyleSource.includes("--image-result-position") &&
    compositeImageJumpItemStyleSource.includes(".image-jump-delta") &&
    compositeImageSearchResultItemStyleSource.includes(".image-jump-result") &&
    compositeImageSearchResultItemStyleSource.includes(".image-jump-result.direction-forward") &&
    compositeImageSearchResultItemStyleSource.includes(".image-jump-result.direction-backward") &&
    compositeImageSearchResultItemStyleSource.includes(".image-jump-result.current") &&
    compositeImageSearchResultItemStyleSource.includes("scroll-margin-block") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result {") &&
    compositeImageSearchPopoverStyleSource.includes(".image-jump-popover") &&
    compositeImageSearchPopoverStyleSource.includes("right: 0") &&
    compositeImageSearchPopoverStyleSource.includes("left: auto") &&
    !compositeImageSearchPopoverStyleSource.includes(".image-jump-command-head") &&
    compositeImageSearchPopoverStyleSource.includes(".image-jump-atlas-panel") &&
    compositeImageSearchPopoverStyleSource.includes(".image-jump-popover-body") &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-results-panel") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-results-head") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-window-meter") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-drag-hint") &&
    compositeImageSearchResultsStyleSource.includes("overscroll-behavior: contain") &&
    compositeImageSearchResultsStyleSource.includes("scrollbar-width: thin") &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-scan-rail") &&
    compositeImageSearchResultsStyleSource.includes("--image-result-scan-top") &&
    compositeImageSearchResultsStyleSource.includes("top: var(--image-result-scan-top, 50%)") &&
    !compositeImageSearchResultsStyleSource.includes("translateY(calc(var(--image-result-scan-progress") &&
    compositeImageSearchResultsStyleSource.includes('.image-jump-scan-rail[data-dragging="true"] span') &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-scan-rail.direction-forward") &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-scan-rail.direction-backward") &&
    !compositeImageSearchResultItemStyleSource.includes(".image-jump-scan-rail") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-status") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-window") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-window-meter") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-window-meter.after i") &&
    !compositeImageSearchStatusStyleSource.includes(".image-jump-search-window span") &&
    !compositeImageSearchStatusStyleSource.includes(".image-jump-search-window strong") &&
    !compositeImageSearchStatusStyleSource.includes(".image-jump-search-window b") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-gesture") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-gesture strong") &&
    compositeImageSearchStatusStyleSource.includes("font-variant-numeric: tabular-nums") &&
    compositeImageSearchStatusStyleSource.includes(
      '.image-jump-search-status[data-dragging="true"] .image-jump-search-gesture',
    ) &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-results.dragging") &&
    compositeImageSearchResultsStyleSource.includes("cursor: grab") &&
    compositeImageSearchResultsStyleSource.includes("cursor: grabbing") &&
    compositeImageSearchPreviewStyleSource.includes(".image-jump-active-preview") &&
    !compositeImageSearchPreviewStyleSource.includes(".image-jump-active-preview > strong") &&
    !compositeImageSearchPreviewStyleSource.includes(".image-jump-active-preview b") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result-index") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result-main") &&
    !compositeImageSearchPopoverStyleSource.includes(".image-jump-active-preview") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-active-preview") &&
    compositeImageSearchResultsStyleSource.includes(".image-jump-empty") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-more") &&
    compositeImageSearchStatusStyleSource.includes(".image-jump-search-more") &&
    compositeImageAtlasStyleSource.includes(".image-jump-atlas") &&
    compositeImageAtlasStyleSource.includes(".image-jump-map") &&
    compositeImageAtlasStyleSource.includes(".image-map-bin") &&
    compositeImageAtlasStyleSource.includes("--match-density") &&
    !compositeImageAtlasStyleSource.includes("translateY(") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result,") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result-position") &&
    !compositeImageSearchResultsStyleSource.includes("--image-result-position") &&
    !compositeImageSearchResultsStyleSource.includes(".image-jump-result-delta") &&
    compositeImageIndexMeter.includes("export function CompositeImageIndexMeter") &&
    compositeImageIndexMeter.includes("imageProgressPercent") &&
    compositeImageIndexMeter.includes('import { CompositeMicroMeter } from "./compositeMicroMeter";') &&
    compositeImageIndexMeter.includes("<CompositeMicroMeter") &&
    compositeImageIndexMeter.includes("progress={progress}") &&
    !compositeImageIndexMeter.includes('import type { CSSProperties } from "react";') &&
    !compositeImageIndexMeter.includes("--image-progress") &&
    compositeImageIndexMeterStyleSource.includes(".image-index-meter") &&
    compositeImageIndexMeterStyleSource.includes("@media (max-width: 940px) and (max-height: 680px)") &&
    !compositeImageIndexMeterStyleSource.includes("--image-progress") &&
    !compositeImageIndexMeterStyleSource.includes("grid-template-columns: auto auto minmax(0, 1fr)") &&
    compositeImageIndexMeterStyleSource.includes(".image-index-meter i") &&
    compositeImageScrubTrackStyleSource.includes(".image-scrub-track") &&
    compositeImageScrubTrackStyleSource.includes(".image-scrub-preview") &&
    compositeImageScrubTrackStyleSource.includes(".image-scrub-preview em") &&
    compositeThemeStyleSource.includes("--composite-interaction-wash") &&
    compositeThemeStyleSource.includes("--composite-overlay-line") &&
    compositeThemeStyleSource.includes("--composite-overlay-shadow") &&
    compositeThemeStyleSource.includes("--composite-scrub-track-height") &&
    compositeThemeStyleSource.includes("--composite-scrub-tick-step") &&
    compositeThemeStyleSource.includes("--composite-scrub-scroll-margin-bottom") &&
    compositeThemeStyleSource.includes("--composite-nearby-column-min") &&
    compositeThemeStyleSource.includes("--composite-accent-strong") &&
    compositeThemeStyleSource.includes("--composite-atlas-background") &&
    compositeThemeStyleSource.includes("--composite-atlas-bin-base") &&
    compositeThemeStyleSource.includes("--composite-atlas-bin-min-height") &&
    compositeThemeStyleSource.includes("--composite-atlas-bin-density-height") &&
    compositeThemeStyleSource.includes("--composite-position-wash") &&
    compositeThemeStyleSource.includes("--composite-position-glow") &&
    compositeThemeStyleSource.includes("--composite-position-track-height") &&
    compositeThemeStyleSource.includes("--composite-navigator-search-line") &&
    compositeThemeStyleSource.includes("--composite-navigator-copy-subtle") &&
    compositeThemeStyleSource.includes("--composite-tool-anchor-surface") &&
    compositeThemeStyleSource.includes("--composite-tool-disabled-ink") &&
    compositeImageAtlasStyleSource.includes("background: var(--composite-atlas-background)") &&
    compositeImageAtlasStyleSource.includes(
      "grid-template-columns: repeat(auto-fit, minmax(var(--composite-atlas-bin-min), 1fr))",
    ) &&
    compositeImageAtlasStyleSource.includes("min-height: var(--composite-atlas-bin-min-height)") &&
    compositeImageAtlasStyleSource.includes("height: calc(var(--match-density) * var(--composite-atlas-bin-density-height))") &&
    compositeImageJumpItemStyleSource.includes("height: var(--composite-position-track-height)") &&
    compositeImageJumpItemStyleSource.includes("box-shadow: var(--composite-position-glow)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeImageAtlasStyleSource) &&
    compositeImageScrubTrackStyleSource.includes("height: var(--composite-scrub-track-height)") &&
    compositeImageScrubTrackStyleSource.includes("scroll-margin-bottom: var(--composite-scrub-scroll-margin-bottom)") &&
    compositeImageScrubTrackStyleSource.includes("box-shadow: var(--composite-overlay-shadow)") &&
    compositeImageNavigationInteractionStyles.every(
      (source) => !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(source),
    ) &&
    [
      compositeImageNavigatorStyleSource,
      compositeImageSearchBarStyleSource,
      compositeInteractionPaletteStyleSource
    ].every((source) => !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(source)) &&
    !compositeImageTimelineStyleSource.includes(".image-scrub-track") &&
    !compositeImageTimelineStyleSource.includes(".image-scrub-preview") &&
    [
      compositeImageNavigatorStyleSource,
      compositeImageJumpControlStyleSource,
      compositeImageAtlasStyleSource,
      compositeImageIndexMeterStyleSource,
      compositeImageScrubTrackStyleSource,
      compositeImageTimelineStyleSource,
      compositeImageSearchBarStyleSource,
      compositeImagePanelStyleSource,
      compositeImageJumpItemStyleSource,
      compositeImageSearchResultItemStyleSource,
      compositeImageSearchPopoverStyleSource,
      compositeImageSearchResultsStyleSource,
      compositeImageSearchPreviewStyleSource,
      compositeInteractionPaletteStyleSource
    ].every((source) => !/font-size:\s*\d/.test(source)) &&
    !compositeImageTimelineStyleSource.includes(".image-filmstrip") &&
    !compositeImageTimelineStyleSource.includes(".image-filmstrip-item") &&
    !compositeImageSearchPopoverStyleSource.includes(".image-filmstrip") &&
    !compositeImageSearchResultsStyleSource.includes(".image-filmstrip") &&
    !compositeImageNavigatorStyleSource.includes(".image-navigator-search-row") &&
    !compositeImageNavigatorStyleSource.includes(".image-navigator-search") &&
    !compositeImageNavigatorStyleSource.includes(".image-navigator-count") &&
    !compositeImageNavigatorStyleSource.includes(".image-jump-field") &&
    !compositeImageNavigatorStyleSource.includes(".image-jump-input") &&
    !compositeImageNavigatorStyleSource.includes(".image-jump-popover") &&
    !compositeImageNavigatorStyleSource.includes(".image-filmstrip") &&
    !compositeImageNavigatorStyleSource.includes(".image-scrub-track") &&
    !compositeImageTimelineStyleSource.includes(".image-index-meter") &&
    !compositeImageNavigatorStyleSource.includes(".image-jump-atlas") &&
    !compositeImageNavigatorStyleSource.includes(".image-map-bin") &&
    !compositeImageNavigatorStyleSource.includes(".composite-interaction-palette") &&
    !compositeReportStyleSource.includes(".composite-image-navigator") &&
    compositeComposerDockStyleSource.includes(".composite-composer-dock") &&
    !compositeComposerDockStyleSource.includes(".composer-dock-stat") &&
    !compositeComposerDockStyleSource.includes(".composer-dock-meter") &&
    compositeComposerDockStyleSource.includes(".composer-dock-grip") &&
    compositeComposerDrawerStyleSource.includes(".composite-sidebar-drawer") &&
    compositeComposerDrawerStyleSource.includes(".composite-sidebar-grid") &&
    !compositeReportPanelStyleSource.includes(".report-panel-head") &&
    !compositeReportPanelStyleSource.includes(".report-panel-actions") &&
    !compositeReportPanelStyleSource.includes(".report-empty-state") &&
    !compositeReportPanelStyleSource.includes(".report-layer-name") &&
    compositeReportRunPoolStyleSource.includes(".report-run-pool") &&
    compositeReportRunPoolStyleSource.includes(".report-run-card") &&
    compositeReportRunPoolStyleSource.includes(".report-run-filter-tabs") &&
    compositeReportRunPoolStyleSource.includes("--report-run-pool-bg") &&
    compositeReportRunPoolStyleSource.includes("--report-run-card-divider") &&
    compositeReportRunPoolStyleSource.includes("content-visibility: auto;") &&
    compositeReportRunPoolStyleSource.includes("contain-intrinsic-size: auto 56px;") &&
    !compositeReportRunPoolStyleSource.includes("#f7f9fc") &&
    !compositeReportRunPoolStyleSource.includes("#e7edf4") &&
    !compositeReportRunPoolStyleSource.includes(".report-layer-tabs") &&
    compositeReportLayerPlanStyleSource.includes(".report-layer-plan") &&
    compositeReportLayerPlanStyleSource.includes(".report-layer-row") &&
    compositeReportLayerPlanStyleSource.includes(".report-layer-name span") &&
    compositeReportLayerPlanStyleSource.includes(':root[data-theme="dark"] .report-layer-plan') &&
    compositeReportLayerPlanStyleSource.includes("--report-layer-head-bg") &&
    compositeReportLayerPlanStyleSource.includes("content-visibility: auto;") &&
    compositeReportLayerPlanStyleSource.includes("contain-intrinsic-size: auto 74px;") &&
    !compositeReportLayerPlanStyleSource.includes("background: #f7f9fc") &&
    !compositeReportLayerPlanStyleSource.includes("border-bottom: 1px solid #e7edf4") &&
    !compositeReportLayerPlanStyleSource.includes("rgba(34, 72, 197, 0.12)") &&
    !compositeComposerDockStyleSource.includes(".report-run-pool") &&
    !compositeComposerDrawerStyleSource.includes(".report-run-card") &&
    !compositeReportRunPoolStyleSource.includes(".report-layer-row") &&
    !compositeReportLayerPlanStyleSource.includes(".report-run-card") &&
    !compositeReportStyleSource.includes(".composite-composer-dock") &&
    !compositeReportStyleSource.includes(".composite-sidebar-drawer") &&
    !compositeReportStyleSource.includes(".report-run-pool") &&
    !compositeReportStyleSource.includes(".report-layer-plan") &&
    compositeReportStyleSource.includes(".composite-report-shell.sidebar-open") &&
    compositeReportStyleSource.includes(".composite-report-shell.sidebar-collapsed") &&
    compositeReportStyleSource.includes(".composite-sidebar-backdrop") &&
    compositeThemeStyleSource.includes("--composite-page-surface") &&
    compositeThemeStyleSource.includes("--composite-surface-glass") &&
    compositeThemeStyleSource.includes("--composite-muted-strong") &&
    compositeThemeStyleSource.includes("--composite-shell-line") &&
    compositeThemeStyleSource.includes("--composite-sidebar-backdrop-z") &&
    compositeThemeStyleSource.includes("--composite-sidebar-backdrop-hover") &&
    compositeThemeStyleSource.includes("--composite-drawer-shadow: 8px 0 20px") &&
    compositeThemeStyleSource.includes("--composite-popover-shadow: 0 10px 24px") &&
    compositeThemeStyleSource.includes("--composite-shell-mobile-min-height") &&
    compositeReportStyleSource.includes("z-index: var(--composite-sidebar-backdrop-z)") &&
    compositeReportStyleSource.includes("background: var(--composite-page-surface)") &&
    compositeReportStyleSource.includes("border-bottom: 1px solid var(--composite-shell-line)") &&
    compositeReportStyleSource.includes("background: var(--composite-sidebar-backdrop)") &&
    compositeReportStyleSource.includes("inset: var(--composite-dock-rail-size) 0 0 0") &&
    compositeReportStyleSource.includes(
      "grid-template-columns: var(--composite-dock-rail-size) minmax(0, 1fr)",
    ) &&
    !compositeReportStyleSource.includes("grid-template-columns: 48px minmax(0, 1fr)") &&
    !compositeReportStyleSource.includes("minmax(560px, 680px)") &&
    !compositeReportStyleSource.includes("minmax(420px, 48vw)") &&
    compositeComposerDrawerStyleSource.includes("position: absolute") &&
    compositeComposerDrawerStyleSource.includes("inset: 0 auto 0 var(--composite-dock-rail-size)") &&
    compositeComposerDrawerStyleSource.includes(
      "width: min(var(--composite-drawer-width), calc(100% - var(--composite-dock-rail-size)))",
    ) &&
    compositeComposerDockStyleSource.includes("width: var(--composite-dock-rail-size)") &&
    compositeThemeStyleSource.includes("--composite-dock-rail-size") &&
    compositeThemeStyleSource.includes("--composite-drawer-width") &&
    !compositeComposerDrawerStyleSource.includes("calc(100% - 48px)") &&
    compositeComposerDrawerStyleSource.includes("box-shadow: var(--composite-drawer-shadow)") &&
    compositeComposerDrawerStyleSource.includes("box-shadow: var(--composite-drawer-shadow-mobile)") &&
    compositeComposerDrawerStyleSource.includes("background: var(--composite-surface-quiet)") &&
    compositeComposerDockPreviewStyleSource.includes("box-shadow: var(--composite-popover-shadow)") &&
    compositeComposerDockPreviewStyleSource.includes("background: var(--composite-surface-glass)") &&
    compositeComposerDockPreviewStyleSource.includes("color: var(--composite-muted-strong)") &&
    compositeImageSearchPopoverStyleSource.includes("box-shadow: var(--composite-popover-shadow)") &&
    compositeImageSearchPopoverStyleSource.includes("box-shadow: var(--composite-popover-shadow-top)") &&
    compositeImageSearchPopoverStyleSource.includes("border: 1px solid var(--composite-line)") &&
    !compositeComposerDrawerStyleSource.includes("14px 0 32px") &&
    !compositeComposerDrawerStyleSource.includes("0 18px 40px") &&
    !compositeComposerDrawerStyleSource.includes("#f7f9fc") &&
    !compositeComposerDrawerStyleSource.includes("#f4f7fa") &&
    !compositeComposerDrawerStyleSource.includes("#cbd7e4") &&
    !compositeComposerDockPreviewStyleSource.includes("rgb(255 255 255 / 96%)") &&
    !compositeComposerDockPreviewStyleSource.includes("#cfdae6") &&
    !compositeComposerDockPreviewStyleSource.includes("#e4ebf2") &&
    !compositeComposerDockPreviewStyleSource.includes("#243243") &&
    !compositeImageSearchPopoverStyleSource.includes("#ccd7e4") &&
    !compositeComposerDockPreviewStyleSource.includes("0 18px 42px") &&
    !compositeImageSearchPopoverStyleSource.includes("0 18px 44px") &&
    !appThemeStyleSource.includes(".suite-report-page") &&
    !appThemeStyleSource.includes(".suite-report-grid") &&
    !appThemeStyleSource.includes(".suite-panel") &&
    !appThemeStyleSource.includes(".composite-view-card") &&
    !appThemeStyleSource.includes(".composite-image-frame") &&
    !appThemeStyleSource.includes(".composite-instance-box") &&
    !appThemeStyleSource.includes(".composite-stage {") &&
    compositeReportStage.includes('import "./compositeReportStage.css";') &&
    compositeStageWorkbench.includes('import "./compositeStageWorkbench.css";') &&
    compositeLayerFocusToolbar === "" &&
    compositeLayerFocusToolbarStyleSource === "" &&
    compositeObjectHud === "" &&
    compositeObjectHudStyleSource === "" &&
    compositeStageWorkbenchStyleSource.includes(".composite-report-workbench") &&
    compositeStageWorkbenchStyleSource.includes(".composite-report-focus") &&
    compositeStageWorkbenchStyleSource.includes(".composite-report-focus.resizable-split") &&
    compositeStageWorkbenchStyleSource.includes(".composite-report-focus.compact") &&
    !compositeReportStageStyleSource.includes(".composite-workbench-toolbar") &&
    !compositeReportStageStyleSource.includes(".composite-layer-focus-strip") &&
    compositeThemeStyleSource.includes("--composite-object-hud-background") &&
    compositeThemeStyleSource.includes("--composite-object-hud-fp-background") &&
    compositeThemeStyleSource.includes("--composite-object-menu-background") &&
    compositeThemeStyleSource.includes("--composite-object-menu-shadow") &&
    compositeThemeStyleSource.includes("--composite-object-menu-safe-height") &&
    compositeThemeStyleSource.includes("--composite-object-menu-safe-right") &&
    !compositeObjectHudStyleSource.includes(".object-hud-cruise") &&
    !compositeObjectHudStyleSource.includes("--object-progress") &&
    !compositeObjectHudStyleSource.includes("width: calc(var(--object-progress) * 100%)") &&
    compositeMicroMeterStyleSource.includes(".composite-micro-meter") &&
    compositeMicroMeterStyleSource.includes("composite-meter-ring") &&
    compositeMicroMeterStyleSource.includes("--composite-meter-progress") &&
    compositeMicroMeterStyleSource.includes("conic-gradient") &&
    compositeMicroMeterStyleSource.includes("composite-meter-sweep") &&
    compositeMicroMeterStyleSource.includes(
      ".composite-micro-meter.idle .composite-meter-ring::before",
    ) &&
    compositeMicroMeterStyleSource.includes(
      ".composite-micro-meter.idle .composite-meter-ring::before {\n  animation: none;",
    ) &&
    !compositeMicroMeterStyleSource.includes("width: calc(var(--composite-meter-progress) * 100%)") &&
    compositeObjectContextMenuStyleSource.includes(".composite-object-context-menu") &&
    compositeObjectContextMenuStyleSource.includes(".object-context-actions") &&
    compositeObjectContextMenuStyleSource.includes(
      "top: min(var(--context-y), calc(100vh - var(--composite-object-menu-safe-height)))",
    ) &&
    compositeObjectContextMenuStyleSource.includes(
      "left: min(var(--context-x), calc(100vw - var(--composite-object-menu-safe-right)))",
    ) &&
    compositeObjectContextMenuStyleSource.includes("box-shadow: var(--composite-object-menu-shadow)") &&
    compositeObjectContextMenuStyleSource.includes(".composite-object-context-menu.status-fp") &&
    [
      compositeObjectHudStyleSource,
      compositeObjectContextMenuStyleSource
    ].every((source) => !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(source)) &&
    !compositeReportStageStyleSource.includes(".composite-object-hud") &&
    !compositeReportStageStyleSource.includes(".composite-object-context-menu") &&
    !compositeReportStage.includes("function CompositeObjectHud") &&
    !compositeReportStage.includes("function CompositeObjectContextMenu") &&
    !compositeReportStageStyleSource.includes(".composite-report-workbench") &&
    !compositeReportStageStyleSource.includes(".composite-report-focus") &&
    compositeOverlayStageStyleSource.includes(".composite-overlay-stage") &&
    compositeOverlayStageStyleSource.includes(".composite-layer-legend") &&
    compositeOverlayStageStyleSource.includes(".composite-layer-legend {\n    display: none;") &&
    compositeLayerCanvasStyleSource.includes(".composite-workbench-canvas") &&
    compositeLayerCanvasStyleSource.includes("position: relative") &&
    compositeLayerCanvasStyleSource.includes(".composite-workbench-canvas.small") &&
    compositeLayerCanvasStyleSource.includes('.composite-workbench-canvas[data-object-wheel-cruise="modified"]') &&
    compositeLayerCanvasStyleSource.includes('.composite-workbench-canvas[data-overlay-surface-pan="modified"] .image-stage') &&
    compositeLayerCanvasStyleSource.includes("touch-action: none") &&
    compositeLayerCanvasStyleSource.includes("overscroll-behavior: contain") &&
    compositeLayerCanvasStyleSource.includes('[data-pointer-reticle="active"]') &&
    compositeThemeStyleSource.includes("--composite-canvas-line") &&
    compositeThemeStyleSource.includes("--composite-canvas-grid-line") &&
    compositeThemeStyleSource.includes("--composite-canvas-grid-line-active") &&
    compositeThemeStyleSource.includes("--composite-canvas-grid-size") &&
    compositeThemeStyleSource.includes("--composite-canvas-small-min-height") &&
    compositeLayerCanvasStyleSource.includes("border: 1px solid var(--composite-canvas-line)") &&
    compositeLayerCanvasStyleSource.includes("var(--composite-canvas-grid-line)") &&
    compositeLayerCanvasStyleSource.includes("var(--composite-canvas-grid-line-active)") &&
    compositeLayerCanvasStyleSource.includes(
      "background-size: var(--composite-canvas-grid-size) var(--composite-canvas-grid-size)",
    ) &&
    compositeLayerCanvasStyleSource.includes("min-height: var(--composite-canvas-small-min-height)") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeLayerCanvasStyleSource) &&
    compositeCanvasGestureHudStyleSource.includes(".composite-canvas-gesture-hud") &&
    !compositeCanvasGestureHudStyleSource.includes("pointer-events: none") &&
    !compositeCanvasGestureHudStyleSource.includes("span.ready") &&
    !compositeCanvasGestureHudStyleSource.includes("span.active") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-overlay-panel") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-overlay-panel.anchor-bottom-right") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-overlay-panel.anchor-full") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-overlay-chip.ready") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-overlay-chip.active") &&
    compositeCanvasOverlayStyleSource.includes(".composite-canvas-coordinate-tag") &&
    compositeCanvasOverlayStyleSource.includes("pointer-events: none") &&
    compositeThemeStyleSource.includes("--composite-canvas-overlay-line") &&
    compositeThemeStyleSource.includes("--composite-canvas-overlay-surface") &&
    compositeThemeStyleSource.includes("--composite-canvas-overlay-shadow") &&
    compositeThemeStyleSource.includes("--composite-canvas-coordinate-safe-y") &&
    compositeCanvasOverlayStyleSource.includes("border: 1px solid var(--composite-canvas-overlay-line)") &&
    compositeCanvasOverlayStyleSource.includes("background: var(--composite-canvas-overlay-surface)") &&
    compositeCanvasOverlayStyleSource.includes("box-shadow: var(--composite-canvas-overlay-shadow)") &&
    compositeCanvasOverlayStyleSource.includes("var(--composite-canvas-coordinate-safe-y)") &&
    !compositeCanvasOverlayStyleSource.includes("--composite-pad-16") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeCanvasOverlayStyleSource) &&
    !compositeLayerCanvasStyleSource.includes(".composite-canvas-gesture-hud") &&
    compositeCanvasPointerReticleStyleSource.includes(".composite-canvas-pointer-reticle") &&
    compositeCanvasPointerReticleStyleSource.includes("--composite-pointer-x") &&
    compositeCanvasPointerReticleStyleSource.includes("--composite-pointer-y") &&
    compositeCanvasPointerReticleStyleSource.includes(".viewer-pointer-surface[data-pointer-reticle=\"active\"]") &&
    compositeThemeStyleSource.includes("--composite-pointer-reticle-z") &&
    compositeThemeStyleSource.includes("--composite-pointer-axis-color") &&
    compositeThemeStyleSource.includes("--composite-pointer-axis-opacity") &&
    compositeThemeStyleSource.includes("--composite-pointer-axis-width") &&
    compositeCanvasPointerReticleStyleSource.includes("z-index: var(--composite-pointer-reticle-z)") &&
    compositeCanvasPointerReticleStyleSource.includes("background: var(--composite-pointer-axis-color)") &&
    compositeCanvasPointerReticleStyleSource.includes("opacity: var(--composite-pointer-axis-opacity)") &&
    compositeCanvasPointerReticleStyleSource.includes("height: var(--composite-pointer-axis-width)") &&
    !compositeCanvasPointerReticleStyleSource.includes("pointer-events: none") &&
    !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(compositeCanvasPointerReticleStyleSource) &&
    compositeCanvasPointerReticleStyleSource.includes(".axis-x") &&
    compositeCanvasPointerReticleStyleSource.includes(".axis-y") &&
    !compositeLayerCanvasStyleSource.includes(".composite-canvas-pointer-reticle") &&
    compositeLayerInspectorStyleSource.includes(".composite-inspector-panel") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-strip") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-chip.fn") &&
    compositeLayerObjectStripStyleSource.includes(".layer-object-chip.fp") &&
    compositeSplitStage === "" &&
    compositeSplitStageStyleSource === "" &&
    compositeSplitPane === "" &&
    compositeSplitPaneStyleSource === "" &&
    compositeSplitLayerCanvas === "" &&
    !compositeSplitStage.includes("function SplitLayerCanvas") &&
    !compositeSplitStage.includes("function MissingLayerPane") &&
    compositeThemeStyleSource.includes("--composite-empty-state-min-height") &&
    compositeThemeStyleSource.includes("--composite-workbench-focus-min") &&
    compositeThemeStyleSource.includes("--composite-split-pane-min-height") &&
    compositeThemeStyleSource.includes("--composite-split-stage-column-min") &&
    compositeThemeStyleSource.includes("--composite-split-focus-ring-width") &&
    compositeReportStageStyleSource.includes("min-height: var(--composite-empty-state-min-height)") &&
    compositeStageWorkbenchStyleSource.includes("var(--composite-workbench-focus-min)") &&
    [
      compositeReportStyleSource,
      compositeReportStageStyleSource,
      compositeStageWorkbenchStyleSource,
      compositeSplitStageStyleSource,
      compositeSplitPaneStyleSource
    ].every((source) => !/(#[0-9a-f]{3,8}\b|rgba?\()/i.test(source)) &&
    !compositeSplitStageStyleSource.includes(".composite-split-pane") &&
    !compositeSplitStageStyleSource.includes(".composite-pane-head") &&
    compositeThemeStyleSource.includes("--composite-surface: #ffffff") &&
    compositeThemeStyleSource.includes("--composite-line: #d8e1ea") &&
    compositeThemeStyleSource.includes("--composite-ink: #172033") &&
    compositeThemeStyleSource.includes("--composite-accent: #2248c5") &&
    compositeThemeStyleSource.includes(':root[data-theme="dark"]') &&
    compositeThemeStyleSource.includes("--composite-surface: var(--bench-surface-raised)") &&
    compositeThemeStyleSource.includes("--composite-page-surface: var(--bench-bg-soft)") &&
    compositeThemeStyleSource.includes("--composite-line: var(--bench-line)") &&
    compositeThemeStyleSource.includes("--composite-ink: var(--bench-ink)") &&
    compositeThemeStyleSource.includes("--composite-gap-2: 2px") &&
    compositeThemeStyleSource.includes("--composite-gap-12: 12px") &&
    compositeThemeStyleSource.includes("--composite-pad-2: 2px") &&
    compositeThemeStyleSource.includes("--composite-pad-12: 12px") &&
    compositeThemeStyleSource.includes("--composite-control-min: 30px") &&
    compositeThemeStyleSource.includes("--composite-dock-track-size") &&
    compositeThemeStyleSource.includes("--composite-dock-grip-min") &&
    compositeThemeStyleSource.includes("--composite-radius-control: 2px") &&
    compositeReportStyleSource.includes("var(--composite-surface)") &&
    compositeImageNavigatorStyleSource.includes("var(--composite-canvas-overlay-surface)") &&
    compositeImageAtlasStyleSource.includes("var(--composite-accent)") &&
    !compositeReportStageStyleSource.includes(".composite-inspector-panel") &&
    !compositeReportStageStyleSource.includes(".layer-object-chip") &&
    !compositeReportStageStyleSource.includes(".composite-split-stage") &&
    !compositeReportStageStyleSource.includes(".composite-split-pane") &&
    !compositeReportStageStyleSource.includes(".composite-workbench-canvas") &&
    !compositeReportStageStyleSource.includes(".composite-overlay-stage") &&
    !compositeStageWorkbenchStyleSource.includes(".composite-report-workbench.mode-overlay") &&
    !compositeStageWorkbenchStyleSource.includes(".composite-report-workbench.mode-split") &&
    [
      compositeReportStyleSource,
      compositeComposerDockStyleSource,
      compositeComposerDrawerStyleSource,
      compositePanelPrimitivesStyleSource,
      compositeReportPanelStyleSource,
      compositeReportRunPoolStyleSource,
      compositeReportLayerPlanStyleSource,
      compositeImageNavigatorStyleSource,
      compositeImageJumpControlStyleSource,
      compositeImageSearchBarStyleSource,
      compositeImagePanelStyleSource,
      compositeImageJumpItemStyleSource,
      compositeImageSearchResultItemStyleSource,
      compositeInteractionPaletteStyleSource,
      compositeImageAtlasStyleSource,
      compositeImageTimelineStyleSource,
      compositeImageIndexMeterStyleSource,
      compositeImageSearchPopoverStyleSource,
      compositeImageSearchResultsStyleSource,
      compositeImageSearchPreviewStyleSource,
      compositeImageSearchStatusStyleSource,
      compositeReportStageStyleSource,
      compositeStageWorkbenchStyleSource,
      compositeLayerFocusToolbarStyleSource,
      compositeMicroMeterStyleSource,
      compositeObjectHudStyleSource,
      compositeObjectContextMenuStyleSource,
      compositeOverlayStageStyleSource,
      compositeLayerCanvasStyleSource,
      compositeLayerInspectorStyleSource,
      compositeLayerObjectStripStyleSource,
      compositeSplitStageStyleSource,
      compositeSplitPaneStyleSource
    ].every((source) => !/font-size:\s*\d/.test(source)) &&
    [
      compositeReportStyleSource,
      compositeComposerDockStyleSource,
      compositeComposerDrawerStyleSource,
      compositePanelPrimitivesStyleSource,
      compositeReportPanelStyleSource,
      compositeReportRunPoolStyleSource,
      compositeReportLayerPlanStyleSource,
      compositeImageNavigatorStyleSource,
      compositeImageJumpControlStyleSource,
      compositeImageSearchBarStyleSource,
      compositeImagePanelStyleSource,
      compositeImageJumpItemStyleSource,
      compositeImageSearchResultItemStyleSource,
      compositeInteractionPaletteStyleSource,
      compositeImageAtlasStyleSource,
      compositeImageTimelineStyleSource,
      compositeImageIndexMeterStyleSource,
      compositeImageSearchPopoverStyleSource,
      compositeImageSearchResultsStyleSource,
      compositeImageSearchPreviewStyleSource,
      compositeImageSearchStatusStyleSource,
      compositeReportStageStyleSource,
      compositeStageWorkbenchStyleSource,
      compositeLayerFocusToolbarStyleSource,
      compositeMicroMeterStyleSource,
      compositeObjectHudStyleSource,
      compositeObjectContextMenuStyleSource,
      compositeOverlayStageStyleSource,
      compositeLayerCanvasStyleSource,
      compositeLayerInspectorStyleSource,
      compositeLayerObjectStripStyleSource,
      compositeSplitStageStyleSource,
      compositeSplitPaneStyleSource
    ].every((source) =>
      !/#(?:ffffff|fbfcfd|f8fafc|f3f6f9|d8e1ea|e2eaf2|dce4ed|d9e2ec|172033|152033|111827|66778c|66768a|2248c5|1f5eff)\b/i.test(source)
    ) &&
    compositeComponentStyleSources.every(
      (source) => !rawCompositeControlGeometryPattern.test(source),
    ) &&
    !compositeReportStyleSource.includes(".composite-report-workbench") &&
    !compositeReportStyleSource.includes(".composite-report-focus") &&
    !compositeReportStyleSource.includes(".composite-inspector-panel") &&
    !compositeImageNavigatorStyleSource.includes(".image-union-local"),
  "composite report styles must keep page, stage, and image navigator CSS in separate modules",
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
  comparisonSamplePage.includes('import "./comparisonSampleStyles.css";') &&
    comparisonSampleStyleSource.includes(".comparison-sample-detail") &&
    comparisonSampleStyleSource.includes(".comparison-sample-title"),
  "comparison sample page must use the shared comparison sample style module",
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

async function collectStyleFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectStyleFiles(entryPath)));
    } else if (entry.name.endsWith(".css")) {
      files.push(entryPath);
    }
  }
  return files;
}

async function readSource(relativePath) {
  return readFile(path.join(root, relativePath), "utf8");
}

async function readCssSource() {
  const entries = await readdir(srcRoot, { withFileTypes: true });
  const cssFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".css"))
    .map((entry) => path.join(srcRoot, entry.name))
    .sort();
  const sources = await Promise.all(cssFiles.map((filePath) => readFile(filePath, "utf8")));
  return sources.join("\n");
}

function assertNoBlockingBrowserDialogs(source, relativePath) {
  const match = source.match(/\b(?:window\.)?(confirm|alert|prompt)\s*\(/);
  assert(!match, `${relativePath}: blocking browser dialog '${match?.[1]}' is not allowed`);
}

function assertNoBusinessDialogShell(source, relativePath) {
  if (relativePath === "src/ui.tsx" || relativePath === "src/uiDialog.tsx") {
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

function assertNoForbiddenUiCopy(source, relativePath, forbiddenItems) {
  for (const item of forbiddenItems) {
    assert(!source.includes(item), `${relativePath}: forbidden UI copy '${item}' is not allowed`);
  }
}

function assertMaxLines(source, relativePath, maxLines) {
  const lineCount = source.split("\n").length;
  assert(
    lineCount <= maxLines,
    `${relativePath}: ${lineCount} lines exceeds the ${maxLines}-line modularity budget`,
  );
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
  if (relativePath === "src/ui.tsx" || relativePath === "src/uiActions.tsx") {
    return;
  }
  assert(!/<button\b/.test(source), `${relativePath}: buttons must use shared UI primitives`);
}

function assertNoRawInputOutsidePrimitives(source, relativePath) {
  if (relativePath === "src/controlPrimitives.tsx" || relativePath === "src/selectPopoverControl.tsx") {
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

function assertQueryFnsUseAbortSignal(source, relativePath) {
  const queryFnMatches = source.matchAll(/\bqueryFn\s*:/g);
  for (const match of queryFnMatches) {
    const snippet = source.slice(match.index, match.index + 80);
    assert(
      snippet.includes("queryFn: ({ signal })"),
      `${relativePath}: React Query queryFn must receive AbortSignal and pass it to GET APIs`,
    );
  }
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
