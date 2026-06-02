import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, X } from "lucide-react";

import type { EvalInstance } from "./api";
import { fetchSettingsPreviewSample } from "./api";
import { SearchInputControl } from "./controlPrimitives";
import { unique } from "./formatters";
import type { SettingsPanelId, SettingsSectionSummary } from "./settingsPreferenceDrawer";
import { SettingsPreferenceDrawer } from "./settingsPreferenceDrawer";
import {
  useWorkspaceShortcuts,
  useWorkspaceSettings
} from "./workspaceSettings";
import { useTypographySettings } from "./typographySettings";
import { displayImageUrl } from "./viewerGeometry";
import { CanvasStage } from "./viewerCanvas";
import { IconActionButton, SelectableCardButton } from "./ui";

import "./settingsTheme.css";
import "./settingsWorkbench.css";
import "./settingsPreview.css";
import "./settingsDrawer.css";
import "./settingsEditor.css";
import "./settingsTypography.css";
import "./settingsLabels.css";
import "./settingsShortcuts.css";

const SETTINGS_PREVIEW_IMAGE_URL = "/static/settings_preview.svg";
const SETTINGS_PREVIEW_LABELS = ["arrow", "icon"];

export function SettingsPage() {
  const previewQuery = useQuery({
    queryKey: ["settings-preview-sample"],
    queryFn: fetchSettingsPreviewSample,
    retry: false,
    staleTime: 60_000
  });
  const fallbackGtInstances = useMemo<EvalInstance[]>(
    () => [
      {
        label: "arrow",
        bbox: [174, 246, 796, 450],
        linestrip: [
          [196, 422],
          [438, 270],
          [760, 292]
        ]
      },
      {
        label: "icon",
        bbox: [118, 122, 180, 184],
        keypoints: [
          [146, 144],
          [132, 168],
          [160, 168]
        ]
      }
    ],
    []
  );
  const previewGtInstances = previewQuery.data?.gt_instances?.length
    ? previewQuery.data.gt_instances
    : fallbackGtInstances;
  const previewLabelsList = useMemo(
    () => unique(previewGtInstances.map((item) => item.label).filter(Boolean)),
    [previewGtInstances]
  );
  const visiblePreviewLabels = useMemo(() => new Set(previewLabelsList), [previewLabelsList]);
  const {
    labels,
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars,
    updateOverlayStyle,
    updateInteractionSetting,
    updateLabelColor,
    removeLabelColor,
    resetOverlayStyle,
    resetInteractionSettings,
    resetLabelColors
  } = useWorkspaceSettings(previewLabelsList.length ? previewLabelsList : SETTINGS_PREVIEW_LABELS);
  const {
    typographySettings,
    typographyVars,
    updateTypographySettings,
    resetTypographySettings
  } = useTypographySettings();
  const shortcutSettings = useWorkspaceShortcuts();
  const previewSample = previewQuery.data?.sample ?? null;
  const previewWidth = previewSample?.image_width ?? 960;
  const previewHeight = previewSample?.image_height ?? 600;
  const previewImageUrl = previewSample ? displayImageUrl(previewSample) : SETTINGS_PREVIEW_IMAGE_URL;
  const previewMeta =
    previewQuery.data && previewSample
      ? `${previewQuery.data.benchmark_id} / #${previewSample.index + 1}`
      : "未找到基准集样本时使用内置示意图";
  const [activeSettingsPanel, setActiveSettingsPanel] = useState<SettingsPanelId>("appearance");
  const [settingsQuery, setSettingsQuery] = useState("");
  const sortedLabels = useMemo(
    () => [...labels].sort((left, right) => left.localeCompare(right)),
    [labels]
  );
  const settingsSections = [
    { id: "appearance", label: "外观", meta: "几何样式" },
    { id: "typography", label: "字体", meta: `${typographySettings.baseFontSize}px` },
    { id: "labels", label: "标签颜色", meta: `${sortedLabels.length} labels` },
    { id: "interaction", label: "交互", meta: "缩放、拖拽和范围" },
    { id: "workflow", label: "快捷键", meta: "Action map" }
  ] satisfies SettingsSectionSummary[];
  const query = settingsQuery.trim().toLowerCase();
  const visiblePanels: SettingsPanelId[] = query
    ? settingsSections
        .filter((section) => `${section.label} ${section.meta}`.toLowerCase().includes(query))
        .map((section) => section.id)
    : [activeSettingsPanel];
  const activeSection = settingsSections.find((section) => section.id === activeSettingsPanel);
  const visibleSectionLabel = query ? "搜索结果" : activeSection?.label ?? "设置";
  const visibleSectionMeta = query
    ? `${visiblePanels.length} 个分组匹配`
    : activeSection?.meta ?? "当前设置分组";

  return (
    <section className="page-stack settings-page settings-workbench-page">
      <div
        className="settings-workbench-shell settings-console-shell"
        style={{ ...overlayVars, ...typographyVars }}
      >
        <header className="settings-command-bar">
          <div className="settings-command-title">
            <div>
              <span>Eval Bench Preferences</span>
              <h2>工作台设置</h2>
            </div>
            <p>以最小控制面板管理视觉偏好，把主空间留给样本检查。</p>
          </div>
          <div className="settings-command-center">
            <SearchInputControl
              className="settings-search-box"
              icon={<Search size={15} />}
              label="搜索设置"
              value={settingsQuery}
              placeholder="搜索设置"
              onChange={setSettingsQuery}
              action={settingsQuery ? (
                <IconActionButton
                  className="settings-search-clear"
                  icon={<X size={13} />}
                  title="清空搜索"
                  onClick={() => setSettingsQuery("")}
                />
              ) : null}
            />
            <nav className="settings-section-nav" aria-label="工作台设置分组">
              {settingsSections.map((section) => (
                <SelectableCardButton
                  key={section.id}
                  active={!query && activeSettingsPanel === section.id}
                  className="settings-section-button"
                  onClick={() => {
                    setActiveSettingsPanel(section.id);
                    setSettingsQuery("");
                  }}
                >
                  <span>{section.label}</span>
                  <small>{section.meta}</small>
                </SelectableCardButton>
              ))}
            </nav>
          </div>
          <div className="settings-profile-strip" title="当前版本使用浏览器本地 profile 保存设置">
            <span>Profile</span>
            <strong>Local Browser</strong>
            <small>{sortedLabels.length} labels</small>
          </div>
        </header>

        <main className="settings-visual-region">
          <div className="settings-preview-stage">
            {previewQuery.isFetching ? <div className="viewer-fetch-chip">正在刷新预览样本</div> : null}
            <CanvasStage
              width={previewWidth}
              height={previewHeight}
              imageUrl={previewImageUrl}
              imageAlt="工作台设置预览"
              gtInstances={previewGtInstances}
              predInstances={[]}
              diagnostics={null}
              visibleLabels={visiblePreviewLabels}
              showGt={true}
              showPred={false}
              showBoxes={true}
              showLines={true}
              showKeypoints={true}
              overlayColors={overlayColors}
              overlayStyle={overlayStyle}
              labelColors={labelColors}
              interactionSettings={interactionSettings}
            />
          </div>
          <div className="settings-preview-dock">
            <div>
              <span>Preview</span>
              <strong>{previewMeta}</strong>
            </div>
            <div className="settings-preview-foot">
              <span style={{ "--swatch": overlayColors.gt } as CSSProperties}>GT</span>
              <span style={{ "--swatch": overlayColors.pred } as CSSProperties}>Pred</span>
              <span style={{ "--swatch": overlayColors.fn } as CSSProperties}>FN</span>
              <span style={{ "--swatch": overlayColors.fp } as CSSProperties}>FP</span>
            </div>
          </div>
        </main>

        <SettingsPreferenceDrawer
          visibleSectionLabel={visibleSectionLabel}
          visibleSectionMeta={visibleSectionMeta}
          visiblePanels={visiblePanels}
          overlayColors={overlayColors}
          overlayStyle={overlayStyle}
          typographySettings={typographySettings}
          labelColors={labelColors}
          sortedLabels={sortedLabels}
          interactionSettings={interactionSettings}
          shortcutBindings={shortcutSettings.bindings}
          onOverlayStyleChange={updateOverlayStyle}
          onResetOverlayStyle={resetOverlayStyle}
          onTypographySettingsChange={updateTypographySettings}
          onResetTypographySettings={resetTypographySettings}
          onLabelColorChange={updateLabelColor}
          onRemoveLabelColor={removeLabelColor}
          onResetLabelColors={resetLabelColors}
          onInteractionSettingChange={updateInteractionSetting}
          onResetInteractionSettings={resetInteractionSettings}
          onShortcutChange={shortcutSettings.updateShortcut}
          onShortcutReset={shortcutSettings.resetShortcut}
          onResetShortcuts={shortcutSettings.resetShortcuts}
        />
      </div>
    </section>
  );
}
