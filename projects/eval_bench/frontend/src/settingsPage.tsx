import React, { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, X } from "lucide-react";

import type { EvalInstance } from "./api";
import { fetchSettingsPreviewSample } from "./api";
import { CompactSelectControl, NumberSettingControl } from "./controlPrimitives";
import { unique } from "./formatters";
import { AppIcon } from "./iconLibrary";
import {
  LabelColorQuickAdd,
  SettingsEditorSection,
  SettingsPreferenceRow,
  ShortcutSettingsPanel
} from "./settingsControls";
import {
  INTERACTION_SETTING_CONTROLS,
  INSTANCE_COLOR_ROLES,
  OVERLAY_STYLE_CONTROLS,
  PRED_LINE_STYLE_OPTIONS,
  explicitLabelColor,
  settingControlValue,
  settingValueFromControl,
  useWorkspaceShortcuts,
  useWorkspaceSettings
} from "./workspaceSettings";
import { displayImageUrl } from "./viewerGeometry";
import { CanvasStage } from "./viewerCanvas";
import { ActionButton, EmptyState, IconActionButton, SelectableCardButton } from "./ui";

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
  const shortcutSettings = useWorkspaceShortcuts();
  const previewSample = previewQuery.data?.sample ?? null;
  const previewWidth = previewSample?.image_width ?? 960;
  const previewHeight = previewSample?.image_height ?? 600;
  const previewImageUrl = previewSample ? displayImageUrl(previewSample) : SETTINGS_PREVIEW_IMAGE_URL;
  const previewMeta =
    previewQuery.data && previewSample
      ? `${previewQuery.data.benchmark_id} / #${previewSample.index + 1}`
      : "未找到基准集样本时使用内置示意图";
  const [activeSettingsPanel, setActiveSettingsPanel] = useState("appearance");
  const [settingsQuery, setSettingsQuery] = useState("");
  const sortedLabels = useMemo(
    () => [...labels].sort((left, right) => left.localeCompare(right)),
    [labels]
  );
  const settingsSections = [
    { id: "appearance", label: "外观", meta: "几何样式" },
    { id: "labels", label: "标签颜色", meta: `${sortedLabels.length} labels` },
    { id: "interaction", label: "交互", meta: "缩放、拖拽和范围" },
    { id: "workflow", label: "快捷键", meta: "Action map" }
  ];
  const query = settingsQuery.trim().toLowerCase();
  const visiblePanels = query
    ? settingsSections
        .filter((section) => `${section.label} ${section.meta}`.toLowerCase().includes(query))
        .map((section) => section.id)
    : [activeSettingsPanel];
  const showPanel = (id: string) => visiblePanels.includes(id);
  const activeSection = settingsSections.find((section) => section.id === activeSettingsPanel);
  const visibleSectionLabel = query ? "搜索结果" : activeSection?.label ?? "设置";
  const visibleSectionMeta = query
    ? `${visiblePanels.length} 个分组匹配`
    : activeSection?.meta ?? "当前设置分组";

  return (
    <section className="page-stack settings-page settings-workbench-page">
      <div className="settings-workbench-shell settings-console-shell" style={overlayVars}>
        <header className="settings-command-bar">
          <div className="settings-command-title">
            <div>
              <span>Eval Bench Preferences</span>
              <h2>工作台设置</h2>
            </div>
            <p>以最小控制面板管理视觉偏好，把主空间留给样本检查。</p>
          </div>
          <div className="settings-command-center">
            <div className="settings-search-box">
              <Search size={15} />
              <input
                value={settingsQuery}
                placeholder="搜索设置"
                onChange={(event) => setSettingsQuery(event.target.value)}
              />
              {settingsQuery ? (
                <IconActionButton
                  className="settings-search-clear"
                  icon={<X size={13} />}
                  title="清空搜索"
                  onClick={() => setSettingsQuery("")}
                />
              ) : null}
            </div>
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
              <span style={{ "--swatch": overlayColors.gt } as React.CSSProperties}>GT</span>
              <span style={{ "--swatch": overlayColors.pred } as React.CSSProperties}>Pred</span>
              <span style={{ "--swatch": overlayColors.fn } as React.CSSProperties}>FN</span>
              <span style={{ "--swatch": overlayColors.fp } as React.CSSProperties}>FP</span>
            </div>
          </div>
        </main>

        <section className="settings-preference-drawer">
          <div className="settings-drawer-head">
            <div>
              <span>Settings</span>
              <strong>{visibleSectionLabel}</strong>
              <small>{visibleSectionMeta}</small>
            </div>
            <p>配置键名、控件和实时预览保持同步；搜索时只展示匹配分组。</p>
          </div>
          <div className="settings-drawer-scroll">
            {showPanel("appearance") ? (
              <SettingsEditorSection title="可视化外观" description="控制框、线、点和标签的几何表达。">
                <SettingsPreferenceRow
                  title="几何样式"
                  settingKey="evalBench.overlay.style"
                  description="控制框、线、点和标签的绘制密度。"
                >
                  <div className="settings-number-grid">
                    {OVERLAY_STYLE_CONTROLS.map((control) => (
                      <NumberSettingControl
                        key={control.key}
                        label={control.label}
                        value={overlayStyle[control.key]}
                        min={control.min}
                        max={control.max}
                        step={control.step}
                        onChange={(value) => updateOverlayStyle(control.key, value)}
                      />
                    ))}
                    <CompactSelectControl
                      dense
                      label="预测线型"
                      value={overlayStyle.predLineStyle}
                      onChange={(value) => updateOverlayStyle("predLineStyle", value)}
                      options={PRED_LINE_STYLE_OPTIONS}
                    />
                  </div>
                  <ActionButton
                    variant="secondary"
                    className="settings-inline-action"
                    icon={<AppIcon name="resetSettings" size={16} />}
                    onClick={resetOverlayStyle}
                  >
                    重置样式
                  </ActionButton>
                </SettingsPreferenceRow>
              </SettingsEditorSection>
            ) : null}

            {showPanel("labels") ? (
              <SettingsEditorSection title="标签颜色" description="用于覆盖特定 label 的颜色；匹配大小写不敏感，但显示保留原始 label。">
                <SettingsPreferenceRow
                  title="新增规则"
                  settingKey="evalBench.overlay.labelColors"
                  description="输入 label 后按 Enter 或点击添加。"
                >
                  <LabelColorQuickAdd onChange={updateLabelColor} />
                </SettingsPreferenceRow>
                <SettingsPreferenceRow
                  title="当前 label"
                  settingKey="evalBench.overlay.labelColors.*"
                  description="来自当前预览样本和已保存的 label 规则。"
                >
                  <div className="settings-label-table">
                    {sortedLabels.length === 0 ? (
                      <div className="muted-line">还没有可配置的 label。</div>
                    ) : (
                      sortedLabels.map((label) => (
                        <div className="settings-label-row" key={label}>
                          <span>{label}</span>
                          <div className="settings-label-role-grid">
                            {INSTANCE_COLOR_ROLES.map((role) => (
                              <label key={role.key}>
                                <small>{role.label}</small>
                                <input
                                  aria-label={`${label} ${role.label} 颜色`}
                                  type="color"
                                  value={explicitLabelColor(labelColors, label, role.key) ?? overlayColors[role.key]}
                                  onChange={(event) =>
                                    updateLabelColor(label, role.key, event.target.value)
                                  }
                                />
                              </label>
                            ))}
                          </div>
                          <ActionButton
                            variant="mini"
                            compact
                            className="settings-label-clear-action"
                            onClick={() => removeLabelColor(label)}
                          >
                            清除
                          </ActionButton>
                        </div>
                      ))
                    )}
                  </div>
                  <ActionButton
                    variant="secondary"
                    className="settings-inline-action"
                    icon={<AppIcon name="clearRules" size={16} />}
                    onClick={resetLabelColors}
                  >
                    清空 label 颜色
                  </ActionButton>
                </SettingsPreferenceRow>
              </SettingsEditorSection>
            ) : null}

            {showPanel("interaction") ? (
              <SettingsEditorSection title="画布交互" description="让缩放和平移适配不同鼠标、触控板和大图场景。">
                <SettingsPreferenceRow
                  title="鼠标操作"
                  settingKey="evalBench.viewer.interaction"
                  description="缩放灵敏度越低，滚轮越稳；平移灵敏度越低，拖拽越慢。"
                >
                  <div className="settings-number-grid">
                    {INTERACTION_SETTING_CONTROLS.map((control) => (
                      <NumberSettingControl
                        key={control.key}
                        label={control.label}
                        value={settingControlValue(interactionSettings[control.key], control)}
                        min={settingControlValue(control.min, control)}
                        max={settingControlValue(control.max, control)}
                        step={settingControlValue(control.step, control)}
                        onChange={(value) =>
                          updateInteractionSetting(control.key, settingValueFromControl(value, control))
                        }
                      />
                    ))}
                  </div>
                  <ActionButton
                    variant="secondary"
                    className="settings-inline-action"
                    icon={<AppIcon name="resetSettings" size={16} />}
                    onClick={resetInteractionSettings}
                  >
                    重置交互
                  </ActionButton>
                </SettingsPreferenceRow>
              </SettingsEditorSection>
            ) : null}

            {showPanel("workflow") ? (
              <SettingsEditorSection title="快捷键" description="按 action 管理键位，适配后续新增图层和工具。">
                <ShortcutSettingsPanel
                  bindings={shortcutSettings.bindings}
                  onChange={shortcutSettings.updateShortcut}
                  onReset={shortcutSettings.resetShortcut}
                  onResetAll={shortcutSettings.resetShortcuts}
                />
              </SettingsEditorSection>
            ) : null}

            {visiblePanels.length === 0 ? <EmptyState title="没有匹配的设置项" /> : null}
          </div>
        </section>
      </div>
    </section>
  );
}
