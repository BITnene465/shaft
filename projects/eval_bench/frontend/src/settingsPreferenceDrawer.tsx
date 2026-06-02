import {
  CompactSelectControl,
  InlineColorControl,
  NumberSettingControl,
  TextInputControl
} from "./controlPrimitives";
import { AppIcon } from "./iconLibrary";
import {
  LabelColorQuickAdd,
  SettingsEditorSection,
  SettingsPreferenceRow,
  ShortcutSettingsPanel
} from "./settingsControls";
import type { TypographySettings } from "./typographySettings";
import { TYPOGRAPHY_PRESETS } from "./typographySettings";
import { ActionButton, EmptyState, SelectableCardButton } from "./ui";
import {
  INTERACTION_SETTING_CONTROLS,
  INSTANCE_COLOR_ROLES,
  OVERLAY_STYLE_CONTROLS,
  PRED_LINE_STYLE_OPTIONS,
  explicitLabelColor,
  settingControlValue,
  settingValueFromControl
} from "./workspaceSettings";
import type {
  InstanceColorRole,
  InteractionSettingKey,
  InteractionSettings,
  LabelColors,
  OverlayColors,
  OverlayStyle,
  OverlayStyleKey,
  ShortcutActionId,
  ShortcutBindings
} from "./workspaceSettings";

export type SettingsPanelId = "appearance" | "typography" | "labels" | "interaction" | "workflow";

export type SettingsSectionSummary = {
  id: SettingsPanelId;
  label: string;
  meta: string;
};

export function SettingsPreferenceDrawer({
  visibleSectionLabel,
  visibleSectionMeta,
  visiblePanels,
  overlayColors,
  overlayStyle,
  typographySettings,
  labelColors,
  sortedLabels,
  interactionSettings,
  shortcutBindings,
  onOverlayStyleChange,
  onResetOverlayStyle,
  onTypographySettingsChange,
  onResetTypographySettings,
  onLabelColorChange,
  onRemoveLabelColor,
  onResetLabelColors,
  onInteractionSettingChange,
  onResetInteractionSettings,
  onShortcutChange,
  onShortcutReset,
  onResetShortcuts
}: {
  visibleSectionLabel: string;
  visibleSectionMeta: string;
  visiblePanels: SettingsPanelId[];
  overlayColors: OverlayColors;
  overlayStyle: OverlayStyle;
  typographySettings: TypographySettings;
  labelColors: LabelColors;
  sortedLabels: string[];
  interactionSettings: InteractionSettings;
  shortcutBindings: ShortcutBindings;
  onOverlayStyleChange: (key: OverlayStyleKey, value: number | string) => void;
  onResetOverlayStyle: () => void;
  onTypographySettingsChange: (patch: Partial<TypographySettings>) => void;
  onResetTypographySettings: () => void;
  onLabelColorChange: (label: string, role: InstanceColorRole, value: string) => void;
  onRemoveLabelColor: (label: string, role?: InstanceColorRole) => void;
  onResetLabelColors: () => void;
  onInteractionSettingChange: (key: InteractionSettingKey, value: number) => void;
  onResetInteractionSettings: () => void;
  onShortcutChange: (action: ShortcutActionId, binding: string) => void;
  onShortcutReset: (action: ShortcutActionId) => void;
  onResetShortcuts: () => void;
}) {
  const showPanel = (id: SettingsPanelId) => visiblePanels.includes(id);

  return (
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
                    onChange={(value) => onOverlayStyleChange(control.key, value)}
                  />
                ))}
                <CompactSelectControl
                  dense
                  label="预测线型"
                  value={overlayStyle.predLineStyle}
                  onChange={(value) => onOverlayStyleChange("predLineStyle", value)}
                  options={PRED_LINE_STYLE_OPTIONS}
                />
              </div>
              <ActionButton
                variant="secondary"
                className="settings-inline-action"
                icon={<AppIcon name="resetSettings" size={16} />}
                onClick={onResetOverlayStyle}
              >
                重置样式
              </ActionButton>
            </SettingsPreferenceRow>
          </SettingsEditorSection>
        ) : null}

        {showPanel("typography") ? (
          <SettingsEditorSection title="字体与字号" description="控制工作台整体信息密度，并支持加载外部字体 CSS。">
            <SettingsPreferenceRow
              title="界面字体"
              settingKey="evalBench.typography"
              description="字体族直接写入 CSS font-family；字体 CSS URL 可使用 https 或相对路径。"
            >
              <div className="typography-preset-grid">
                {TYPOGRAPHY_PRESETS.map((preset) => (
                  <SelectableCardButton
                    key={preset.id}
                    className="typography-preset-card"
                    active={isTypographyPresetActive(typographySettings, preset.settings)}
                    onClick={() => onTypographySettingsChange(preset.settings)}
                  >
                    <span>{preset.label}</span>
                    <small>{preset.description}</small>
                  </SelectableCardButton>
                ))}
              </div>
              <div className="settings-typography-grid">
                <TextInputControl
                  className="settings-text-control"
                  label="界面字体族"
                  value={typographySettings.fontFamily}
                  placeholder='"IBM Plex Sans", "Noto Sans SC", sans-serif'
                  onChange={(value) => onTypographySettingsChange({ fontFamily: value })}
                />
                <TextInputControl
                  className="settings-text-control"
                  label="等宽字体族"
                  value={typographySettings.monoFontFamily}
                  placeholder='"JetBrains Mono", Consolas, monospace'
                  onChange={(value) => onTypographySettingsChange({ monoFontFamily: value })}
                />
                <TextInputControl
                  className="settings-text-control"
                  label="字体 CSS URL"
                  type="url"
                  value={typographySettings.fontCssUrl}
                  placeholder="https://.../font.css 或 /fonts/custom.css"
                  onChange={(value) => onTypographySettingsChange({ fontCssUrl: value })}
                />
                <TextInputControl
                  className="settings-text-control"
                  label="自定义字体名称"
                  value={typographySettings.customFontName}
                  placeholder="EvalBenchText"
                  onChange={(value) => onTypographySettingsChange({ customFontName: value })}
                />
                <TextInputControl
                  className="settings-text-control"
                  label="字体文件 URL"
                  type="url"
                  value={typographySettings.customFontFileUrl}
                  placeholder="/fonts/EvalBenchText.woff2"
                  onChange={(value) => onTypographySettingsChange({ customFontFileUrl: value })}
                />
                <NumberSettingControl
                  label="基础字号"
                  value={typographySettings.baseFontSize}
                  min={10}
                  max={20}
                  step={0.5}
                  onChange={(value) => onTypographySettingsChange({ baseFontSize: value })}
                />
              </div>
              <div className="typography-preview-strip">
                <strong>Run banana_v2_5_step8000_px4m</strong>
                <span>组合报告 / layout + arrow / P@50 0.934</span>
                <code>evalBench.typography.fontCssUrl</code>
              </div>
              <ActionButton
                variant="secondary"
                className="settings-inline-action"
                icon={<AppIcon name="resetSettings" size={16} />}
                onClick={onResetTypographySettings}
              >
                重置字体
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
              <LabelColorQuickAdd onChange={onLabelColorChange} />
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
                          <InlineColorControl
                            key={role.key}
                            label={`${label} ${role.label} 颜色`}
                            caption={role.label}
                            value={explicitLabelColor(labelColors, label, role.key) ?? overlayColors[role.key]}
                            onChange={(value) => onLabelColorChange(label, role.key, value)}
                          />
                        ))}
                      </div>
                      <ActionButton
                        variant="mini"
                        compact
                        className="settings-label-clear-action"
                        onClick={() => onRemoveLabelColor(label)}
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
                onClick={onResetLabelColors}
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
                      onInteractionSettingChange(control.key, settingValueFromControl(value, control))
                    }
                  />
                ))}
              </div>
              <ActionButton
                variant="secondary"
                className="settings-inline-action"
                icon={<AppIcon name="resetSettings" size={16} />}
                onClick={onResetInteractionSettings}
              >
                重置交互
              </ActionButton>
            </SettingsPreferenceRow>
          </SettingsEditorSection>
        ) : null}

        {showPanel("workflow") ? (
          <SettingsEditorSection title="快捷键" description="按 action 管理键位，适配后续新增图层和工具。">
            <ShortcutSettingsPanel
              bindings={shortcutBindings}
              onChange={onShortcutChange}
              onReset={onShortcutReset}
              onResetAll={onResetShortcuts}
            />
          </SettingsEditorSection>
        ) : null}

        {visiblePanels.length === 0 ? <EmptyState title="没有匹配的设置项" /> : null}
      </div>
    </section>
  );
}

function isTypographyPresetActive(
  current: TypographySettings,
  preset: TypographySettings
) {
  return (
    current.fontFamily === preset.fontFamily &&
    current.monoFontFamily === preset.monoFontFamily &&
    current.fontCssUrl === preset.fontCssUrl &&
    current.customFontName === preset.customFontName &&
    current.customFontFileUrl === preset.customFontFileUrl &&
    current.baseFontSize === preset.baseFontSize
  );
}
