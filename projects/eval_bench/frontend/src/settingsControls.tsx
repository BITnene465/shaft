import { useState } from "react";
import type { ReactNode } from "react";

import {
  INSTANCE_COLOR_ROLES,
  SHORTCUT_ACTIONS,
  shortcutEventBinding
} from "./workspaceSettings";
import type {
  InstanceColorRole,
  ShortcutActionId,
  ShortcutBindings
} from "./workspaceSettings";
import { ActionButton } from "./ui";

export function SettingsEditorSection({
  title,
  description,
  children
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="settings-editor-section">
      <div className="settings-section-title">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      {children}
    </section>
  );
}

export function SettingsPreferenceRow({
  title,
  settingKey,
  description,
  children
}: {
  title: string;
  settingKey?: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="settings-preference-row">
      <div className="settings-preference-copy">
        <strong>{title}</strong>
        {settingKey ? <code>{settingKey}</code> : null}
        <span>{description}</span>
      </div>
      <div className="settings-preference-control">{children}</div>
    </div>
  );
}

export function LabelColorQuickAdd({
  onChange
}: {
  onChange: (label: string, role: InstanceColorRole, value: string) => void;
}) {
  const [draftLabel, setDraftLabel] = useState("");
  const [draftRole, setDraftRole] = useState<InstanceColorRole>("gt");
  const [draftColor, setDraftColor] = useState("#2563eb");

  function addLabelColor() {
    const label = draftLabel.trim();
    if (!label) {
      return;
    }
    onChange(label, draftRole, draftColor);
    setDraftLabel("");
  }

  return (
    <div className="label-color-add-row settings-label-add-row">
      <input
        value={draftLabel}
        placeholder="label，例如 arrow"
        onChange={(event) => setDraftLabel(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            addLabelColor();
          }
        }}
      />
      <input
        aria-label="新增 label 颜色"
        type="color"
        value={draftColor}
        onChange={(event) => setDraftColor(event.target.value)}
      />
      <select
        aria-label="新增 label 颜色角色"
        value={draftRole}
        onChange={(event) => setDraftRole(event.target.value as InstanceColorRole)}
      >
        {INSTANCE_COLOR_ROLES.map((role) => (
          <option key={role.key} value={role.key}>
            {role.label}
          </option>
        ))}
      </select>
      <ActionButton variant="secondary" className="dense" onClick={addLabelColor}>
        添加
      </ActionButton>
    </div>
  );
}

export function ShortcutSettingsPanel({
  bindings,
  onChange,
  onReset,
  onResetAll
}: {
  bindings: ShortcutBindings;
  onChange: (actionId: ShortcutActionId, binding: string) => void;
  onReset: (actionId: ShortcutActionId) => void;
  onResetAll: () => void;
}) {
  const bindingCounts = SHORTCUT_ACTIONS.reduce<Record<string, number>>((counts, action) => {
    const binding = bindings[action.id];
    if (binding) {
      counts[binding] = (counts[binding] ?? 0) + 1;
    }
    return counts;
  }, {});

  return (
    <SettingsPreferenceRow
      title="键位映射"
      settingKey="evalBench.shortcuts"
      description="每个命令 action 维护一个键位，冲突项会在表格中标记。"
    >
      <div className="shortcut-map-table">
        {SHORTCUT_ACTIONS.map((action) => {
          const binding = bindings[action.id];
          const conflict = Boolean(binding && bindingCounts[binding] > 1);
          return (
            <div className={conflict ? "shortcut-map-row conflict" : "shortcut-map-row"} key={action.id}>
              <div>
                <span>{action.group}</span>
                <strong>{action.label}</strong>
                <code>{action.id}</code>
              </div>
              <button
                className="shortcut-capture"
                type="button"
                onKeyDown={(event) => {
                  event.preventDefault();
                  const binding = shortcutEventBinding(event.nativeEvent);
                  if (binding) {
                    onChange(action.id, binding);
                  }
                }}
              >
                {binding || "未设置"}
              </button>
              <ActionButton variant="mini" onClick={() => onReset(action.id)}>
                重置
              </ActionButton>
            </div>
          );
        })}
      </div>
      <ActionButton variant="secondary" className="settings-inline-action" onClick={onResetAll}>
        重置全部快捷键
      </ActionButton>
    </SettingsPreferenceRow>
  );
}
