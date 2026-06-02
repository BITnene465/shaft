import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FileText, MessageSquarePlus, Save, Sparkles } from "lucide-react";

import type { RunSummary } from "./api";
import { appendRunNote, isApiError, updateRunNote } from "./api";
import {
  FormSelectControl,
  StandaloneTextareaControl
} from "./controlPrimitives";
import {
  errorMessage,
  formatDate,
  inferenceValue,
  requestPixelBudgetValue,
  samplingValue,
  stringValue
} from "./formatters";
import { ActionButton, ConfigItem, DisclosurePanel } from "./ui";

const RUN_NOTE_TEMPLATES = [
  {
    id: "reproduce",
    label: "复现",
    body: "## reproduce\n- checkpoint:\n- command:\n- data split:\n- seed:\n"
  },
  {
    id: "idea",
    label: "Idea",
    body: "## idea\n- origin:\n- hypothesis:\n- expected signal:\n"
  },
  {
    id: "diagnosis",
    label: "异常",
    body: "## diagnosis\n- symptom:\n- suspected cause:\n- evidence:\n- next check:\n"
  },
  {
    id: "next",
    label: "Next",
    body: "## next\n- action:\n- owner:\n- blocking:\n"
  }
];

const RUN_NOTE_APPEND_HEADINGS = [
  { value: "follow-up", label: "follow-up" },
  { value: "reproduce", label: "reproduce" },
  { value: "idea", label: "idea" },
  { value: "diagnosis", label: "diagnosis" },
  { value: "next", label: "next" }
];

export function RunConfigPanel({
  run,
  defaultOpen = false
}: {
  run: RunSummary;
  defaultOpen?: boolean;
}) {
  const queryClient = useQueryClient();
  const [configOpen, setConfigOpen] = useState(defaultOpen);
  const [noteDraft, setNoteDraft] = useState(run.note || "");
  const [savedNote, setSavedNote] = useState(run.note || "");
  const [noteVersion, setNoteVersion] = useState(run.note_updated_at);
  const [appendHeading, setAppendHeading] = useState(RUN_NOTE_APPEND_HEADINGS[0].value);
  const [appendDraft, setAppendDraft] = useState("");
  const noteMutation = useMutation({
    mutationFn: (note: string) => updateRunNote(run.run_id, note, noteVersion),
    onSuccess: (note) => {
      setNoteDraft(note.note);
      setSavedNote(note.note);
      setNoteVersion(note.updated_at);
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    },
    onError: (error) => {
      if (isApiError(error) && error.status === 409) {
        void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      }
    }
  });
  const appendMutation = useMutation({
    mutationFn: ({ note, heading }: { note: string; heading: string }) =>
      appendRunNote(run.run_id, note, heading, noteVersion),
    onSuccess: (note) => {
      setAppendDraft("");
      setNoteDraft(note.note);
      setSavedNote(note.note);
      setNoteVersion(note.updated_at);
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    },
    onError: (error) => {
      if (isApiError(error) && error.status === 409) {
        void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      }
    }
  });
  const promptSource = stringValue(run.prompt_metadata.source) || (run.prompt_path ? "file" : "inline");
  const systemPrompt = stringValue(run.prompt_metadata.system_prompt);
  const userPrompt = stringValue(run.prompt_metadata.user_prompt);
  const noteDirty = noteDraft !== savedNote;
  const noteMaxLength = run.note_max_length;

  useEffect(() => {
    const nextNote = run.note || "";
    setNoteDraft(nextNote);
    setSavedNote(nextNote);
    setNoteVersion(run.note_updated_at);
    setAppendDraft("");
  }, [run.run_id, run.note, run.note_updated_at]);

  useEffect(() => {
    setConfigOpen(defaultOpen);
  }, [defaultOpen, run.run_id]);

  function insertNoteTemplate(template: (typeof RUN_NOTE_TEMPLATES)[number]) {
    setNoteDraft((current) => {
      const normalized = current.trimEnd();
      const separator = normalized ? "\n\n" : "";
      return `${normalized}${separator}${template.body}`.slice(0, noteMaxLength);
    });
  }

  return (
    <DisclosurePanel
      id="run-note"
      className="run-config-panel"
      open={configOpen}
      onToggle={(event) => setConfigOpen(event.currentTarget.open)}
      summary={
        <>
          <span>记录配置</span>
          <strong>
            {run.model_id} / {run.prompt_id || "-"} / {inferenceValue(run.inference, "backend")}
          </strong>
        </>
      }
    >
      <div className="run-note-editor">
        <div className="run-note-editor-head">
          <FileText size={16} />
          <div>
            <strong>Run note</strong>
            <span>
              {run.note_updated_at
                ? `更新于 ${formatDate(run.note_updated_at)}`
                : "记录复现线索、idea 来源和排障细节"}
            </span>
          </div>
        </div>
        <StandaloneTextareaControl
          label="Run note"
          value={noteDraft}
          onChange={setNoteDraft}
          placeholder="记录 checkpoint、prompt 改动、复现实验入口、异常判断和下一步 idea。"
          maxLength={noteMaxLength}
        />
        <div className="run-note-template-bar" aria-label="Run note 模板">
          <span>
            <Sparkles size={13} />
            模板
          </span>
          {RUN_NOTE_TEMPLATES.map((template) => (
            <ActionButton
              key={template.id}
              compact
              variant="mini"
              onClick={() => insertNoteTemplate(template)}
              disabled={noteDraft.length + template.body.length > noteMaxLength}
            >
              {template.label}
            </ActionButton>
          ))}
        </div>
        <div className="run-note-append-panel">
          <div>
            <FormSelectControl
              className="run-note-heading-select"
              label="追加标题"
              value={appendHeading}
              options={RUN_NOTE_APPEND_HEADINGS}
              onChange={setAppendHeading}
            />
            <StandaloneTextareaControl
              label="追加 run note"
              value={appendDraft}
              onChange={setAppendDraft}
              placeholder="追加新的复现线索、观察或下一步检查，不覆盖已有 note。"
              maxLength={noteMaxLength}
            />
          </div>
          <ActionButton
            compact
            variant="secondary"
            icon={<MessageSquarePlus size={14} />}
            disabled={!appendDraft.trim() || appendMutation.isPending || noteMutation.isPending}
            onClick={() => appendMutation.mutate({ note: appendDraft, heading: appendHeading })}
          >
            追加线索
          </ActionButton>
        </div>
        <div className="run-note-actions">
          <span>
            {noteDraft.length.toLocaleString()} / {noteMaxLength.toLocaleString()}
          </span>
          {noteMutation.error ? <strong>{errorMessage(noteMutation.error)}</strong> : null}
          {appendMutation.error ? <strong>{errorMessage(appendMutation.error)}</strong> : null}
          {noteMutation.data ? <em>已保存</em> : null}
          {appendMutation.data ? <em>已追加</em> : null}
          <ActionButton
            compact
            variant="primary"
            icon={<Save size={14} />}
            disabled={!noteDirty || noteMutation.isPending}
            onClick={() => noteMutation.mutate(noteDraft)}
          >
            保存备注
          </ActionButton>
        </div>
      </div>
      <div className="run-config-grid">
        <ConfigBlock title="模型">
          <ConfigItem label="ID" value={run.model_id} />
          <ConfigItem label="路径" value={run.model_path || "-"} />
        </ConfigBlock>
        <ConfigBlock title="Prompt">
          <ConfigItem label="ID" value={run.prompt_id || "-"} />
          <ConfigItem label="来源" value={promptSource} />
          <ConfigItem label="路径" value={run.prompt_path || "-"} />
          <ConfigItem label="Hash" value={run.prompt_hash ? run.prompt_hash.slice(0, 12) : "-"} />
        </ConfigBlock>
        <ConfigBlock title="服务">
          <ConfigItem label="后端" value={inferenceValue(run.inference, "backend")} />
          <ConfigItem label="服务 ID" value={inferenceValue(run.inference, "service_id")} />
          <ConfigItem label="端点" value={inferenceValue(run.inference, "endpoint")} />
          <ConfigItem label="服务模型" value={inferenceValue(run.inference, "served_model_name")} />
          <ConfigItem label="CUDA" value={inferenceValue(run.inference, "cuda_visible_devices")} />
          <ConfigItem label="TP" value={inferenceValue(run.inference, "tensor_parallel_size")} />
          <ConfigItem label="端口" value={inferenceValue(run.inference, "port")} />
        </ConfigBlock>
        <ConfigBlock title="生成">
          <ConfigItem label="最大输出" value={inferenceValue(run.inference, "max_tokens")} />
          <ConfigItem label="上下文" value={inferenceValue(run.inference, "max_model_len")} />
          <ConfigItem label="并发序列" value={inferenceValue(run.inference, "max_num_seqs")} />
          <ConfigItem label="显存占比" value={inferenceValue(run.inference, "gpu_memory_utilization")} />
          <ConfigItem label="批大小" value={inferenceValue(run.inference, "batch_size")} />
          <ConfigItem label="请求前像素预算" value={requestPixelBudgetValue(run.inference)} />
          <ConfigItem label="采样" value={samplingValue(run.inference)} />
        </ConfigBlock>
        <ConfigBlock title="评测">
          <ConfigItem label="解析器" value={run.parser || "-"} />
          <ConfigItem label="指标" value={run.metric_profile || "-"} />
          <ConfigItem label="可视化" value={run.visualization_profile || "-"} />
        </ConfigBlock>
      </div>
      {systemPrompt || userPrompt ? (
        <DisclosurePanel className="prompt-details" summary="Prompt 快照">
          {systemPrompt ? (
            <pre>
              <strong>system</strong>
              {"\n"}
              {systemPrompt}
            </pre>
          ) : null}
          {userPrompt ? (
            <pre>
              <strong>user</strong>
              {"\n"}
              {userPrompt}
            </pre>
          ) : null}
        </DisclosurePanel>
      ) : null}
    </DisclosurePanel>
  );
}

export function shouldOpenRunNotePanel() {
  if (typeof window === "undefined") {
    return false;
  }
  return window.location.hash === "#run-note";
}

function ConfigBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="config-block">
      <div className="config-title">{title}</div>
      <div className="config-items">{children}</div>
    </div>
  );
}
