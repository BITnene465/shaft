import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import type { BenchmarkSummary, PromptTemplate } from "./api";
import {
  createJob,
  fetchJobTemplates,
  fetchPromptTemplates,
  preflightJob,
  upsertPromptTemplate
} from "./api";
import { CompactSelectControl, TextareaControl } from "./controlPrimitives";
import { errorMessage, unique } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { DetectionLabelSubtaskPanel } from "./labelSubtaskControls";
import {
  applyBenchmarkDefault,
  applyPromptTemplateToManifest,
  formatManifest,
  manifestBenchmarkId,
  manifestBenchmarkSplit,
  manifestEvalTask,
  manifestHasTargetLabelScope,
  manifestTargetLabels,
  normalizeManifestTargetLabelsForTask,
  promptTemplateFromManifest,
  targetLabelsFromPrompt,
  updateManifestBenchmarkSplit,
  updateManifestTargetLabels
} from "./manifestTools";
import { ActionButton, Badge, DisclosurePanel, PanelTitle } from "./ui";
import { ResizableSplit } from "./workspaceLayout";
import "./formControls.css";

export function JobCreatePanel({ benchmarks, bare }: { benchmarks: BenchmarkSummary[]; bare?: boolean }) {
  const queryClient = useQueryClient();
  const templatesQuery = useQuery({
    queryKey: ["job-templates"],
    queryFn: ({ signal }) => fetchJobTemplates({ signal })
  });
  const promptTemplatesQuery = useQuery({
    queryKey: ["prompt-templates"],
    queryFn: ({ signal }) => fetchPromptTemplates({ signal })
  });
  const templates = templatesQuery.data?.templates ?? {};
  const promptTemplates = promptTemplatesQuery.data?.templates ?? [];
  const templateIds = Object.keys(templates);
  const promptIds = promptTemplates.map((template) => template.prompt_id);
  const [templateId, setTemplateId] = useState("eval_job");
  const [promptId, setPromptId] = useState("grounding_arrow.v2.4.main");
  const [manifestText, setManifestText] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: createJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  const promptMutation = useMutation({
    mutationFn: upsertPromptTemplate,
    onSuccess: (record) => {
      setPromptId(record.prompt_id);
      void queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
    }
  });
  const preflightMutation = useMutation({ mutationFn: preflightJob });
  const selectedTemplate = templates[templateId] ?? templates[templateIds[0] ?? ""];
  const selectedPrompt =
    promptTemplates.find((template) => template.prompt_id === promptId) ?? promptTemplates[0];
  const manifestDraft = useMemo(() => parseManifestDraft(manifestText), [manifestText]);
  const manifestTaskValue = manifestEvalTask(manifestDraft);
  const manifestBenchmarkValue = manifestBenchmarkId(manifestDraft);
  const manifestBenchmarkSplitValue = manifestBenchmarkSplit(manifestDraft) || "auto";
  const selectedBenchmark = benchmarks.find((benchmark) => benchmark.benchmark_id === manifestBenchmarkValue);
  const benchmarkSplitOptions = jobBenchmarkSplitOptions(selectedBenchmark, manifestBenchmarkSplitValue);
  const selectedTargetLabels = manifestTargetLabels(manifestDraft);
  const labelOptions = unique([
    ...(selectedBenchmark?.labels ?? []),
    ...targetLabelsFromPrompt(selectedPrompt),
    ...selectedTargetLabels
  ]);

  useEffect(() => {
    if (!manifestText && selectedTemplate?.manifest) {
      setManifestText(formatManifest(applyBenchmarkDefault(selectedTemplate.manifest, benchmarks)));
    }
  }, [benchmarks, manifestText, selectedTemplate]);

  useEffect(() => {
    if (promptIds.length > 0 && !promptIds.includes(promptId)) {
      setPromptId(promptIds[0]);
    }
  }, [promptId, promptIds.join("|")]);

  useEffect(() => {
    if (
      !manifestDraft ||
      manifestTaskValue === "detection" ||
      !manifestHasTargetLabelScope(manifestDraft)
    ) {
      return;
    }
    setManifestText(formatManifest(normalizeManifestTargetLabelsForTask(manifestDraft)));
    setParseError(null);
    resetPreflightResult();
  }, [manifestDraft, manifestTaskValue]);

  function resetPreflightResult() {
    preflightMutation.reset();
  }

  function loadTemplate(nextTemplateId = templateId) {
    const template = templates[nextTemplateId];
    if (!template) {
      return;
    }
    setTemplateId(nextTemplateId);
    setManifestText(formatManifest(applyBenchmarkDefault(template.manifest, benchmarks)));
    setParseError(null);
    resetPreflightResult();
  }

  function applySelectedPrompt(nextPromptId = promptId) {
    const promptTemplate =
      promptTemplates.find((template) => template.prompt_id === nextPromptId) ?? selectedPrompt;
    if (!promptTemplate) {
      return;
    }
    const manifest = parseManifest() ?? applyBenchmarkDefault(selectedTemplate?.manifest ?? {}, benchmarks);
    setPromptId(promptTemplate.prompt_id);
    setManifestText(
      formatManifest(applyBenchmarkDefault(applyPromptTemplateToManifest(manifest, promptTemplate), benchmarks))
    );
    setParseError(null);
    resetPreflightResult();
  }

  function savePromptFromManifest() {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    const draft = promptTemplateFromManifest(manifest, selectedPrompt);
    promptMutation.mutate(draft);
  }

  function parseManifest(): Record<string, unknown> | null {
    try {
      const parsed = JSON.parse(manifestText) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setParseError("Manifest 必须是 JSON object。");
        return null;
      }
      setParseError(null);
      return parsed as Record<string, unknown>;
    } catch (error) {
      setParseError(errorMessage(error));
      return null;
    }
  }

  function validateManifest() {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    preflightMutation.mutate({ manifest });
  }

  function updateTargetLabels(nextLabels: string[]) {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    setManifestText(formatManifest(updateManifestTargetLabels(manifest, nextLabels)));
    setParseError(null);
    resetPreflightResult();
  }

  function updateBenchmarkSplit(nextSplit: string) {
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    setManifestText(formatManifest(updateManifestBenchmarkSplit(manifest, nextSplit)));
    setParseError(null);
    resetPreflightResult();
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    mutation.mutate({ manifest });
  }

  return (
    <div className={bare ? "manifest-card bare" : "workspace-card manifest-card"}>
      {bare ? null : <PanelTitle title="新建评测任务" meta="模板 manifest + 后端预检查" />}
      <form className="manifest-job-form" onSubmit={submit}>
        <div className="manifest-toolbar">
          <CompactSelectControl
            label="模板"
            value={templateId}
            onChange={loadTemplate}
            disabled={templatesQuery.isLoading}
            options={
              templateIds.length === 0
                ? [{ value: "eval_job", label: "加载中" }]
                : templateIds.map((id) => ({ value: id, label: templates[id]?.label ?? id }))
            }
          />
          <CompactSelectControl
            label="Prompt"
            value={selectedPrompt?.prompt_id ?? promptId}
            onChange={applySelectedPrompt}
            disabled={promptTemplatesQuery.isLoading || promptTemplates.length === 0}
            options={
              promptTemplates.length === 0
                ? [{ value: promptId, label: "加载中" }]
                : promptTemplates.map((template) => ({
                    value: template.prompt_id,
                    label: template.label || template.prompt_id
                  }))
            }
          />
          <CompactSelectControl
            label="Benchmark split"
            value={manifestBenchmarkSplitValue}
            onChange={updateBenchmarkSplit}
            disabled={!selectedBenchmark}
            options={benchmarkSplitOptions}
          />
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="restoreTemplate" size={16} />}
            onClick={() => loadTemplate()}
          >
            恢复模板
          </ActionButton>
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="applyPrompt" size={16} />}
            onClick={() => applySelectedPrompt()}
            disabled={!selectedPrompt}
          >
            应用 Prompt
          </ActionButton>
          <ActionButton
            variant="secondary"
            icon={<AppIcon name="preflightValidate" size={16} />}
            onClick={validateManifest}
            disabled={preflightMutation.isPending}
          >
            {preflightMutation.isPending ? "检查中" : "预检查"}
          </ActionButton>
          <ActionButton
            variant="primary"
            type="submit"
            icon={<AppIcon name="enqueueJob" size={16} />}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "加入中" : "加入队列"}
          </ActionButton>
        </div>
        <DetectionLabelSubtaskPanel
          task={manifestTaskValue}
          benchmarkId={manifestBenchmarkValue}
          labelOptions={labelOptions}
          selectedLabels={selectedTargetLabels}
          onChange={updateTargetLabels}
        />
        <ResizableSplit
          className="manifest-split"
          storageKey="eval_bench_manifest_result_width"
          fixedPane="second"
          defaultSize={360}
          minSize={240}
          maxSize={820}
          first={
            <div className="manifest-editor-pane">
              {selectedTemplate ? (
                <p className="manifest-template-note">{selectedTemplate.description}</p>
              ) : null}
              <TextareaControl
                className="manifest-editor-field"
                label="可编辑任务 Manifest"
                spellCheck={false}
                value={manifestText}
                onChange={(value) => {
                  setManifestText(value);
                  setParseError(null);
                  resetPreflightResult();
                }}
              />
            </div>
          }
          second={
            <div className="manifest-result-pane">
              <PanelTitle title="预检查" meta="提交前的参数与运行时校验" />
              {selectedPrompt ? (
                <PromptTemplatePanel
                  prompt={selectedPrompt}
                  onSaveFromManifest={savePromptFromManifest}
                  saving={promptMutation.isPending}
                  saveErrorMessage={errorMessage(promptMutation.error)}
                />
              ) : null}
              {parseError ? <div className="form-error">JSON 解析错误：{parseError}</div> : null}
              {preflightMutation.data ? <PreflightPanel result={preflightMutation.data} /> : null}
              {preflightMutation.error ? (
                <div className="form-error">预检查请求失败：{errorMessage(preflightMutation.error)}</div>
              ) : null}
              {mutation.error ? (
                <div className="form-error">任务入队失败：{errorMessage(mutation.error)}</div>
              ) : null}
              {!parseError && !preflightMutation.data && !preflightMutation.isError && !mutation.isError ? (
                <div className="manifest-placeholder">
                  编辑 manifest 后执行预检查；通过后再加入队列。
                </div>
              ) : null}
            </div>
          }
        />
      </form>
    </div>
  );
}

function parseManifestDraft(value: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function PreflightPanel({
  result
}: {
  result: { ok: boolean; errors: string[]; warnings: string[]; runtime_command?: string[] | null };
}) {
  return (
    <div className={result.ok ? "preflight-panel ok" : "preflight-panel failed"}>
      <div className="preflight-heading">
        <strong>{result.ok ? "预检查通过" : "预检查失败"}</strong>
        <span>{result.errors.length} 个错误 / {result.warnings.length} 个警告</span>
      </div>
      {result.errors.length > 0 ? (
        <ul>
          {result.errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      ) : null}
      {result.warnings.length > 0 ? (
        <ul>
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
      {result.runtime_command && result.runtime_command.length > 0 ? (
        <pre>{result.runtime_command.join(" ")}</pre>
      ) : null}
    </div>
  );
}

function jobBenchmarkSplitOptions(
  benchmark: BenchmarkSummary | undefined,
  currentSplit: string
) {
  const values = new Set<string>();
  if (benchmark?.split) {
    values.add(benchmark.split);
  }
  Object.keys(benchmark?.split_manifests ?? {}).forEach((value) => {
    if (value.trim()) {
      values.add(value);
    }
  });
  if (currentSplit && currentSplit !== "auto") {
    values.add(currentSplit);
  }
  return [
    { value: "auto", label: "自动推断" },
    ...Array.from(values)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ value, label: value }))
  ];
}

function PromptTemplatePanel({
  prompt,
  onSaveFromManifest,
  saving,
  saveErrorMessage
}: {
  prompt: PromptTemplate;
  onSaveFromManifest: () => void;
  saving: boolean;
  saveErrorMessage: string;
}) {
  const targetLabels = targetLabelsFromPrompt(prompt);
  return (
    <DisclosurePanel
      className="prompt-template-panel"
      open
      summary={
        <>
          <span>{prompt.label || prompt.prompt_id}</span>
          <Badge value={prompt.task} />
        </>
      }
    >
      <div className="prompt-template-meta">
        <span>{prompt.prompt_id}</span>
        <span>{prompt.parser ?? "parser 未设置"}</span>
        <span>{prompt.metric_profile ?? "metric 未设置"}</span>
        <span>目标 {targetLabels.length ? targetLabels.join(" / ") : "全部 label"}</span>
      </div>
      <div className="prompt-template-text">
        <strong>System</strong>
        <p>{prompt.system_prompt || "-"}</p>
        <strong>User</strong>
        <p>{prompt.user_prompt || "-"}</p>
      </div>
      <ActionButton
        variant="secondary"
        compact
        icon={<AppIcon name="applyPrompt" size={16} />}
        onClick={onSaveFromManifest}
        disabled={saving}
      >
        {saving ? "保存中" : "将当前 Manifest 的 Prompt 保存为模板"}
      </ActionButton>
      {saveErrorMessage ? (
        <div className="form-error">Prompt 模板保存失败：{saveErrorMessage}</div>
      ) : null}
    </DisclosurePanel>
  );
}
