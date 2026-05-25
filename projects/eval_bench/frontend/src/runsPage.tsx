import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { FileText, Save, Sparkles } from "lucide-react";

import type { BenchmarkSummary, RunSampleSummary, RunSummary } from "./api";
import {
  fetchRunSampleDetail,
  fetchRunSamples,
  fetchRuns,
  importPredictions,
  isApiError,
  updateRunNote
} from "./api";
import { useDashboardState } from "./dashboardState";
import { FormSelectControl } from "./controlPrimitives";
import { AdvancedFilterBar } from "./filterControls";
import {
  basename,
  formatDate,
  inferenceValue,
  isTextInputTarget,
  pixelBudgetValue,
  samplingValue,
  stringValue,
  unique
} from "./formatters";
import { AppIcon } from "./iconLibrary";
import { DetectionLabelSubtaskPanel } from "./labelSubtaskControls";
import { RunTable } from "./runTables";
import {
  SAMPLE_PAGE_SIZE,
  clampSamplePageOffset,
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { PagerControl, SamplePager, clampListPageOffset } from "./samplePager";
import { SampleViewer } from "./sampleViewer";
import {
  ActionButton,
  CommandButton,
  ConfigItem,
  EmptyState,
  SelectableRowButton,
  WorkspaceDialog
} from "./ui";
import { preloadSampleImages } from "./viewerGeometry";
import { ResizableSplit } from "./workspaceLayout";
import { useWorkspaceShortcuts } from "./workspaceSettings";

const RUN_PAGE_SIZE = 80;
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

export function RunsPage() {
  const dashboardQuery = useDashboardState();
  const [importOpen, setImportOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [promptFilter, setPromptFilter] = useState("all");
  const [metricProfileFilter, setMetricProfileFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const runFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: RUN_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      task: taskFilter !== "all" ? taskFilter : undefined,
      benchmarkId: benchmarkFilter !== "all" ? benchmarkFilter : undefined,
      label: labelFilter !== "all" ? labelFilter : undefined,
      modelId: modelFilter !== "all" ? modelFilter : undefined,
      promptId: promptFilter !== "all" ? promptFilter : undefined,
      metricProfile: metricProfileFilter !== "all" ? metricProfileFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [
      benchmarkFilter,
      labelFilter,
      metricProfileFilter,
      modelFilter,
      pageOffset,
      promptFilter,
      searchText,
      statusFilter,
      taskFilter
    ]
  );
  const runsQuery = useQuery({
    queryKey: ["runs", runFilters],
    queryFn: () => fetchRuns(runFilters)
  });
  const runFacetsQuery = useQuery({
    queryKey: ["runs", "facets"],
    queryFn: () => fetchRuns({ limit: 500 })
  });
  const runs = runsQuery.data?.runs ?? [];
  const runFacets = runFacetsQuery.data?.runs ?? runs;
  const tasks = unique(runFacets.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runFacets.map((run) => run.benchmark_id).filter(Boolean));
  const statuses = unique(runFacets.map((run) => run.status).filter(Boolean));
  const labels = unique(runFacets.flatMap((run) => run.target_labels).filter(Boolean));
  const models = unique(runFacets.map((run) => run.model_id).filter(Boolean));
  const prompts = unique(runFacets.map((run) => run.prompt_id).filter(Boolean));
  const metricProfiles = unique(runFacets.map((run) => run.metric_profile).filter(Boolean));
  const totalRuns = runsQuery.data?.total ?? runs.length;
  useEffect(() => {
    setPageOffset(0);
  }, [
    searchText,
    statusFilter,
    taskFilter,
    benchmarkFilter,
    labelFilter,
    modelFilter,
    promptFilter,
    metricProfileFilter
  ]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, totalRuns, RUN_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [pageOffset, totalRuns]);
  if (runsQuery.isLoading || dashboardQuery.isLoading) {
    return <EmptyState title="正在加载评测记录" />;
  }
  if (runsQuery.error || !runsQuery.data) {
    return <EmptyState title="评测记录加载失败" tone="danger" />;
  }
  const benchmarkOptions = dashboardQuery.data?.benchmarks ?? [];
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>评测记录库</h2>
          <span>{totalRuns.toLocaleString()} 条 run snapshot</span>
        </div>
        <CommandButton
          variant="secondary"
          icon={<AppIcon name="importPrediction" size={17} />}
          onClick={() => setImportOpen(true)}
        >
          导入预测
        </CommandButton>
      </div>
      <div className="workspace-card fill">
        <RunTable
          runs={runs}
          filterMeta={`${runs.length.toLocaleString()} / ${totalRuns.toLocaleString()} 条 run`}
          filterControls={[
            {
              type: "search",
              id: "run-query",
              label: "全文检索",
              value: searchText,
              onChange: setSearchText,
              placeholder: "搜索 run、模型、基准集、备注"
            },
            {
              type: "select",
              id: "run-status",
              label: "状态",
              value: statusFilter,
              values: ["all", ...statuses],
              labels: { all: "全部" },
              onChange: setStatusFilter
            },
            {
              type: "select",
              id: "run-task",
              label: "任务",
              value: taskFilter,
              values: ["all", ...tasks],
              labels: { all: "全部" },
              onChange: setTaskFilter
            },
            {
              type: "select",
              id: "run-benchmark",
              label: "基准集",
              value: benchmarkFilter,
              values: ["all", ...benchmarks],
              labels: { all: "全部" },
              onChange: setBenchmarkFilter
            },
            {
              type: "select",
              id: "run-label",
              label: "标签",
              value: labelFilter,
              values: ["all", ...labels],
              labels: { all: "全部" },
              onChange: setLabelFilter
            },
            {
              type: "select",
              id: "run-model",
              label: "模型",
              value: modelFilter,
              values: ["all", ...models],
              labels: { all: "全部" },
              onChange: setModelFilter
            },
            {
              type: "select",
              id: "run-prompt",
              label: "Prompt",
              value: promptFilter,
              values: ["all", ...prompts],
              labels: { all: "全部" },
              onChange: setPromptFilter
            },
            {
              type: "select",
              id: "run-metric",
              label: "Metric",
              value: metricProfileFilter,
              values: ["all", ...metricProfiles],
              labels: { all: "全部" },
              onChange: setMetricProfileFilter
            }
          ]}
          footer={
            <PagerControl
              className="rank-board-pager run-list-pager"
              offset={runsQuery.data.offset ?? pageOffset}
              limit={runsQuery.data.limit ?? RUN_PAGE_SIZE}
              total={totalRuns}
              onPageChange={setPageOffset}
            />
          }
        />
      </div>
      <WorkspaceDialog
        open={importOpen}
        title="导入预测快照"
        meta="把外部预测目录导入为 run，并和 GT 对比"
        onClose={() => setImportOpen(false)}
      >
        <ImportPredictionsPanel benchmarks={benchmarkOptions} bare />
      </WorkspaceDialog>
    </section>
  );
}

function ImportPredictionsPanel({ benchmarks, bare }: { benchmarks: BenchmarkSummary[]; bare?: boolean }) {
  const queryClient = useQueryClient();
  const [runId, setRunId] = useState("");
  const [benchmarkId, setBenchmarkId] = useState(benchmarks[0]?.benchmark_id ?? "");
  const [predictionRoot, setPredictionRoot] = useState("");
  const [task, setTask] = useState("detection");
  const [modelId, setModelId] = useState("");
  const [modelPath, setModelPath] = useState("imported");
  const [promptId, setPromptId] = useState("imported");
  const [targetLabels, setTargetLabels] = useState<string[]>([]);
  const [specId, setSpecId] = useState("");
  const [strict, setStrict] = useState(false);
  const [overwrite, setOverwrite] = useState(false);
  const [evaluate, setEvaluate] = useState(true);
  const mutation = useMutation({
    mutationFn: importPredictions,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
      void queryClient.invalidateQueries({ queryKey: ["rank-board"] });
      void queryClient.invalidateQueries({ queryKey: ["comparisons"] });
    }
  });
  const effectiveBenchmarkId = benchmarkId || benchmarks[0]?.benchmark_id || "";
  const selectedBenchmark = benchmarks.find((benchmark) => benchmark.benchmark_id === effectiveBenchmarkId);
  const labelOptions = selectedBenchmark?.labels ?? [];
  useEffect(() => {
    if (task !== "detection") {
      setTargetLabels([]);
      return;
    }
    if (labelOptions.length > 0) {
      setTargetLabels((current) => current.filter((label) => labelOptions.includes(label)));
    }
  }, [task, effectiveBenchmarkId, labelOptions.join("\u0000")]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate({
      run_id: runId.trim(),
      benchmark_id: effectiveBenchmarkId,
      prediction_root: predictionRoot.trim(),
      task,
      model_id: modelId.trim(),
      model_path: modelPath.trim() || "imported",
      prompt_id: promptId.trim() || "imported",
      spec_id: specId.trim() || undefined,
      target_labels: targetLabels,
      strict,
      overwrite,
      evaluate
    });
  }

  const benchmarkSelectOptions =
    benchmarks.length === 0
      ? [{ value: "", label: "暂无基准集" }]
      : benchmarks.map((benchmark) => ({
          value: benchmark.benchmark_id,
          label: benchmark.benchmark_id
        }));
  const content = (
    <form className="job-form import-form" onSubmit={submit}>
      <label>
        <span>记录 ID</span>
        <input
          value={runId}
          onChange={(event) => setRunId(event.target.value)}
          placeholder="model-a_val_import"
          required
        />
      </label>
      <FormSelectControl
        label="基准集"
        value={effectiveBenchmarkId}
        options={benchmarkSelectOptions}
        required
        onChange={setBenchmarkId}
      />
      <label className="wide-field">
        <span>预测目录</span>
        <input
          value={predictionRoot}
          onChange={(event) => setPredictionRoot(event.target.value)}
          placeholder="/path/to/prediction_json_dir"
          required
        />
      </label>
      <FormSelectControl
        label="任务"
        value={task}
        options={[
          { value: "detection", label: "检测" },
          { value: "keypoint", label: "关键点" }
        ]}
        onChange={setTask}
      />
      <label>
        <span>模型 ID</span>
        <input
          value={modelId}
          onChange={(event) => setModelId(event.target.value)}
          placeholder="qwen3vl-best"
          required
        />
      </label>
      <label className="wide-field">
        <span>模型路径</span>
        <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
      </label>
      <label>
        <span>Prompt</span>
        <input value={promptId} onChange={(event) => setPromptId(event.target.value)} />
      </label>
      <DetectionLabelSubtaskPanel
        className="full-field"
        task={task}
        benchmarkId={effectiveBenchmarkId}
        labelOptions={labelOptions}
        selectedLabels={targetLabels}
        onChange={setTargetLabels}
      />
      <label>
        <span>规格</span>
        <input
          value={specId}
          onChange={(event) => setSpecId(event.target.value)}
          placeholder="optional"
        />
      </label>
      <label className="checkbox-field">
        <input
          type="checkbox"
          checked={strict}
          onChange={(event) => setStrict(event.target.checked)}
        />
        <span>严格导入</span>
      </label>
      <label className="checkbox-field">
        <input
          type="checkbox"
          checked={overwrite}
          onChange={(event) => setOverwrite(event.target.checked)}
        />
        <span>覆盖已有 run</span>
      </label>
      <label className="checkbox-field">
        <input
          type="checkbox"
          checked={evaluate}
          onChange={(event) => setEvaluate(event.target.checked)}
        />
        <span>导入后计算指标</span>
      </label>
      <ActionButton
        type="submit"
        variant="primary"
        icon={<AppIcon name="importPrediction" size={16} />}
        disabled={mutation.isPending || benchmarks.length === 0}
      >
        导入
      </ActionButton>
      {mutation.data ? (
        <div className="form-result full-field">
          已导入 {mutation.data.imported_predictions.toLocaleString()} 条预测，缺失{" "}
          {mutation.data.missing_prediction_count.toLocaleString()} 条。{" "}
          <Link to="/runs/$runId" params={{ runId: mutation.data.run_id }}>
            打开 run
          </Link>
        </div>
      ) : null}
      {mutation.error ? (
        <div className="form-result error full-field">{mutation.error.message}</div>
      ) : null}
    </form>
  );
  return bare ? content : <div className="workspace-card compact-form-card">{content}</div>;
}

export function RunDetailPage() {
  const { runId } = useParams({ from: "/runs/$runId" });
  const queryClient = useQueryClient();
  const { data: dashboardState } = useDashboardState();
  const runSummary = dashboardState?.runs.find((run) => run.run_id === runId) ?? null;
  const [selectedIndex, setSelectedIndex] = useState(() => sampleIndexFromLocation());
  const [pageOffset, setPageOffset] = useState(() => samplePageOffsetFromLocation(SAMPLE_PAGE_SIZE));
  const [errorFilter, setErrorFilter] = useState("all");
  const [labelFilter, setLabelFilter] = useState("all");
  const samplesQuery = useQuery({
    queryKey: ["run-samples", runId, pageOffset, errorFilter, labelFilter],
    queryFn: () =>
      fetchRunSamples(runId, {
        offset: pageOffset,
        limit: SAMPLE_PAGE_SIZE,
        label: labelFilter,
        errorFilter
      })
  });
  const page = samplesQuery.data;
  const samples = page?.samples ?? [];
  const labels = page?.labels ?? [];
  const activeSample = samples.find((sample) => sample.index === selectedIndex) ?? samples[0] ?? null;
  const activeIndex = activeSample?.index ?? selectedIndex;
  const hasActiveSampleFilter = errorFilter !== "all" || labelFilter !== "all";
  const { actionForEvent } = useWorkspaceShortcuts();
  const detailQuery = useQuery({
    queryKey: ["run-sample-detail", runId, activeIndex],
    queryFn: () => fetchRunSampleDetail(runId, activeIndex),
    enabled: Boolean(activeSample),
    placeholderData: (previousData) => (previousData?.run_id === runId ? previousData : undefined),
    staleTime: 30_000
  });

  function selectSample(index: number) {
    setSelectedIndex(index);
    updateSampleIndexInLocation(index);
  }

  function changeErrorFilter(value: string) {
    setErrorFilter(value);
    setPageOffset(0);
  }

  function changeLabelFilter(value: string) {
    setLabelFilter(value);
    setPageOffset(0);
  }

  function moveSample(delta: number) {
    if (samples.length === 0) {
      return;
    }
    const position = samples.findIndex((sample) => sample.index === activeIndex);
    const next = samples[position + delta];
    if (next) {
      selectSample(next.index);
      return;
    }
    const nextOffset = pageOffset + delta * SAMPLE_PAGE_SIZE;
    if (nextOffset >= 0 && page && nextOffset < page.total) {
      setPageOffset(nextOffset);
    }
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      const actionId = actionForEvent(event);
      if (actionId === "sample.previous") {
        event.preventDefault();
        moveSample(-1);
      }
      if (actionId === "sample.next") {
        event.preventDefault();
        moveSample(1);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actionForEvent, activeIndex, page?.total, pageOffset, samples]);

  useEffect(() => {
    if (activeSample && activeSample.index !== selectedIndex) {
      selectSample(activeSample.index);
    }
  }, [activeSample, selectedIndex]);

  useEffect(() => {
    if (!page) {
      return;
    }
    const nextOffset = clampSamplePageOffset(pageOffset, page.total, SAMPLE_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [page?.total, pageOffset]);

  useEffect(() => {
    return preloadSampleImages(samples, activeIndex);
  }, [activeIndex, samples]);

  useEffect(() => {
    if (samples.length === 0) {
      return;
    }
    const position = Math.max(0, samples.findIndex((sample) => sample.index === activeIndex));
    const preload = samples.slice(Math.max(0, position - 1), position + 2);
    preload.forEach((sample) => {
      void queryClient.prefetchQuery({
        queryKey: ["run-sample-detail", runId, sample.index],
        queryFn: () => fetchRunSampleDetail(runId, sample.index),
        staleTime: 30_000
      });
    });
  }, [activeIndex, queryClient, runId, samples]);

  if (samplesQuery.isLoading) {
    return <EmptyState title="正在加载评测样本" />;
  }
  if (samplesQuery.error) {
    return <EmptyState title="评测样本加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack visual-inspector-page run-inspector-page">
      {runSummary ? <RunConfigPanel run={runSummary} /> : null}
      {page?.total === 0 && !hasActiveSampleFilter ? (
        <EmptyState title="这条评测记录没有基准集样本。" />
      ) : (
        <ResizableSplit
          className="inspector-grid"
          storageKey="eval_bench_run_sidebar_width"
          defaultSize={224}
          minSize={148}
          maxSize={520}
          first={
            <div className="inspector-sidebar">
              <SampleFilters
                errorFilter={errorFilter}
                labelFilter={labelFilter}
                labels={labels}
                onErrorFilterChange={changeErrorFilter}
                onLabelFilterChange={changeLabelFilter}
              />
              <SampleList
                samples={samples}
                selectedIndex={activeIndex}
                onSelect={selectSample}
                emptyText="没有符合过滤条件的样本。"
              />
              {page ? (
                <SamplePager
                  offset={page.offset}
                  limit={page.limit}
                  total={page.total}
                  onPageChange={setPageOffset}
                />
              ) : null}
            </div>
          }
          second={
            <div className="viewer-panel">
              {samples.length === 0 ? (
                <div className="empty-panel">没有符合过滤条件的样本。</div>
              ) : detailQuery.error ? (
                <div className="empty-panel">样本详情加载失败</div>
              ) : detailQuery.isLoading || !detailQuery.data ? (
                <div className="empty-panel">正在加载样本详情</div>
              ) : (
                <>
                  {detailQuery.isFetching ? <div className="viewer-fetch-chip">正在刷新样本详情</div> : null}
                  <SampleViewer detail={detailQuery.data} />
                </>
              )}
            </div>
          }
        />
      )}
    </section>
  );
}

function RunConfigPanel({ run }: { run: RunSummary }) {
  const queryClient = useQueryClient();
  const [noteDraft, setNoteDraft] = useState(run.note || "");
  const noteMutation = useMutation({
    mutationFn: (note: string) => updateRunNote(run.run_id, note, run.note_updated_at),
    onSuccess: () => {
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
  const noteDirty = noteDraft !== (run.note || "");
  const noteMaxLength = run.note_max_length;

  useEffect(() => {
    setNoteDraft(run.note || "");
  }, [run.run_id, run.note]);

  function insertNoteTemplate(template: (typeof RUN_NOTE_TEMPLATES)[number]) {
    setNoteDraft((current) => {
      const normalized = current.trimEnd();
      const separator = normalized ? "\n\n" : "";
      return `${normalized}${separator}${template.body}`.slice(0, noteMaxLength);
    });
  }

  return (
    <details className="run-config-panel">
      <summary>
        <span>记录配置</span>
        <strong>
          {run.model_id} / {run.prompt_id || "-"} / {inferenceValue(run.inference, "backend")}
        </strong>
      </summary>
      <div className="run-note-editor">
        <div className="run-note-editor-head">
          <FileText size={16} />
          <div>
            <strong>Run note</strong>
            <span>
              {run.note_updated_at ? `更新于 ${formatDate(run.note_updated_at)}` : "记录复现线索、idea 来源和排障细节"}
            </span>
          </div>
        </div>
        <textarea
          value={noteDraft}
          onChange={(event) => setNoteDraft(event.target.value)}
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
        <div className="run-note-actions">
          <span>
            {noteDraft.length.toLocaleString()} / {noteMaxLength.toLocaleString()}
          </span>
          {noteMutation.error ? <strong>{noteMutation.error.message}</strong> : null}
          {noteMutation.data ? <em>已保存</em> : null}
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
          <ConfigItem label="像素预算" value={pixelBudgetValue(run.inference)} />
          <ConfigItem label="采样" value={samplingValue(run.inference)} />
        </ConfigBlock>
        <ConfigBlock title="评测">
          <ConfigItem label="解析器" value={run.parser || "-"} />
          <ConfigItem label="指标" value={run.metric_profile || "-"} />
          <ConfigItem label="可视化" value={run.visualization_profile || "-"} />
        </ConfigBlock>
      </div>
      {systemPrompt || userPrompt ? (
        <details className="prompt-details">
          <summary>Prompt 快照</summary>
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
        </details>
      ) : null}
    </details>
  );
}

function ConfigBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="config-block">
      <div className="config-title">{title}</div>
      <div className="config-items">{children}</div>
    </div>
  );
}

function SampleFilters({
  errorFilter,
  labelFilter,
  labels,
  onErrorFilterChange,
  onLabelFilterChange
}: {
  errorFilter: string;
  labelFilter: string;
  labels: string[];
  onErrorFilterChange: (value: string) => void;
  onLabelFilterChange: (value: string) => void;
}) {
  return (
    <AdvancedFilterBar
      title="样本检索"
      meta={`${labels.length.toLocaleString()} labels`}
      controls={[
        {
          type: "select",
          id: "error",
          label: "状态",
          value: errorFilter,
          values: ["all", "fn", "fp", "missing", "clean"],
          labels: { all: "全部", fn: "漏检", fp: "误检", missing: "缺失预测", clean: "正常" },
          onChange: onErrorFilterChange
        },
        {
          type: "select",
          id: "label",
          label: "标签",
          value: labelFilter,
          values: ["all", ...labels],
          labels: { all: "全部" },
          onChange: onLabelFilterChange
        }
      ]}
    />
  );
}

function SampleList({
  samples,
  selectedIndex,
  onSelect,
  emptyText
}: {
  samples: RunSampleSummary[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  emptyText: string;
}) {
  if (samples.length === 0) {
    return <div className="sample-list empty">{emptyText}</div>;
  }
  return (
    <div className="sample-list">
      {samples.map((sample) => (
        <SelectableRowButton
          key={sample.index}
          selected={sample.index === selectedIndex}
          onClick={() => onSelect(sample.index)}
        >
          <span className="sample-row-main">
            <strong>{sample.index + 1}</strong>
            <span title={sample.image}>{basename(sample.image)}</span>
          </span>
          <span className="sample-row-meta">
            真实 {sample.gt_instance_count.toLocaleString()} / 预测{" "}
            {sample.pred_instance_count.toLocaleString()}
          </span>
          <span className={sample.has_prediction ? "sample-status ok" : "sample-status missing"}>
            {sample.has_prediction ? "已预测" : "缺预测"}
          </span>
        </SelectableRowButton>
      ))}
    </div>
  );
}
