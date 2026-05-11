import React from "react";
import ReactDOM from "react-dom/client";
import { useEffect, useMemo, useState } from "react";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient
} from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";
import {
  Link,
  Outlet,
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
  useLocation,
  useParams
} from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  Archive,
  BarChart3,
  Database,
  Eye,
  FileSearch,
  Gauge,
  GitCompare,
  Layers,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RotateCw,
  Search,
  Server,
  SlidersHorizontal,
  Trash2,
  X
} from "lucide-react";

import {
  BenchmarkSampleDetail,
  BenchmarkSampleSummary,
  BenchmarkSummary,
  ComparisonLabelDelta,
  ComparisonReport,
  ComparisonSample,
  ComparisonSampleDetail,
  ComparisonSummary,
  DashboardState,
  EvalInstance,
  JobLog,
  JobSummary,
  PromptTemplate,
  RunSampleDetail,
  RunSampleSummary,
  RunSummary,
  SchedulerStatus,
  ServiceLog,
  ServiceSummary,
  archiveRun,
  cancelJob,
  checkServiceHealth,
  createBenchmark,
  createJob,
  createService,
  deleteJob,
  deleteRun,
  deleteService,
  evaluateRun,
  fetchBenchmarkSampleDetail,
  fetchBenchmarkSamples,
  fetchComparison,
  fetchComparisonSample,
  fetchComparisons,
  fetchJobLogs,
  fetchJobs,
  fetchJobTemplates,
  fetchPromptTemplates,
  fetchRunSampleDetail,
  fetchRunSamples,
  fetchSchedulerStatus,
  fetchServiceLogs,
  fetchServices,
  fetchSettingsPreviewSample,
  fetchState,
  importPredictions,
  preflightJob,
  startService,
  stopService,
  upsertPromptTemplate
} from "./api";
import {
  buildObjectRows,
  countInstancesByLabel,
  formatBbox,
  objectMetricText,
  objectStatusLabel,
  visibleLabelMetrics,
  visibleSampleMetrics
} from "./viewerMetrics";
import type {
  LabelMetricRow,
  ObjectKind,
  ObjectRow,
  VisibleMetrics
} from "./viewerMetrics";
import {
  DEFAULT_INTERACTION_SETTINGS,
  DEFAULT_OVERLAY_STYLE,
  fallbackLabelColor,
  loadSplitSize,
  useSidebarPreference,
  useWorkspaceSettings
} from "./workspaceSettings";
import {
  ActionPanel,
  Badge,
  DataTable,
  EmptyState,
  PanelTitle,
  SectionHeader,
  WorkspaceTabs
} from "./ui";
import type {
  InteractionSettingKey,
  InteractionSettings,
  LabelColors,
  OverlayColorKey,
  OverlayColors,
  OverlayStyle,
  OverlayStyleKey
} from "./workspaceSettings";
import "./styles.css";
import "./design.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 10_000,
      staleTime: 5_000,
      retry: 1
    }
  }
});
const SAMPLE_PAGE_SIZE = 80;
const CANVAS_FIT_PADDING = 18;
const PRELOAD_RADIUS = 4;
const SETTINGS_PREVIEW_IMAGE_URL = "/static/settings_preview.svg";
const SETTINGS_PREVIEW_LABELS = ["arrow", "icon"];

class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: unknown) {
    console.error("Eval Bench dashboard failed to render", error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fatal-panel">
          <div className="fatal-panel-inner">
            <strong>看板渲染失败</strong>
            <span>{this.state.error}</span>
            <button type="button" onClick={() => window.location.reload()}>
              重新加载
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function useDashboardState() {
  return useQuery({ queryKey: ["dashboard-state"], queryFn: fetchState });
}

function Shell() {
  const stateQuery = useDashboardState();
  const state = stateQuery.data;
  const location = useLocation();
  const pageTitle = getShellTitle(location.pathname);
  const { sidebarCollapsed, setSidebarCollapsed } = useSidebarPreference();

  return (
    <div className={sidebarCollapsed ? "app-shell sidebar-collapsed" : "app-shell"}>
      <aside className={sidebarCollapsed ? "sidebar collapsed" : "sidebar"}>
        <div className="brand">
          <ShaftMark />
          <div className="brand-copy">
            <div className="brand-title">Shaft Eval Bench</div>
            <div className="brand-subtitle">视觉结构评测中心</div>
          </div>
          <button
            className="sidebar-toggle"
            type="button"
            title={sidebarCollapsed ? "展开导航栏" : "收起导航栏"}
            aria-label={sidebarCollapsed ? "展开导航栏" : "收起导航栏"}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>
        <nav className="nav-list">
          <NavItem to="/" icon={<Gauge size={17} />} label="总览" />
          <NavItem to="/benchmarks" icon={<Database size={17} />} label="基准集" />
          <NavItem to="/services" icon={<Server size={17} />} label="模型服务" />
          <NavItem to="/jobs" icon={<Play size={17} />} label="评测中心" />
          <NavItem to="/runs" icon={<BarChart3 size={17} />} label="结果库" />
          <NavItem to="/compare" icon={<GitCompare size={17} />} label="对比分析" />
          <NavItem to="/settings" icon={<SlidersHorizontal size={17} />} label="工作台设置" />
        </nav>
        <div className="store-chip">
          <span>数据目录</span>
          <strong title={state?.store_root ?? "加载中"}>{state?.store_root ?? "加载中"}</strong>
        </div>
      </aside>
      <main className="content">
        <header className="topbar">
          <div>
            <div className="eyebrow">{pageTitle.kicker}</div>
            <h1>{pageTitle.title}</h1>
          </div>
          <div className="topbar-actions">
            <StatusPill loading={stateQuery.isFetching} error={stateQuery.isError} />
          </div>
        </header>
        <Outlet />
      </main>
      <ToastHub />
    </div>
  );
}

type ToastMessage = {
  id: string;
  message: string;
  tone: "danger" | "info";
};

function ToastHub() {
  const [items, setItems] = useState<ToastMessage[]>([]);
  useEffect(() => {
    function handleError(event: Event) {
      const detail = (event as CustomEvent<{ message?: string }>).detail;
      const message = detail?.message || "请求失败。";
      const id = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
      setItems((current) => [...current.slice(-3), { id, message, tone: "danger" }]);
      window.setTimeout(() => {
        setItems((current) => current.filter((item) => item.id !== id));
      }, 8000);
    }
    window.addEventListener("eval-bench-api-error", handleError);
    return () => window.removeEventListener("eval-bench-api-error", handleError);
  }, []);
  if (items.length === 0) {
    return null;
  }
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {items.map((item) => (
        <div className={`toast-message ${item.tone}`} key={item.id}>
          <strong>操作失败</strong>
          <span>{item.message}</span>
          <button
            className="icon-button dense"
            type="button"
            title="关闭提醒"
            onClick={() => setItems((current) => current.filter((entry) => entry.id !== item.id))}
          >
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}

function getShellTitle(pathname: string) {
  if (pathname.startsWith("/benchmarks")) {
    return { kicker: "真值样本库", title: "基准集" };
  }
  if (pathname.startsWith("/services")) {
    return { kicker: "推理运行时", title: "模型服务" };
  }
  if (pathname.startsWith("/jobs")) {
    return { kicker: "评测任务与结果", title: "评测中心" };
  }
  if (pathname.startsWith("/runs")) {
    return { kicker: "可复查的评测结果", title: "结果库" };
  }
  if (pathname.startsWith("/compare")) {
    return { kicker: "双模型对比", title: "对比分析" };
  }
  if (pathname.startsWith("/settings")) {
    return { kicker: "个人显示偏好", title: "工作台设置" };
  }
  return { kicker: "评测运营台", title: "总览" };
}

function ShaftMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 48 48" aria-hidden="true">
      <defs>
        <linearGradient id="shaftMarkGradient" x1="8" x2="40" y1="6" y2="42">
          <stop offset="0" stopColor="#5ed3f3" />
          <stop offset="0.52" stopColor="#8d7cf6" />
          <stop offset="1" stopColor="#f7c948" />
        </linearGradient>
      </defs>
      <path
        d="M10 15.5 24 7l14 8.5v17L24 41l-14-8.5v-17Z"
        fill="#101820"
        stroke="url(#shaftMarkGradient)"
        strokeWidth="3"
      />
      <path d="M17 18h15l-9 12h9" fill="none" stroke="#f7f9fb" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function NavItem({
  to,
  icon,
  label
}: {
  to: string;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <Link to={to} className="nav-item" activeProps={{ className: "nav-item active" }} title={label}>
      {icon}
      <span>{label}</span>
    </Link>
  );
}

function StatusPill({ loading, error }: { loading: boolean; error: boolean }) {
  if (error) {
    return <div className="status-pill danger">接口异常</div>;
  }
  return <div className="status-pill">{loading ? "同步中" : "在线"}</div>;
}

function OverviewPage() {
  const { data, isLoading, error } = useDashboardState();
  if (isLoading) {
    return <EmptyState title="正在加载看板状态" />;
  }
  if (error || !data) {
    return <EmptyState title="看板状态加载失败" tone="danger" />;
  }
  return (
    <section className="page-stack dashboard-home">
      <SummaryGrid state={data} />
      <div className="home-grid">
        <div className="workspace-card span-2">
          <PanelTitle
            title="最近评测记录"
            meta={`共 ${data.runs.length.toLocaleString()} 条`}
          />
          <RunTable runs={data.runs.slice(0, 8)} compact />
        </div>
        <div className="workspace-card">
          <PanelTitle title="任务队列" meta="持久化 job" />
          <JobQueuePanel compact />
        </div>
        <div className="workspace-card">
          <PanelTitle title="基准集" meta="GT 样本库" />
          <BenchmarkTable benchmarks={data.benchmarks.slice(0, 8)} compact />
        </div>
      </div>
    </section>
  );
}

function SummaryGrid({ state }: { state: DashboardState }) {
  return (
    <div className="summary-grid">
      <MetricCard icon={<Database size={18} />} label="基准集" value={state.benchmark_count} />
      <MetricCard icon={<Layers size={18} />} label="样本数" value={state.total_benchmark_samples} />
      <MetricCard icon={<Activity size={18} />} label="评测记录" value={state.run_count} />
      <MetricCard icon={<FileSearch size={18} />} label="预测实例" value={state.prediction_count} />
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value.toLocaleString()}</div>
      </div>
    </div>
  );
}

function BenchmarksPage() {
  const { data, isLoading, error } = useDashboardState();
  if (isLoading) {
    return <EmptyState title="正在加载基准集" />;
  }
  if (error || !data) {
    return <EmptyState title="基准集加载失败" tone="danger" />;
  }
  return (
    <section className="page-stack">
      <WorkspaceTabs defaultValue="catalog" label="基准集工作区">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="catalog">样本目录</Tabs.Trigger>
          <Tabs.Trigger value="create">创建副本</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="catalog" className="workspace-tab-panel">
          <div className="workspace-card fill">
            <PanelTitle
              title="基准集目录"
              meta={`${data.benchmarks.length.toLocaleString()} 个副本`}
            />
            <BenchmarkTable benchmarks={data.benchmarks} />
          </div>
        </Tabs.Content>
        <Tabs.Content value="create" className="workspace-tab-panel">
          <BenchmarkCreatePanel />
        </Tabs.Content>
      </WorkspaceTabs>
    </section>
  );
}

function BenchmarkCreatePanel() {
  const queryClient = useQueryClient();
  const [benchmarkId, setBenchmarkId] = useState("");
  const [sourceRoot, setSourceRoot] = useState("data/raw_data");
  const [sourceManifest, setSourceManifest] = useState("data/raw_data/splits/layout_val.txt");
  const [split, setSplit] = useState("val");
  const [tasks, setTasks] = useState<string[]>(["detection", "keypoint"]);
  const [layers, setLayers] = useState("layout,arrow");
  const [overwrite, setOverwrite] = useState(false);
  const mutation = useMutation({
    mutationFn: createBenchmark,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });

  function toggleTask(task: string) {
    setTasks((current) => {
      if (current.includes(task)) {
        return current.filter((item) => item !== task);
      }
      return [...current, task];
    });
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate({
      benchmark_id: benchmarkId.trim(),
      source_root: sourceRoot.trim(),
      source_manifest: sourceManifest.trim(),
      split: split.trim() || "val",
      tasks,
      layers: layers
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      overwrite
    });
  }

  return (
    <ActionPanel title="创建 benchmark 副本" meta="从 raw_data split 复制不可变 test/val 集">
      <form className="job-form benchmark-form" onSubmit={submit}>
        <label>
        <span>基准集 ID</span>
          <input
            value={benchmarkId}
            onChange={(event) => setBenchmarkId(event.target.value)}
            placeholder="multitask_val_v1"
            required
          />
        </label>
        <label className="wide-field">
          <span>数据根目录</span>
          <input
            value={sourceRoot}
            onChange={(event) => setSourceRoot(event.target.value)}
            required
          />
        </label>
        <label className="wide-field">
          <span>Split 文件</span>
          <input
            value={sourceManifest}
            onChange={(event) => setSourceManifest(event.target.value)}
            required
          />
        </label>
        <label>
          <span>Split 名称</span>
          <input value={split} onChange={(event) => setSplit(event.target.value)} required />
        </label>
        <label>
          <span>标注层</span>
          <input value={layers} onChange={(event) => setLayers(event.target.value)} />
        </label>
        <label className="checkbox-field">
          <input
            type="checkbox"
            checked={tasks.includes("detection")}
            onChange={() => toggleTask("detection")}
          />
          <span>检测</span>
        </label>
        <label className="checkbox-field">
          <input
            type="checkbox"
            checked={tasks.includes("keypoint")}
            onChange={() => toggleTask("keypoint")}
          />
          <span>关键点</span>
        </label>
        <label className="checkbox-field">
          <input
            type="checkbox"
            checked={overwrite}
            onChange={(event) => setOverwrite(event.target.checked)}
          />
          <span>覆盖已有副本</span>
        </label>
        <button type="submit" disabled={mutation.isPending || tasks.length === 0}>
          创建
        </button>
        {mutation.data ? (
          <div className="form-result wide-field">
            已创建 {mutation.data.benchmark_id}，包含 {mutation.data.sample_count.toLocaleString()} 个样本。{" "}
            <Link
              to="/benchmarks/$benchmarkId"
              params={{ benchmarkId: mutation.data.benchmark_id }}
            >
              打开
            </Link>
          </div>
        ) : null}
        {mutation.error ? (
          <div className="form-result error wide-field">{mutation.error.message}</div>
        ) : null}
      </form>
    </ActionPanel>
  );
}

function BenchmarkDetailPage() {
  const { benchmarkId } = useParams({ from: "/benchmarks/$benchmarkId" });
  const queryClient = useQueryClient();
  const [selectedIndex, setSelectedIndex] = useState(() => sampleIndexFromLocation());
  const [pageOffset, setPageOffset] = useState(() => samplePageOffsetFromLocation(SAMPLE_PAGE_SIZE));
  const [labelFilter, setLabelFilter] = useState("all");
  const samplesQuery = useQuery({
    queryKey: ["benchmark-samples", benchmarkId, pageOffset, labelFilter],
    queryFn: () =>
      fetchBenchmarkSamples(benchmarkId, {
        offset: pageOffset,
        limit: SAMPLE_PAGE_SIZE,
        label: labelFilter
      })
  });
  const page = samplesQuery.data;
  const samples = page?.samples ?? [];
  const labels = page?.labels ?? [];
  const activeSample = samples.find((sample) => sample.index === selectedIndex) ?? samples[0] ?? null;
  const activeIndex = activeSample?.index ?? selectedIndex;
  const detailQuery = useQuery({
    queryKey: ["benchmark-sample-detail", benchmarkId, activeIndex],
    queryFn: () => fetchBenchmarkSampleDetail(benchmarkId, activeIndex),
    enabled: Boolean(activeSample),
    placeholderData: (previousData) =>
      previousData?.benchmark_id === benchmarkId ? previousData : undefined,
    staleTime: 30_000
  });

  function selectSample(index: number) {
    setSelectedIndex(index);
    updateSampleIndexInLocation(index);
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
      if (event.key === "[") {
        event.preventDefault();
        moveSample(-1);
      }
      if (event.key === "]") {
        event.preventDefault();
        moveSample(1);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeIndex, page?.total, pageOffset, samples]);

  useEffect(() => {
    if (activeSample && activeSample.index !== selectedIndex) {
      selectSample(activeSample.index);
    }
  }, [activeSample, selectedIndex]);

  useEffect(() => {
    preloadSampleImages(samples, activeIndex);
  }, [activeIndex, samples]);

  useEffect(() => {
    if (samples.length === 0) {
      return;
    }
    const position = Math.max(0, samples.findIndex((sample) => sample.index === activeIndex));
    const preload = samples.slice(Math.max(0, position - 3), position + 4);
    preload.forEach((sample) => {
      void queryClient.prefetchQuery({
        queryKey: ["benchmark-sample-detail", benchmarkId, sample.index],
        queryFn: () => fetchBenchmarkSampleDetail(benchmarkId, sample.index),
        staleTime: 30_000
      });
    });
  }, [activeIndex, benchmarkId, queryClient, samples]);

  if (samplesQuery.isLoading) {
    return <EmptyState title="正在加载样本" />;
  }
  if (samplesQuery.error) {
    return <EmptyState title="样本加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack visual-inspector-page">
      <SectionHeader
        title="基准集检查"
        subtitle={`${benchmarkId} 的真值样本浏览器。`}
      />
      {samples.length === 0 ? (
        <EmptyState title="这个基准集没有样本。" />
      ) : (
        <ResizableSplit
          className="inspector-grid"
          storageKey="eval_bench_benchmark_sidebar_width"
          defaultSize={224}
          minSize={148}
          maxSize={520}
          first={
            <div className="inspector-sidebar">
            <BenchmarkSampleFilters
              labelFilter={labelFilter}
              labels={labels}
              onLabelFilterChange={changeLabelFilter}
            />
            <BenchmarkSampleList
              samples={samples}
              selectedIndex={activeIndex}
              onSelect={selectSample}
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
                <BenchmarkSampleViewer detail={detailQuery.data} />
              </>
            )}
            </div>
          }
        />
      )}
    </section>
  );
}

function BenchmarkSampleFilters({
  labelFilter,
  labels,
  onLabelFilterChange
}: {
  labelFilter: string;
  labels: string[];
  onLabelFilterChange: (value: string) => void;
}) {
  return (
    <div className="sample-filters single">
      <FilterSelect
        label="标签"
        value={labelFilter}
        values={["all", ...labels]}
        labels={{ all: "全部" }}
        onChange={onLabelFilterChange}
      />
    </div>
  );
}

function BenchmarkSampleList({
  samples,
  selectedIndex,
  onSelect
}: {
  samples: BenchmarkSampleSummary[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="sample-list">
      {samples.map((sample) => (
        <button
          key={sample.index}
          className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}
          type="button"
          onClick={() => onSelect(sample.index)}
        >
          <span className="sample-row-main">
            <strong>{sample.index + 1}</strong>
            <span title={sample.image}>{basename(sample.image)}</span>
          </span>
          <span className="sample-row-meta">
            真值 {sample.instance_count.toLocaleString()} / 标签 {sample.labels.join(", ") || "-"}
          </span>
        </button>
      ))}
    </div>
  );
}

function BenchmarkSampleViewer({ detail }: { detail: BenchmarkSampleDetail }) {
  const width = detail.sample.image_width ?? 1000;
  const height = detail.sample.image_height ?? 1000;
  const labels = useMemo(() => unique(detail.gt_instances.map((instance) => instance.label)), [detail.gt_instances]);
  const {
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars
  } = useWorkspaceSettings(labels);

  return (
    <div className="viewer-stack" style={overlayVars}>
      <div className="viewer-toolbar">
        <div>
          <h2>{basename(detail.sample.image)}</h2>
          <p>{detail.sample.image}</p>
        </div>
        <div className="legend-row">
          <span className="legend-item gt">真值</span>
        </div>
      </div>
      <div className="diagnostic-strip">
        <span>实例 {detail.sample.instance_count.toLocaleString()}</span>
        <span>标签 {detail.sample.labels.join(", ") || "-"}</span>
      </div>
      <CanvasStage
        width={width}
        height={height}
        imageUrl={detail.sample.image_url}
        imageAlt={detail.sample.image}
        gtInstances={detail.gt_instances}
        predInstances={[]}
        diagnostics={null}
        visibleLabels={new Set(labels)}
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
      <div className="instance-summary">
        <InstanceStats title="真值实例" instances={detail.gt_instances} />
      </div>
    </div>
  );
}

function RunsPage() {
  const { data, isLoading, error } = useDashboardState();
  if (isLoading) {
    return <EmptyState title="正在加载评测记录" />;
  }
  if (error || !data) {
    return <EmptyState title="评测记录加载失败" tone="danger" />;
  }
  return (
    <section className="page-stack">
      <WorkspaceTabs defaultValue="runs" label="评测记录工作区">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="runs">记录库</Tabs.Trigger>
          <Tabs.Trigger value="import">导入预测</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="runs" className="workspace-tab-panel">
          <div className="workspace-card fill">
            <PanelTitle
              title="评测记录库"
              meta={`${data.runs.length.toLocaleString()} 条记录`}
            />
            <RunTable runs={data.runs} />
          </div>
        </Tabs.Content>
        <Tabs.Content value="import" className="workspace-tab-panel">
          <ImportPredictionsPanel benchmarks={data.benchmarks} />
        </Tabs.Content>
      </WorkspaceTabs>
    </section>
  );
}

function ImportPredictionsPanel({ benchmarks }: { benchmarks: BenchmarkSummary[] }) {
  const queryClient = useQueryClient();
  const [runId, setRunId] = useState("");
  const [benchmarkId, setBenchmarkId] = useState(benchmarks[0]?.benchmark_id ?? "");
  const [predictionRoot, setPredictionRoot] = useState("");
  const [task, setTask] = useState("detection");
  const [modelId, setModelId] = useState("");
  const [modelPath, setModelPath] = useState("imported");
  const [promptId, setPromptId] = useState("imported");
  const [specId, setSpecId] = useState("");
  const [strict, setStrict] = useState(false);
  const [overwrite, setOverwrite] = useState(false);
  const [evaluate, setEvaluate] = useState(true);
  const mutation = useMutation({
    mutationFn: importPredictions,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const effectiveBenchmarkId = benchmarkId || benchmarks[0]?.benchmark_id || "";

  function submit(event: React.FormEvent<HTMLFormElement>) {
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
      strict,
      overwrite,
      evaluate
    });
  }

  return (
    <ActionPanel title="导入预测快照" meta="把外部预测目录导入为 run，并和 GT 对比">
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
        <label>
          <span>基准集</span>
          <select
            value={effectiveBenchmarkId}
            onChange={(event) => setBenchmarkId(event.target.value)}
            required
          >
            {benchmarks.length === 0 ? <option value="">暂无基准集</option> : null}
            {benchmarks.map((benchmark) => (
              <option key={benchmark.benchmark_id} value={benchmark.benchmark_id}>
                {benchmark.benchmark_id}
              </option>
            ))}
          </select>
        </label>
        <label className="wide-field">
          <span>预测目录</span>
          <input
            value={predictionRoot}
            onChange={(event) => setPredictionRoot(event.target.value)}
            placeholder="/path/to/prediction_json_dir"
            required
          />
        </label>
        <label>
          <span>任务</span>
          <select value={task} onChange={(event) => setTask(event.target.value)}>
            <option value="detection">检测</option>
            <option value="keypoint">关键点</option>
          </select>
        </label>
        <label>
          <span>模型 ID</span>
          <input
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
            placeholder="qwen3vl-best"
            required
          />
        </label>
        <label>
          <span>模型路径</span>
          <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
        </label>
        <label>
          <span>Prompt</span>
          <input value={promptId} onChange={(event) => setPromptId(event.target.value)} />
        </label>
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
        <button type="submit" disabled={mutation.isPending || benchmarks.length === 0}>
          导入
        </button>
        {mutation.data ? (
          <div className="form-result wide-field">
            已导入 {mutation.data.imported_predictions.toLocaleString()} 条预测，缺失{" "}
            {mutation.data.missing_prediction_count.toLocaleString()} 条。{" "}
            <Link to="/runs/$runId" params={{ runId: mutation.data.run_id }}>
              打开 run
            </Link>
          </div>
        ) : null}
        {mutation.error ? (
          <div className="form-result error wide-field">{mutation.error.message}</div>
        ) : null}
      </form>
    </ActionPanel>
  );
}

function RunDetailPage() {
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
      if (event.key === "[") {
        event.preventDefault();
        moveSample(-1);
      }
      if (event.key === "]") {
        event.preventDefault();
        moveSample(1);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeIndex, page?.total, pageOffset, samples]);

  useEffect(() => {
    if (activeSample && activeSample.index !== selectedIndex) {
      selectSample(activeSample.index);
    }
  }, [activeSample, selectedIndex]);

  useEffect(() => {
    preloadSampleImages(samples, activeIndex);
  }, [activeIndex, samples]);

  useEffect(() => {
    if (samples.length === 0) {
      return;
    }
    const position = Math.max(0, samples.findIndex((sample) => sample.index === activeIndex));
    const preload = samples.slice(Math.max(0, position - 3), position + 4);
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
      {samples.length === 0 ? (
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
  const promptSource = stringValue(run.prompt_metadata.source) || (run.prompt_path ? "file" : "inline");
  const systemPrompt = stringValue(run.prompt_metadata.system_prompt);
  const userPrompt = stringValue(run.prompt_metadata.user_prompt);
  return (
    <details className="run-config-panel">
      <summary>
        <span>记录配置</span>
        <strong>
          {run.model_id} / {run.prompt_id || "-"} / {inferenceValue(run.inference, "backend")}
        </strong>
      </summary>
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

function ConfigBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="config-block">
      <div className="config-title">{title}</div>
      <div className="config-items">{children}</div>
    </div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="config-item">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
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
    <div className="sample-filters">
      <FilterSelect
        label="状态"
        value={errorFilter}
        values={["all", "fn", "fp", "missing", "clean"]}
        labels={{ all: "全部", fn: "漏检", fp: "误检", missing: "缺失预测", clean: "正常" }}
        onChange={onErrorFilterChange}
      />
      <FilterSelect
        label="标签"
        value={labelFilter}
        values={["all", ...labels]}
        labels={{ all: "全部" }}
        onChange={onLabelFilterChange}
      />
    </div>
  );
}

function FilterSelect({
  label,
  value,
  values,
  labels,
  onChange,
  compact = false
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  onChange: (value: string) => void;
  compact?: boolean;
}) {
  return (
    <label className={compact ? "filter-select compact" : "filter-select"}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} title={label}>
        {values.map((item) => (
          <option key={item} value={item}>
            {labels?.[item] ?? item}
          </option>
        ))}
      </select>
    </label>
  );
}

function ChipGroup({
  label,
  value,
  values,
  labels,
  onChange
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  onChange: (value: string) => void;
}) {
  return (
    <div className="chip-group">
      <span>{label}</span>
      <div className="chip-row">
        {values.map((item) => (
          <button
            key={item}
            className={item === value ? "query-chip active" : "query-chip"}
            type="button"
            onClick={() => onChange(item)}
            title={item}
          >
            {labels?.[item] ?? item}
          </button>
        ))}
      </div>
    </div>
  );
}

function SampleList({
  samples,
  selectedIndex,
  onSelect
}: {
  samples: RunSampleSummary[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="sample-list">
      {samples.map((sample) => (
        <button
          key={sample.index}
          className={sample.index === selectedIndex ? "sample-row selected" : "sample-row"}
          type="button"
          onClick={() => onSelect(sample.index)}
        >
          <span className="sample-row-main">
            <strong>{sample.index + 1}</strong>
            <span title={sample.image}>{basename(sample.image)}</span>
          </span>
          <span className="sample-row-meta">
            {sample.diagnostics ? (
              <>
                TP {sample.diagnostics.matched_count.toLocaleString()} / FP{" "}
                {sample.diagnostics.false_positive_count.toLocaleString()} / FN{" "}
                {sample.diagnostics.false_negative_count.toLocaleString()}
              </>
            ) : (
              <>
                真值 {sample.gt_instance_count.toLocaleString()} / 预测{" "}
                {sample.pred_instance_count.toLocaleString()}
              </>
            )}
          </span>
          <span className={sample.has_prediction ? "sample-status ok" : "sample-status missing"}>
            {sample.has_prediction ? "已预测" : "缺预测"}
          </span>
        </button>
      ))}
    </div>
  );
}

function SamplePager({
  offset,
  limit,
  total,
  onPageChange
}: {
  offset: number;
  limit: number;
  total: number;
  onPageChange: (offset: number) => void;
}) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(total, offset + limit);
  const previousOffset = Math.max(0, offset - limit);
  const nextOffset = offset + limit;
  return (
    <div className="sample-pager">
      <span>
        {start.toLocaleString()}-{end.toLocaleString()} / {total.toLocaleString()}
      </span>
      <div>
        <button
          className="mini-button"
          type="button"
          onClick={() => onPageChange(previousOffset)}
          disabled={offset <= 0}
        >
          上一页
        </button>
        <button
          className="mini-button"
          type="button"
          onClick={() => onPageChange(nextOffset)}
          disabled={nextOffset >= total}
        >
          下一页
        </button>
      </div>
    </div>
  );
}

function ResizableSplit({
  className,
  storageKey,
  fixedPane = "first",
  defaultSize,
  minSize,
  maxSize,
  first,
  second
}: {
  className: string;
  storageKey: string;
  fixedPane?: "first" | "second";
  defaultSize: number;
  minSize: number;
  maxSize: number;
  first: React.ReactNode;
  second: React.ReactNode;
}) {
  const rootRef = React.useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState(() => loadSplitSize(storageKey, defaultSize, minSize, maxSize));
  const [containerWidth, setContainerWidth] = useState(0);
  const dragRef = React.useRef<{ pointerId: number; startX: number; startSize: number } | null>(null);
  const effectiveMaxSize = Math.max(
    minSize,
    Math.min(maxSize, containerWidth > 0 ? containerWidth - minSize - 8 : maxSize)
  );

  useEffect(() => {
    localStorage.setItem(storageKey, String(size));
  }, [size, storageKey]);

  useEffect(() => {
    const node = rootRef.current;
    if (!node) {
      return undefined;
    }
    function updateWidth() {
      setContainerWidth(Math.max(0, node?.getBoundingClientRect().width ?? 0));
    }
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setSize((current) => clampNumber(current, minSize, effectiveMaxSize));
  }, [effectiveMaxSize, minSize]);

  function startResize(event: React.PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startSize: size
    };
  }

  function moveResize(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const delta = event.clientX - drag.startX;
    const signedDelta = fixedPane === "first" ? delta : -delta;
    setSize(clampNumber(drag.startSize + signedDelta, minSize, effectiveMaxSize));
  }

  function endResize(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div
      ref={rootRef}
      className={`${className} resizable-split ${fixedPane === "first" ? "fixed-first" : "fixed-second"}`}
      style={{
        gridTemplateColumns:
          fixedPane === "first"
            ? `${size}px 8px minmax(0, 1fr)`
            : `minmax(0, 1fr) 8px ${size}px`
      }}
    >
      {first}
      <div
        className="split-resizer"
        role="separator"
        aria-orientation="vertical"
        tabIndex={0}
        onPointerDown={startResize}
        onPointerMove={moveResize}
        onPointerUp={endResize}
        onPointerCancel={endResize}
      />
      {second}
    </div>
  );
}

function SampleViewer({ detail }: { detail: RunSampleDetail }) {
  return <InteractiveSampleViewer detail={detail} />;
}

function CanvasStage({
  width,
  height,
  imageUrl,
  imageAlt,
  gtInstances,
  predInstances,
  diagnostics,
  visibleLabels,
  showGt,
  showPred,
  showBoxes,
  showLines,
  showKeypoints,
  overlayColors,
  overlayStyle,
  labelColors,
  interactionSettings = DEFAULT_INTERACTION_SETTINGS,
  activeObjectId = null,
  onHover,
  onLock
}: {
  width: number;
  height: number;
  imageUrl: string;
  imageAlt: string;
  gtInstances: EvalInstance[];
  predInstances: EvalInstance[];
  diagnostics: RunSampleDetail["diagnostics"];
  visibleLabels?: Set<string>;
  showGt: boolean;
  showPred: boolean;
  showBoxes: boolean;
  showLines: boolean;
  showKeypoints: boolean;
  overlayColors: OverlayColors;
  overlayStyle: OverlayStyle;
  labelColors: LabelColors;
  interactionSettings?: InteractionSettings;
  activeObjectId?: string | null;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
}) {
  const stageRef = React.useRef<HTMLDivElement | null>(null);
  const contentRef = React.useRef<HTMLDivElement | null>(null);
  const dragRef = React.useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPan: { x: number; y: number };
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [stageSize, setStageSize] = useState({ width: 1, height: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const fitSize = useMemo(
    () => computeFitSize(width, height, stageSize),
    [height, stageSize, width]
  );

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    dragRef.current = null;
    setIsPanning(false);
  }, [imageUrl]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return undefined;
    }
    const stageNode = node;
    function updateStageSize() {
      const rect = stageNode.getBoundingClientRect();
      setStageSize({
        width: Math.max(1, rect.width),
        height: Math.max(1, rect.height)
      });
    }
    updateStageSize();
    const observer = new ResizeObserver(updateStageSize);
    observer.observe(stageNode);
    return () => observer.disconnect();
  }, []);

  function clampStagePan(nextPan: { x: number; y: number }, nextZoom = zoom) {
    return clampPan(nextPan, nextZoom, stageRef.current, contentRef.current);
  }

  function applyZoom(nextZoom: number, anchor?: { x: number; y: number }) {
    const clampedZoom = clampNumber(
      nextZoom,
      interactionSettings.minZoom,
      interactionSettings.maxZoom
    );
    setZoom((currentZoom) => {
      if (Math.abs(clampedZoom - currentZoom) < 0.001) {
        return currentZoom;
      }
      setPan((currentPan) => {
        if (!anchor) {
          return clampStagePan(currentPan, clampedZoom);
        }
        const stage = stageRef.current;
        const center = stage
          ? { x: stage.clientWidth / 2, y: stage.clientHeight / 2 }
          : { x: 0, y: 0 };
        const scale = clampedZoom / currentZoom;
        const relativeAnchor = {
          x: anchor.x - center.x,
          y: anchor.y - center.y
        };
        return clampStagePan(
          {
            x: relativeAnchor.x - (relativeAnchor.x - currentPan.x) * scale,
            y: relativeAnchor.y - (relativeAnchor.y - currentPan.y) * scale
          },
          clampedZoom
        );
      });
      return clampedZoom;
    });
  }

  function resetViewport() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    dragRef.current = null;
    setIsPanning(false);
  }

  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    const node = stageRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const anchor = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    applyZoom(zoom * Math.exp(-event.deltaY * interactionSettings.wheelZoomSensitivity), anchor);
  }

  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return undefined;
    }
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [interactionSettings.wheelZoomSensitivity, zoom]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      if (event.key.toLowerCase() === "f") {
        resetViewport();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startPan: pan
    };
    setIsPanning(true);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    setPan(
      clampStagePan({
        x: drag.startPan.x + (event.clientX - drag.startX) * interactionSettings.panSensitivity,
        y: drag.startPan.y + (event.clientY - drag.startY) * interactionSettings.panSensitivity
      })
    );
  }

  function endPan(event: React.PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div
      ref={stageRef}
      className={isPanning ? "image-stage panning" : "image-stage pannable"}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
    >
      <div
        ref={contentRef}
        className="image-zoom-layer"
        style={{
          width: `${fitSize.width}px`,
          height: `${fitSize.height}px`,
          transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom})`
        }}
      >
        <img
          src={imageUrl}
          alt={imageAlt}
          draggable={false}
          loading="eager"
          decoding="async"
        />
        <svg className="overlay-svg interactive" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {showGt ? (
            <InstanceLayer
              instances={gtInstances}
              kind="gt"
              diagnostics={diagnostics}
              visibleLabels={visibleLabels}
              showBoxes={showBoxes}
              showLines={showLines}
              showKeypoints={showKeypoints}
              activeObjectId={activeObjectId}
              overlayColors={overlayColors}
              overlayStyle={overlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
            />
          ) : null}
          {showPred ? (
            <InstanceLayer
              instances={predInstances}
              kind="pred"
              diagnostics={diagnostics}
              visibleLabels={visibleLabels}
              showBoxes={showBoxes}
              showLines={showLines}
              showKeypoints={showKeypoints}
              activeObjectId={activeObjectId}
              overlayColors={overlayColors}
              overlayStyle={overlayStyle}
              labelColors={labelColors}
              onHover={onHover}
              onLock={onLock}
            />
          ) : null}
        </svg>
      </div>
      <div className="canvas-hud">
        <span>{Math.round(zoom * 100)}%</span>
        {Math.abs(zoom - 1) > 0.01 || pan.x !== 0 || pan.y !== 0 ? (
          <button type="button" onClick={resetViewport}>
            复位
          </button>
        ) : null}
      </div>
    </div>
  );
}

function InteractiveSampleViewer({ detail }: { detail: RunSampleDetail }) {
  const width = detail.sample.image_width ?? 1000;
  const height = detail.sample.image_height ?? 1000;
  const labels = useMemo(
    () => unique([...detail.gt_instances, ...detail.pred_instances].map((instance) => instance.label)),
    [detail.gt_instances, detail.pred_instances]
  );
  const [activeLabels, setActiveLabels] = useState<string[]>(labels);
  const [showGt, setShowGt] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLines, setShowLines] = useState(true);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [lockedObjectId, setLockedObjectId] = useState<string | null>(null);
  const {
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars,
    updateOverlayColor,
    updateOverlayStyle,
    updateLabelColor,
    removeLabelColor,
    resetOverlayColors,
    resetOverlayStyle,
    resetLabelColors
  } = useWorkspaceSettings(labels);
  const activeObjectId = lockedObjectId ?? hoveredObjectId;
  const activeLabelSet = useMemo(() => new Set(activeLabels), [activeLabels]);
  const visibleGtInstances = useMemo(
    () => detail.gt_instances.filter((instance) => activeLabelSet.has(instance.label)),
    [activeLabelSet, detail.gt_instances]
  );
  const visiblePredInstances = useMemo(
    () => detail.pred_instances.filter((instance) => activeLabelSet.has(instance.label)),
    [activeLabelSet, detail.pred_instances]
  );
  const objectRows = useMemo(
    () =>
      buildObjectRows({
        gtInstances: detail.gt_instances,
        predInstances: detail.pred_instances,
        labels: activeLabelSet,
        diagnostics: detail.diagnostics
      }),
    [activeLabelSet, detail.diagnostics, detail.gt_instances, detail.pred_instances]
  );
  const visibleMetrics = visibleSampleMetrics(detail, activeLabelSet);
  const labelMetrics = visibleLabelMetrics(detail, activeLabelSet);

  useEffect(() => {
    setActiveLabels(labels);
    setLockedObjectId(null);
    setHoveredObjectId(null);
  }, [detail.sample.index, labels.join("|")]);

  function toggleLabel(label: string) {
    setActiveLabels((current) => {
      if (current.includes(label)) {
        return current.filter((item) => item !== label);
      }
      return unique([...current, label]);
    });
  }

  function toggleLockedObject(objectId: string | null) {
    if (objectId === null) {
      setLockedObjectId(null);
      return;
    }
    setLockedObjectId((current) => (current === objectId ? null : objectId));
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (isTextInputTarget(event.target)) {
        return;
      }
      if (event.key === "Escape") {
        setLockedObjectId(null);
        setHoveredObjectId(null);
      } else if (event.key.toLowerCase() === "g") {
        setShowGt((value) => !value);
      } else if (event.key.toLowerCase() === "p") {
        setShowPred((value) => !value);
      } else if (event.key.toLowerCase() === "b") {
        setShowBoxes((value) => !value);
      } else if (event.key.toLowerCase() === "l") {
        setShowLines((value) => !value);
      } else if (event.key.toLowerCase() === "k") {
        setShowKeypoints((value) => !value);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const canvasStage = (
    <CanvasStage
      width={width}
      height={height}
      imageUrl={detail.sample.image_url}
      imageAlt={detail.sample.image}
      gtInstances={detail.gt_instances}
      predInstances={detail.pred_instances}
      diagnostics={detail.diagnostics}
      visibleLabels={activeLabelSet}
      showGt={showGt}
      showPred={showPred}
      showBoxes={showBoxes}
      showLines={showLines}
      showKeypoints={showKeypoints}
      activeObjectId={activeObjectId}
      overlayColors={overlayColors}
      overlayStyle={overlayStyle}
      labelColors={labelColors}
      interactionSettings={interactionSettings}
      onHover={setHoveredObjectId}
      onLock={toggleLockedObject}
    />
  );
  const inspectorPanel = (
    <aside className="viewer-side-panel">
      <ViewerControlPanel
        labels={labels}
        activeLabels={activeLabels}
        colors={overlayColors}
        styleConfig={overlayStyle}
        labelColors={labelColors}
        showGt={showGt}
        showPred={showPred}
        showBoxes={showBoxes}
        showLines={showLines}
        showKeypoints={showKeypoints}
        onToggleLabel={toggleLabel}
            onColorChange={updateOverlayColor}
            onStyleChange={updateOverlayStyle}
            onLabelColorChange={updateLabelColor}
            onLabelColorRemove={removeLabelColor}
            onResetColors={resetOverlayColors}
            onResetStyle={resetOverlayStyle}
            onResetLabelColors={resetLabelColors}
        onShowGtChange={setShowGt}
        onShowPredChange={setShowPred}
        onShowBoxesChange={setShowBoxes}
        onShowLinesChange={setShowLines}
        onShowKeypointsChange={setShowKeypoints}
      />
      <VisibleMetricStrip metrics={visibleMetrics} />
      <LabelMetricTable rows={labelMetrics} />
      <div className="instance-workbench">
        <InstanceStats title="真值实例" instances={visibleGtInstances} />
        <InstanceStats title="预测实例" instances={visiblePredInstances} />
        <ObjectList
          objects={objectRows}
          activeObjectId={activeObjectId}
          lockedObjectId={lockedObjectId}
          onHover={setHoveredObjectId}
          onLock={toggleLockedObject}
        />
      </div>
    </aside>
  );

  return (
    <div className="viewer-stack" style={overlayVars}>
      <div className="viewer-toolbar">
        <div>
          <h2>{basename(detail.sample.image)}</h2>
          <p>{detail.sample.image}</p>
        </div>
        <div className="legend-row">
          <span className="legend-item gt">真值匹配</span>
          <span className="legend-item fn">漏检</span>
          <span className="legend-item pred">预测匹配</span>
          <span className="legend-item fp">误检</span>
          <button
            className="query-chip"
            type="button"
            onClick={() => setInspectorCollapsed((value) => !value)}
          >
            {inspectorCollapsed ? "显示检查器" : "收起检查器"}
          </button>
        </div>
      </div>
      {inspectorCollapsed ? (
        <div className="viewer-canvas-layout side-collapsed">{canvasStage}</div>
      ) : (
        <ResizableSplit
          className="viewer-canvas-layout"
          storageKey="eval_bench_viewer_inspector_width"
          fixedPane="second"
          defaultSize={224}
          minSize={176}
          maxSize={560}
          first={canvasStage}
          second={inspectorPanel}
        />
      )}
    </div>
  );
}

function DiagnosticStrip({ diagnostics }: { diagnostics: NonNullable<RunSampleDetail["diagnostics"]> }) {
  return (
    <div className="diagnostic-strip">
      <span>TP {diagnostics.matched_count.toLocaleString()}</span>
      <span>FP {diagnostics.false_positive_count.toLocaleString()}</span>
      <span>FN {diagnostics.false_negative_count.toLocaleString()}</span>
      <span>平均 IoU {formatMetric(diagnostics.mean_iou)}</span>
    </div>
  );
}

function ViewerControlPanel({
  labels,
  activeLabels,
  colors,
  styleConfig,
  labelColors,
  showGt,
  showPred,
  showBoxes,
  showLines,
  showKeypoints,
  onToggleLabel,
  onColorChange,
  onStyleChange,
  onLabelColorChange,
  onLabelColorRemove,
  onResetColors,
  onResetStyle,
  onResetLabelColors,
  onShowGtChange,
  onShowPredChange,
  onShowBoxesChange,
  onShowLinesChange,
  onShowKeypointsChange
}: {
  labels: string[];
  activeLabels: string[];
  colors: OverlayColors;
  styleConfig: OverlayStyle;
  labelColors: LabelColors;
  showGt: boolean;
  showPred: boolean;
  showBoxes: boolean;
  showLines: boolean;
  showKeypoints: boolean;
  onToggleLabel: (label: string) => void;
  onColorChange: (key: OverlayColorKey, value: string) => void;
  onStyleChange: (key: OverlayStyleKey, value: number | string) => void;
  onLabelColorChange: (label: string, value: string) => void;
  onLabelColorRemove: (label: string) => void;
  onResetColors: () => void;
  onResetStyle: () => void;
  onResetLabelColors: () => void;
  onShowGtChange: (value: boolean) => void;
  onShowPredChange: (value: boolean) => void;
  onShowBoxesChange: (value: boolean) => void;
  onShowLinesChange: (value: boolean) => void;
  onShowKeypointsChange: (value: boolean) => void;
}) {
  function layerPresetValue() {
    if (showGt && showPred && showBoxes && showLines && showKeypoints) {
      return "all";
    }
    if (showGt && !showPred && showBoxes && showLines && showKeypoints) {
      return "gt";
    }
    if (!showGt && showPred && showBoxes && showLines && showKeypoints) {
      return "pred";
    }
    if (showGt && showPred && showBoxes && !showLines && !showKeypoints) {
      return "boxes";
    }
    if (showGt && showPred && !showBoxes && showLines && !showKeypoints) {
      return "lines";
    }
    return "custom";
  }

  function applyLayerPreset(value: string) {
    if (value === "gt") {
      onShowGtChange(true);
      onShowPredChange(false);
      onShowBoxesChange(true);
      onShowLinesChange(true);
      onShowKeypointsChange(true);
      return;
    }
    if (value === "pred") {
      onShowGtChange(false);
      onShowPredChange(true);
      onShowBoxesChange(true);
      onShowLinesChange(true);
      onShowKeypointsChange(true);
      return;
    }
    if (value === "boxes") {
      onShowGtChange(true);
      onShowPredChange(true);
      onShowBoxesChange(true);
      onShowLinesChange(false);
      onShowKeypointsChange(false);
      return;
    }
    if (value === "lines") {
      onShowGtChange(true);
      onShowPredChange(true);
      onShowBoxesChange(false);
      onShowLinesChange(true);
      onShowKeypointsChange(false);
      return;
    }
    onShowGtChange(true);
    onShowPredChange(true);
    onShowBoxesChange(true);
    onShowLinesChange(true);
    onShowKeypointsChange(true);
  }

  return (
    <div className="viewer-controls">
      <label className="compact-select">
        <span>视图</span>
        <select value={layerPresetValue()} onChange={(event) => applyLayerPreset(event.target.value)}>
          <option value="all">真值 + 预测 / 全部几何</option>
          <option value="gt">仅真值</option>
          <option value="pred">仅预测</option>
          <option value="boxes">只看框</option>
          <option value="lines">只看线</option>
          <option value="custom">自定义</option>
        </select>
      </label>
      <div className="layer-toggle-strip" aria-label="图层开关">
        <ToggleButton label="真值" active={showGt} onChange={onShowGtChange} />
        <ToggleButton label="预测" active={showPred} onChange={onShowPredChange} />
        <ToggleButton label="框" active={showBoxes} onChange={onShowBoxesChange} />
        <ToggleButton label="线" active={showLines} onChange={onShowLinesChange} />
        <ToggleButton label="点" active={showKeypoints} onChange={onShowKeypointsChange} />
      </div>
      <details className="control-popover">
        <summary>
          标签 <strong>{activeLabels.length}/{labels.length}</strong>
        </summary>
        <div className="label-select-grid">
          {labels.map((label) => {
            const active = activeLabels.includes(label);
            return (
              <button
                key={label}
                className={active ? "label-select active" : "label-select"}
                type="button"
                onClick={() => onToggleLabel(label)}
              >
                {label}
              </button>
            );
          })}
        </div>
      </details>
      <OverlayAppearancePanel
        colors={colors}
        styleConfig={styleConfig}
        onColorChange={onColorChange}
        onStyleChange={onStyleChange}
        onResetColors={onResetColors}
        onResetStyle={onResetStyle}
      />
      <LabelColorPanel
        labels={labels}
        labelColors={labelColors}
        onChange={onLabelColorChange}
        onRemove={onLabelColorRemove}
        onReset={onResetLabelColors}
      />
    </div>
  );
}

function OverlayAppearancePanel({
  colors,
  styleConfig,
  onColorChange,
  onStyleChange,
  onResetColors,
  onResetStyle,
  defaultOpen = false
}: {
  colors: OverlayColors;
  styleConfig: OverlayStyle;
  onColorChange: (key: OverlayColorKey, value: string) => void;
  onStyleChange: (key: OverlayStyleKey, value: number | string) => void;
  onResetColors: () => void;
  onResetStyle: () => void;
  defaultOpen?: boolean;
}) {
  return (
    <>
      <details className="control-popover" open={defaultOpen}>
        <summary>
          样式 <strong>框 / 线 / 点</strong>
        </summary>
        <div className="control-title-row">
          <span>可视化参数</span>
          <button className="text-button" type="button" onClick={onResetStyle}>
            重置
          </button>
        </div>
        <div className="style-control-grid">
          <StyleSlider
            label="框线宽"
            value={styleConfig.boxStrokeWidth}
            min={1}
            max={10}
            step={0.5}
            onChange={(value) => onStyleChange("boxStrokeWidth", value)}
          />
          <StyleSlider
            label="骨架线宽"
            value={styleConfig.lineStrokeWidth}
            min={1}
            max={12}
            step={0.5}
            onChange={(value) => onStyleChange("lineStrokeWidth", value)}
          />
          <StyleSlider
            label="点半径"
            value={styleConfig.pointRadius}
            min={1}
            max={12}
            step={0.5}
            onChange={(value) => onStyleChange("pointRadius", value)}
          />
          <StyleSlider
            label="标签字号"
            value={styleConfig.labelFontSize}
            min={9}
            max={28}
            step={1}
            onChange={(value) => onStyleChange("labelFontSize", value)}
          />
          <StyleSlider
            label="高亮线宽"
            value={styleConfig.activeStrokeWidth}
            min={2}
            max={16}
            step={0.5}
            onChange={(value) => onStyleChange("activeStrokeWidth", value)}
          />
          <StyleSlider
            label="标签描边"
            value={styleConfig.labelStrokeWidth}
            min={0}
            max={8}
            step={0.5}
            onChange={(value) => onStyleChange("labelStrokeWidth", value)}
          />
          <StyleSlider
            label="标签底色"
            value={styleConfig.labelBackgroundOpacity}
            min={0}
            max={1}
            step={0.05}
            onChange={(value) => onStyleChange("labelBackgroundOpacity", value)}
          />
          <StyleSlider
            label="框填充"
            value={styleConfig.boxFillOpacity}
            min={0}
            max={0.5}
            step={0.02}
            onChange={(value) => onStyleChange("boxFillOpacity", value)}
          />
          <StyleSlider
            label="箭头大小"
            value={styleConfig.directionHeadScale}
            min={0.5}
            max={2.5}
            step={0.05}
            onChange={(value) => onStyleChange("directionHeadScale", value)}
          />
          <StyleSlider
            label="整体透明度"
            value={styleConfig.opacity}
            min={0.2}
            max={1}
            step={0.05}
            onChange={(value) => onStyleChange("opacity", value)}
          />
          <label className="compact-select dense">
            <span>预测线型</span>
            <select
              value={styleConfig.predLineStyle}
              onChange={(event) => onStyleChange("predLineStyle", event.target.value)}
            >
              <option value="dashed">虚线</option>
              <option value="solid">实线</option>
            </select>
          </label>
        </div>
      </details>
      <details className="control-popover" open={defaultOpen}>
        <summary>
          颜色 <strong>叠图</strong>
        </summary>
        <div className="control-title-row">
          <span />
          <button className="text-button" type="button" onClick={onResetColors}>
            重置
          </button>
        </div>
        <div className="color-control-grid">
          <ColorControl
            label="真值"
            value={colors.gt}
            onChange={(value) => onColorChange("gt", value)}
          />
          <ColorControl
            label="预测"
            value={colors.pred}
            onChange={(value) => onColorChange("pred", value)}
          />
          <ColorControl
            label="漏检"
            value={colors.fn}
            onChange={(value) => onColorChange("fn", value)}
          />
          <ColorControl
            label="误检"
            value={colors.fp}
            onChange={(value) => onColorChange("fp", value)}
          />
          <ColorControl
            label="高亮"
            value={colors.active}
            onChange={(value) => onColorChange("active", value)}
          />
        </div>
      </details>
    </>
  );
}

function LabelColorPanel({
  labels,
  labelColors,
  onChange,
  onRemove,
  onReset,
  defaultOpen = false
}: {
  labels: string[];
  labelColors: LabelColors;
  onChange: (label: string, value: string) => void;
  onRemove: (label: string) => void;
  onReset: () => void;
  defaultOpen?: boolean;
}) {
  const [draftLabel, setDraftLabel] = useState("");
  const [draftColor, setDraftColor] = useState("#2563eb");
  const sortedLabels = useMemo(
    () => [...labels].sort((left, right) => left.localeCompare(right)),
    [labels]
  );

  function addLabelColor() {
    const label = draftLabel.trim();
    if (!label) {
      return;
    }
    onChange(label, draftColor);
    setDraftLabel("");
  }

  return (
    <details className="control-popover" open={defaultOpen}>
      <summary>
        标签颜色 <strong>{sortedLabels.length}</strong>
      </summary>
      <div className="control-title-row">
        <span>按运行时 label 匹配</span>
        <button className="text-button" type="button" onClick={onReset}>
          重置
        </button>
      </div>
      <div className="label-color-add-row">
        <input
          value={draftLabel}
          placeholder="输入 label"
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
        <button className="secondary-button dense" type="button" onClick={addLabelColor}>
          添加
        </button>
      </div>
      <div className="label-color-grid">
        {sortedLabels.length === 0 ? (
          <div className="muted-line">还没有自定义 label 颜色。</div>
        ) : (
          sortedLabels.map((label) => (
            <div className="label-color-row" key={label}>
              <ColorControl
                label={label}
                value={labelColors[label] ?? fallbackLabelColor(label)}
                onChange={(value) => onChange(label, value)}
              />
              <button
                className="icon-button dense"
                type="button"
                title={`移除 ${label} 颜色规则`}
                onClick={() => onRemove(label)}
              >
                <X size={13} />
              </button>
            </div>
          ))
        )}
      </div>
    </details>
  );
}

function InteractionSettingsPanel({
  settings,
  onChange,
  onReset,
  defaultOpen = false
}: {
  settings: InteractionSettings;
  onChange: (key: InteractionSettingKey, value: number) => void;
  onReset: () => void;
  defaultOpen?: boolean;
}) {
  return (
    <details className="control-popover" open={defaultOpen}>
      <summary>
        交互 <strong>4</strong>
      </summary>
      <div className="control-title-row">
        <span>鼠标和画布范围</span>
        <button className="text-button" type="button" onClick={onReset}>
          重置
        </button>
      </div>
      <div className="style-control-grid">
        <StyleSlider
          label="滚轮缩放灵敏度"
          value={Math.round(settings.wheelZoomSensitivity * 100000)}
          min={5}
          max={300}
          step={1}
          onChange={(value) => onChange("wheelZoomSensitivity", value / 100000)}
        />
        <StyleSlider
          label="拖拽平移灵敏度"
          value={settings.panSensitivity}
          min={0.2}
          max={3}
          step={0.05}
          onChange={(value) => onChange("panSensitivity", value)}
        />
        <StyleSlider
          label="最小缩放"
          value={settings.minZoom}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(value) => onChange("minZoom", value)}
        />
        <StyleSlider
          label="最大缩放"
          value={settings.maxZoom}
          min={2}
          max={20}
          step={0.25}
          onChange={(value) => onChange("maxZoom", value)}
        />
      </div>
    </details>
  );
}

function StyleSlider({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="style-slider">
      <span>
        {label}
        <strong>{Number.isInteger(value) ? value : value.toFixed(2)}</strong>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ColorControl({
  label,
  value,
  onChange
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="color-control">
      <span>{label}</span>
      <input type="color" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function ToggleButton({
  label,
  active,
  onChange
}: {
  label: string;
  active: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className={active ? "control-check active" : "control-check"}>
      <input type="checkbox" checked={active} onChange={() => onChange(!active)} />
      {label}
    </label>
  );
}

function VisibleMetricStrip({ metrics }: { metrics: VisibleMetrics }) {
  return (
    <div className="diagnostic-strip">
      <span>真值 {metrics.gtCount.toLocaleString()}</span>
      <span>预测 {metrics.predCount.toLocaleString()}</span>
      <span>TP {metrics.matchedCount.toLocaleString()}</span>
      <span>FP {metrics.falsePositiveCount.toLocaleString()}</span>
      <span>FN {metrics.falseNegativeCount.toLocaleString()}</span>
      <span>平均 IoU {formatMetric(metrics.meanIou)}</span>
    </div>
  );
}

function LabelMetricTable({ rows }: { rows: LabelMetricRow[] }) {
  return (
    <details className="label-metric-card">
      <summary>分标签指标</summary>
      {rows.length === 0 ? (
        <div className="muted-line">没有可见标签。</div>
      ) : (
        <div className="label-metric-table">
          <table>
            <thead>
              <tr>
                <th>标签</th>
                <th>真值</th>
                <th>预测</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>P@.50</th>
                <th>R@.50</th>
                <th>平均 IoU</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td title={row.label}>{row.label}</td>
                  <td>{row.gtCount.toLocaleString()}</td>
                  <td>{row.predCount.toLocaleString()}</td>
                  <td>{row.matchedCount.toLocaleString()}</td>
                  <td>{row.falsePositiveCount.toLocaleString()}</td>
                  <td>{row.falseNegativeCount.toLocaleString()}</td>
                  <td>{formatMetric(row.precision)}</td>
                  <td>{formatMetric(row.recall)}</td>
                  <td>{formatMetric(row.meanIou)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </details>
  );
}

function InstanceLayer({
  instances,
  kind,
  diagnostics,
  visibleLabels,
  showBoxes = true,
  showLines = true,
  showKeypoints = true,
  activeObjectId = null,
  overlayColors,
  overlayStyle = DEFAULT_OVERLAY_STYLE,
  labelColors,
  onHover,
  onLock
}: {
  instances: EvalInstance[];
  kind: ObjectKind;
  diagnostics: RunSampleDetail["diagnostics"];
  visibleLabels?: Set<string>;
  showBoxes?: boolean;
  showLines?: boolean;
  showKeypoints?: boolean;
  activeObjectId?: string | null;
  overlayColors: OverlayColors;
  overlayStyle?: OverlayStyle;
  labelColors: LabelColors;
  onHover?: (objectId: string | null) => void;
  onLock?: (objectId: string | null) => void;
}) {
  const matched = new Set(
    (diagnostics?.matches ?? []).map((match) => (kind === "gt" ? match.gt_index : match.pred_index))
  );
  const errorItems =
    kind === "gt" ? diagnostics?.false_negatives ?? [] : diagnostics?.false_positives ?? [];
  const errors = new Set(errorItems.map((item) => item.index));
  return (
    <>
      {instances.map((instance, index) => {
        if (visibleLabels && !visibleLabels.has(instance.label)) {
          return null;
        }
        const objectId = `${kind}:${index}`;
        const bbox = normalizeBbox((instance as { bbox?: unknown }).bbox);
        const linePoints = normalizePointList(
          (instance as { linestrip?: unknown; line_strip?: unknown; points?: unknown }).linestrip ??
            (instance as { line_strip?: unknown }).line_strip
        );
        const keypoints = normalizePointList((instance as { keypoints?: unknown }).keypoints);
        const anchorBox = bbox ?? boundsFromPoints(linePoints ?? keypoints);
        if (!bbox && (!linePoints || linePoints.length === 0) && (!keypoints || keypoints.length === 0)) {
          return null;
        }
        const status = errors.has(index)
          ? kind === "gt"
            ? "fn"
            : "fp"
          : matched.has(index)
            ? "match"
            : "neutral";
        const color = resolveInstanceColor(instance.label, status, kind, overlayColors, labelColors);
        const directionHead =
          linePoints && linePoints.length >= 2
            ? arrowHeadPoints(
                linePoints,
                overlayStyle.lineStrokeWidth,
                overlayStyle.directionHeadScale
              )
            : null;
        const lineRadius = Math.max(overlayStyle.pointRadius, overlayStyle.lineStrokeWidth * 0.75);
        const labelX = anchorBox ? anchorBox[0] + 3 : 0;
        const labelY = anchorBox ? Math.max(12, anchorBox[1] - 4) : 0;
        const labelWidth = Math.max(
          28,
          instance.label.length * overlayStyle.labelFontSize * 0.62 + 10
        );
        const labelHeight = overlayStyle.labelFontSize + 6;
        return (
          <g
            key={objectId}
            className={
              objectId === activeObjectId
                ? `overlay-instance ${kind} ${status} active`
                : `overlay-instance ${kind} ${status}`
            }
            style={{ "--instance-color": color } as React.CSSProperties}
            onPointerEnter={() => onHover?.(objectId)}
            onPointerLeave={() => onHover?.(null)}
            onClick={(event) => {
              event.stopPropagation();
              onLock?.(objectId);
            }}
          >
            {showBoxes && bbox ? (
              <rect x={bbox[0]} y={bbox[1]} width={bbox[2] - bbox[0]} height={bbox[3] - bbox[1]} />
            ) : null}
            {showBoxes && anchorBox ? (
              <g className="overlay-label">
                <rect
                  className="label-backplate"
                  x={labelX - 3}
                  y={labelY - overlayStyle.labelFontSize - 3}
                  width={labelWidth}
                  height={labelHeight}
                  rx={2}
                />
                <text x={labelX} y={labelY}>
                  {instance.label}
                </text>
              </g>
            ) : null}
            {showLines && linePoints && linePoints.length >= 2 ? (
              <>
                <polyline points={linePoints.map((point) => `${point[0]},${point[1]}`).join(" ")} />
                <circle
                  className="line-endpoint start"
                  cx={linePoints[0][0]}
                  cy={linePoints[0][1]}
                  r={lineRadius}
                />
                <circle
                  className="line-endpoint end"
                  cx={linePoints[linePoints.length - 1][0]}
                  cy={linePoints[linePoints.length - 1][1]}
                  r={lineRadius}
                />
                {directionHead ? (
                  <polygon
                    className="direction-head"
                    points={directionHead.map((point) => `${point[0]},${point[1]}`).join(" ")}
                  />
                ) : null}
              </>
            ) : null}
            {showKeypoints && keypoints && keypoints.length > 0 ? (
              keypoints.map((point, pointIndex) => (
                <circle
                  key={`${objectId}-point-${pointIndex}`}
                  cx={point[0]}
                  cy={point[1]}
                  r={overlayStyle.pointRadius}
                />
              ))
            ) : null}
          </g>
        );
      })}
    </>
  );
}

function ObjectList({
  objects,
  activeObjectId,
  lockedObjectId,
  onHover,
  onLock
}: {
  objects: ObjectRow[];
  activeObjectId: string | null;
  lockedObjectId: string | null;
  onHover: (objectId: string | null) => void;
  onLock: (objectId: string | null) => void;
}) {
  return (
    <div className="object-list">
      <div className="instance-card-title">对象列表</div>
      {objects.length === 0 ? (
        <div className="muted-line">没有可见对象。</div>
      ) : (
        <div className="object-list-scroll">
          {objects.map((object) => (
            <button
              key={object.id}
              className={object.id === activeObjectId ? "object-row active" : "object-row"}
              type="button"
              onPointerEnter={() => onHover(object.id)}
              onPointerLeave={() => onHover(null)}
              onClick={() => onLock(object.id)}
            >
              <span className={`object-kind ${object.kind}`}>{objectKindLabel(object.kind)}</span>
              <span className="object-main">
                <span className="object-label">
                  {object.label}
                  <span className="object-index">#{object.index + 1}</span>
                </span>
                <span className="object-bbox">{formatBbox(object.bbox)}</span>
              </span>
              <span className={`object-status ${object.status}`}>
                {objectStatusLabel(object.status)}
              </span>
              <span className="object-match">{objectMetricText(object, formatMetric)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function InstanceStats({ title, instances }: { title: string; instances: EvalInstance[] }) {
  const counts = countInstancesByLabel(instances);
  const entries = Object.entries(counts).sort(([left], [right]) => left.localeCompare(right));
  return (
    <div className="instance-card">
      <div className="instance-card-title">{title}</div>
      {entries.length === 0 ? (
        <div className="muted-line">没有实例。</div>
      ) : (
        <div className="label-chip-row">
          {entries.map(([label, count]) => (
            <span className="label-chip" key={label}>
              {label} {count}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function JobsPage() {
  const { data } = useDashboardState();
  const recentRuns = data?.runs.slice(0, 12) ?? [];
  return (
    <section className="page-stack">
      <WorkspaceTabs defaultValue="activity" label="评测中心">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="activity">活动流</Tabs.Trigger>
          <Tabs.Trigger value="new">新建评测</Tabs.Trigger>
          <Tabs.Trigger value="runs">结果库</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="activity" className="workspace-tab-panel">
          <div className="job-activity-grid">
            <div className="workspace-card fill">
              <PanelTitle title="任务队列" meta="创建、执行、失败排障和 runtime log" />
              <JobQueuePanel />
            </div>
            <div className="workspace-card fill">
              <PanelTitle title="最近结果" meta="任务完成后会沉淀为可复查 run" />
              <RecentRunList runs={recentRuns} />
            </div>
          </div>
        </Tabs.Content>
        <Tabs.Content value="new" className="workspace-tab-panel">
          <JobCreatePanel benchmarks={data?.benchmarks ?? []} />
        </Tabs.Content>
        <Tabs.Content value="runs" className="workspace-tab-panel">
          <div className="workspace-card fill">
            <PanelTitle title="结果库" meta={`${(data?.runs.length ?? 0).toLocaleString()} 条记录`} />
            <RunTable runs={data?.runs ?? []} />
          </div>
        </Tabs.Content>
      </WorkspaceTabs>
    </section>
  );
}

function RecentRunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-panel">还没有评测结果。</div>;
  }
  return (
    <div className="recent-run-list">
      {runs.map((run) => (
        <Link
          className="recent-run-card"
          key={run.run_id}
          to="/runs/$runId"
          params={{ runId: run.run_id }}
        >
          <span className="recent-run-head">
            <strong title={run.run_id}>{run.run_id}</strong>
            <Badge value={run.status} />
          </span>
          <span className="recent-run-meta" title={run.model_id}>
            {run.model_id || "unknown model"}
          </span>
          <span className="recent-run-metrics">
            <em>P {formatMetric(run.precision_iou50)}</em>
            <em>R {formatMetric(run.recall_iou50)}</em>
            <em>IoU {formatMetric(run.mean_iou)}</em>
          </span>
        </Link>
      ))}
    </div>
  );
}

function ServicesPage() {
  const servicesQuery = useQuery({ queryKey: ["services"], queryFn: fetchServices });
  return (
    <section className="page-stack">
      <WorkspaceTabs defaultValue="registry" label="服务工作区">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="registry">服务目录</Tabs.Trigger>
          <Tabs.Trigger value="register">登记服务</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="registry" className="workspace-tab-panel">
          {servicesQuery.isLoading ? (
            <EmptyState title="正在加载服务" />
          ) : servicesQuery.error || !servicesQuery.data ? (
            <EmptyState title="服务加载失败" tone="danger" />
          ) : (
            <ServiceGrid services={servicesQuery.data.services} />
          )}
        </Tabs.Content>
        <Tabs.Content value="register" className="workspace-tab-panel">
          <ServiceCreatePanel />
        </Tabs.Content>
      </WorkspaceTabs>
    </section>
  );
}

function SettingsPage() {
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
    updateOverlayColor,
    updateOverlayStyle,
    updateInteractionSetting,
    updateLabelColor,
    removeLabelColor,
    resetOverlayColors,
    resetOverlayStyle,
    resetInteractionSettings,
    resetLabelColors
  } = useWorkspaceSettings(previewLabelsList.length ? previewLabelsList : SETTINGS_PREVIEW_LABELS);
  const previewSample = previewQuery.data?.sample ?? null;
  const previewWidth = previewSample?.image_width ?? 960;
  const previewHeight = previewSample?.image_height ?? 600;
  const previewImageUrl = previewSample?.image_url ?? SETTINGS_PREVIEW_IMAGE_URL;
  const previewMeta =
    previewQuery.data && previewSample
      ? `${previewQuery.data.benchmark_id} / #${previewSample.index + 1}`
      : "未找到基准集样本时使用内置示意图";

  return (
    <section className="page-stack settings-page">
      <WorkspaceTabs defaultValue="display" label="工作台设置">
        <Tabs.List className="workspace-tab-list">
          <Tabs.Trigger value="display">显示偏好</Tabs.Trigger>
          <Tabs.Trigger value="workflow">使用习惯</Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="display" className="workspace-tab-panel">
          <ResizableSplit
            className="settings-grid"
            storageKey="eval_bench_settings_controls_width"
            fixedPane="second"
            defaultSize={420}
            minSize={300}
            maxSize={820}
            first={
              <div className="workspace-card settings-preview-card" style={overlayVars}>
              <PanelTitle title="叠图预览" meta={previewMeta} />
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
              <div className="settings-preview-foot">
                <span style={{ "--swatch": overlayColors.gt } as React.CSSProperties}>GT / label</span>
                <span style={{ "--swatch": overlayColors.pred } as React.CSSProperties}>Pred</span>
                <span style={{ "--swatch": overlayColors.fn } as React.CSSProperties}>FN</span>
                <span style={{ "--swatch": overlayColors.fp } as React.CSSProperties}>FP</span>
                <strong>滚轮缩放，拖拽检查局部</strong>
              </div>
            </div>
            }
            second={
              <div className="workspace-card settings-control-card">
              <PanelTitle title="可视化外观" meta="颜色、线宽、点和标签" />
              <div className="viewer-controls settings-controls">
                <OverlayAppearancePanel
                  colors={overlayColors}
                  styleConfig={overlayStyle}
                  onColorChange={updateOverlayColor}
                  onStyleChange={updateOverlayStyle}
                  onResetColors={resetOverlayColors}
                  onResetStyle={resetOverlayStyle}
                  defaultOpen
                />
                <LabelColorPanel
                  labels={labels}
                  labelColors={labelColors}
                  onChange={updateLabelColor}
                  onRemove={removeLabelColor}
                  onReset={resetLabelColors}
                  defaultOpen
                />
                <InteractionSettingsPanel
                  settings={interactionSettings}
                  onChange={updateInteractionSetting}
                  onReset={resetInteractionSettings}
                  defaultOpen
                />
              </div>
            </div>
            }
          />
        </Tabs.Content>
        <Tabs.Content value="workflow" className="workspace-tab-panel">
          <div className="workspace-card settings-workflow-card">
            <PanelTitle title="交互约定" meta="面向标注检查和评测排障" />
            <div className="settings-note-grid">
              <div>
                <strong>画布</strong>
                <span>滚轮缩放，拖拽平移，F 复位视图，Esc 取消对象锁定。</span>
              </div>
              <div>
                <strong>图层</strong>
                <span>G/P/B/L/K 可快速切换真值、预测、框、线和点。</span>
              </div>
              <div>
                <strong>样本</strong>
                <span>列表分页加载，使用样本列表或快捷键切换当前样本。</span>
              </div>
            </div>
          </div>
        </Tabs.Content>
      </WorkspaceTabs>
    </section>
  );
}

function ServiceCreatePanel() {
  const queryClient = useQueryClient();
  const [kind, setKind] = useState("local_vllm");
  const [serviceId, setServiceId] = useState("local-vllm-0");
  const [modelPath, setModelPath] = useState("");
  const [servedModelName, setServedModelName] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [cudaVisibleDevices, setCudaVisibleDevices] = useState("");
  const [tensorParallelSize, setTensorParallelSize] = useState(1);
  const [port, setPort] = useState(8000);
  const [maxModelLen, setMaxModelLen] = useState(32768);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState(0.9);
  const [maxNumSeqs, setMaxNumSeqs] = useState(8);
  const mutation = useMutation({
    mutationFn: createService,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate({
      kind,
      service_id: serviceId.trim() || undefined,
      model_path: modelPath.trim() || undefined,
      served_model_name: servedModelName.trim() || undefined,
      endpoint: endpoint.trim() || undefined,
      cuda_visible_devices: cudaVisibleDevices.trim() || undefined,
      tensor_parallel_size: tensorParallelSize,
      port,
      max_model_len: maxModelLen,
      gpu_memory_utilization: gpuMemoryUtilization,
      max_num_seqs: maxNumSeqs
    });
  }

  return (
    <ActionPanel title="登记模型服务" meta="保存本地或外部 vLLM 服务参数">
      <form className="job-form service-form" onSubmit={submit}>
        <label>
          <span>类型</span>
          <select value={kind} onChange={(event) => setKind(event.target.value)}>
            <option value="local_vllm">本地 vLLM</option>
            <option value="external_vllm">外部 vLLM</option>
          </select>
        </label>
        <label>
          <span>服务 ID</span>
          <input value={serviceId} onChange={(event) => setServiceId(event.target.value)} />
        </label>
        <label>
          <span>模型路径</span>
          <input
            value={modelPath}
            onChange={(event) => setModelPath(event.target.value)}
            placeholder="outputs/qwen3vl-sft/run/best"
          />
        </label>
        <label>
          <span>服务模型名</span>
          <input
            value={servedModelName}
            onChange={(event) => setServedModelName(event.target.value)}
            placeholder="qwen3vl-best"
          />
        </label>
        <label>
          <span>端点</span>
          <input
            value={endpoint}
            onChange={(event) => setEndpoint(event.target.value)}
            placeholder="http://127.0.0.1:8000"
          />
        </label>
        <label>
          <span>CUDA</span>
          <input
            value={cudaVisibleDevices}
            onChange={(event) => setCudaVisibleDevices(event.target.value)}
            placeholder="0,2"
          />
        </label>
        <label>
          <span>TP 大小</span>
          <input
            type="number"
            min={1}
            value={tensorParallelSize}
            onChange={(event) => setTensorParallelSize(Number(event.target.value))}
          />
        </label>
        <label>
          <span>端口</span>
          <input
            type="number"
            min={1}
            value={port}
            onChange={(event) => setPort(Number(event.target.value))}
          />
        </label>
        <label>
          <span>最大上下文</span>
          <input
            type="number"
            min={1}
            value={maxModelLen}
            onChange={(event) => setMaxModelLen(Number(event.target.value))}
          />
        </label>
        <label>
          <span>显存占比</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={gpuMemoryUtilization}
            onChange={(event) => setGpuMemoryUtilization(Number(event.target.value))}
          />
        </label>
        <label>
          <span>最大并发序列</span>
          <input
            type="number"
            min={1}
            value={maxNumSeqs}
            onChange={(event) => setMaxNumSeqs(Number(event.target.value))}
          />
        </label>
        <button className="primary-button" type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "保存中" : "保存服务"}
        </button>
        {mutation.isError ? <div className="form-error">服务保存失败。</div> : null}
      </form>
    </ActionPanel>
  );
}

function ServiceGrid({ services }: { services: ServiceSummary[] }) {
  if (services.length === 0) {
    return <EmptyState title="还没有登记模型服务。" />;
  }
  return (
    <div className="service-grid">
      {services.map((service) => (
        <ServiceCard key={service.service_id} service={service} />
      ))}
    </div>
  );
}

function ServiceCard({ service }: { service: ServiceSummary }) {
  const queryClient = useQueryClient();
  const [showLog, setShowLog] = useState(false);
  const startMutation = useMutation({
    mutationFn: () => startService(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const healthMutation = useMutation({
    mutationFn: () => checkServiceHealth(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const stopMutation = useMutation({
    mutationFn: () => stopService(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: () => deleteService(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const logQuery = useQuery({
    queryKey: ["service-log", service.service_id],
    queryFn: () => fetchServiceLogs(service.service_id),
    enabled: showLog
  });
  const command = Array.isArray(service.runtime.command)
    ? service.runtime.command.map(String).join(" ")
    : "";
  const health = serviceHealth(service);
  return (
    <div className="service-card">
      <div className="service-card-heading">
        <div>
          <h2>{service.service_id}</h2>
          <p>{serviceConfigValue(service, "model_path") || serviceConfigValue(service, "endpoint")}</p>
        </div>
        <Badge value={service.status} />
      </div>
      <div className="service-config-grid">
        <ConfigItem label="类型" value={service.kind} />
        <ConfigItem label="服务模型" value={serviceConfigValue(service, "served_model_name")} />
        <ConfigItem label="端点" value={serviceEndpointValue(service)} />
        <ConfigItem label="CUDA" value={serviceConfigValue(service, "cuda_visible_devices")} />
        <ConfigItem label="TP" value={serviceConfigValue(service, "tensor_parallel_size")} />
        <ConfigItem label="端口" value={serviceConfigValue(service, "port")} />
        <ConfigItem label="上下文" value={serviceConfigValue(service, "max_model_len")} />
        <ConfigItem label="显存占比" value={serviceConfigValue(service, "gpu_memory_utilization")} />
        <ConfigItem label="并发序列" value={serviceConfigValue(service, "max_num_seqs")} />
        <ConfigItem label="PID" value={runtimeValue(service, "pid")} />
        <ConfigItem label="健康状态" value={health.status} />
        <ConfigItem label="探测时间" value={health.checkedAt} />
        <ConfigItem label="更新时间" value={formatDate(service.updated_at)} />
      </div>
      <div className={health.ok ? "service-health ok" : "service-health"}>
        <span>{health.ok ? "就绪" : health.status}</span>
        <strong title={health.message}>{health.message}</strong>
      </div>
      {command ? <pre className="service-command">{command}</pre> : null}
      {service.error ? <div className="form-error">{service.error}</div> : null}
      <div className="row-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={
            service.kind !== "local_vllm" ||
            service.status === "starting" ||
            service.status === "running" ||
            startMutation.isPending
          }
          onClick={() => startMutation.mutate()}
        >
          {startMutation.isPending ? "启动中" : "启动"}
        </button>
        <button
          className="mini-button"
          type="button"
          disabled={healthMutation.isPending}
          onClick={() => healthMutation.mutate()}
        >
          {healthMutation.isPending ? "探测中" : "探测"}
        </button>
        <button
          className="mini-button"
          type="button"
          disabled={!["starting", "running"].includes(service.status) || stopMutation.isPending}
          onClick={() => stopMutation.mutate()}
        >
          {stopMutation.isPending ? "停止中" : "停止"}
        </button>
        <button className="mini-button" type="button" onClick={() => setShowLog((value) => !value)}>
          {showLog ? "隐藏日志" : "日志"}
        </button>
        <button
          className="icon-button dense danger"
          type="button"
          disabled={deleteMutation.isPending}
          title="删除服务记录"
          onClick={() => {
            if (confirm(`删除服务 ${service.service_id}？`)) {
              deleteMutation.mutate();
            }
          }}
        >
          <Trash2 size={14} />
        </button>
      </div>
      {showLog ? <ServiceLogPanel query={logQuery} /> : null}
    </div>
  );
}

function ServiceLogPanel({
  query
}: {
  query: UseQueryResult<ServiceLog, Error>;
}) {
  if (query.isLoading) {
    return <div className="service-log-panel muted-line">正在加载日志</div>;
  }
  if (query.isError || !query.data) {
    return <div className="service-log-panel form-error">日志加载失败。</div>;
  }
  return (
    <div className="service-log-panel">
      <div className="service-log-heading">
        <span>日志尾部</span>
        <strong title={query.data.log_path ?? ""}>{query.data.log_path ?? "没有日志文件"}</strong>
      </div>
      <pre>{query.data.text || "没有日志内容。"}</pre>
    </div>
  );
}

function ComparePage() {
  const { data, isLoading, error } = useDashboardState();
  const comparisonListQuery = useQuery({
    queryKey: ["comparisons"],
    queryFn: fetchComparisons
  });
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [baselineRunId, setBaselineRunId] = useState(
    () => new URLSearchParams(window.location.search).get("baseline") ?? ""
  );
  const [candidateRunId, setCandidateRunId] = useState(
    () => new URLSearchParams(window.location.search).get("candidate") ?? ""
  );
  const runs = data?.runs ?? [];
  const tasks = unique(runs.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runs.map((run) => run.benchmark_id).filter(Boolean));
  const filteredRuns = runs
    .filter((run) => taskFilter === "all" || run.spec_task === taskFilter)
    .filter((run) => benchmarkFilter === "all" || run.benchmark_id === benchmarkFilter)
    .sort((left, right) => scoreRun(right) - scoreRun(left));
  const comparableRuns = filteredRuns.filter((run) => run.report_path);
  const fallbackCandidate = comparableRuns[0]?.run_id ?? "";
  const fallbackBaseline =
    comparableRuns.find((run) => run.run_id !== fallbackCandidate)?.run_id ?? "";
  const effectiveBaseline = runIdExists(comparableRuns, baselineRunId)
    ? baselineRunId
    : fallbackBaseline;
  const candidateFallback =
    comparableRuns.find((run) => run.run_id !== effectiveBaseline)?.run_id ?? "";
  const effectiveCandidate =
    runIdExists(comparableRuns, candidateRunId) && candidateRunId !== effectiveBaseline
      ? candidateRunId
      : candidateFallback;
  const comparisonQuery = useQuery({
    queryKey: ["comparison", effectiveBaseline, effectiveCandidate],
    queryFn: () => fetchComparison(effectiveBaseline, effectiveCandidate),
    enabled: Boolean(effectiveBaseline && effectiveCandidate && effectiveBaseline !== effectiveCandidate)
  });

  useEffect(() => {
    if (comparisonQuery.data?.comparison_id) {
      void comparisonListQuery.refetch();
    }
  }, [comparisonListQuery.refetch, comparisonQuery.data?.comparison_id]);

  if (isLoading) {
    return <EmptyState title="正在加载对比状态" />;
  }
  if (error || !data) {
    return <EmptyState title="对比状态加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack compare-page">
      <div className="compare-topbar">
        <div className="compare-title">
          <span>对比工作区</span>
          <strong>{filteredRuns.length.toLocaleString()} 条 run</strong>
        </div>
        <div className="compare-chip-strip">
          <FilterSelect
            label="任务"
            value={taskFilter}
            values={["all", ...tasks]}
            labels={{ all: "全部" }}
            onChange={setTaskFilter}
            compact
          />
          <FilterSelect
            label="基准集"
            value={benchmarkFilter}
            values={["all", ...benchmarks]}
            labels={{ all: "全部" }}
            onChange={setBenchmarkFilter}
            compact
          />
        </div>
      </div>
      <ResizableSplit
        className="compare-workspace"
        storageKey="eval_bench_compare_rail_width"
        defaultSize={292}
        minSize={180}
        maxSize={680}
        first={
          <aside className="compare-run-rail">
            <RunSelectRail
              title="基线"
              value={effectiveBaseline}
              runs={comparableRuns}
              disabled={comparableRuns.length < 2}
              onChange={setBaselineRunId}
            />
            <RunSelectRail
              title="候选"
              value={effectiveCandidate}
              runs={comparableRuns}
              disabled={comparableRuns.length < 2}
              onChange={setCandidateRunId}
            />
            <ComparisonHistoryPanel comparisons={comparisonListQuery.data?.comparisons ?? []} />
          </aside>
        }
        second={
          <ResizableSplit
            className="compare-main-split"
            storageKey="eval_bench_compare_leaderboard_width"
            fixedPane="second"
            defaultSize={372}
            minSize={260}
            maxSize={780}
            first={
              <main className="compare-report-pane">
                {comparableRuns.length < 2 ? (
                  <div className="empty-panel">至少需要两个已完成评测的 run 才能对比。</div>
                ) : effectiveBaseline === effectiveCandidate ? (
                  <div className="empty-panel">请选择两个不同的 run。</div>
                ) : comparisonQuery.isLoading ? (
                  <div className="empty-panel">正在加载对比报告</div>
                ) : comparisonQuery.isError || !comparisonQuery.data ? (
                  <div className="empty-panel danger-text">对比报告加载失败。</div>
                ) : (
                  <ComparisonPanel report={comparisonQuery.data} />
                )}
              </main>
            }
            second={
              <aside className="compare-leaderboard-pane">
                <div className="comparison-sample-title">排行榜</div>
                <LeaderboardTable runs={filteredRuns} />
              </aside>
            }
          />
        }
      />
    </section>
  );
}

function RunSelectRail({
  title,
  value,
  runs,
  disabled,
  onChange
}: {
  title: string;
  value: string;
  runs: RunSummary[];
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const selected = runs.find((run) => run.run_id === value);
  return (
    <div className="compare-run-select">
      <label>
        <span>{title}</span>
        <select value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
          {disabled ? <option value="">需要两个报告</option> : null}
          {runs.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {formatRunOption(run)}
            </option>
          ))}
        </select>
      </label>
      {selected ? (
        <div className="compare-run-card">
          <strong title={selected.run_id}>{selected.run_id}</strong>
          <span>{selected.model_id}</span>
          <div>
            <Badge value={selected.status} />
            <em>R {formatMetric(selected.recall_iou50)}</em>
            <em>P {formatMetric(selected.precision_iou50)}</em>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ComparisonPanel({ report }: { report: ComparisonReport }) {
  const [activeLabel, setActiveLabel] = useState("all");
  const labelDeltas = report.labels ?? [];
  const labelValues = labelDeltas.map((item) => item.label);
  useEffect(() => {
    if (activeLabel !== "all" && !labelValues.includes(activeLabel)) {
      setActiveLabel("all");
    }
  }, [activeLabel, labelValues.join("|")]);
  const filteredImprovements = useMemo(
    () => filterComparisonSamplesByLabel(report.top_improvements, activeLabel),
    [activeLabel, report.top_improvements]
  );
  const filteredRegressions = useMemo(
    () => filterComparisonSamplesByLabel(report.top_regressions, activeLabel),
    [activeLabel, report.top_regressions]
  );
  const firstImprovement = firstComparableSample(filteredImprovements);
  const firstRegression = firstComparableSample(filteredRegressions);
  return (
    <div className="comparison-panel">
      <div className="comparison-title-row">
        <div>
          <div className="eyebrow">双模型对比报告</div>
          <h2>
            {report.baseline_run_id} vs {report.candidate_run_id}
          </h2>
        </div>
        <div className="compare-title-meta">
          <div className="sample-count-chip">{report.sample_count.toLocaleString()} 个样本</div>
          {report.target_labels?.length ? (
            <div className="sample-count-chip subtle">{report.target_labels.join(" / ")}</div>
          ) : null}
        </div>
      </div>
      {report.warnings?.length ? (
        <div className="comparison-warning-strip">
          {report.warnings.map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </div>
      ) : null}
      <div className="comparison-delta-grid">
        <DeltaCard label="P@.50" value={report.delta.precision_iou50} />
        <DeltaCard label="R@.50" value={report.delta.recall_iou50} />
        <DeltaCard label="平均 IoU" value={report.delta.mean_iou} />
        <DeltaCard label="匹配数" value={report.delta.matched_count} integer />
        <DeltaCard label="误检" value={report.delta.false_positive_count} integer inverted />
        <DeltaCard label="漏检" value={report.delta.false_negative_count} integer inverted />
      </div>
      <div className="comparison-summary-row">
        <SummaryPill label="提升" value={report.summary.improved_samples} tone="positive" />
        <SummaryPill label="退化" value={report.summary.regressed_samples} tone="negative" />
        <SummaryPill label="变化" value={report.summary.changed_samples} />
        <SummaryPill label="不变" value={report.summary.unchanged_samples} />
      </div>
      <ComparisonQuickActions
        baselineRunId={report.baseline_run_id}
        candidateRunId={report.candidate_run_id}
        firstImprovement={firstImprovement}
        firstRegression={firstRegression}
      />
      <ComparisonLabelDeltaStrip
        labels={labelDeltas}
        activeLabel={activeLabel}
        onChange={setActiveLabel}
      />
      <div className="comparison-columns">
        <ComparisonSampleTable
          title="提升最多"
          samples={filteredImprovements}
          baselineRunId={report.baseline_run_id}
          candidateRunId={report.candidate_run_id}
          tone="positive"
        />
        <ComparisonSampleTable
          title="退化最多"
          samples={filteredRegressions}
          baselineRunId={report.baseline_run_id}
          candidateRunId={report.candidate_run_id}
          tone="negative"
        />
      </div>
    </div>
  );
}

function ComparisonQuickActions({
  baselineRunId,
  candidateRunId,
  firstImprovement,
  firstRegression
}: {
  baselineRunId: string;
  candidateRunId: string;
  firstImprovement: ComparisonSample | null;
  firstRegression: ComparisonSample | null;
}) {
  return (
    <div className="comparison-quick-actions">
      {firstRegression ? (
        <a
          className="mini-link compare-alert"
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstRegression.candidate_index ?? firstRegression.sample_index ?? 0
          )}
        >
          <Eye size={13} />
          看首个退化样本
        </a>
      ) : null}
      {firstImprovement ? (
        <a
          className="mini-link compare-ready"
          href={comparisonSampleHref(
            baselineRunId,
            candidateRunId,
            firstImprovement.candidate_index ?? firstImprovement.sample_index ?? 0
          )}
        >
          <Eye size={13} />
          看首个提升样本
        </a>
      ) : null}
    </div>
  );
}

function DeltaCard({
  label,
  value,
  integer,
  inverted
}: {
  label: string;
  value: number;
  integer?: boolean;
  inverted?: boolean;
}) {
  const positive = value > 0;
  const negative = value < 0;
  const good = inverted ? negative : positive;
  const bad = inverted ? positive : negative;
  const className = good ? "delta-card positive" : bad ? "delta-card negative" : "delta-card";
  return (
    <div className={className}>
      <span>{label}</span>
      <strong>{integer ? formatSignedInteger(value) : formatSignedMetric(value)}</strong>
    </div>
  );
}

function SummaryPill({
  label,
  value,
  tone
}: {
  label: string;
  value: number;
  tone?: "positive" | "negative";
}) {
  return (
    <div className={tone ? `summary-pill ${tone}` : "summary-pill"}>
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}

function ComparisonLabelDeltaStrip({
  labels,
  activeLabel,
  onChange
}: {
  labels: ComparisonLabelDelta[];
  activeLabel: string;
  onChange: (label: string) => void;
}) {
  const visible = labels.slice(0, 8);
  if (visible.length === 0) {
    return null;
  }
  return (
    <div className="comparison-label-strip">
      <button
        className={activeLabel === "all" ? "label-delta-card active" : "label-delta-card"}
        type="button"
        onClick={() => onChange("all")}
      >
        <span>全部标签</span>
        <strong>All</strong>
        <em>查看全量变化样本</em>
      </button>
      {visible.map((item) => {
        const tone =
          item.delta_score > 0 ? "positive" : item.delta_score < 0 ? "negative" : "neutral";
        return (
          <button
            className={
              activeLabel === item.label
                ? `label-delta-card ${tone} active`
                : `label-delta-card ${tone}`
            }
            type="button"
            onClick={() => onChange(item.label)}
            key={item.label}
          >
            <span>{item.label}</span>
            <strong>R {formatSignedMetric(item.delta.recall_iou50)}</strong>
            <em>
              TP {formatSignedInteger(item.delta.matched_count)} · FP{" "}
              {formatSignedInteger(item.delta.false_positive_count)} · FN{" "}
              {formatSignedInteger(item.delta.false_negative_count)}
            </em>
          </button>
        );
      })}
    </div>
  );
}

function filterComparisonSamplesByLabel(samples: ComparisonSample[], label: string) {
  if (label === "all") {
    return samples;
  }
  return samples.filter((sample) => Boolean(sample.labels?.[label]));
}

function firstComparableSample(samples: ComparisonSample[]) {
  return (
    samples.find((sample) => sample.candidate_index !== null || sample.sample_index !== null) ?? null
  );
}

function ComparisonSampleTable({
  title,
  samples,
  baselineRunId,
  candidateRunId,
  tone
}: {
  title: string;
  samples: ComparisonSample[];
  baselineRunId: string;
  candidateRunId: string;
  tone: "positive" | "negative";
}) {
  return (
    <div className={`comparison-sample-block ${tone}`}>
      <div className="comparison-sample-title">{title}</div>
      {samples.length === 0 ? (
        <div className="comparison-sample-empty">没有变化样本。</div>
      ) : (
        <div className="comparison-sample-list">
          {samples.map((sample) => {
            const index = sample.candidate_index ?? sample.sample_index;
            const name = basename(sample.image ?? sample.key);
            const sampleLabels = Object.keys(sample.labels ?? {}).slice(0, 4);
            const content = (
              <>
                <span className="comparison-sample-row-head">
                  <strong title={sample.image ?? sample.key}>{name}</strong>
                  <em>{index === null ? "未对齐" : `#${index + 1}`}</em>
                  <span>{sample.status}</span>
                </span>
                {sampleLabels.length > 0 ? (
                  <span className="comparison-sample-labels">
                    {sampleLabels.map((label) => (
                      <em key={label}>{label}</em>
                    ))}
                  </span>
                ) : null}
                <span className="comparison-sample-metrics">
                  <MetricDelta label="Score" value={sample.delta_score} />
                  <MetricDelta label="TP" value={sample.delta.matched_count} integer />
                  <MetricDelta
                    label="FP"
                    value={sample.delta.false_positive_count}
                    integer
                    inverted
                  />
                  <MetricDelta
                    label="FN"
                    value={sample.delta.false_negative_count}
                    integer
                    inverted
                  />
                  <MetricDelta label="IoU" value={sample.delta.mean_iou} />
                </span>
              </>
            );
            if (index === null) {
              return (
                <div className="comparison-sample-row disabled" key={sample.key}>
                  {content}
                </div>
              );
            }
            return (
              <a
                className="comparison-sample-row"
                href={comparisonSampleHref(baselineRunId, candidateRunId, index)}
                key={sample.key}
              >
                {content}
                <Eye size={14} />
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MetricDelta({
  label,
  value,
  integer,
  inverted
}: {
  label: string;
  value: number;
  integer?: boolean;
  inverted?: boolean;
}) {
  const positive = value > 0;
  const negative = value < 0;
  const good = inverted ? negative : positive;
  const bad = inverted ? positive : negative;
  const className = good ? "metric-delta positive" : bad ? "metric-delta negative" : "metric-delta";
  return (
    <span className={className}>
      <em>{label}</em>
      <strong>{integer ? formatSignedInteger(value) : formatCompactSignedMetric(value)}</strong>
    </span>
  );
}

function ComparisonSamplePage() {
  const { baselineRunId, candidateRunId, sampleIndex } = useParams({
    from: "/compare/$baselineRunId/$candidateRunId/$sampleIndex"
  });
  const numericIndex = Number(sampleIndex);
  const validIndex = Number.isInteger(numericIndex) && numericIndex >= 0;
  const query = useQuery({
    queryKey: ["comparison-sample", baselineRunId, candidateRunId, numericIndex],
    queryFn: () => fetchComparisonSample(baselineRunId, candidateRunId, numericIndex),
    enabled: validIndex
  });

  if (!validIndex) {
    return <EmptyState title="样本序号无效" tone="danger" />;
  }
  if (query.isLoading) {
    return <EmptyState title="正在加载对比样本" />;
  }
  if (query.isError || !query.data) {
    return <EmptyState title="对比样本加载失败" tone="danger" />;
  }

  return (
    <section className="page-stack comparison-sample-page">
      <div className="compare-topbar">
        <div className="compare-title">
          <span>样本对比</span>
          <strong>#{numericIndex + 1}</strong>
        </div>
        <div className="compare-chip-strip">
          <span className="sample-count-chip">{baselineRunId}</span>
          <span className="sample-count-chip">{candidateRunId}</span>
        </div>
      </div>
      <ComparisonSampleViewer detail={query.data} />
    </section>
  );
}

function ComparisonSampleViewer({ detail }: { detail: ComparisonSampleDetail }) {
  return (
    <ResizableSplit
      className="comparison-sample-detail"
      storageKey="eval_bench_comparison_sample_candidate_width"
      fixedPane="second"
      defaultSize={520}
      minSize={280}
      maxSize={1180}
      first={
        <ComparisonRunPanel
          title="基线"
          runId={detail.baseline_run_id}
          detail={detail.baseline}
        />
      }
      second={
        <ComparisonRunPanel
          title="候选"
          runId={detail.candidate_run_id}
          detail={detail.candidate}
        />
      }
    />
  );
}

function ComparisonRunPanel({
  title,
  runId,
  detail
}: {
  title: string;
  runId: string;
  detail: RunSampleDetail;
}) {
  return (
    <div className="comparison-run-panel">
      <div className="comparison-run-heading">
        <div>
          <div className="eyebrow">{title}</div>
          <h2>{runId}</h2>
        </div>
        <a className="mini-link" href={runSampleHref(runId, detail.sample.index)}>
          <Eye size={13} />
          打开 run
        </a>
      </div>
      <SampleViewer detail={detail} />
    </div>
  );
}

function ComparisonHistoryPanel({ comparisons }: { comparisons: ComparisonSummary[] }) {
  if (comparisons.length === 0) {
    return null;
  }
  const columns: ColumnDef<ComparisonSummary>[] = [
    { header: "对比记录", accessorKey: "comparison_id" },
    { header: "任务", accessorKey: "task" },
    { header: "样本数", cell: ({ row }) => row.original.sample_count.toLocaleString() },
    { header: "Delta R", cell: ({ row }) => formatSignedMetric(row.original.delta.recall_iou50) },
    { header: "提升", cell: ({ row }) => row.original.summary.improved_samples.toLocaleString() },
    { header: "退化", cell: ({ row }) => row.original.summary.regressed_samples.toLocaleString() },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) }
  ];
  return (
    <div className="history-block">
      <div className="comparison-sample-title">历史对比</div>
      <DataTable
        columns={columns}
        data={comparisons.slice(0, 8)}
        emptyText="暂无历史对比。"
        compact
      />
    </div>
  );
}

function LeaderboardTable({ runs }: { runs: RunSummary[] }) {
  const columns: ColumnDef<RunSummary>[] = [
    {
      header: "排名",
      cell: ({ row }) => row.index + 1
    },
    {
      header: "记录",
      cell: ({ row }) => (
        <Link to="/runs/$runId" params={{ runId: row.original.run_id }}>
          {row.original.run_id}
        </Link>
      )
    },
    { header: "模型", accessorKey: "model_id" },
    { header: "任务", accessorKey: "spec_task" },
    { header: "基准集", accessorKey: "benchmark_id" },
    { header: "P@.50", cell: ({ row }) => formatMetric(row.original.precision_iou50) },
    { header: "R@.50", cell: ({ row }) => formatMetric(row.original.recall_iou50) },
    { header: "平均 IoU", cell: ({ row }) => formatMetric(row.original.mean_iou) },
    {
      header: "预测数",
      cell: ({ row }) => row.original.prediction_count.toLocaleString()
    }
  ];
  return <DataTable columns={columns} data={runs} emptyText="没有符合过滤条件的 run。" />;
}

function JobQueuePanel({ compact = false }: { compact?: boolean }) {
  const queryClient = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: 2_000
  });
  const schedulerQuery = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  const runningJobs = data?.jobs.filter((job) => job.status === "running") ?? [];
  const selectedJob = data?.jobs.find((job) => job.job_id === selectedJobId) ?? null;
  const selectedRuntimeLogPath =
    selectedJob && typeof selectedJob.metadata.runtime_log_path === "string"
      ? selectedJob.metadata.runtime_log_path
      : "";
  const jobLogsQuery = useQuery({
    queryKey: ["job-logs", selectedJob?.job_id ?? ""],
    queryFn: () => fetchJobLogs(selectedJob?.job_id ?? "", 0),
    enabled: Boolean(selectedJob?.job_id && selectedRuntimeLogPath),
    refetchInterval: selectedJob?.status === "running" ? 3_000 : false
  });
  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: deleteJob,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
    }
  });
  if (isLoading) {
    return <div className="empty-panel">正在加载队列状态</div>;
  }
  if (error || !data) {
    return <div className="empty-panel danger-text">队列状态加载失败</div>;
  }
  return (
    <div className={compact ? "queue-stack compact" : "queue-stack"}>
      <SchedulerStrip
        jobs={data.jobs}
        scheduler={schedulerQuery.data ?? { enabled: false }}
      />
      {data.jobs.length === 0 ? (
        <div className="empty-panel">当前没有任务。</div>
      ) : (
        <div className={compact ? "table-shell compact" : "table-shell"}>
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>类型</th>
                <th>状态</th>
                <th>目标</th>
                <th>创建时间</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.jobs.map((job) => (
                <tr
                  key={job.job_id}
                  className={job.job_id === selectedJob?.job_id ? "selectable-row selected" : "selectable-row"}
                  onClick={() => setSelectedJobId(job.job_id)}
                >
                  <td>{job.job_id}</td>
                  <td>{job.kind}</td>
                  <td>
                    <Badge value={job.status} />
                    <JobProgressInline job={job} />
                  </td>
                  <td>
                    <div className="job-target-cell">
                      <span>{jobTarget(job.payload)}</span>
                      {job.error ? <em title={job.error}>{job.error}</em> : null}
                      {typeof job.metadata.runtime_log_path === "string" ? (
                        <small title={job.metadata.runtime_log_path}>
                          runtime log: {basename(job.metadata.runtime_log_path)}
                        </small>
                      ) : null}
                    </div>
                  </td>
                  <td>{formatDate(job.created_at)}</td>
                  <td>
                    <div className="row-actions">
                      <button
                        className="icon-button dense"
                        type="button"
                        disabled={job.status !== "queued" || cancelMutation.isPending}
                        title="取消排队任务"
                        onClick={(event) => {
                          event.stopPropagation();
                          cancelMutation.mutate(job.job_id);
                        }}
                      >
                        <X size={14} />
                      </button>
                      <button
                        className="icon-button dense danger"
                        type="button"
                        disabled={deleteMutation.isPending}
                        title="删除任务记录"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (confirm(`删除任务记录 ${job.job_id}？`)) {
                            deleteMutation.mutate(job.job_id);
                          }
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {selectedJob ? <JobDetailPanel job={selectedJob} logs={jobLogsQuery.data ?? null} /> : null}
    </div>
  );
}

function SchedulerStrip({
  jobs,
  scheduler
}: {
  jobs: JobSummary[];
  scheduler: SchedulerStatus;
}) {
  const queued = jobs.filter((job) => job.status === "queued").length;
  const running = jobs.filter((job) => job.status === "running").length;
  const failed = jobs.filter((job) => job.status === "failed").length;
  const reservedDevices = scheduler.reserved_cuda_devices ?? [];
  const reservedPorts = scheduler.reserved_runtime_ports ?? [];
  return (
    <div className="scheduler-strip">
      <div>
        <span className={scheduler.enabled ? "status-dot live" : "status-dot"} />
        <strong>{scheduler.enabled ? "自动调度运行中" : "自动调度未启用"}</strong>
      </div>
      <span>运行 {running}</span>
      <span>排队 {queued}</span>
      {failed > 0 ? <span className="danger-text">失败 {failed}</span> : null}
      <span>并发上限 {scheduler.max_concurrent_jobs ?? "-"}</span>
      {reservedDevices.length > 0 ? <span>占用 CUDA {reservedDevices.join(",")}</span> : null}
      {reservedPorts.length > 0 ? <span>占用端口 {reservedPorts.join(",")}</span> : null}
    </div>
  );
}

function JobDetailPanel({ job, logs }: { job: JobSummary; logs: JobLog | null }) {
  const progress = jobProgress(job);
  const lines = logs?.lines ?? [];
  const linkedRunId = stringValue(job.metadata.run_id);
  return (
    <div className="job-detail-panel">
      <div className="job-monitor-header">
        <div>
          <div className="eyebrow">任务详情</div>
          <strong>{job.job_id}</strong>
        </div>
        <div className="job-monitor-actions">
          {linkedRunId ? (
            <Link className="mini-link" to="/runs/$runId" params={{ runId: linkedRunId }}>
              打开结果
            </Link>
          ) : null}
          <Badge value={job.status} />
        </div>
      </div>
      <div className="job-progress-row">
        <div className="job-progress-track" aria-label="任务进度">
          <span style={{ width: `${progress.percent ?? 8}%` }} />
        </div>
        <span>{progress.text}</span>
      </div>
      <div className="job-monitor-meta">
        <span>{progressPhaseText(progress.phase)}</span>
        {progress.message ? <span>{progress.message}</span> : null}
        {progress.currentSample ? <span title={progress.currentSample}>{progress.currentSample}</span> : null}
      </div>
      <div className="job-detail-grid">
        <span>目标</span>
        <strong>{jobTarget(job.payload)}</strong>
        <span>创建</span>
        <strong>{formatDate(job.created_at)}</strong>
        <span>更新</span>
        <strong>{formatDate(job.updated_at)}</strong>
        <span>日志</span>
        <strong>
          {typeof job.metadata.runtime_log_path === "string"
            ? job.metadata.runtime_log_path
            : "runtime log 尚未创建"}
        </strong>
      </div>
      {lines.length > 0 ? (
        <pre className="job-log-tail">{lines.join("")}</pre>
      ) : (
        <div className="job-log-empty">
          {logs?.log_path ? "runtime log 还没有新内容。" : "等待 runtime log。"}
        </div>
      )}
    </div>
  );
}

function JobProgressInline({ job }: { job: JobSummary }) {
  if (job.status !== "running" && job.status !== "failed" && job.status !== "succeeded") {
    return null;
  }
  const progress = jobProgress(job);
  return (
    <div className="job-progress-inline">
      <div className="job-progress-mini">
        <span style={{ width: `${progress.percent ?? (job.status === "succeeded" ? 100 : 0)}%` }} />
      </div>
      <small>{progress.text}</small>
    </div>
  );
}

function jobProgress(job: JobSummary) {
  const metadata = job.metadata ?? {};
  const done = metadataNumber(metadata.progress_done);
  const total = metadataNumber(metadata.progress_total);
  const phase = typeof metadata.progress_phase === "string" ? metadata.progress_phase : job.status;
  const message = typeof metadata.progress_message === "string" ? metadata.progress_message : "";
  const currentSample =
    typeof metadata.progress_current_sample === "string" ? metadata.progress_current_sample : "";
  const percent =
    total && total > 0 && done !== null
      ? Math.max(0, Math.min(100, Math.round((done / total) * 100)))
      : job.status === "succeeded"
        ? 100
        : null;
  const text =
    total && total > 0 && done !== null
      ? `${done}/${total} (${percent}%)`
      : progressPhaseText(phase);
  return { currentSample, done, message, percent, phase, text, total };
}

function metadataNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function progressPhaseText(value: string) {
  const labels: Record<string, string> = {
    resolving: "解析配置",
    worker_starting: "启动后台 worker",
    starting_runtime: "启动模型服务",
    runtime_ready: "模型服务就绪",
    prepare_run: "准备 run",
    inference: "推理中",
    evaluating: "计算指标",
    succeeded: "完成",
    failed: "失败",
    running: "运行中",
    queued: "排队中"
  };
  return labels[value] ?? value;
}

function JobCreatePanel({ benchmarks }: { benchmarks: BenchmarkSummary[] }) {
  const queryClient = useQueryClient();
  const templatesQuery = useQuery({ queryKey: ["job-templates"], queryFn: fetchJobTemplates });
  const promptTemplatesQuery = useQuery({
    queryKey: ["prompt-templates"],
    queryFn: fetchPromptTemplates
  });
  const templates = templatesQuery.data?.templates ?? {};
  const promptTemplates = promptTemplatesQuery.data?.templates ?? [];
  const templateIds = Object.keys(templates);
  const promptIds = promptTemplates.map((template) => template.prompt_id);
  const [templateId, setTemplateId] = useState("eval_job");
  const [promptId, setPromptId] = useState("grounding_layout.latest");
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

  function loadTemplate(nextTemplateId = templateId) {
    const template = templates[nextTemplateId];
    if (!template) {
      return;
    }
    setTemplateId(nextTemplateId);
    setManifestText(formatManifest(applyBenchmarkDefault(template.manifest, benchmarks)));
    setParseError(null);
    preflightMutation.reset();
  }

  function applySelectedPrompt(nextPromptId = promptId) {
    const promptTemplate =
      promptTemplates.find((template) => template.prompt_id === nextPromptId) ?? selectedPrompt;
    if (!promptTemplate) {
      return;
    }
    const manifest = parseManifest() ?? applyBenchmarkDefault(selectedTemplate?.manifest ?? {}, benchmarks);
    setPromptId(promptTemplate.prompt_id);
    setManifestText(formatManifest(applyPromptTemplateToManifest(manifest, promptTemplate)));
    setParseError(null);
    preflightMutation.reset();
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
      setParseError(error instanceof Error ? error.message : String(error));
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

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const manifest = parseManifest();
    if (!manifest) {
      return;
    }
    mutation.mutate({ manifest });
  }

  return (
    <div className="workspace-card manifest-card">
      <PanelTitle title="新建评测任务" meta="模板 manifest + 后端预检查" />
      <form className="manifest-job-form" onSubmit={submit}>
        <div className="manifest-toolbar">
          <label className="filter-select compact">
            <span>模板</span>
            <select
              value={templateId}
              onChange={(event) => loadTemplate(event.target.value)}
              disabled={templatesQuery.isLoading}
            >
              {templateIds.length === 0 ? <option value="eval_job">加载中</option> : null}
              {templateIds.map((id) => (
                <option key={id} value={id}>
                  {templates[id]?.label ?? id}
                </option>
              ))}
            </select>
          </label>
          <label className="filter-select compact">
            <span>Prompt</span>
            <select
              value={selectedPrompt?.prompt_id ?? promptId}
              onChange={(event) => applySelectedPrompt(event.target.value)}
              disabled={promptTemplatesQuery.isLoading || promptTemplates.length === 0}
            >
              {promptTemplates.length === 0 ? <option value={promptId}>加载中</option> : null}
              {promptTemplates.map((template) => (
                <option key={template.prompt_id} value={template.prompt_id}>
                  {template.label || template.prompt_id}
                </option>
              ))}
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={() => loadTemplate()}>
            恢复模板
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => applySelectedPrompt()}
            disabled={!selectedPrompt}
          >
            应用 Prompt
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={validateManifest}
            disabled={preflightMutation.isPending}
          >
            {preflightMutation.isPending ? "检查中" : "预检查"}
          </button>
          <button className="primary-button" type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? "加入中" : "加入队列"}
          </button>
        </div>
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
              <label className="manifest-editor-field">
                <span>可编辑任务 Manifest</span>
                <textarea
                  spellCheck={false}
                  value={manifestText}
                  onChange={(event) => {
                    setManifestText(event.target.value);
                    setParseError(null);
                    preflightMutation.reset();
                  }}
                />
              </label>
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
                  saveError={promptMutation.isError}
                />
              ) : null}
              {parseError ? <div className="form-error">JSON 解析错误：{parseError}</div> : null}
              {preflightMutation.data ? <PreflightPanel result={preflightMutation.data} /> : null}
              {preflightMutation.isError ? (
                <div className="form-error">预检查请求失败。</div>
              ) : null}
              {mutation.isError ? <div className="form-error">任务入队失败。</div> : null}
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

function PreflightPanel({ result }: { result: { ok: boolean; errors: string[]; warnings: string[]; runtime_command?: string[] | null } }) {
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

function PromptTemplatePanel({
  prompt,
  onSaveFromManifest,
  saving,
  saveError
}: {
  prompt: PromptTemplate;
  onSaveFromManifest: () => void;
  saving: boolean;
  saveError: boolean;
}) {
  const targetLabels = targetLabelsFromPrompt(prompt);
  return (
    <details className="prompt-template-panel" open>
      <summary>
        <span>{prompt.label || prompt.prompt_id}</span>
        <Badge value={prompt.task} />
      </summary>
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
      <button
        className="secondary-button dense"
        type="button"
        onClick={onSaveFromManifest}
        disabled={saving}
      >
        {saving ? "保存中" : "将当前 Manifest 的 Prompt 保存为模板"}
      </button>
      {saveError ? <div className="form-error">Prompt 模板保存失败。</div> : null}
    </details>
  );
}

function BenchmarkTable({
  benchmarks,
  compact = false
}: {
  benchmarks: BenchmarkSummary[];
  compact?: boolean;
}) {
  const columns: ColumnDef<BenchmarkSummary>[] = [
    {
      header: "基准集",
      cell: ({ row }) => (
        <Link to="/benchmarks/$benchmarkId" params={{ benchmarkId: row.original.benchmark_id }}>
          {row.original.benchmark_id}
        </Link>
      )
    },
    { header: "任务", cell: ({ row }) => row.original.tasks.join(", ") || "-" },
    { header: "标注层", cell: ({ row }) => row.original.layers.join(", ") || "-" },
    { header: "Split", accessorKey: "split" },
    {
      header: "样本数",
      accessorKey: "sample_count",
      cell: ({ row }) => row.original.sample_count.toLocaleString()
    },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) },
    {
      header: "",
      id: "actions",
      cell: ({ row }) => (
        <Link
          className="mini-link"
          to="/benchmarks/$benchmarkId"
          params={{ benchmarkId: row.original.benchmark_id }}
          title="检查基准集真值样本"
        >
          <Eye size={13} />
          检查
        </Link>
      )
    }
  ];
  return (
    <DataTable
      columns={columns}
      data={benchmarks}
      emptyText="还没有登记基准集。"
      compact={compact}
    />
  );
}

function RunTable({ runs, compact = false }: { runs: RunSummary[]; compact?: boolean }) {
  const queryClient = useQueryClient();
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const evaluateMutation = useMutation({
    mutationFn: evaluateRun,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const archiveMutation = useMutation({
    mutationFn: archiveRun,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: deleteRun,
    onSuccess: () => {
      setSelectedRunIds([]);
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
    }
  });
  const statuses = unique(runs.map((run) => run.status).filter(Boolean));
  const tasks = unique(runs.map((run) => run.spec_task).filter(Boolean));
  const benchmarks = unique(runs.map((run) => run.benchmark_id).filter(Boolean));
  const filteredRuns = compact
    ? runs
    : runs
        .filter((run) => statusFilter === "all" || run.status === statusFilter)
        .filter((run) => taskFilter === "all" || run.spec_task === taskFilter)
        .filter((run) => benchmarkFilter === "all" || run.benchmark_id === benchmarkFilter)
        .filter((run) => {
          const query = searchText.trim().toLowerCase();
          if (!query) {
            return true;
          }
          return [
            run.run_id,
            run.model_id,
            run.benchmark_id,
            run.spec_task,
            run.prompt_id
          ].some((value) => String(value).toLowerCase().includes(query));
        });
  const comparableSelection = selectedRunIds.slice(0, 2);
  const compareHref =
    comparableSelection.length === 2
      ? `/compare?baseline=${encodeURIComponent(comparableSelection[0])}&candidate=${encodeURIComponent(
          comparableSelection[1]
        )}`
      : "/compare";
  const columns: ColumnDef<RunSummary>[] = [
    ...(compact
      ? []
      : [
          {
            header: "",
            id: "select",
            cell: ({ row }) => (
              <input
                className="row-select-checkbox"
                aria-label={`选择 ${row.original.run_id} 进行对比`}
                type="checkbox"
                checked={selectedRunIds.includes(row.original.run_id)}
                onChange={() => toggleRunSelection(row.original.run_id)}
              />
            )
          } satisfies ColumnDef<RunSummary>
        ]),
    {
      header: "记录",
      cell: ({ row }) => (
        <Link to="/runs/$runId" params={{ runId: row.original.run_id }}>
          {row.original.run_id}
        </Link>
      )
    },
    { header: "状态", cell: ({ row }) => <Badge value={row.original.status} /> },
    { header: "任务", accessorKey: "spec_task" },
    { header: "基准集", accessorKey: "benchmark_id" },
    { header: "模型", accessorKey: "model_id" },
    {
      header: "预测数",
      accessorKey: "prediction_count",
      cell: ({ row }) => row.original.prediction_count.toLocaleString()
    },
    { header: "P@.50", cell: ({ row }) => formatMetric(row.original.precision_iou50) },
    { header: "R@.50", cell: ({ row }) => formatMetric(row.original.recall_iou50) },
    { header: "报告数", accessorKey: "report_count" },
    { header: "创建时间", cell: ({ row }) => formatDate(row.original.created_at) },
    {
      header: "",
      id: "actions",
      cell: ({ row }) => (
        <div className="row-actions">
          <Link
            className="icon-button dense"
            to="/runs/$runId"
            params={{ runId: row.original.run_id }}
            title="检查样本级预测"
          >
            <Eye size={13} />
          </Link>
          <button
            className="icon-button dense"
            type="button"
            onClick={() => evaluateMutation.mutate(row.original.run_id)}
            disabled={evaluateMutation.isPending}
            title="计算预测指标"
          >
            <RotateCw size={13} />
          </button>
          {!compact ? (
            <>
              <button
                className="icon-button dense"
                type="button"
                onClick={() => archiveMutation.mutate(row.original.run_id)}
                disabled={archiveMutation.isPending || row.original.status === "archived"}
                title="归档 run"
              >
                <Archive size={14} />
              </button>
              <button
                className="icon-button dense danger"
                type="button"
                onClick={() => {
                  if (confirm(`将 run ${row.original.run_id} 移入回收站？`)) {
                    deleteMutation.mutate(row.original.run_id);
                  }
                }}
                disabled={deleteMutation.isPending}
                title="删除 run"
              >
                <Trash2 size={14} />
              </button>
            </>
          ) : null}
        </div>
      )
    }
  ];
  function toggleRunSelection(runId: string) {
    setSelectedRunIds((current) => {
      if (current.includes(runId)) {
        return current.filter((item) => item !== runId);
      }
      return [...current, runId].slice(-2);
    });
  }
  return (
    <div className={compact ? "run-table-stack compact" : "run-table-stack"}>
      {!compact ? (
        <div className="run-query-bar">
          <label className="search-box">
            <Search size={15} />
            <input
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder="搜索 run、模型、基准集"
            />
          </label>
          <FilterSelect
            label="状态"
            value={statusFilter}
            values={["all", ...statuses]}
            labels={{ all: "全部" }}
            onChange={setStatusFilter}
            compact
          />
          <FilterSelect
            label="任务"
            value={taskFilter}
            values={["all", ...tasks]}
            labels={{ all: "全部" }}
            onChange={setTaskFilter}
            compact
          />
          <FilterSelect
            label="基准集"
            value={benchmarkFilter}
            values={["all", ...benchmarks]}
            labels={{ all: "全部" }}
            onChange={setBenchmarkFilter}
            compact
          />
          <a
            className={
              comparableSelection.length === 2 ? "mini-link compare-ready" : "mini-link disabled"
            }
            href={compareHref}
          >
            <GitCompare size={13} />
            对比 {comparableSelection.length}/2
          </a>
        </div>
      ) : null}
      <DataTable
        columns={columns}
        data={filteredRuns}
        emptyText="还没有评测记录。"
        compact={compact}
      />
    </div>
  );
}

function objectKindLabel(kind: ObjectKind) {
  return kind === "gt" ? "真值" : "预测";
}

function formatDate(value: string | null) {
  if (!value) {
    return "-";
  }
  return value.replace("T", " ").replace("Z", "");
}

function formatMetric(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(3);
}

function jobTarget(payload: Record<string, unknown>) {
  const model = typeof payload.model_id === "string" ? payload.model_id : "model";
  const benchmark =
    typeof payload.benchmark_id === "string" ? payload.benchmark_id : "benchmark";
  const task = typeof payload.task === "string" ? payload.task : "task";
  return `${model} / ${benchmark} / ${task}`;
}

function basename(path: string) {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function unique(values: string[]) {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function isTextInputTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
}

function scoreRun(run: RunSummary) {
  const precision = run.precision_iou50 ?? 0;
  const recall = run.recall_iou50 ?? 0;
  const meanIou = run.mean_iou ?? 0;
  return precision * 0.45 + recall * 0.45 + meanIou * 0.1;
}

function formatRunOption(run: RunSummary) {
  return `${run.run_id} / ${run.model_id} / R ${formatMetric(run.recall_iou50)}`;
}

function runIdExists(runs: RunSummary[], runId: string) {
  return runs.some((run) => run.run_id === runId);
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function inferenceValue(inference: Record<string, unknown>, key: string) {
  const value = inference[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function serviceConfigValue(service: ServiceSummary, key: string) {
  const value = service.config[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function runtimeValue(service: ServiceSummary, key: string) {
  const value = service.runtime[key];
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function serviceHealth(service: ServiceSummary) {
  const health = service.runtime.health;
  if (!health || typeof health !== "object" || Array.isArray(health)) {
    return {
      ok: false,
      status: "unchecked",
      message: "health has not been checked",
      checkedAt: "-"
    };
  }
  const payload = health as Record<string, unknown>;
  return {
    ok: payload.ok === true,
    status: stringValue(payload.status) || "unchecked",
    message: stringValue(payload.message) || "-",
    checkedAt: formatDate(stringValue(payload.checked_at) || null)
  };
}

function serviceEndpointValue(service: ServiceSummary) {
  const configured = serviceConfigValue(service, "endpoint");
  if (configured !== "-") {
    return configured;
  }
  const runtime = runtimeValue(service, "endpoint");
  if (runtime !== "-") {
    return runtime;
  }
  const host = serviceConfigValue(service, "host");
  const port = serviceConfigValue(service, "port");
  return port === "-" ? "-" : `http://${host === "-" ? "127.0.0.1" : host}:${port}`;
}

function pixelBudgetValue(inference: Record<string, unknown>) {
  const minPixels = inferenceValue(inference, "min_pixels");
  const maxPixels = inferenceValue(inference, "max_pixels");
  if (minPixels === "-" && maxPixels === "-") {
    return "-";
  }
  return `${minPixels} / ${maxPixels}`;
}

function samplingValue(inference: Record<string, unknown>) {
  return `T ${inferenceValue(inference, "temperature")} / top_p ${inferenceValue(inference, "top_p")}`;
}

function resolveInstanceColor(
  label: string,
  status: "match" | "neutral" | "fn" | "fp",
  kind: ObjectKind,
  overlayColors: OverlayColors,
  labelColors: LabelColors
) {
  if (status === "fn") {
    return overlayColors.fn;
  }
  if (status === "fp") {
    return overlayColors.fp;
  }
  return labelColors[label] ?? overlayColors[kind] ?? fallbackLabelColor(label);
}

function arrowHeadPoints(points: number[][], lineWidth: number, scale = 1): number[][] | null {
  if (points.length < 2) {
    return null;
  }
  const segments = points
    .slice(0, -1)
    .map((start, index) => {
      const end = points[index + 1];
      return { start, end, length: Math.hypot(end[0] - start[0], end[1] - start[1]) };
    })
    .filter((segment) => segment.length > 1);
  if (segments.length === 0) {
    return null;
  }
  const totalLength = segments.reduce((total, segment) => total + segment.length, 0);
  const target = totalLength * 0.5;
  let accumulated = 0;
  let selected = segments[Math.floor(segments.length / 2)];
  for (const segment of segments) {
    if (accumulated + segment.length >= target) {
      selected = segment;
      break;
    }
    accumulated += segment.length;
  }
  const [x1, y1] = selected.start;
  const [x2, y2] = selected.end;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = selected.length;
  const baseSize = Math.max(6, Math.min(18, lineWidth * 2.4)) * scale;
  const size = Math.min(baseSize, length * 0.22);
  if (length < size * 1.8) {
    return null;
  }
  const segmentOffset = Math.max(0, target - accumulated);
  const localRatio = clampNumber(segmentOffset / length, 0.32, 0.68);
  const unitX = dx / length;
  const unitY = dy / length;
  const tipX = x1 + dx * localRatio;
  const tipY = y1 + dy * localRatio;
  const baseX = tipX - unitX * size;
  const baseY = tipY - unitY * size;
  const wing = size * 0.45;
  return [
    [tipX, tipY],
    [baseX - unitY * wing, baseY + unitX * wing],
    [baseX + unitY * wing, baseY - unitX * wing]
  ];
}

function preloadSampleImages(
  samples: Array<Pick<RunSampleSummary, "index" | "image_url"> | Pick<BenchmarkSampleSummary, "index" | "image_url">>,
  selectedIndex: number
) {
  const nearby = samples.filter(
    (sample) => Math.abs(sample.index - selectedIndex) <= PRELOAD_RADIUS && sample.image_url
  );
  for (const sample of nearby) {
    const image = new Image();
    image.decoding = "async";
    image.src = sample.image_url;
  }
}

function computeFitSize(
  width: number,
  height: number,
  stageSize: { width: number; height: number }
) {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const availableWidth = Math.max(1, stageSize.width - CANVAS_FIT_PADDING * 2);
  const availableHeight = Math.max(1, stageSize.height - CANVAS_FIT_PADDING * 2);
  const scale = Math.min(availableWidth / safeWidth, availableHeight / safeHeight);
  return {
    width: Math.max(1, Math.floor(safeWidth * scale)),
    height: Math.max(1, Math.floor(safeHeight * scale))
  };
}

function normalizeBbox(value: unknown): [number, number, number, number] | null {
  if (Array.isArray(value) && value.length >= 4 && value.slice(0, 4).every(isFiniteNumber)) {
    const [x1, y1, x2, y2] = value.slice(0, 4) as number[];
    return normalizeBoxNumbers(x1, y1, x2, y2);
  }
  if (
    Array.isArray(value) &&
    value.length >= 2 &&
    Array.isArray(value[0]) &&
    Array.isArray(value[1]) &&
    value[0].length >= 2 &&
    value[1].length >= 2 &&
    [value[0][0], value[0][1], value[1][0], value[1][1]].every(isFiniteNumber)
  ) {
    return normalizeBoxNumbers(value[0][0], value[0][1], value[1][0], value[1][1]);
  }
  return null;
}

function normalizeBoxNumbers(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): [number, number, number, number] | null {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const right = Math.max(x1, x2);
  const bottom = Math.max(y1, y2);
  if (right <= left || bottom <= top) {
    return null;
  }
  return [left, top, right, bottom];
}

function normalizePointList(value: unknown): number[][] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const points = value
    .filter((point): point is [number, number] => {
      return (
        Array.isArray(point) &&
        point.length >= 2 &&
        isFiniteNumber(point[0]) &&
        isFiniteNumber(point[1])
      );
    })
    .map((point) => [point[0], point[1]]);
  return points.length > 0 ? points : null;
}

function boundsFromPoints(points: number[][] | null): [number, number, number, number] | null {
  if (!points || points.length === 0) {
    return null;
  }
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function clampPan(
  pan: { x: number; y: number },
  zoom: number,
  stage: HTMLDivElement | null,
  content: HTMLDivElement | null
) {
  if (!stage || !content) {
    return { x: 0, y: 0 };
  }
  const viewportWidth = stage.clientWidth;
  const viewportHeight = stage.clientHeight;
  const contentWidth = content.offsetWidth * zoom;
  const contentHeight = content.offsetHeight * zoom;
  const maxX = Math.max(0, Math.abs(contentWidth - viewportWidth) / 2);
  const maxY = Math.max(0, Math.abs(contentHeight - viewportHeight) / 2);
  return {
    x: clampNumber(pan.x, -maxX, maxX),
    y: clampNumber(pan.y, -maxY, maxY)
  };
}

function formatSignedMetric(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(3)}`;
}

function formatCompactSignedMetric(value: number) {
  const prefix = value > 0 ? "+" : "";
  const absValue = Math.abs(value);
  const digits = absValue >= 100 ? 0 : absValue >= 10 ? 1 : 2;
  return `${prefix}${value.toFixed(digits)}`;
}

function formatSignedInteger(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toLocaleString()}`;
}

function sampleIndexFromLocation() {
  if (typeof window === "undefined") {
    return 0;
  }
  const value = new URLSearchParams(window.location.search).get("sample");
  const index = Number(value);
  return Number.isInteger(index) && index >= 0 ? index : 0;
}

function samplePageOffsetFromLocation(pageSize: number) {
  const index = sampleIndexFromLocation();
  return Math.floor(index / pageSize) * pageSize;
}

function updateSampleIndexInLocation(index: number) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("sample", String(index));
  window.history.replaceState(null, "", url);
}

function runSampleHref(runId: string, sampleIndex: number) {
  return `/runs/${encodeURIComponent(runId)}?sample=${sampleIndex}`;
}

function comparisonSampleHref(baselineRunId: string, candidateRunId: string, sampleIndex: number) {
  return `/compare/${encodeURIComponent(baselineRunId)}/${encodeURIComponent(
    candidateRunId
  )}/${sampleIndex}`;
}

function formatManifest(value: unknown) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function applyBenchmarkDefault(
  manifest: Record<string, unknown>,
  benchmarks: BenchmarkSummary[]
): Record<string, unknown> {
  const cloned = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const benchmarkIds = benchmarks.map((benchmark) => benchmark.benchmark_id);
  if (benchmarkIds.length === 0 || !isRecord(cloned.eval)) {
    return cloned;
  }
  const currentBenchmarkId = cloned.eval.benchmark_id;
  if (typeof currentBenchmarkId !== "string" || !benchmarkIds.includes(currentBenchmarkId)) {
    cloned.eval.benchmark_id = benchmarkIds[0];
  }
  return cloned;
}

function applyPromptTemplateToManifest(
  manifest: Record<string, unknown>,
  prompt: PromptTemplate
): Record<string, unknown> {
  const cloned = JSON.parse(JSON.stringify(manifest)) as Record<string, unknown>;
  const section = manifestPromptSection(cloned);
  if (!section) {
    cloned.eval = {};
    return applyPromptTemplateToManifest(cloned, prompt);
  }
  section.prompt_id = prompt.prompt_id;
  if (prompt.task) {
    section.task = prompt.task;
  }
  section.system_prompt = prompt.system_prompt;
  section.prompt_text = prompt.user_prompt;
  if (prompt.parser) {
    section.parser = prompt.parser;
  }
  if (prompt.metric_profile) {
    section.metric_profile = prompt.metric_profile;
  }
  if (prompt.visualization_profile) {
    section.visualization_profile = prompt.visualization_profile;
  }
  const targetLabels = targetLabelsFromPrompt(prompt);
  if (targetLabels.length > 0) {
    section.target_labels = targetLabels;
  }
  section.generation = mergeRecordDefaults(section.generation, prompt.generation);
  section.data = mergeRecordDefaults(section.data, prompt.data);
  section.prompt_template = {
    prompt_id: prompt.prompt_id,
    label: prompt.label,
    task: prompt.task
  };
  return cloned;
}

function promptTemplateFromManifest(
  manifest: Record<string, unknown>,
  fallback?: PromptTemplate
): Partial<PromptTemplate> {
  const section = manifestPromptSection(manifest) ?? {};
  const promptId = promptStringValue(section.prompt_id) ?? fallback?.prompt_id ?? "custom.prompt";
  return {
    prompt_id: promptId,
    label: promptStringValue(section.label) ?? fallback?.label ?? promptId,
    task: promptStringValue(section.task) ?? fallback?.task ?? "detection",
    system_prompt: promptStringValue(section.system_prompt) ?? fallback?.system_prompt ?? "",
    user_prompt:
      promptStringValue(section.prompt_text) ??
      promptStringValue(section.user_prompt) ??
      fallback?.user_prompt ??
      "",
    parser: promptStringValue(section.parser) ?? fallback?.parser ?? null,
    metric_profile: promptStringValue(section.metric_profile) ?? fallback?.metric_profile ?? null,
    visualization_profile:
      promptStringValue(section.visualization_profile) ?? fallback?.visualization_profile ?? null,
    generation: isRecord(section.generation) ? section.generation : fallback?.generation ?? {},
    data: isRecord(section.data) ? section.data : fallback?.data ?? {},
    metadata: {
      ...(fallback?.metadata ?? {}),
      target_labels: isStringArray(section.target_labels)
        ? section.target_labels
        : targetLabelsFromPrompt(fallback),
      source: "dashboard_manifest"
    }
  };
}

function targetLabelsFromPrompt(prompt?: Partial<PromptTemplate>) {
  const labels = prompt?.metadata?.target_labels;
  return isStringArray(labels) ? labels : [];
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function manifestPromptSection(manifest: Record<string, unknown>): Record<string, unknown> | null {
  if (isRecord(manifest.eval)) {
    return manifest.eval;
  }
  if (isRecord(manifest.preannotate)) {
    return manifest.preannotate;
  }
  return null;
}

function mergeRecordDefaults(current: unknown, defaults: Record<string, unknown>) {
  return {
    ...defaults,
    ...(isRecord(current) ? current : {})
  };
}

function promptStringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

const rootRoute = createRootRoute({ component: Shell });
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: OverviewPage
});
const benchmarksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks",
  component: BenchmarksPage
});
const benchmarkDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks/$benchmarkId",
  component: BenchmarkDetailPage
});
const jobsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/jobs",
  component: JobsPage
});
const servicesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/services",
  component: ServicesPage
});
const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs",
  component: RunsPage
});
const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: RunDetailPage
});
const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: ComparePage
});
const comparisonSampleRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare/$baselineRunId/$candidateRunId/$sampleIndex",
  component: ComparisonSamplePage
});
const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  benchmarksRoute,
  benchmarkDetailRoute,
  servicesRoute,
  jobsRoute,
  runsRoute,
  runDetailRoute,
  compareRoute,
  comparisonSampleRoute,
  settingsRoute
]);
const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </AppErrorBoundary>
  </React.StrictMode>
);
