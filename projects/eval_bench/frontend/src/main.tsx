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
  Search,
  Server,
  SlidersHorizontal,
  X
} from "lucide-react";

import { StyleSlider } from "./controlPrimitives";
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
  RunSampleDetail,
  RunSampleSummary,
  RunSummary,
  createBenchmark,
  fetchBenchmarkSampleDetail,
  fetchBenchmarkSamples,
  fetchComparison,
  fetchComparisonSample,
  fetchComparisons,
  fetchRunSampleDetail,
  fetchRunSamples,
  fetchSettingsPreviewSample,
  importPredictions
} from "./api";
import {
  buildObjectRows,
  visibleLabelMetrics,
  visibleSampleMetrics
} from "./viewerMetrics";
import {
  INTERACTION_SETTING_CONTROLS,
  INSTANCE_COLOR_ROLES,
  OVERLAY_STYLE_CONTROLS,
  PRED_LINE_STYLE_OPTIONS,
  explicitLabelColor,
  settingControlValue,
  settingValueFromControl,
  useSidebarPreference,
  useWorkspaceShortcuts,
  useWorkspaceSettings
} from "./workspaceSettings";
import {
  basename,
  comparisonSampleHref,
  formatCompactSignedMetric,
  formatDate,
  formatMetric,
  formatRunOption,
  formatSignedInteger,
  formatSignedMetric,
  inferenceValue,
  isTextInputTarget,
  pixelBudgetValue,
  runIdExists,
  runSampleHref,
  samplingValue,
  scoreRun,
  stringValue,
  unique
} from "./formatters";
import { useDashboardState } from "./dashboardState";
import { FilterSelect } from "./filterControls";
import { JobQueuePanel, JobsPage } from "./jobsPage";
import { BenchmarkTable, RunTable } from "./runTables";
import { ServicesPage } from "./servicesPage";
import {
  LabelColorQuickAdd,
  SettingsEditorSection,
  SettingsPreferenceRow,
  ShortcutSettingsPanel
} from "./settingsControls";
import {
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { preloadSampleImages } from "./viewerGeometry";
import { CanvasStage } from "./viewerCanvas";
import {
  DiagnosticStrip,
  InstanceStats,
  LabelMetricTable,
  ObjectList,
  ViewerControlPanel,
  VisibleMetricStrip,
  handleViewerShortcutAction
} from "./viewerPanels";
import {
  ActionPanel,
  Badge,
  ConfigItem,
  DataTable,
  EmptyState,
  PanelTitle,
  SectionHeader,
  WorkspaceTabs
} from "./ui";
import { ResizableSplit } from "./workspaceLayout";
import type {
  InteractionSettingKey,
  InteractionSettings,
  InstanceColorRole,
  LabelColors,
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
          <img className="brand-logo" src="/logo.png" alt="" aria-hidden="true" />
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
            <div className="user-profile-chip" title="当前版本使用浏览器本地 profile 保存偏好">
              <span>用户</span>
              <strong>local</strong>
            </div>
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
  const { actionForEvent } = useWorkspaceShortcuts();
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

function SampleViewer({ detail }: { detail: RunSampleDetail }) {
  return <InteractiveSampleViewer detail={detail} />;
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
    updateOverlayStyle,
    updateLabelColor,
    removeLabelColor,
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
  const { actionForEvent } = useWorkspaceShortcuts();

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
      const actionId = actionForEvent(event);
      if (!actionId) {
        return;
      }
      if (handleViewerShortcutAction(actionId, {
        clearSelection: () => {
          setLockedObjectId(null);
          setHoveredObjectId(null);
        },
        toggleGt: () => setShowGt((value) => !value),
        togglePred: () => setShowPred((value) => !value),
        toggleBoxes: () => setShowBoxes((value) => !value),
        toggleLines: () => setShowLines((value) => !value),
        toggleKeypoints: () => setShowKeypoints((value) => !value)
      })) {
        event.preventDefault();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [actionForEvent]);

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
        onStyleChange={updateOverlayStyle}
        onLabelColorChange={updateLabelColor}
        onLabelColorRemove={removeLabelColor}
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
  const previewImageUrl = previewSample?.image_url ?? SETTINGS_PREVIEW_IMAGE_URL;
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
                <button type="button" onClick={() => setSettingsQuery("")} title="清空搜索">
                  <X size={13} />
                </button>
              ) : null}
            </div>
            <nav className="settings-section-nav" aria-label="工作台设置分组">
              {settingsSections.map((section) => (
                <button
                  key={section.id}
                  className={
                    !query && activeSettingsPanel === section.id
                      ? "settings-section-button active"
                      : "settings-section-button"
                  }
                  type="button"
                  onClick={() => {
                    setActiveSettingsPanel(section.id);
                    setSettingsQuery("");
                  }}
                >
                  <span>{section.label}</span>
                  <small>{section.meta}</small>
                </button>
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
                <div className="settings-slider-grid">
                  {OVERLAY_STYLE_CONTROLS.map((control) => (
                    <StyleSlider
                      key={control.key}
                      label={control.label}
                      value={overlayStyle[control.key]}
                      min={control.min}
                      max={control.max}
                      step={control.step}
                      onChange={(value) => updateOverlayStyle(control.key, value)}
                    />
                  ))}
                  <label className="compact-select dense">
                    <span>预测线型</span>
                    <select
                      value={overlayStyle.predLineStyle}
                      onChange={(event) => updateOverlayStyle("predLineStyle", event.target.value)}
                    >
                      {PRED_LINE_STYLE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <button className="settings-inline-action" type="button" onClick={resetOverlayStyle}>
                  重置样式
                </button>
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
                        <button type="button" onClick={() => removeLabelColor(label)}>
                          清除
                        </button>
                      </div>
                    ))
                  )}
                </div>
                <button className="settings-inline-action" type="button" onClick={resetLabelColors}>
                  清空 label 颜色
                </button>
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
                <div className="settings-slider-grid">
                  {INTERACTION_SETTING_CONTROLS.map((control) => (
                    <StyleSlider
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
                <button className="settings-inline-action" type="button" onClick={resetInteractionSettings}>
                  重置交互
                </button>
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
            <Badge value={selected.status} domain="run" />
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
