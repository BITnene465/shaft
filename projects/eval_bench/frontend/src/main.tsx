import React from "react";
import ReactDOM from "react-dom/client";
import { useEffect, useMemo, useRef, useState } from "react";
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
  lazyRouteComponent,
  useLocation,
  useParams
} from "@tanstack/react-router";
import {
  Eye,
  FileText,
  PanelLeftClose,
  PanelLeftOpen,
  Save,
  X
} from "lucide-react";

import {
  BenchmarkSampleDetail,
  BenchmarkSampleSummary,
  BenchmarkSummary,
  ComparisonSampleDetail,
  RunSampleDetail,
  RunSampleSummary,
  RunSummary,
  createBenchmark,
  fetchBenchmarks,
  fetchBenchmarkSampleDetail,
  fetchBenchmarkSamples,
  fetchComparisonSample,
  fetchRuns,
  fetchRunSampleDetail,
  fetchRunSamples,
  importPredictions,
  updateRunNote
} from "./api";
import {
  buildObjectRows,
  visibleLabelMetrics,
  visibleSampleMetrics
} from "./viewerMetrics";
import {
  useSidebarPreference,
  useViewerLayerPreferences,
  useWorkspaceShortcuts,
  useWorkspaceSettings
} from "./workspaceSettings";
import {
  basename,
  comparisonSampleHref,
  formatDate,
  formatMetric,
  formatSignedMetric,
  inferenceValue,
  isTextInputTarget,
  pixelBudgetValue,
  runSampleHref,
  samplingValue,
  stringValue,
  unique
} from "./formatters";
import { useDashboardState } from "./dashboardState";
import { AdvancedFilterBar } from "./filterControls";
import { AppIcon } from "./iconLibrary";
import { JobsPage } from "./jobsPage";
import { OverviewPage } from "./overviewPage";
import { BenchmarkTable, RunTable } from "./runTables";
import { ServicesPage } from "./servicesPage";
import { SettingsPage } from "./settingsPage";
import {
  sampleIndexFromLocation,
  samplePageOffsetFromLocation,
  updateSampleIndexInLocation
} from "./sampleNavigation";
import { displayImageUrl, preloadSampleImages } from "./viewerGeometry";
import { CanvasStage } from "./viewerCanvas";
import {
  InstanceStats,
  LabelMetricTable,
  ObjectList,
  ViewerControlPanel,
  VisibleMetricStrip,
  handleViewerShortcutAction
} from "./viewerPanels";
import {
  ActionButton,
  Badge,
  CommandButton,
  ConfigItem,
  DataTable,
  EmptyState,
  IconActionButton,
  PanelTitle,
  SectionHeader,
  WorkspaceDialog
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
            <ActionButton variant="secondary" onClick={() => window.location.reload()}>
              重新加载
            </ActionButton>
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
          <NavItem to="/" icon={<AppIcon name="overview" size={21} />} label="总览" />
          <NavItem to="/benchmarks" icon={<AppIcon name="benchmark" size={21} />} label="基准集" />
          <NavItem to="/services" icon={<AppIcon name="service" size={21} />} label="模型服务" />
          <NavItem to="/jobs" icon={<AppIcon name="evalJob" size={21} />} label="评测中心" />
          <NavItem to="/runs" icon={<AppIcon name="runResults" size={21} />} label="结果库" />
          <NavItem to="/rank-board" icon={<AppIcon name="rankBoard" size={21} />} label="排行榜" />
          <NavItem to="/compare" icon={<AppIcon name="compareAnalysis" size={21} />} label="对比分析" />
          <NavItem to="/settings" icon={<AppIcon name="workspaceSettings" size={21} />} label="工作台设置" />
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
              <span>Profile</span>
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
          <IconActionButton
            icon={<X size={13} />}
            title="关闭提醒"
            onClick={() => setItems((current) => current.filter((entry) => entry.id !== item.id))}
          />
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
  if (pathname.startsWith("/rank-board")) {
    return { kicker: "模型排名工作台", title: "排行榜" };
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
  return (
    <div className={loading ? "status-pill loading" : "status-pill online"}>
      {loading ? "同步中" : "在线"}
    </div>
  );
}

function BenchmarksPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [taskFilter, setTaskFilter] = useState("all");
  const [layerFilter, setLayerFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const benchmarkFilters = useMemo(
    () => ({
      offset: 0,
      limit: 200,
      task: taskFilter !== "all" ? taskFilter : undefined,
      layer: layerFilter !== "all" ? layerFilter : undefined,
      split: splitFilter !== "all" ? splitFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [layerFilter, searchText, splitFilter, taskFilter]
  );
  const benchmarksQuery = useQuery({
    queryKey: ["benchmarks", benchmarkFilters],
    queryFn: () => fetchBenchmarks(benchmarkFilters)
  });
  const benchmarkFacetsQuery = useQuery({
    queryKey: ["benchmarks", "facets"],
    queryFn: () => fetchBenchmarks({ limit: 500 })
  });
  const benchmarks = benchmarksQuery.data?.benchmarks ?? [];
  const benchmarkFacets = benchmarkFacetsQuery.data?.benchmarks ?? benchmarks;
  const tasks = unique(benchmarkFacets.flatMap((benchmark) => benchmark.tasks).filter(Boolean));
  const layers = unique(benchmarkFacets.flatMap((benchmark) => benchmark.layers).filter(Boolean));
  const splits = unique(benchmarkFacets.map((benchmark) => benchmark.split).filter(Boolean));
  if (benchmarksQuery.isLoading) {
    return <EmptyState title="正在加载基准集" />;
  }
  if (benchmarksQuery.error || !benchmarksQuery.data) {
    return <EmptyState title="基准集加载失败" tone="danger" />;
  }
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>基准集目录</h2>
          <span>{(benchmarksQuery.data.total ?? benchmarks.length).toLocaleString()} 个不可变副本</span>
        </div>
        <CommandButton
          icon={<AppIcon name="createBenchmark" size={17} />}
          onClick={() => setCreateOpen(true)}
        >
          创建副本
        </CommandButton>
      </div>
      <AdvancedFilterBar
        title="基准集高级检索"
        meta={`${benchmarks.length.toLocaleString()} / ${(benchmarksQuery.data.total ?? benchmarks.length).toLocaleString()} 个 benchmark`}
        controls={[
          {
            type: "search",
            id: "benchmark-query",
            label: "全文检索",
            value: searchText,
            onChange: setSearchText,
            placeholder: "搜索 benchmark、manifest、root、来源"
          },
          {
            type: "select",
            id: "benchmark-task",
            label: "任务",
            value: taskFilter,
            values: ["all", ...tasks],
            labels: { all: "全部" },
            onChange: setTaskFilter
          },
          {
            type: "select",
            id: "benchmark-layer",
            label: "标注层",
            value: layerFilter,
            values: ["all", ...layers],
            labels: { all: "全部" },
            onChange: setLayerFilter
          },
          {
            type: "select",
            id: "benchmark-split",
            label: "Split",
            value: splitFilter,
            values: ["all", ...splits],
            labels: { all: "全部" },
            onChange: setSplitFilter
          }
        ]}
      />
      <div className="workspace-card fill">
        <BenchmarkTable benchmarks={benchmarks} />
      </div>
      <WorkspaceDialog
        open={createOpen}
        title="创建 benchmark 副本"
        meta="从 raw_data split 复制不可变 test/val 集"
        onClose={() => setCreateOpen(false)}
      >
        <BenchmarkCreatePanel bare />
      </WorkspaceDialog>
    </section>
  );
}

function BenchmarkCreatePanel({ bare }: { bare?: boolean }) {
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
      void queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
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

  const content = (
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
        <ActionButton
          className="form-submit-button"
          type="submit"
          variant="primary"
          icon={<AppIcon name="submitCreate" size={16} />}
          disabled={mutation.isPending || tasks.length === 0}
        >
          创建
        </ActionButton>
        {mutation.data ? (
          <div className="form-result full-field">
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
          <div className="form-result error full-field">{mutation.error.message}</div>
        ) : null}
      </form>
  );
  return bare ? content : <div className="workspace-card compact-form-card">{content}</div>;
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
    <AdvancedFilterBar
      title="样本检索"
      meta={`${labels.length.toLocaleString()} labels`}
      controls={[
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
        imageUrl={displayImageUrl(detail.sample)}
        imageAlt={detail.sample.image}
        imageTileUrlTemplate={detail.sample.image_tile_url_template}
        imageTileSize={detail.sample.image_tile_size}
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
  const runFilters = useMemo(
    () => ({
      offset: 0,
      limit: 200,
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
          <span>{(runsQuery.data.total ?? runs.length).toLocaleString()} 条 run snapshot</span>
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
          filterMeta={`${runs.length.toLocaleString()} / ${(runsQuery.data.total ?? runs.length).toLocaleString()} 条 run`}
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
  const [targetLabels, setTargetLabels] = useState("");
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
      target_labels: parseTargetLabels(targetLabels),
      strict,
      overwrite,
      evaluate
    });
  }

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
        <label className="wide-field">
          <span>模型路径</span>
          <input value={modelPath} onChange={(event) => setModelPath(event.target.value)} />
        </label>
        <label>
          <span>Prompt</span>
          <input value={promptId} onChange={(event) => setPromptId(event.target.value)} />
        </label>
        <label>
          <span>目标标签</span>
          <input
            value={targetLabels}
            onChange={(event) => setTargetLabels(event.target.value)}
            placeholder="arrow 或 icon,image,shape"
          />
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
        <ActionButton
          className="form-submit-button"
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

function parseTargetLabels(value: string) {
  return value
    .replace(/,/g, " ")
    .split(/\s+/)
    .map((item) => item.trim())
    .filter(Boolean);
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
  const queryClient = useQueryClient();
  const [noteDraft, setNoteDraft] = useState(run.note || "");
  const noteMutation = useMutation({
    mutationFn: (note: string) => updateRunNote(run.run_id, note),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["dashboard-state"] });
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
            真实 {sample.gt_instance_count.toLocaleString()} / 预测{" "}
            {sample.pred_instance_count.toLocaleString()}
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
        <ActionButton
          variant="mini"
          onClick={() => onPageChange(previousOffset)}
          disabled={offset <= 0}
        >
          上一页
        </ActionButton>
        <ActionButton
          variant="mini"
          onClick={() => onPageChange(nextOffset)}
          disabled={nextOffset >= total}
        >
          下一页
        </ActionButton>
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
  const {
    activeLabels,
    setActiveLabels,
    showGt,
    setShowGt,
    showPred,
    setShowPred,
    showBoxes,
    setShowBoxes,
    showLines,
    setShowLines,
    showKeypoints,
    setShowKeypoints
  } = useViewerLayerPreferences(labels);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [lockedObjectId, setLockedObjectId] = useState<string | null>(null);
  const {
    overlayColors,
    overlayStyle,
    labelColors,
    interactionSettings,
    overlayVars
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
    setLockedObjectId(null);
    setHoveredObjectId(null);
  }, [detail.sample.index]);

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
      imageUrl={displayImageUrl(detail.sample)}
      imageAlt={detail.sample.image}
      imageTileUrlTemplate={detail.sample.image_tile_url_template}
      imageTileSize={detail.sample.image_tile_size}
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
        showGt={showGt}
        showPred={showPred}
        showBoxes={showBoxes}
        showLines={showLines}
        showKeypoints={showKeypoints}
        onToggleLabel={toggleLabel}
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
const rankBoardRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/rank-board",
  component: lazyRouteComponent(() => import("./rankBoardPage"), "RankBoardPage")
});
const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: lazyRouteComponent(() => import("./comparePage"), "ComparePage")
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
  rankBoardRoute,
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
