import React from "react";
import ReactDOM from "react-dom/client";
import { useEffect, useState } from "react";
import {
  QueryClient,
  QueryClientProvider
} from "@tanstack/react-query";
import {
  Link,
  Outlet,
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  useLocation
} from "@tanstack/react-router";
import {
  PanelLeftClose,
  PanelLeftOpen,
  X
} from "lucide-react";

import { useSidebarPreference } from "./workspaceSettings";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { errorMessage } from "./formatters";
import { JobsPage } from "./jobsPage";
import { OverviewPage } from "./overviewPage";
import { ServicesPage } from "./servicesPage";
import { SettingsPage } from "./settingsPage";
import {
  ActionButton,
  EmptyState,
  IconActionButton
} from "./ui";
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

class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: unknown) {
    return { error: errorMessage(error) };
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
          <IconActionButton
            className="sidebar-toggle"
            title={sidebarCollapsed ? "展开导航栏" : "收起导航栏"}
            dense={false}
            icon={sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
            onClick={() => setSidebarCollapsed((value) => !value)}
          />
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

const rootRoute = createRootRoute({ component: Shell });
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: OverviewPage
});
const benchmarksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks",
  component: lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarksPage")
});
const benchmarkDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks/$benchmarkId",
  component: lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarkDetailPage")
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
  component: lazyRouteComponent(() => import("./runsPage"), "RunsPage")
});
const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: lazyRouteComponent(() => import("./runsPage"), "RunDetailPage")
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
  component: lazyRouteComponent(() => import("./comparisonSamplePage"), "ComparisonSamplePage")
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
