import React, { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "@tanstack/react-router";
import { Moon, PanelLeftClose, PanelLeftOpen, Sun, X } from "lucide-react";

import { resetCompareViewState } from "./compareViewState";
import { useDashboardState } from "./dashboardState";
import { errorMessage } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { resetJobsViewState } from "./jobsViewState";
import { resetRankBoardViewState } from "./rankBoardViewState";
import { resetRunsViewState } from "./runsViewState";
import { bootstrapTypographySettings, useTypographySettings } from "./typographySettings";
import { ActionButton, IconActionButton } from "./ui";
import {
  bootstrapThemePreference,
  useSidebarPreference,
  useThemePreference
} from "./workspaceSettings";

bootstrapTypographySettings();
bootstrapThemePreference();

export class AppErrorBoundary extends React.Component<
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
            <ActionButton variant="secondary" onClick={() => this.setState({ error: null })}>
              重试渲染
            </ActionButton>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export function AppShell() {
  const stateQuery = useDashboardState();
  const state = stateQuery.data;
  const location = useLocation();
  const pageTitle = getShellTitle(location.pathname);
  const { sidebarCollapsed, setSidebarCollapsed } = useSidebarPreference();
  const { themeMode, toggleThemeMode } = useThemePreference();
  useTypographySettings();

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
          <NavItem
            to="/jobs"
            icon={<AppIcon name="evalJob" size={21} />}
            label="评测中心"
            onNavigate={resetJobsViewState}
          />
          <NavItem
            to="/runs"
            icon={<AppIcon name="runResults" size={21} />}
            label="结果库"
            onNavigate={resetRunsViewState}
          />
          <NavItem
            to="/rank-board"
            icon={<AppIcon name="rankBoard" size={21} />}
            label="排行榜"
            onNavigate={resetRankBoardViewState}
          />
          <NavItem
            to="/suite-report"
            icon={<AppIcon name="diagnostics" size={21} />}
            label="组合报告"
          />
          <NavItem
            to="/compare"
            icon={<AppIcon name="compareAnalysis" size={21} />}
            label="对比分析"
            onNavigate={resetCompareViewState}
          />
          <NavItem
            to="/settings"
            icon={<AppIcon name="workspaceSettings" size={21} />}
            label="工作台设置"
          />
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
            <IconActionButton
              className="theme-toggle"
              title={themeMode === "dark" ? "切换到日间主题" : "切换到夜间主题"}
              icon={themeMode === "dark" ? <Sun size={15} /> : <Moon size={15} />}
              onClick={toggleThemeMode}
            />
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
  if (pathname.startsWith("/suite-report")) {
    return { kicker: "多结果分层检查", title: "组合报告" };
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
  label,
  onNavigate
}: {
  to: string;
  icon: React.ReactNode;
  label: string;
  onNavigate?: () => void;
}) {
  return (
    <Link
      to={to}
      className="nav-item"
      activeProps={{ className: "nav-item active" }}
      preload="intent"
      preloadDelay={80}
      title={label}
      onClick={onNavigate}
    >
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
