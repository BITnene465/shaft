import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Activity, BarChart3 } from "lucide-react";

import type { RunSummary, SchedulerStatus } from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { Badge, EmptyState, PanelTitle } from "./ui";

export function OverviewPage() {
  const { data, isLoading, error } = useDashboardState();
  const jobsQuery = useQuery({
    queryKey: ["overview-jobs"],
    queryFn: () => fetchJobs({ limit: 500 }),
    refetchInterval: 2_000
  });
  const servicesQuery = useQuery({
    queryKey: ["overview-services"],
    queryFn: () => fetchServices({ limit: 500 }),
    refetchInterval: 5_000
  });
  const schedulerQuery = useQuery({
    queryKey: ["overview-scheduler"],
    queryFn: fetchSchedulerStatus,
    refetchInterval: 2_000
  });
  if (isLoading) {
    return <EmptyState title="正在加载看板状态" />;
  }
  if (error || !data) {
    return <EmptyState title="看板状态加载失败" tone="danger" />;
  }
  const jobs = jobsQuery.data?.jobs ?? [];
  const services = servicesQuery.data?.services ?? [];
  const schedulerStatus = schedulerQuery.data;
  const queuedJobs = jobs.filter((job) => job.status === "queued").length;
  const runningJobs = jobs.filter((job) => job.status === "running").length;
  const liveServices = services.filter((service) => service.status === "running").length;
  const activeRuns = data.runs.filter((run) =>
    ["created", "queued", "running"].includes(run.status)
  ).length;
  const overviewSyncing =
    jobsQuery.isFetching || servicesQuery.isFetching || schedulerQuery.isFetching;
  const statusRows = countBy(data.runs, (run) => run.status || "unknown");
  const taskRows = countBy(data.runs, (run) => run.spec_task || "unknown");
  const modelRows = countBy(data.runs, (run) => run.model_id || "unknown");
  const promptRows = countBy(data.runs, (run) => run.prompt_id || "unknown");
  const parserRows = countBy(data.runs, (run) => run.parser || "unknown");
  const viewRows = countBy(data.runs, (run) => run.visualization_profile || "default");
  const benchmarkTaskRows = countMany(data.benchmarks, (benchmark) => benchmark.tasks);
  const benchmarkLabelRows = countMany(data.benchmarks, (benchmark) => benchmark.labels);
  const benchmarkLayerRows = countMany(data.benchmarks, (benchmark) => benchmark.layers);
  const splitRows = countBy(data.benchmarks, (benchmark) => benchmark.split || "unknown");
  const coverageRows = runCoverageRows(data.runs);
  const sampleScaleRows = countBy(data.benchmarks, (benchmark) =>
    sampleScaleBucket(benchmark.sample_count)
  );
  const targetLabelRows = countMany(data.runs, (run) =>
    run.target_labels.length > 0 ? run.target_labels : ["unscoped"]
  );
  const freshnessRows = runFreshnessRows(data.runs);
  const predictionRows = countBy(data.runs, (run) => predictionScaleBucket(run.prediction_count));
  const noteRows = runNoteRows(data.runs);
  const jobStatusRows = countBy(jobs, (job) => job.status || "unknown");
  const jobKindRows = countBy(jobs, (job) => job.kind || "unknown");
  const serviceStatusRows = countBy(services, (service) => service.status || "unknown");
  const serviceKindRows = countBy(services, (service) => service.kind || "unknown");
  const liveSignalRows = [
    { key: "queued jobs", count: queuedJobs },
    { key: "running jobs", count: runningJobs },
    { key: "live services", count: liveServices },
    { key: "active runs", count: activeRuns }
  ];
  const schedulerRows = schedulerResourceRows(schedulerStatus);
  const timelineRows = runTimeline(data.runs, 12);
  const notedRuns = data.runs.filter((run) => run.note.trim()).length;
  const evaluatedRuns = data.runs.filter((run) => run.report_path).length;
  const overviewCharts = [
    { title: "Run 生命周期", meta: "status", rows: statusRows },
    { title: "评测覆盖", meta: "report state", rows: coverageRows },
    { title: "Run 任务", meta: "spec_task", rows: taskRows },
    { title: "模型分布", meta: "model_id", rows: modelRows },
    { title: "Prompt 分布", meta: "prompt_id", rows: promptRows },
    { title: "Parser", meta: "decode path", rows: parserRows },
    { title: "Viewer profile", meta: "visual mode", rows: viewRows },
    { title: "Benchmark 任务", meta: "task set", rows: benchmarkTaskRows },
    { title: "Label footprint", meta: "benchmark labels", rows: benchmarkLabelRows },
    { title: "样本规模", meta: "sample buckets", rows: sampleScaleRows },
    { title: "数据层", meta: "benchmark layers", rows: benchmarkLayerRows },
    { title: "Split 分布", meta: "dataset split", rows: splitRows },
    { title: "Label scope", meta: "run labels", rows: targetLabelRows },
    { title: "Run 新鲜度", meta: "created_at", rows: freshnessRows },
    { title: "预测规模", meta: "prediction files", rows: predictionRows },
    { title: "备注覆盖", meta: "run notes", rows: noteRows },
    { title: "Job 状态", meta: "queue state", rows: jobStatusRows },
    { title: "Job 类型", meta: "job kind", rows: jobKindRows },
    { title: "Service 状态", meta: "runtime state", rows: serviceStatusRows },
    { title: "Service 类型", meta: "runtime kind", rows: serviceKindRows },
    { title: "实时信号", meta: "live counters", rows: liveSignalRows },
    { title: "Scheduler", meta: "resource slots", rows: schedulerRows }
  ];
  return (
    <section className="page-stack dashboard-home">
      <div className="overview-console">
        <div className="overview-console-main">
          <div className="overview-title-block">
            <div className="eyebrow">Eval Bench Control</div>
            <h2>总览</h2>
          </div>
          <div className="overview-stat-row">
            <OverviewStat label="Bench" value={data.benchmark_count} />
            <OverviewStat label="Runs" value={data.run_count} />
            <OverviewStat label="Done" value={evaluatedRuns} />
            <OverviewStat label="Live" value={activeRuns} tone={activeRuns > 0 ? "live" : "idle"} />
          </div>
        </div>
        <div className="overview-console-side">
          <div className="overview-store-line">
            <span>Store</span>
            <strong title={data.store_root}>{data.store_root}</strong>
          </div>
          <div className="overview-console-links">
            <Link to="/rank-board">
              <AppIcon name="rankBoard" size={14} />
              排行榜
            </Link>
            <Link to="/runs">
              <AppIcon name="runResults" size={14} />
              结果库
            </Link>
            <Link to="/jobs">
              <Activity size={14} />
              任务
            </Link>
          </div>
        </div>
      </div>
      <div className="overview-signal-deck">
        <div className="overview-ops-strip">
          <OverviewDatum label="GT samples" value={data.total_benchmark_samples} />
          <OverviewDatum label="Predictions" value={data.prediction_count} />
          <OverviewDatum label="Notes" value={notedRuns} />
          <OverviewDatum label="Tasks" value={taskRows.length} />
        </div>
        <OverviewWriteRhythm rows={timelineRows} />
        <OverviewTelemetryPanel
          queuedJobs={queuedJobs}
          runningJobs={runningJobs}
          liveServices={liveServices}
          totalJobs={jobs.length}
          totalServices={services.length}
          schedulerEnabled={Boolean(schedulerStatus?.enabled)}
          syncing={overviewSyncing}
        />
      </div>
      <div className="overview-grid refined">
        <div className="overview-chart-matrix">
          {overviewCharts.map((chart) => (
            <OverviewMiniChartPanel
              key={chart.title}
              title={chart.title}
              meta={chart.meta}
              rows={chart.rows}
            />
          ))}
        </div>
        <div className="workspace-card overview-recent-panel">
          <PanelTitle title="最近 run" meta={`最新 ${Math.min(6, data.runs.length)} 条`} />
          <OverviewRunList runs={data.runs.slice(0, 6)} />
        </div>
      </div>
    </section>
  );
}

function OverviewStat({
  label,
  value,
  tone = "idle"
}: {
  label: string;
  value: number;
  tone?: "idle" | "live";
}) {
  return (
    <div className={tone === "live" ? "overview-stat live" : "overview-stat"}>
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}

function OverviewDatum({ label, value }: { label: string; value: number }) {
  return (
    <div className="overview-datum">
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}

function OverviewTelemetryPanel({
  queuedJobs,
  runningJobs,
  liveServices,
  totalJobs,
  totalServices,
  schedulerEnabled,
  syncing
}: {
  queuedJobs: number;
  runningJobs: number;
  liveServices: number;
  totalJobs: number;
  totalServices: number;
  schedulerEnabled: boolean;
  syncing: boolean;
}) {
  const cells = [
    {
      label: "Scheduler",
      value: schedulerEnabled ? "AUTO" : "MANUAL",
      tone: schedulerEnabled ? "live" : "idle"
    },
    {
      label: "Queued jobs",
      value: queuedJobs.toLocaleString(),
      tone: queuedJobs > 0 ? "warm" : "idle"
    },
    {
      label: "Running jobs",
      value: runningJobs.toLocaleString(),
      tone: runningJobs > 0 ? "live" : "idle"
    },
    {
      label: "Services",
      value: `${liveServices}/${totalServices}`,
      tone: liveServices > 0 ? "live" : "idle"
    },
    { label: "Job records", value: totalJobs.toLocaleString(), tone: "idle" }
  ] as const;
  return (
    <div className="overview-telemetry-panel">
      <div className={syncing ? "telemetry-signal syncing" : "telemetry-signal"}>
        <span />
        <strong>{syncing ? "syncing" : "stable"}</strong>
      </div>
      <div className="overview-telemetry-grid">
        {cells.map((cell) => (
          <div className={`telemetry-cell ${cell.tone}`} key={cell.label}>
            <span>{cell.label}</span>
            <strong>{cell.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function OverviewMiniChartPanel({
  title,
  meta,
  rows
}: {
  title: string;
  meta: string;
  rows: Array<{ key: string; count: number }>;
}) {
  const maxCount = Math.max(1, ...rows.map((row) => row.count));
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const topRows = rows.slice(0, 4);
  const ringStyle = {
    "--overview-ring": overviewConicGradient(rows)
  } as React.CSSProperties;
  return (
    <div className="workspace-card overview-chart-card compact">
      <PanelTitle title={title} meta={meta} />
      <div className="overview-mini-chart">
        <div className="overview-chart-ring" style={ringStyle}>
          <span>{rows.length.toLocaleString()}</span>
          <strong>{total.toLocaleString()}</strong>
        </div>
        <div className="overview-bar-list">
          {topRows.length === 0 ? (
            <div className="empty-inline">暂无数据</div>
          ) : (
            topRows.map((row, index) => (
              <div className="overview-bar-row" key={row.key}>
                <span>{row.key}</span>
                <div>
                  <i
                    style={
                      {
                        width:
                          row.count > 0
                            ? `${Math.max(4, (row.count / maxCount) * 100)}%`
                            : "0%",
                        "--overview-bar-fill": overviewChartColor(index)
                      } as React.CSSProperties
                    }
                  />
                </div>
                <strong>{row.count.toLocaleString()}</strong>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function overviewConicGradient(rows: Array<{ key: string; count: number }>) {
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  if (total <= 0) {
    return "conic-gradient(#d8e4ec 0deg 360deg)";
  }
  let cursor = 0;
  const segments = rows.slice(0, 6).map((row, index) => {
    const start = cursor;
    const end = cursor + (row.count / total) * 360;
    cursor = end;
    return `${overviewChartColor(index)} ${start.toFixed(1)}deg ${end.toFixed(1)}deg`;
  });
  if (cursor < 360) {
    segments.push(`#dbe5ed ${cursor.toFixed(1)}deg 360deg`);
  }
  return `conic-gradient(${segments.join(", ")})`;
}

function overviewChartColor(index: number) {
  const colors = ["#23a36f", "#1d5d7a", "#d68722", "#8d5fb8", "#d04f66", "#5e7892"];
  return colors[index % colors.length];
}

function OverviewWriteRhythm({ rows }: { rows: Array<{ key: string; count: number }> }) {
  const maxCount = Math.max(1, ...rows.map((row) => row.count));
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const activeBuckets = rows.filter((row) => row.count > 0).length;
  const latest = rows.at(-1);
  return (
    <div className="overview-rhythm-strip">
      <div className="overview-rhythm-copy">
        <span>
          <BarChart3 size={14} />
          写入节奏
        </span>
        <strong>{total.toLocaleString()} runs / 12d</strong>
      </div>
      <div className="overview-rhythm-bars" aria-label="最近 12 个日期桶的 run 写入节奏">
        {rows.map((row) => (
          <span
            key={row.key}
            style={
              {
                "--rhythm-height": `${Math.max(10, (row.count / maxCount) * 100)}%`
              } as React.CSSProperties
            }
            title={`${row.key}: ${row.count.toLocaleString()} runs`}
          >
            <i />
          </span>
        ))}
      </div>
      <div className="overview-rhythm-meta">
        <span>{latest?.key ?? "-"}</span>
        <strong>
          活跃 {activeBuckets}/{rows.length}
        </strong>
      </div>
    </div>
  );
}

function OverviewRunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <div className="empty-inline">暂无 run。</div>;
  }
  return (
    <div className="overview-run-list">
      {runs.map((run, index) => (
        <Link key={run.run_id} to="/runs/$runId" params={{ runId: run.run_id }}>
          <em>{String(index + 1).padStart(2, "0")}</em>
          <span>
            <strong>{run.run_id}</strong>
            <small>{run.model_id || "-"}</small>
          </span>
          <Badge value={run.status} domain="run" />
        </Link>
      ))}
    </div>
  );
}

function countBy<T>(items: T[], keyForItem: (item: T) => string) {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyForItem(item).trim() || "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([key, count]) => ({ key, count }))
    .sort((left, right) => right.count - left.count || left.key.localeCompare(right.key));
}

function countMany<T>(items: T[], keysForItem: (item: T) => string[]) {
  return countBy(
    items.flatMap((item) => {
      const keys = keysForItem(item)
        .map((key) => key.trim())
        .filter(Boolean);
      return keys.length > 0 ? keys : ["unknown"];
    }),
    (key) => key
  );
}

function runCoverageRows(runs: RunSummary[]) {
  const rows = [
    {
      key: "已评估",
      count: runs.filter((run) => Boolean(run.report_path)).length
    },
    {
      key: "有预测",
      count: runs.filter((run) => !run.report_path && run.prediction_count > 0).length
    },
    {
      key: "仅记录",
      count: runs.filter((run) => !run.report_path && run.prediction_count <= 0).length
    }
  ];
  return rows.filter((row) => row.count > 0);
}

function sampleScaleBucket(sampleCount: number) {
  if (sampleCount <= 0) {
    return "0";
  }
  if (sampleCount < 100) {
    return "1-99";
  }
  if (sampleCount < 1_000) {
    return "100-999";
  }
  if (sampleCount < 10_000) {
    return "1k-9.9k";
  }
  return "10k+";
}

function predictionScaleBucket(predictionCount: number) {
  if (predictionCount <= 0) {
    return "0";
  }
  if (predictionCount < 10) {
    return "1-9";
  }
  if (predictionCount < 100) {
    return "10-99";
  }
  if (predictionCount < 1_000) {
    return "100-999";
  }
  return "1k+";
}

function runNoteRows(runs: RunSummary[]) {
  return [
    { key: "有备注", count: runs.filter((run) => run.note.trim()).length },
    { key: "无备注", count: runs.filter((run) => !run.note.trim()).length }
  ];
}

function schedulerResourceRows(status: SchedulerStatus | undefined) {
  if (!status) {
    return [{ key: "unknown", count: 1 }];
  }
  const liveJobs = status.live_running_count ?? status.live_running_jobs?.length ?? 0;
  return [
    { key: status.enabled ? "auto" : "manual", count: 1 },
    { key: "live jobs", count: liveJobs },
    { key: "workers", count: status.active_worker_threads?.length ?? 0 },
    { key: "cuda", count: status.reserved_cuda_devices?.length ?? 0 },
    { key: "ports", count: status.reserved_runtime_ports?.length ?? 0 }
  ];
}

function runFreshnessRows(runs: RunSummary[]) {
  const anchor = latestRunDate(runs) ?? new Date();
  return countBy(runs, (run) => {
    const timestamp = run.created_at ? Date.parse(run.created_at) : Number.NaN;
    if (!Number.isFinite(timestamp)) {
      return "unknown";
    }
    const days = Math.max(
      0,
      Math.floor((anchor.getTime() - timestamp) / (24 * 60 * 60 * 1_000))
    );
    if (days === 0) {
      return "latest day";
    }
    if (days <= 3) {
      return "1-3d";
    }
    if (days <= 7) {
      return "4-7d";
    }
    return "older";
  });
}

function runTimeline(runs: RunSummary[], bucketCount: number) {
  const endDate = latestRunDate(runs) ?? new Date();
  const keys = Array.from({ length: bucketCount }, (_, index) => {
    const date = new Date(endDate);
    date.setUTCDate(endDate.getUTCDate() - (bucketCount - 1 - index));
    return date.toISOString().slice(0, 10);
  });
  const counts = countBy(runs, (run) => (run.created_at ? run.created_at.slice(0, 10) : "unknown"));
  const countMap = new Map(counts.map((row) => [row.key, row.count]));
  return keys.map((key) => ({ key, count: countMap.get(key) ?? 0 }));
}

function latestRunDate(runs: RunSummary[]) {
  const timestamps = runs
    .map((run) => (run.created_at ? Date.parse(run.created_at) : Number.NaN))
    .filter(Number.isFinite);
  if (timestamps.length === 0) {
    return null;
  }
  return new Date(Math.max(...timestamps));
}
