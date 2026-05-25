import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Activity, BarChart3 } from "lucide-react";

import type {
  JobSummary,
  RunSummary,
  SchedulerStatus,
  ServiceSummary
} from "./api";
import { fetchJobs, fetchSchedulerStatus, fetchServices } from "./api";
import { useDashboardState } from "./dashboardState";
import { AppIcon } from "./iconLibrary";
import { Badge, EmptyState, PanelTitle } from "./ui";

type OverviewChartKind = "ring" | "rails" | "cells" | "meter" | "spark" | "mosaic";
type OverviewChartRow = { key: string; count: number };
type OverviewActivityLane = {
  label: string;
  tone: "run" | "job" | "service";
  rows: OverviewChartRow[];
  total: number;
};
type OverviewChartSpec = {
  title: string;
  meta: string;
  rows: OverviewChartRow[];
  kind?: OverviewChartKind;
};

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
  const benchmarkTaskRows = countMany(data.benchmarks, (benchmark) => benchmark.tasks);
  const benchmarkLabelRows = countMany(data.benchmarks, (benchmark) => benchmark.labels);
  const coverageRows = runCoverageRows(data.runs);
  const sampleScaleRows = countBy(data.benchmarks, (benchmark) =>
    sampleScaleBucket(benchmark.sample_count)
  );
  const jobStatusRows = countBy(jobs, (job) => job.status || "unknown");
  const serviceStatusRows = countBy(services, (service) => service.status || "unknown");
  const jobTimelineRows = itemTimeline(jobs, 12, (job) => job.created_at);
  const sampleLabelWeightRows = sumMany(
    data.benchmarks,
    (benchmark) => benchmark.labels,
    (benchmark) => benchmark.sample_count
  );
  const schedulerRows = schedulerResourceRows(schedulerStatus);
  const timelineRows = runTimeline(data.runs, 12);
  const activityLanes = overviewActivityLanes(data.runs, jobs, services, 12);
  const notedRuns = data.runs.filter((run) => run.note.trim()).length;
  const evaluatedRuns = data.runs.filter((run) => run.report_path).length;
  const overviewCharts: OverviewChartSpec[] = [
    { title: "Run 生命周期", meta: "status", rows: statusRows, kind: "ring" },
    { title: "评测覆盖", meta: "report state", rows: coverageRows, kind: "meter" },
    { title: "Run 任务", meta: "spec_task", rows: taskRows, kind: "rails" },
    { title: "Run 日历", meta: "12d write", rows: timelineRows, kind: "spark" },
    { title: "Benchmark 任务", meta: "task set", rows: benchmarkTaskRows, kind: "rails" },
    { title: "Label footprint", meta: "benchmark labels", rows: benchmarkLabelRows, kind: "cells" },
    { title: "样本规模", meta: "sample buckets", rows: sampleScaleRows, kind: "ring" },
    { title: "样本/label", meta: "sample weight", rows: sampleLabelWeightRows, kind: "mosaic" },
    { title: "模型分布", meta: "model_id", rows: modelRows, kind: "cells" },
    { title: "Job 状态", meta: "queue state", rows: jobStatusRows, kind: "ring" },
    { title: "Job 日历", meta: "12d queue", rows: jobTimelineRows, kind: "spark" },
    { title: "Service 状态", meta: "runtime state", rows: serviceStatusRows, kind: "ring" },
    { title: "Scheduler 资源", meta: "resource slots", rows: schedulerRows, kind: "cells" }
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
        <OverviewActivityMatrix lanes={activityLanes} />
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
              kind={chart.kind}
            />
          ))}
          <OverviewRecentRunsPanel runs={data.runs.slice(0, 4)} />
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
  rows,
  kind = "ring"
}: {
  title: string;
  meta: string;
  rows: OverviewChartRow[];
  kind?: OverviewChartKind;
}) {
  const maxCount = Math.max(1, ...rows.map((row) => row.count));
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const visibleRows = overviewVisibleRows(kind, rows);
  const ringStyle = {
    "--overview-ring": overviewConicGradient(rows)
  } as React.CSSProperties;
  return (
    <div className={`workspace-card overview-chart-card compact ${kind}`}>
      <PanelTitle title={title} meta={meta} />
      <div className={`overview-mini-chart ${kind}`}>
        {kind === "rails" ? (
          <OverviewRailsChart rows={visibleRows} maxCount={maxCount} />
        ) : kind === "cells" ? (
          <OverviewCellsChart rows={visibleRows} maxCount={maxCount} />
        ) : kind === "meter" ? (
          <OverviewMeterChart rows={visibleRows} total={total} maxCount={maxCount} />
        ) : kind === "spark" ? (
          <OverviewSparkChart rows={visibleRows} total={total} maxCount={maxCount} />
        ) : kind === "mosaic" ? (
          <OverviewMosaicChart rows={visibleRows} maxCount={maxCount} />
        ) : (
          <OverviewRingChart
            rows={visibleRows}
            total={total}
            maxCount={maxCount}
            ringStyle={ringStyle}
            groupCount={rows.length}
          />
        )}
      </div>
    </div>
  );
}

function overviewVisibleRows(kind: OverviewChartKind, rows: OverviewChartRow[]) {
  if (kind === "spark" || kind === "mosaic") {
    return rows.slice(0, 12);
  }
  const positiveRows = rows.filter((row) => row.count > 0);
  return (positiveRows.length > 0 ? positiveRows : rows).slice(0, 3);
}

function OverviewRingChart({
  rows,
  total,
  maxCount,
  ringStyle,
  groupCount
}: {
  rows: OverviewChartRow[];
  total: number;
  maxCount: number;
  ringStyle: React.CSSProperties;
  groupCount: number;
}) {
  return (
    <>
      <div className="overview-chart-ring" style={ringStyle}>
        <span>{groupCount.toLocaleString()}</span>
        <strong>{total.toLocaleString()}</strong>
      </div>
      <OverviewBarList rows={rows} maxCount={maxCount} />
    </>
  );
}

function OverviewRailsChart({ rows, maxCount }: { rows: OverviewChartRow[]; maxCount: number }) {
  return (
    <>
      <div className="overview-rail-plot" aria-hidden="true">
        {railRows(rows).map((row, index) => (
          <span
            key={`${row.key}-${index}`}
            style={
              {
                "--overview-rail-height":
                  row.count > 0 ? `${Math.max(12, (row.count / maxCount) * 100)}%` : "6%",
                "--overview-bar-fill": overviewChartColor(index)
              } as React.CSSProperties
            }
          />
        ))}
      </div>
      <OverviewBarList rows={rows} maxCount={maxCount} />
    </>
  );
}

function OverviewCellsChart({ rows, maxCount }: { rows: OverviewChartRow[]; maxCount: number }) {
  return (
    <>
      <div className="overview-cell-grid" aria-hidden="true">
        {cellRows(rows).map((row, index) => (
          <span
            key={`${row.key}-${index}`}
            style={
              {
                opacity: row.count > 0 ? Math.max(0.34, row.count / maxCount) : 0.16,
                background: overviewChartColor(index)
              } as React.CSSProperties
            }
          />
        ))}
      </div>
      <OverviewBarList rows={rows} maxCount={maxCount} />
    </>
  );
}

function OverviewMeterChart({
  rows,
  total,
  maxCount
}: {
  rows: OverviewChartRow[];
  total: number;
  maxCount: number;
}) {
  return (
    <div className="overview-meter-chart">
      <div className="overview-stack-meter" aria-hidden="true">
        {rows.length === 0 || total <= 0 ? (
          <i style={{ width: "100%", background: "#d8e4ec" }} />
        ) : (
          rows.map((row, index) => (
            <i
              key={row.key}
              style={{
                width: `${Math.max(6, (row.count / total) * 100)}%`,
                background: overviewChartColor(index)
              }}
            />
          ))
        )}
      </div>
      <OverviewBarList rows={rows} maxCount={maxCount} />
    </div>
  );
}

function OverviewSparkChart({
  rows,
  total,
  maxCount
}: {
  rows: OverviewChartRow[];
  total: number;
  maxCount: number;
}) {
  const displayRows = rows.length > 0 ? rows : [{ key: "empty", count: 0 }];
  const peak = Math.max(0, ...displayRows.map((row) => row.count));
  return (
    <div className="overview-spark-chart">
      <div className="overview-spark-bars" aria-hidden="true">
        {displayRows.map((row, index) => (
          <span key={`${row.key}-${index}`} title={`${row.key}: ${row.count.toLocaleString()}`}>
            <i
              style={
                {
                  "--overview-spark-height":
                    row.count > 0 ? `${Math.max(12, (row.count / maxCount) * 100)}%` : "5%",
                  "--overview-bar-fill": overviewChartColor(index)
                } as React.CSSProperties
              }
            />
          </span>
        ))}
      </div>
      <div className="overview-spark-caption">
        <span>sum</span>
        <strong>{total.toLocaleString()}</strong>
        <span>peak</span>
        <strong>{peak.toLocaleString()}</strong>
      </div>
    </div>
  );
}

function OverviewMosaicChart({ rows, maxCount }: { rows: OverviewChartRow[]; maxCount: number }) {
  const displayRows = rows.length > 0 ? rows : [{ key: "empty", count: 0 }];
  const cells = Array.from({ length: 12 }, (_, index) => displayRows[index] ?? null);
  return (
    <div className="overview-mosaic-chart">
      <div className="overview-mosaic-grid" aria-hidden="true">
        {cells.map((row, index) => (
          <span
            key={row ? `${row.key}-${index}` : `empty-${index}`}
            title={row ? `${row.key}: ${row.count.toLocaleString()}` : "empty"}
            style={
              {
                opacity: row && row.count > 0 ? Math.max(0.28, row.count / maxCount) : 0.12,
                background: row ? overviewChartColor(index) : "#dce7ef"
              } as React.CSSProperties
            }
          />
        ))}
      </div>
      <OverviewBarList rows={displayRows.slice(0, 2)} maxCount={maxCount} />
    </div>
  );
}

function OverviewBarList({ rows, maxCount }: { rows: OverviewChartRow[]; maxCount: number }) {
  return (
    <div className="overview-bar-list">
      {rows.length === 0 ? (
        <div className="empty-inline">暂无数据</div>
      ) : (
        rows.map((row, index) => (
          <div className="overview-bar-row" key={row.key}>
            <span>{row.key}</span>
            <div>
              <i
                style={
                  {
                    width:
                      row.count > 0 ? `${Math.max(4, (row.count / maxCount) * 100)}%` : "0%",
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
  );
}

function railRows(rows: OverviewChartRow[]) {
  return rows.length > 0 ? rows : [{ key: "empty", count: 0 }];
}

function cellRows(rows: OverviewChartRow[]) {
  const fallback = rows.length > 0 ? rows : [{ key: "empty", count: 0 }];
  return Array.from({ length: 6 }, (_, index) => fallback[index % fallback.length]);
}

function overviewConicGradient(rows: OverviewChartRow[]) {
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

function OverviewActivityMatrix({ lanes }: { lanes: OverviewActivityLane[] }) {
  const maxCount = Math.max(
    1,
    ...lanes.flatMap((lane) => lane.rows.map((row) => row.count))
  );
  const total = lanes.reduce((sum, lane) => sum + lane.total, 0);
  const activeCells = lanes.reduce(
    (sum, lane) => sum + lane.rows.filter((row) => row.count > 0).length,
    0
  );
  const bucketCount = lanes[0]?.rows.length ?? 0;
  const latest = lanes[0]?.rows.at(-1);
  return (
    <div className="overview-rhythm-strip overview-activity-matrix">
      <div className="overview-rhythm-copy">
        <span>
          <BarChart3 size={14} />
          活动矩阵
        </span>
        <strong>{total.toLocaleString()} events / {bucketCount}d</strong>
      </div>
      <div className="overview-activity-lanes" aria-label="最近日期桶的 run job service 活动">
        {lanes.map((lane) => (
          <div className={`overview-activity-lane ${lane.tone}`} key={lane.label}>
            <span>{lane.label}</span>
            <div className="overview-activity-cells">
              {lane.rows.map((row) => (
                <i
                  key={row.key}
                  style={
                    {
                      opacity: row.count > 0 ? Math.max(0.32, row.count / maxCount) : 0.12
                    } as React.CSSProperties
                  }
                  title={`${lane.label} ${row.key}: ${row.count.toLocaleString()}`}
                />
              ))}
            </div>
            <strong>{lane.total.toLocaleString()}</strong>
          </div>
        ))}
      </div>
      <div className="overview-rhythm-meta">
        <span>{latest?.key ?? "-"}</span>
        <strong>
          active {activeCells}/{lanes.length * bucketCount}
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

function OverviewRecentRunsPanel({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="workspace-card overview-chart-card overview-recent-card">
      <PanelTitle title="最近 run" meta={`latest ${runs.length}`} />
      <OverviewRunList runs={runs} />
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

function sumBy<T>(
  items: T[],
  keyForItem: (item: T) => string,
  valueForItem: (item: T) => number
) {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyForItem(item).trim() || "unknown";
    const value = valueForItem(item);
    counts.set(key, (counts.get(key) ?? 0) + (Number.isFinite(value) ? Math.max(0, value) : 0));
  }
  return Array.from(counts.entries())
    .map(([key, count]) => ({ key, count }))
    .sort((left, right) => right.count - left.count || left.key.localeCompare(right.key));
}

function sumMany<T>(
  items: T[],
  keysForItem: (item: T) => string[],
  valueForItem: (item: T) => number
) {
  return sumBy(
    items.flatMap((item) => {
      const keys = keysForItem(item)
        .map((key) => key.trim())
        .filter(Boolean);
      const value = valueForItem(item);
      return (keys.length > 0 ? keys : ["unknown"]).map((key) => ({ key, value }));
    }),
    (item) => item.key,
    (item) => item.value
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

function overviewActivityLanes(
  runs: RunSummary[],
  jobs: JobSummary[],
  services: ServiceSummary[],
  bucketCount: number
): OverviewActivityLane[] {
  const keys = overviewTimelineKeys(
    latestActivityDate(runs, jobs, services) ?? new Date(),
    bucketCount
  );
  return [
    {
      label: "Runs",
      tone: "run",
      rows: timelineRowsForItems(runs, keys, (run) => run.created_at),
      total: runs.length
    },
    {
      label: "Jobs",
      tone: "job",
      rows: timelineRowsForItems(jobs, keys, (job) => job.created_at),
      total: jobs.length
    },
    {
      label: "Svc",
      tone: "service",
      rows: timelineRowsForItems(
        services,
        keys,
        (service) => service.updated_at ?? service.created_at
      ),
      total: services.length
    }
  ];
}

function itemTimeline<T>(
  items: T[],
  bucketCount: number,
  timestampForItem: (item: T) => string | null | undefined
) {
  const endDate = latestItemDate(items, timestampForItem) ?? new Date();
  const keys = overviewTimelineKeys(endDate, bucketCount);
  return timelineRowsForItems(items, keys, timestampForItem);
}

function timelineRowsForItems<T>(
  items: T[],
  keys: string[],
  timestampForItem: (item: T) => string | null | undefined
) {
  const counts = countBy(items, (item) => {
    const timestamp = timestampForItem(item);
    return timestamp ? timestamp.slice(0, 10) : "unknown";
  });
  const countMap = new Map(counts.map((row) => [row.key, row.count]));
  return keys.map((key) => ({ key, count: countMap.get(key) ?? 0 }));
}

function overviewTimelineKeys(endDate: Date, bucketCount: number) {
  return Array.from({ length: bucketCount }, (_, index) => {
    const date = new Date(endDate);
    date.setUTCDate(endDate.getUTCDate() - (bucketCount - 1 - index));
    return date.toISOString().slice(0, 10);
  });
}

function latestActivityDate(
  runs: RunSummary[],
  jobs: JobSummary[],
  services: ServiceSummary[]
) {
  const timestamps = [
    ...runs.map((run) => run.created_at),
    ...jobs.map((job) => job.created_at),
    ...services.map((service) => service.updated_at ?? service.created_at)
  ]
    .map((timestamp) => (timestamp ? Date.parse(timestamp) : Number.NaN))
    .filter(Number.isFinite);
  if (timestamps.length === 0) {
    return null;
  }
  return new Date(Math.max(...timestamps));
}

function runTimeline(runs: RunSummary[], bucketCount: number) {
  return itemTimeline(runs, bucketCount, (run) => run.created_at);
}

function latestItemDate<T>(
  items: T[],
  timestampForItem: (item: T) => string | null | undefined
) {
  const timestamps = items
    .map((item) => {
      const timestamp = timestampForItem(item);
      return timestamp ? Date.parse(timestamp) : Number.NaN;
    })
    .filter(Number.isFinite);
  if (timestamps.length === 0) {
    return null;
  }
  return new Date(Math.max(...timestamps));
}
