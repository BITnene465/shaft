import React from "react";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Database,
  FileCheck2,
  Layers3,
  PlayCircle,
  Server,
  Trophy
} from "lucide-react";

import type { RunSummary } from "./api";
import { errorMessage, formatMetric, runF1Score } from "./formatters";
import type { BestRun, OverviewAction, OverviewActionIcon, OverviewLens, OverviewRoute } from "./overviewModel";
import { overviewLensLabel, overviewSignalNodes, trackPercent, useOverviewModel } from "./overviewModel";
import { runAgeLabel, runArtifactReadiness } from "./runArtifactSignals";
import { Badge, EmptyState, OptionChipButton } from "./ui";

import "./overviewPage.css";

export function OverviewPage() {
  const overview = useOverviewModel();

  if (overview.isLoading) {
    return <EmptyState title="正在加载看板状态" />;
  }
  if (overview.error || !overview.data) {
    return <EmptyState title={`看板状态加载失败：${errorMessage(overview.error)}`} tone="danger" />;
  }

  return (
    <section className="page-stack dashboard-home overview-home-v18">
      <div className="overview-v18-grid">
        <OverviewPrimaryCard
          action={overview.nextAction}
          bestRun={overview.bestRun}
          reportCoverage={overview.reportCoverage}
          benchmarkCount={overview.data.benchmark_count}
          benchmarkSamples={overview.data.total_benchmark_samples}
          predictionCount={overview.data.prediction_count}
          evaluatedRuns={overview.evaluatedRuns}
          queuedJobs={overview.queuedJobs}
          runningJobs={overview.runningJobs}
          failedJobs={overview.failedJobs}
          activeQueue={overview.activeQueue}
          liveServices={overview.liveServices}
          serviceCount={overview.serviceCount}
          schedulerLive={overview.schedulerLive}
        />
        <OverviewQueueCard
          activeQueue={overview.activeQueue}
          queuedJobs={overview.queuedJobs}
          runningJobs={overview.runningJobs}
          failedJobs={overview.failedJobs}
          totalJobs={overview.totalJobs}
        />
        <OverviewRecentRunsPanel runs={overview.recentRuns} />
        <OverviewResourceCard
          liveServices={overview.liveServices}
          serviceCount={overview.serviceCount}
          workerCount={overview.workerCount}
          reservedDevices={overview.reservedDevices}
          reservedPorts={overview.reservedPorts}
          schedulerEnabled={overview.schedulerEnabled}
          schedulerLive={overview.schedulerLive}
        />
      </div>
    </section>
  );
}

function OverviewPrimaryCard({
  action,
  bestRun,
  reportCoverage,
  benchmarkCount,
  benchmarkSamples,
  predictionCount,
  evaluatedRuns,
  queuedJobs,
  runningJobs,
  failedJobs,
  activeQueue,
  liveServices,
  serviceCount,
  schedulerLive
}: {
  action: OverviewAction;
  bestRun: BestRun | null;
  reportCoverage: number;
  benchmarkCount: number;
  benchmarkSamples: number;
  predictionCount: number;
  evaluatedRuns: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  activeQueue: number;
  liveServices: number;
  serviceCount: number;
  schedulerLive: boolean;
}) {
  const actionLine = bestRun
    ? `当前最佳 ${bestRun.run.run_id}，F1 ${formatMetric(bestRun.f1)}。`
    : "还没有可用 F1 报告。";
  const [lens, setLens] = React.useState<OverviewLens>("health");
  const signalNodes = overviewSignalNodes({
    lens,
    action,
    bestRun,
    benchmarkCount,
    benchmarkSamples,
    predictionCount,
    evaluatedRuns,
    reportCoverage,
    queuedJobs,
    runningJobs,
    failedJobs,
    activeQueue,
    liveServices,
    serviceCount,
    schedulerLive
  });
  const [selectedSignalId, setSelectedSignalId] = React.useState(signalNodes[0]?.id ?? "");
  const selectedSignal = signalNodes.find((node) => node.id === selectedSignalId) ?? signalNodes[0];
  React.useEffect(() => {
    if (!signalNodes.some((node) => node.id === selectedSignalId)) {
      setSelectedSignalId(signalNodes[0]?.id ?? "");
    }
  }, [signalNodes, selectedSignalId]);
  return (
    <section className={`overview-v18-card overview-v18-primary ${action.tone}`}>
      <div className="overview-v18-card-head">
        <div>
          <span>运营态势</span>
          <h3>{action.label}</h3>
        </div>
        <Link to={action.to} className="overview-v18-icon-link" aria-label={action.label}>
          {overviewActionIcon(action.icon)}
          <ArrowRight size={15} />
        </Link>
      </div>
      <div className="overview-v18-primary-body">
        <div className="overview-v18-action-copy">
          <strong>{action.value}</strong>
          <p>{action.detail}</p>
          <small>{actionLine}</small>
        </div>
        <Link
          to={bestRun ? "/rank-board" : "/runs"}
          className="overview-v18-score"
          style={{ "--overview-score": `${bestRun ? Math.round(bestRun.f1 * 100) : 0}%` } as React.CSSProperties}
        >
          <span>
            <b>{bestRun ? formatMetric(bestRun.f1) : "-"}</b>
            <em>F1</em>
          </span>
          <i>
            <b />
          </i>
        </Link>
      </div>
      <div className="overview-v18-console" aria-label="首页控制台">
        <div className="overview-v18-console-head">
          <span>全局信号</span>
          <div role="group" aria-label="切换总览视角">
            {(["health", "quality", "throughput"] as OverviewLens[]).map((mode) => (
              <OptionChipButton
                key={mode}
                active={lens === mode}
                className="overview-v18-surface-tab"
                onClick={() => setLens(mode)}
              >
                {overviewLensLabel(mode)}
              </OptionChipButton>
            ))}
          </div>
        </div>
        <div className="overview-v18-surface-body">
          <div className="overview-v18-signal-map" role="list" aria-label={`${overviewLensLabel(lens)}状态地图`}>
            {signalNodes.map((node, index) => (
              <OptionChipButton
                key={node.id}
                active={selectedSignal?.id === node.id}
                className={`overview-v18-signal-node ${node.tone}`}
                onClick={() => setSelectedSignalId(node.id)}
              >
                <span>{node.label}</span>
                <strong>{node.value}</strong>
                <em>{node.caption}</em>
                {index < signalNodes.length - 1 ? <i aria-hidden="true" /> : null}
              </OptionChipButton>
            ))}
          </div>
          <div className={`overview-v18-signal-inspector ${selectedSignal?.tone ?? "idle"}`}>
            <span>{selectedSignal?.label ?? "总览"}</span>
            <strong>{selectedSignal?.value ?? "-"}</strong>
            <p>{selectedSignal?.detail ?? "暂无可展示的全局信号。"}</p>
            <Link to={selectedSignal?.to ?? action.to}>
              打开
              <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </div>
      <div className="overview-v18-flow" aria-label="评测闭环">
        <OverviewFlowItem
          icon={<Database size={16} />}
          label="Benchmarks"
          value={benchmarkCount.toLocaleString()}
          detail={`${benchmarkSamples.toLocaleString()} samples`}
          to="/benchmarks"
        />
        <OverviewFlowItem
          icon={<Layers3 size={16} />}
          label="Predictions"
          value={predictionCount.toLocaleString()}
          detail="prediction snapshots"
          to="/runs"
        />
        <OverviewFlowItem
          icon={<FileCheck2 size={16} />}
          label="Reports"
          value={evaluatedRuns.toLocaleString()}
          detail={`${reportCoverage}% coverage`}
          to="/rank-board"
        />
      </div>
    </section>
  );
}

function OverviewFlowItem({
  icon,
  label,
  value,
  detail,
  to
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  to: OverviewRoute;
}) {
  return (
    <Link className="overview-v18-flow-item" to={to}>
      <span>{icon}</span>
      <strong>{value}</strong>
      <em>{label}</em>
      <small>{detail}</small>
    </Link>
  );
}

function OverviewQueueCard({
  activeQueue,
  queuedJobs,
  runningJobs,
  failedJobs,
  totalJobs
}: {
  activeQueue: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  totalJobs: number;
}) {
  return (
    <section className={failedJobs > 0 ? "overview-v18-card overview-v18-queue danger" : "overview-v18-card overview-v18-queue"}>
      <div className="overview-v18-card-head">
        <div>
          <span>Queue</span>
          <h3>任务队列</h3>
        </div>
        <Link to="/jobs" className="overview-v18-icon-link" aria-label="打开任务队列">
          <ClipboardList size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      <strong className="overview-v18-big-number">{activeQueue.toLocaleString()}</strong>
      <div className="overview-v18-meter" style={{ "--meter": `${trackPercent(activeQueue, Math.max(totalJobs, 1))}%` } as React.CSSProperties}>
        <b />
      </div>
      <dl className="overview-v18-pairs">
        <div>
          <dt>running</dt>
          <dd>{runningJobs.toLocaleString()}</dd>
        </div>
        <div>
          <dt>queued</dt>
          <dd>{queuedJobs.toLocaleString()}</dd>
        </div>
        <div>
          <dt>failed</dt>
          <dd>{failedJobs.toLocaleString()}</dd>
        </div>
      </dl>
    </section>
  );
}

function OverviewRecentRunsPanel({ runs }: { runs: RunSummary[] }) {
  return (
    <section className="overview-v18-card overview-v18-recent">
      <div className="overview-v18-card-head">
        <div>
          <span>Latest Runs</span>
          <h3>最近 run</h3>
        </div>
        <Link to="/runs" className="overview-v18-icon-link" aria-label="打开 run 列表">
          <Activity size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      {runs.length === 0 ? <div className="empty-inline">暂无 run。</div> : <OverviewRunList runs={runs} />}
    </section>
  );
}

function OverviewRunList({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="overview-v18-run-list">
      {runs.map((run) => {
        const readiness = runArtifactReadiness(run);
        const f1 = runF1Score(run);
        return (
          <Link key={run.run_id} to="/runs/$runId" params={{ runId: run.run_id }} className={readiness.tone}>
            <span className="overview-v18-run-id">
              <strong className="run-id-text">{run.run_id}</strong>
              <small>{run.model_id || "-"}</small>
            </span>
            <span className="overview-v18-run-artifacts">
              <i style={{ "--run-readiness": `${readiness.percent}%` } as React.CSSProperties}>
                <b />
              </i>
              <small>{readiness.label}</small>
            </span>
            <span className="overview-v18-run-score">
              <b>{f1 === null ? "F1 -" : `F1 ${formatMetric(f1)}`}</b>
              <small>{runAgeLabel(run.created_at)}</small>
            </span>
            <Badge value={run.status} domain="run" />
          </Link>
        );
      })}
    </div>
  );
}

function OverviewResourceCard({
  liveServices,
  serviceCount,
  workerCount,
  reservedDevices,
  reservedPorts,
  schedulerEnabled,
  schedulerLive
}: {
  liveServices: number;
  serviceCount: number;
  workerCount: number;
  reservedDevices: number;
  reservedPorts: number;
  schedulerEnabled: boolean;
  schedulerLive: boolean;
}) {
  return (
    <section className="overview-v18-card overview-v18-resources">
      <div className="overview-v18-card-head">
        <div>
          <span>Runtime</span>
          <h3>服务与资源</h3>
        </div>
        <Link to="/services" className="overview-v18-icon-link" aria-label="打开服务管理">
          <Server size={17} />
          <ArrowRight size={15} />
        </Link>
      </div>
      <div className="overview-v18-service-line">
        <strong>{liveServices}/{serviceCount}</strong>
        <span>services running</span>
      </div>
      <dl className="overview-v18-pairs">
        <div>
          <dt>workers</dt>
          <dd>{workerCount}</dd>
        </div>
        <div>
          <dt>devices</dt>
          <dd>{reservedDevices}</dd>
        </div>
        <div>
          <dt>ports</dt>
          <dd>{reservedPorts}</dd>
        </div>
      </dl>
      <Link to="/jobs" className={schedulerLive ? "overview-v18-scheduler live" : "overview-v18-scheduler"}>
        {schedulerLive ? <CheckCircle2 size={15} /> : <PlayCircle size={15} />}
        <span>{schedulerLive ? "scheduler live" : schedulerEnabled ? "scheduler idle" : "scheduler off"}</span>
      </Link>
    </section>
  );
}

function overviewActionIcon(icon: OverviewActionIcon) {
  if (icon === "alert") {
    return <AlertTriangle size={17} />;
  }
  if (icon === "report") {
    return <FileCheck2 size={17} />;
  }
  if (icon === "activity") {
    return <Activity size={17} />;
  }
  if (icon === "trophy") {
    return <Trophy size={17} />;
  }
  if (icon === "server") {
    return <Server size={17} />;
  }
  return <PlayCircle size={17} />;
}
