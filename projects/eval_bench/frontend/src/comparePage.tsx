import { Link } from "@tanstack/react-router";

import { useCompareController } from "./compareController";
import { CompareFilterBar } from "./compareFilters";
import { ComparisonPanel } from "./compareReportComponents";
import { ComparisonHistoryPanel, RunSelectRail } from "./compareRunRailComponents";
import { compactIdentifier, errorMessage } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { PagerControl } from "./samplePager";
import { EmptyState, InlineNavLink } from "./ui";
import { ResizableSplit } from "./workspaceLayout";

import "./comparePage.css";
import "./compareTheme.css";
import "./compareRunRail.css";
import "./compareReportPanel.css";
import "./comparisonSampleStyles.css";

export function ComparePage() {
  const controller = useCompareController();
  const {
    runs,
    comparableRuns,
    filteredCount,
    runPageOffset,
    runPageLimit,
    comparisonHistoryTotal,
    comparisonHistoryOffset,
    comparisonHistoryLimit,
    hasComparisonHistoryFilters,
    effectiveBaseline,
    effectiveCandidate,
    comparisonReport,
    comparisonReportRefreshing,
    comparisonList,
    runsRefreshing,
    comparisonHistoryRefreshing,
    runsLoading,
    runsErrorTitle,
    comparisonLoading,
    comparisonError,
    comparisonIsError,
    filterValues,
    filterOptions,
    filterSetters,
    activeLabel,
    setActiveLabel,
    setBaselineRunId,
    setCandidateRunId,
    setPageOffset,
    setHistoryOffset
  } = controller;

  if (runsLoading) {
    return <EmptyState title="正在加载对比状态" />;
  }
  if (runsErrorTitle) {
    return <EmptyState title={runsErrorTitle} tone="danger" />;
  }

  return (
    <section className="page-stack compare-page">
      <div className="compare-topbar">
        <div className="compare-title">
          <span>对比工作区</span>
          <strong>{filteredCount.toLocaleString()} 条 run</strong>
        </div>
        <InlineNavLink className="compare-ready" icon={<AppIcon name="rankBoard" size={13} />} to="/rank-board">
          排行榜
        </InlineNavLink>
      </div>
      <CompareFilterBar values={filterValues} options={filterOptions} setters={filterSetters} />
      <ResizableSplit
        className="compare-workspace"
        storageKey="eval_bench_compare_rail_width"
        defaultSize={292}
        minSize={180}
        maxSize={680}
        first={
          <aside className="compare-run-rail">
            {runsRefreshing ? (
              <div className="viewer-fetch-chip">正在更新 run 列表</div>
            ) : null}
            <RunSelectRail
              title="基线"
              value={effectiveBaseline}
              runs={comparableRuns}
              disabled={filteredCount < 2 && !effectiveBaseline}
              onChange={setBaselineRunId}
            />
            <RunSelectRail
              title="候选"
              value={effectiveCandidate}
              runs={comparableRuns}
              disabled={filteredCount < 2 && !effectiveCandidate}
              onChange={setCandidateRunId}
            />
            <PagerControl
              className="rank-board-pager compare-run-pager"
              offset={runPageOffset}
              limit={runPageLimit}
              total={filteredCount}
              meta={
                <>
                  {" · "}
                  {runs.length.toLocaleString()} visible / {comparableRuns.length.toLocaleString()} reports
                </>
              }
              onPageChange={setPageOffset}
            />
            <ComparisonHistoryPanel
              comparisons={comparisonList}
              total={comparisonHistoryTotal}
              offset={comparisonHistoryOffset}
              limit={comparisonHistoryLimit}
              active={hasComparisonHistoryFilters}
              refreshing={comparisonHistoryRefreshing}
              onPageChange={setHistoryOffset}
            />
          </aside>
        }
        second={
          <ResizableSplit
            className="compare-main-split"
            storageKey="eval_bench_compare_context_width"
            fixedPane="second"
            defaultSize={340}
            minSize={260}
            maxSize={620}
            first={
              <main className="compare-report-pane">
                {!effectiveBaseline || !effectiveCandidate ? (
                  <div className="empty-panel">至少需要两个已完成评测的 run 才能对比。</div>
                ) : effectiveBaseline === effectiveCandidate ? (
                  <div className="empty-panel">请选择两个不同的 run。</div>
                ) : comparisonReport ? (
                  <>
                    {comparisonReportRefreshing ? (
                      <div className="viewer-fetch-chip">正在切换对比报告</div>
                    ) : null}
                    <ComparisonPanel
                      report={comparisonReport}
                      activeLabel={activeLabel}
                      onActiveLabelChange={setActiveLabel}
                    />
                  </>
                ) : comparisonLoading ? (
                  <div className="empty-panel">正在加载对比报告</div>
                ) : comparisonIsError ? (
                  <div className="empty-panel danger-text">
                    对比报告加载失败：{errorMessage(comparisonError)}
                  </div>
                ) : null}
              </main>
            }
            second={
              <aside className="compare-context-pane">
                <CompareContextPanel
                  filteredCount={filteredCount}
                  comparableCount={comparableRuns.length}
                  baselineRunId={effectiveBaseline}
                  candidateRunId={effectiveCandidate}
                />
              </aside>
            }
          />
        }
      />
    </section>
  );
}

function CompareContextPanel({
  filteredCount,
  comparableCount,
  baselineRunId,
  candidateRunId
}: {
  filteredCount: number;
  comparableCount: number;
  baselineRunId: string;
  candidateRunId: string;
}) {
  return (
    <div className="compare-context-stack">
      <div className="compare-context-grid">
        <div className="compare-context-card">
          <span>Runs</span>
          <strong>{filteredCount.toLocaleString()}</strong>
        </div>
        <div className="compare-context-card">
          <span>Reports</span>
          <strong>{comparableCount.toLocaleString()}</strong>
        </div>
      </div>
      <div className="compare-context-links">
        {baselineRunId ? (
          <Link to="/runs/$runId" params={{ runId: baselineRunId }} title={baselineRunId}>
            <span>Base</span>
            <strong>{compactIdentifier(baselineRunId, 16, 8)}</strong>
          </Link>
        ) : null}
        {candidateRunId ? (
          <Link to="/runs/$runId" params={{ runId: candidateRunId }} title={candidateRunId}>
            <span>Cand</span>
            <strong>{compactIdentifier(candidateRunId, 16, 8)}</strong>
          </Link>
        ) : null}
      </div>
      <InlineNavLink
        className="compare-ready compare-rank-link"
        icon={<AppIcon name="rankBoard" size={13} />}
        to="/rank-board"
      >
        排行榜
      </InlineNavLink>
    </div>
  );
}
