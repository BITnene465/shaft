import { PagerControl } from "./samplePager";
import { RANK_PAGE_SIZE } from "./rankBoardModel";
import { useRankBoardController } from "./rankBoardController";
import { RankBoardFilterBar } from "./rankBoardFilters";
import {
  RankBoardSummary,
  RankFacetRail,
  SuiteRankFacetRail,
  SuiteRankSummary
} from "./rankBoardFacets";
import { RankBoardTable, SuiteRankBoardTable } from "./rankBoardTables";
import { EmptyState, OptionChipButton } from "./ui";

import "./rankTheme.css";
import "./rankBoardPage.css";

export function RankBoardPage() {
  const controller = useRankBoardController();
  const {
    boardMode,
    setRunMode,
    setSuiteMode,
    runs,
    board,
    suiteBoard,
    entries,
    loading,
    errorTitle,
    filterValues,
    filterOptions,
    filterSetters,
    facetFilters,
    facetHandlers,
    sortBy,
    sortOrder,
    suiteSortBy,
    suiteSortOrder,
    tableRefreshing,
    suiteTableRefreshing,
    setPageOffset,
    handleSortChange,
    handleSuiteSortChange
  } = controller;

  if (loading) {
    return <EmptyState title="正在加载排行榜" />;
  }
  if (errorTitle) {
    return <EmptyState title={errorTitle} tone="danger" />;
  }

  return (
    <section className="page-stack density-page rank-board-page">
      <div className="rank-mode-switch" aria-label="排行榜模式">
        <OptionChipButton
          active={boardMode === "run"}
          title="单任务 run 排名"
          onClick={setRunMode}
        >
          Single task
        </OptionChipButton>
        <OptionChipButton
          active={boardMode === "suite"}
          title="Suite aggregate 排名"
          onClick={setSuiteMode}
        >
          Suite aggregate
        </OptionChipButton>
      </div>
      <div className="workspace-card fill rank-board-table-card">
        {boardMode === "suite" && suiteBoard ? (
          <>
            <div className="rank-board-table-toolbar">
              <SuiteRankSummary board={suiteBoard} />
              <PagerControl
                className="rank-board-pager"
                offset={suiteBoard.offset}
                limit={suiteBoard.limit}
                total={suiteBoard.total}
                onPageChange={setPageOffset}
              />
            </div>
            <SuiteRankBoardTable
              entries={suiteBoard.entries}
              sortBy={suiteSortBy}
              sortOrder={suiteSortOrder}
              onSortChange={handleSuiteSortChange}
              refreshing={suiteTableRefreshing}
            />
          </>
        ) : board ? (
          <>
            <div className="rank-board-table-toolbar">
              <RankBoardSummary board={board} runCount={runs.length} />
              <PagerControl
                className="rank-board-pager"
                offset={board.offset}
                limit={board.limit}
                total={board.total}
                onPageChange={setPageOffset}
              />
            </div>
            <RankBoardFilterBar
              values={filterValues}
              options={filterOptions}
              setters={filterSetters}
            />
            <RankBoardTable
              entries={entries}
              primaryMetric={board.primary_metric}
              sortBy={sortBy}
              sortOrder={sortOrder}
              onSortChange={handleSortChange}
              refreshing={tableRefreshing}
            />
          </>
        ) : null}
      </div>
      {boardMode === "run" && board ? (
        <RankFacetRail
          board={board}
          filters={facetFilters}
          onFilterChange={facetHandlers}
        />
      ) : suiteBoard ? (
        <SuiteRankFacetRail board={suiteBoard} />
      ) : null}
    </section>
  );
}
