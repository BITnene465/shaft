import { CompositeReportCommandBar } from "./compositeReportCommandBar";
import { useCompositeReportController } from "./compositeReportController";
import { CompositeReportShell } from "./compositeReportShell";

import "./compositeReport.css";

export function SuiteReportPage() {
  const report = useCompositeReportController();

  return (
    <section className="page-stack composite-report-page">
      <CompositeReportCommandBar
        activeSlots={report.activeSlots.length}
        readyLayerCount={report.readyLayerCount}
        missingLayerCount={report.missingLayerCount}
        composite={report.compositeQuery.data}
        enabled={report.compositeEnabled}
        error={report.stateQuery.error || report.compositeQuery.error}
        stageMode={report.stageMode}
        onStageModeChange={report.setStageMode}
      />

      <CompositeReportShell report={report} />
    </section>
  );
}
