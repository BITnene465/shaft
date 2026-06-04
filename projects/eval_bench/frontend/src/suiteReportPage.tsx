import { useCompositeReportController } from "./compositeReportController";
import { CompositeReportShell } from "./compositeReportShell";

import "./compositeReport.css";

export function SuiteReportPage() {
  const report = useCompositeReportController();

  return (
    <section className="page-stack composite-report-page">
      <CompositeReportShell report={report} />
    </section>
  );
}
