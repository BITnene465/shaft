export const APP_ICON_PATHS = {
  appMark: "/icons/eval-bench/app-mark.png",
  overview: "/icons/eval-bench/overview.png",
  benchmark: "/icons/eval-bench/benchmark.png",
  service: "/icons/eval-bench/service.png",
  evalJob: "/icons/eval-bench/eval-job.png",
  runResults: "/icons/eval-bench/run-results.png",
  compareAnalysis: "/icons/eval-bench/compare-analysis.png",
  workspaceSettings: "/icons/eval-bench/workspace-settings.png",
  createEval: "/icons/eval-bench/create-eval.png",
  createBenchmark: "/icons/eval-bench/create-benchmark.png",
  importPrediction: "/icons/eval-bench/import-prediction.png",
  registerService: "/icons/eval-bench/register-service.png",
  samples: "/icons/eval-bench/samples.png",
  predictions: "/icons/eval-bench/predictions.png",
  metrics: "/icons/eval-bench/metrics.png",
  diagnostics: "/icons/eval-bench/diagnostics.png",
  restoreTemplate: "/icons/eval-bench/restore-template.png",
  applyPrompt: "/icons/eval-bench/apply-prompt.png",
  preflightValidate: "/icons/eval-bench/preflight-validate.png",
  enqueueJob: "/icons/eval-bench/enqueue-job.png",
  submitCreate: "/icons/eval-bench/submit-create.png",
  saveService: "/icons/eval-bench/save-service.png",
  resetSettings: "/icons/eval-bench/reset-settings.png",
  clearRules: "/icons/eval-bench/clear-rules.png"
} as const;

export type AppIconName = keyof typeof APP_ICON_PATHS;

export function AppIcon({
  name,
  size = 20,
  className = "",
  alt = ""
}: {
  name: AppIconName;
  size?: number;
  className?: string;
  alt?: string;
}) {
  return (
    <img
      className={className ? `app-icon ${className}` : "app-icon"}
      src={APP_ICON_PATHS[name]}
      alt={alt}
      aria-hidden={alt ? undefined : true}
      draggable={false}
      style={{ width: size, height: size }}
    />
  );
}
