export function recordViewerRenderMetric(name: string) {
  if (typeof window === "undefined" || !window.location.search.includes("perf=1")) {
    return;
  }
  const target = window as Window & { __evalBenchRenderMetrics?: Record<string, number> };
  target.__evalBenchRenderMetrics ??= {};
  target.__evalBenchRenderMetrics[name] = (target.__evalBenchRenderMetrics[name] ?? 0) + 1;
}
