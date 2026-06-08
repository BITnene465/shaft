const ROUTE_WARMUP_TIMEOUT_MS = 2_500;
const ROUTE_WARMUP_FALLBACK_DELAY_MS = 1_200;

type IdleWindow = Window &
  typeof globalThis & {
    requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number;
    cancelIdleCallback?: (handle: number) => void;
  };

type SaveDataNavigator = Navigator & {
  connection?: {
    saveData?: boolean;
  };
};

const WARMUP_CORE_ROUTE_MODULES = [
  () => import("./benchmarksPage"),
  () => import("./rankBoardPage"),
  () => import("./runsPage"),
  () => import("./jobsPage"),
  () => import("./suiteReportPage"),
  () => import("./comparePage"),
  () => import("./comparisonSamplePage"),
  () => import("./servicesPage"),
  () => import("./settingsPage")
];

let routeWarmupCompleted = false;

export function warmupEvalBenchRoutes() {
  if (routeWarmupCompleted || shouldSkipRouteWarmup()) {
    return () => undefined;
  }
  const idleWindow = window as IdleWindow;
  let cancelled = false;
  const runWarmup = () => {
    if (cancelled || routeWarmupCompleted) {
      return;
    }
    routeWarmupCompleted = true;
    void warmupRouteModules();
  };
  if (idleWindow.requestIdleCallback) {
    const idleHandle = idleWindow.requestIdleCallback(runWarmup, {
      timeout: ROUTE_WARMUP_TIMEOUT_MS
    });
    return () => {
      cancelled = true;
      idleWindow.cancelIdleCallback?.(idleHandle);
    };
  }
  const timeoutHandle = window.setTimeout(runWarmup, ROUTE_WARMUP_FALLBACK_DELAY_MS);
  return () => {
    cancelled = true;
    window.clearTimeout(timeoutHandle);
  };
}

async function warmupRouteModules() {
  await Promise.allSettled(WARMUP_CORE_ROUTE_MODULES.map((loadRouteModule) => loadRouteModule()));
}

function shouldSkipRouteWarmup() {
  return Boolean((navigator as SaveDataNavigator).connection?.saveData);
}
