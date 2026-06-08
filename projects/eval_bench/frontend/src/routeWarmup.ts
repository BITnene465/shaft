import { shouldAvoidSpeculativeNetworkWork } from "./networkHints";

const ROUTE_WARMUP_TIMEOUT_MS = 2_500;
const ROUTE_WARMUP_FALLBACK_DELAY_MS = 1_200;
const ROUTE_WARMUP_BATCH_SIZE = 3;
const ROUTE_WARMUP_BATCH_GAP_MS = 120;

type IdleWindow = Window &
  typeof globalThis & {
    requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number;
    cancelIdleCallback?: (handle: number) => void;
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
let routeWarmupInProgress = false;
let routeWarmupSubscribers = 0;

export function warmupEvalBenchRoutes() {
  if (routeWarmupCompleted || shouldSkipRouteWarmup()) {
    return () => undefined;
  }
  routeWarmupSubscribers += 1;
  if (routeWarmupInProgress) {
    return releaseRouteWarmupSubscriber;
  }
  const idleWindow = window as IdleWindow;
  const runWarmup = () => {
    if (!hasRouteWarmupSubscribers() || routeWarmupCompleted || routeWarmupInProgress) {
      return;
    }
    routeWarmupInProgress = true;
    void warmupRouteModules(hasRouteWarmupSubscribers).then((completed) => {
      routeWarmupCompleted = completed;
      routeWarmupInProgress = false;
    });
  };
  if (idleWindow.requestIdleCallback) {
    const idleHandle = idleWindow.requestIdleCallback(runWarmup, {
      timeout: ROUTE_WARMUP_TIMEOUT_MS
    });
    return () => {
      releaseRouteWarmupSubscriber();
      idleWindow.cancelIdleCallback?.(idleHandle);
    };
  }
  const timeoutHandle = window.setTimeout(runWarmup, ROUTE_WARMUP_FALLBACK_DELAY_MS);
  return () => {
    releaseRouteWarmupSubscriber();
    window.clearTimeout(timeoutHandle);
  };
}

async function warmupRouteModules(shouldContinue: () => boolean) {
  for (let index = 0; index < WARMUP_CORE_ROUTE_MODULES.length; index += ROUTE_WARMUP_BATCH_SIZE) {
    if (!shouldContinue()) {
      return false;
    }
    const routeModuleBatch = WARMUP_CORE_ROUTE_MODULES.slice(
      index,
      index + ROUTE_WARMUP_BATCH_SIZE
    );
    await Promise.allSettled(routeModuleBatch.map((loadRouteModule) => loadRouteModule()));
    if (!shouldContinue()) {
      return false;
    }
    if (index + ROUTE_WARMUP_BATCH_SIZE < WARMUP_CORE_ROUTE_MODULES.length) {
      await waitForWarmupBatchGap();
    }
  }
  return true;
}

function shouldSkipRouteWarmup() {
  return shouldAvoidSpeculativeNetworkWork();
}

function hasRouteWarmupSubscribers() {
  return routeWarmupSubscribers > 0;
}

function releaseRouteWarmupSubscriber() {
  routeWarmupSubscribers = Math.max(0, routeWarmupSubscribers - 1);
}

function waitForWarmupBatchGap() {
  const idleWindow = window as IdleWindow;
  return new Promise((resolve) => {
    if (idleWindow.requestIdleCallback) {
      idleWindow.requestIdleCallback(() => resolve(undefined), {
        timeout: ROUTE_WARMUP_BATCH_GAP_MS
      });
      return;
    }
    window.setTimeout(resolve, ROUTE_WARMUP_BATCH_GAP_MS);
  });
}
